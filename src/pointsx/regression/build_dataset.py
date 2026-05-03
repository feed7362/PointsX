"""Build the training .npz for the CircumferenceRegressor.

For every `body_XXXXX` in the synthetic dataset:
  1. Load front + side photos from the pose dataset
  2. Run YOLO-pose + YOLO-seg → keypoints + silhouette masks
  3. Run `MeasurementPipeline` width-extraction pipeline (pose + seg + calibration)
  4. Build 28-dim feature vector via `regression.features.build_feature_vector`
  5. Load ground-truth circumferences from `<pose-root>/measurements/body_XXXXX.json`
  6. Save (features, targets) to a single .npz ready for `regression/train.py`

Input layout (produced by `synthetic/pipeline.py --mode both`):
    <pose-root>/
        train/images/s00001_front.jpg    (one of train/val, either split works)
        val/images/s00001_side.jpg
        measurements/body_00001.json
        manifest.json

Usage:
    python -m pointsx.regression.build_dataset \
        --pose-root  data/synthetic-pose \
        --pose-model models/yolo11n-pose.pt \
        --seg-model  models/yolo11n-seg.pt \
        --output     data/regression_features.npz
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from pointsx.calibration import calibrate
from pointsx.measurements import extract_measurements
from pointsx.models import BodyModels
from pointsx.regression.features import build_feature_vector
from pointsx.schemas import Keypoints

logger = logging.getLogger(__name__)

# Target vector order — must match CircumferenceRegressor output layout
TARGET_FIELDS = [
    "neck_circumference_cm",
    "waist_circumference_cm",
    "hips_circumference_cm",
    "thigh_circumference_cm",
    "calf_circumference_cm",
    "wrist_circumference_cm",
]


def _pair_images(pose_root: Path) -> dict[int, dict[str, Path]]:
    """Group image paths by body_id → {"front": path, "side": path}.

    Images live under train/images/ OR val/images/ depending on the random split
    at render time; we accept either location.
    """
    pairs: dict[int, dict[str, Path]] = defaultdict(dict)
    for split in ("train", "val"):
        for img in (pose_root / split / "images").glob("*.jpg"):
            # Filename format: s00001_front.jpg or s00001_side.jpg
            stem = img.stem
            try:
                body_part, view = stem.rsplit("_", 1)
                body_id = int(body_part.lstrip("s"))
            except (ValueError, AttributeError):
                continue
            if view in ("front", "side"):
                pairs[body_id][view] = img
    return pairs


def _reference_point(kp: Keypoints) -> tuple[float, float]:
    """Estimate subject center for seg-mask selection."""
    valid = kp.confidence >= 0.3
    if np.any(valid):
        center = kp.points[valid].mean(axis=0)
    else:
        center = kp.points.mean(axis=0)
    return float(center[0]), float(center[1])


def build_dataset(
    pose_root: Path,
    pose_model: Path,
    seg_model: Path,
    output: Path,
    img_size: int = 640,
    device: str = "auto",
    limit: int | None = None,
) -> tuple[int, int]:
    """Walk the synthetic dataset, run inference, build (features, targets).

    Returns (n_written, n_skipped).
    """
    models = BodyModels(
        pose_model_path=pose_model,
        seg_model_path=seg_model,
        img_size=img_size,
        device=device,
    )

    meas_dir = pose_root / "measurements"
    if not meas_dir.exists():
        raise FileNotFoundError(f"Missing ground-truth measurements dir: {meas_dir}")

    pairs = _pair_images(pose_root)
    logger.info("Found %d bodies with at least one view.", len(pairs))

    body_ids = sorted(pairs.keys())
    if limit is not None:
        body_ids = body_ids[:limit]

    feats, targets = [], []
    skipped_no_pair = skipped_no_gt = skipped_no_detect = skipped_features = 0
    first_errors: list[str] = []  # keep first few error messages for quick diagnosis

    def _remember(msg: str) -> None:
        if len(first_errors) < 5:
            first_errors.append(msg)

    with Progress(
        TextColumn("[bold blue]Building dataset"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TextColumn("eta"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("build", total=len(body_ids))

        for body_id in body_ids:
            progress.update(task, advance=1)
            views = pairs[body_id]

            if "front" not in views or "side" not in views:
                skipped_no_pair += 1
                _remember(f"body {body_id}: missing view(s), have {list(views.keys())}")
                continue

            gt_path = meas_dir / f"body_{body_id:05d}.json"
            if not gt_path.exists():
                skipped_no_gt += 1
                _remember(f"body {body_id}: ground-truth JSON not found at {gt_path}")
                continue
            gt = json.loads(gt_path.read_text())

            # Run pose + seg on both views
            try:
                import cv2
                front_img = cv2.imread(str(views["front"]))
                side_img  = cv2.imread(str(views["side"]))
                if front_img is None or side_img is None:
                    skipped_no_detect += 1
                    _remember(f"body {body_id}: cv2.imread returned None")
                    continue

                front_kp = models.predict_pose(front_img, view="front")
                side_kp  = models.predict_pose(side_img,  view="side")
                if front_kp is None or side_kp is None:
                    skipped_no_detect += 1
                    _remember(
                        f"body {body_id}: pose detection returned None "
                        f"(front={front_kp is not None}, side={side_kp is not None})"
                    )
                    continue

                front_mask = models.predict_segmentation(
                    front_img, view="front", reference_point=_reference_point(front_kp)
                )
                side_mask = models.predict_segmentation(
                    side_img, view="side", reference_point=_reference_point(side_kp)
                )
                if front_mask is None or side_mask is None:
                    skipped_no_detect += 1
                    _remember(
                        f"body {body_id}: seg returned None "
                        f"(front={front_mask is not None}, side={side_mask is not None})"
                    )
                    continue
            except Exception as exc:
                skipped_no_detect += 1
                _remember(f"body {body_id}: inference exception: {exc!r}")
                continue

            # Calibrate using the ground-truth height (we know it exactly)
            height_cm = float(gt.get("actual_height_cm") or gt.get("height_cm") or 0.0)
            if height_cm <= 0:
                skipped_no_gt += 1
                _remember(f"body {body_id}: no height in GT JSON (keys={list(gt.keys())[:8]}…)")
                continue

            try:
                cal = calibrate(front_kp, side_kp, height_cm)
                measurements = extract_measurements(
                    front_kp, side_kp, front_mask, side_mask, cal
                )
            except Exception as exc:
                skipped_features += 1
                _remember(f"body {body_id}: extract_measurements failed: {exc!r}")
                continue

            feat = build_feature_vector(measurements)
            if feat is None:
                skipped_features += 1
                _remember(f"body {body_id}: build_feature_vector returned None (too many zeros)")
                continue

            # Build the 6-dim target vector
            try:
                target = np.array(
                    [float(gt[k]) for k in TARGET_FIELDS], dtype=np.float32
                )
            except KeyError as exc:
                skipped_no_gt += 1
                _remember(
                    f"body {body_id}: GT JSON missing key {exc}; "
                    f"available keys: {list(gt.keys())}"
                )
                continue

            feats.append(feat)
            targets.append(target)

    # Always print the skip summary so failures are diagnosable
    print(
        f"\nSkip summary:  no_pair={skipped_no_pair}  no_gt={skipped_no_gt}  "
        f"no_detect={skipped_no_detect}  bad_features={skipped_features}  "
        f"ok={len(feats)}"
    )
    if first_errors:
        print("\nFirst few failures (diagnosis):")
        for line in first_errors:
            print(f"  • {line}")

    if not feats:
        raise RuntimeError(
            "No samples built. See skip summary + first-failure messages above. "
            "Common causes:\n"
            "  - pose/seg models don't detect: try running the pipeline on one image manually\n"
            "  - measurements/ field names differ from TARGET_FIELDS\n"
            "  - images all ended up in one split and the other has no matching side/front"
        )

    feats_arr   = np.stack(feats)   # (N, 28)
    targets_arr = np.stack(targets) # (N, 6)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, features=feats_arr, targets=targets_arr)

    logger.info("Wrote %s  features=%s  targets=%s",
                output, feats_arr.shape, targets_arr.shape)
    logger.info(
        "Skipped: no_pair=%d  no_gt=%d  no_detect=%d  bad_features=%d",
        skipped_no_pair, skipped_no_gt, skipped_no_detect, skipped_features,
    )
    return len(feats), skipped_no_pair + skipped_no_gt + skipped_no_detect + skipped_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build regression training .npz from synthetic data")
    parser.add_argument("--pose-root", required=True, type=Path,
                        help="Synthetic pose dataset root (contains train/, val/, measurements/)")
    parser.add_argument("--pose-model", required=True, type=Path,
                        help="Trained YOLO-pose weights (yolo11n-pose.pt)")
    parser.add_argument("--seg-model", required=True, type=Path,
                        help="Trained YOLO-seg weights (yolo11n-pose.pt)")
    parser.add_argument("--output", type=Path, default=Path("data/regression_features.npz"))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="auto",
                        help="'cuda', 'cpu', 'auto' (default)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N bodies (debug)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    n_ok, n_skipped = build_dataset(
        pose_root=args.pose_root,
        pose_model=args.pose_model,
        seg_model=args.seg_model,
        output=args.output,
        img_size=args.img_size,
        device=args.device,
        limit=args.limit,
    )
    print(f"\nDone. Wrote {n_ok} samples, skipped {n_skipped}.")


if __name__ == "__main__":
    main()
