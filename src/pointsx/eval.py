"""Evaluate PointsX measurement accuracy on labelled real-world subjects.

Takes a CSV of subjects (front photo, side photo, height, sex, ground-truth
measurements) and runs the full pipeline with the requested model combo, then
reports per-measurement Mean Absolute Error and RMSE plus a per-subject diff
table.

Subjects CSV format (header required, units = cm):

    subject_id,front,side,height_cm,sex,chest_circumference,waist_circumference,...

Any column whose name matches a CANONICAL_MEASUREMENTS id is treated as ground
truth; missing or blank cells are skipped (no penalty). Paths can be absolute
or relative to the CSV file's directory.

Example:

    python -m pointsx.eval \
        --subjects data/eval/subjects.csv \
        --pose-backend coco \
        --pose-model-coco models/yolo26-pose.pt \
        --seg-model models/yolo12l-person-seg-extended.pt \
        --regression-model models/circumference_regressor.pt \
        --output runs/eval/report.csv

    # Same data, no regressor (Ramanujan fallback):
    python -m pointsx.eval --subjects data/eval/subjects.csv \
        --pose-backend coco --no-regressor \
        --output runs/eval/report-ramanujan.csv

Combine multiple runs by piping different --label values into one folder for
A/B comparison.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from webui.envelope import CANONICAL_MEASUREMENTS, body_to_envelope
from webui.inference import WebuiPipeline

logger = logging.getLogger(__name__)

CANONICAL_IDS = [mid for mid, _label, _src in CANONICAL_MEASUREMENTS]


@dataclass
class SubjectRow:
    subject_id: str
    front: Path
    side: Path
    height_cm: float
    sex: str
    gt: dict[str, float]  # measurement_id -> ground-truth cm


def _read_subjects(csv_path: Path) -> list[SubjectRow]:
    base = csv_path.resolve().parent
    rows: list[SubjectRow] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            sys.exit(f"{csv_path}: empty or missing header")
        gt_cols = [c for c in reader.fieldnames if c in CANONICAL_IDS]
        for raw in reader:
            try:
                front = Path(raw["front"])
                side = Path(raw["side"])
            except KeyError as e:
                sys.exit(f"{csv_path}: missing required column {e}")
            if not front.is_absolute():
                front = base / front
            if not side.is_absolute():
                side = base / side
            gt: dict[str, float] = {}
            for col in gt_cols:
                v = (raw.get(col) or "").strip()
                if not v:
                    continue
                try:
                    gt[col] = float(v)
                except ValueError:
                    logger.warning("Subject %s: bad value for %s=%r", raw.get("subject_id"), col, v)
            rows.append(
                SubjectRow(
                    subject_id=str(raw.get("subject_id") or front.stem),
                    front=front,
                    side=side,
                    height_cm=float(raw["height_cm"]),
                    sex=str(raw.get("sex") or "other").lower(),
                    gt=gt,
                )
            )
    if not rows:
        sys.exit(f"{csv_path}: no subjects found")
    return rows


def _envelope_predictions(env: Any) -> dict[str, float]:
    """Pull the {id: value_cm} dict from a MeasurementEnvelope."""
    out: dict[str, float] = {}
    for item in env.measurements:
        out[item.id] = float(item.value_cm)
    return out


def _summarise(errors: dict[str, list[float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mid in CANONICAL_IDS:
        vals = errors.get(mid) or []
        if not vals:
            rows.append({"measurement": mid, "n": 0, "mae_cm": None,
                         "rmse_cm": None, "bias_cm": None})
            continue
        abs_arr = np.array([abs(v) for v in vals])
        rmse = math.sqrt(float(np.mean(np.square(abs_arr))))
        rows.append({
            "measurement": mid,
            "n": len(vals),
            "mae_cm": round(float(np.mean(abs_arr)), 2),
            "rmse_cm": round(rmse, 2),
            "bias_cm": round(float(np.mean(vals)), 2),
        })
    return rows


def _print_summary(label: str, summary: list[dict[str, Any]]) -> None:
    print(f"\n=== {label} — per-measurement error ===")
    print(f"{'measurement':<32} {'n':>4} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    for r in summary:
        if r["n"] == 0:
            print(f"{r['measurement']:<32} {0:>4}      —        —        —")
            continue
        print(f"{r['measurement']:<32} {r['n']:>4} "
              f"{r['mae_cm']:>7.2f}  {r['rmse_cm']:>7.2f}  {r['bias_cm']:>+7.2f}")


def _aggregate(summary: list[dict[str, Any]]) -> dict[str, float]:
    abs_errs: list[float] = []
    sq_errs: list[float] = []
    for r in summary:
        if r["n"]:
            abs_errs.extend([r["mae_cm"]] * r["n"])
            sq_errs.extend([r["rmse_cm"] ** 2] * r["n"])
    if not abs_errs:
        return {"overall_mae_cm": float("nan"), "overall_rmse_cm": float("nan"), "n_obs": 0}
    return {
        "overall_mae_cm": round(float(np.mean(abs_errs)), 2),
        "overall_rmse_cm": round(math.sqrt(float(np.mean(sq_errs))), 2),
        "n_obs": len(abs_errs),
    }


def _write_report(
    output: Path, label: str, per_subject: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# label", label])
        w.writerow([])
        w.writerow(["# per-subject errors (cm)"])
        cols = ["subject_id", "measurement", "predicted_cm", "gt_cm", "error_cm"]
        w.writerow(cols)
        for r in per_subject:
            w.writerow([r["subject_id"], r["measurement"], r["predicted_cm"],
                        r["gt_cm"], r["error_cm"]])
        w.writerow([])
        w.writerow(["# per-measurement aggregate"])
        w.writerow(["measurement", "n", "mae_cm", "rmse_cm", "bias_cm"])
        for r in summary:
            w.writerow([r["measurement"], r["n"], r["mae_cm"],
                        r["rmse_cm"], r["bias_cm"]])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PointsX measurement accuracy against ground truth"
    )
    parser.add_argument("--subjects", required=True, type=Path,
                        help="CSV with subject_id, front, side, height_cm, sex, <measurement>_cm columns")
    parser.add_argument("--pose-backend", choices=("custom", "coco"), default="coco")
    parser.add_argument("--pose-model-custom", type=Path, default=Path("models/pose-cus.pt"))
    parser.add_argument("--pose-model-coco", type=Path, default=Path("models/yolo26-pose.pt"))
    parser.add_argument("--seg-model", type=Path,
                        default=Path("models/yolo12l-person-seg-extended.pt"))
    parser.add_argument("--regression-model", type=Path,
                        default=Path("models/circumference_regressor.pt"),
                        help="Path to circumference regressor .pt (ignored if --no-regressor).")
    parser.add_argument("--no-regressor", action="store_true",
                        help="Skip the regressor; use Ramanujan ellipse fallback for circumferences.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/eval/report.csv"))
    parser.add_argument("--label", default=None,
                        help="Run label for the report header (default: derived from flags)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    subjects = _read_subjects(args.subjects)

    reg_path: Path | None = None
    if not args.no_regressor:
        if args.regression_model.is_file():
            reg_path = args.regression_model
        else:
            logger.warning("Regressor file not found at %s — falling back to Ramanujan",
                           args.regression_model)

    label = args.label or (
        f"{args.pose_backend}+{'reg' if reg_path else 'ramanujan'}"
    )
    logger.info("Eval run %s on %d subjects", label, len(subjects))

    pipeline = WebuiPipeline(
        pose_custom_path=args.pose_model_custom if args.pose_model_custom.is_file() else None,
        pose_coco_path=args.pose_model_coco if args.pose_model_coco.is_file() else None,
        seg_model_path=args.seg_model,
        regression_model_path=reg_path,
        device=args.device,
    )
    avail = pipeline.models.available_pose_backends()
    if args.pose_backend not in avail:
        sys.exit(f"Pose backend {args.pose_backend!r} unavailable. Loaded: {sorted(avail)}")

    per_subject_rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {mid: [] for mid in CANONICAL_IDS}
    n_failed = 0

    for s in subjects:
        if not s.front.is_file() or not s.side.is_file():
            logger.warning("%s: missing image (front=%s side=%s)", s.subject_id, s.front, s.side)
            n_failed += 1
            continue
        front_img = cv2.imread(str(s.front), cv2.IMREAD_COLOR)
        side_img = cv2.imread(str(s.side), cv2.IMREAD_COLOR)
        if front_img is None or side_img is None:
            logger.warning("%s: cv2.imread returned None", s.subject_id)
            n_failed += 1
            continue
        try:
            result = pipeline.measure(
                front_img, side_img, s.height_cm, pose_backend=args.pose_backend,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s: pipeline failed: %s", s.subject_id, exc)
            n_failed += 1
            continue

        envelope = body_to_envelope(
            result, subject_height_cm=s.height_cm, sex=s.sex,
            request_id=f"eval-{s.subject_id}",
        )
        preds = _envelope_predictions(envelope)

        for mid, gt_cm in s.gt.items():
            pred_cm = preds.get(mid)
            if pred_cm is None:
                per_subject_rows.append({
                    "subject_id": s.subject_id, "measurement": mid,
                    "predicted_cm": "", "gt_cm": gt_cm, "error_cm": "",
                })
                continue
            err = pred_cm - gt_cm
            errors[mid].append(err)
            per_subject_rows.append({
                "subject_id": s.subject_id, "measurement": mid,
                "predicted_cm": round(pred_cm, 2), "gt_cm": round(gt_cm, 2),
                "error_cm": round(err, 2),
            })

    summary = _summarise(errors)
    agg = _aggregate(summary)

    _print_summary(label, summary)
    print(f"\nOverall: MAE={agg['overall_mae_cm']} cm  RMSE={agg['overall_rmse_cm']} cm  "
          f"observations={agg['n_obs']}  failed_subjects={n_failed}/{len(subjects)}")

    _write_report(args.output, label, per_subject_rows, summary)
    logger.info("Wrote report to %s", args.output)


if __name__ == "__main__":
    main()
