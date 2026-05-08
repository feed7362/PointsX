"""Evaluate PointsX measurement accuracy on labelled real-world subjects.

Two modes:

  Single combo (default):
      pointsx-eval --subjects subjects.csv --pose-backend coco
      Prints the same Ukrainian-labelled measurement table the webui shows,
      per subject, side-by-side with ground truth.

  Grid (--grid):
      pointsx-eval --subjects subjects.csv --grid
      Iterates over all combinations of pose-backend × regressor × sex-offsets,
      ranks them by overall MAE, and emits a head-to-head diff so you can spot
      which combo wins on which measurement. Pose+seg run only once per
      pose-backend per subject — circumferences and envelope assembly are
      cheap, so the whole grid costs roughly 2× a single eval (one per pose
      backend), not 8× combo count.

Subjects CSV (header required, units = cm):

    subject_id,front,side,height_cm,sex,chest_circumference,...

Any column matching a CANONICAL_MEASUREMENTS id is treated as ground truth;
missing/blank cells are skipped (no penalty). Paths can be absolute or
relative to the CSV file's directory.
"""
from __future__ import annotations

import argparse
import copy
import csv
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pointsx.circumference import estimate_circumferences
from pointsx.postprocess import validate_measurements
from pointsx.schemas import BodyMeasurements
from webui.envelope import CANONICAL_MEASUREMENTS, body_to_envelope
from webui.inference import InferenceResult, WebuiPipeline

logger = logging.getLogger(__name__)

CANONICAL_IDS = [mid for mid, _label, _src in CANONICAL_MEASUREMENTS]
LABEL_BY_ID = {mid: label for mid, label, _src in CANONICAL_MEASUREMENTS}


@dataclass
class SubjectRow:
    subject_id: str
    front: Path
    side: Path
    height_cm: float
    sex: str
    gt: dict[str, float]


@dataclass
class Combo:
    label: str
    pose_backend: str           # "custom" | "coco"
    use_regressor: bool
    apply_sex_offsets: bool


@dataclass
class ComboStats:
    label: str
    errors: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in CANONICAL_IDS})
    n_failed: int = 0


# ── CSV ─────────────────────────────────────────────────────────────────────

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
                if v:
                    try:
                        gt[col] = float(v)
                    except ValueError:
                        logger.warning("Subject %s: bad %s=%r",
                                       raw.get("subject_id"), col, v)
            rows.append(SubjectRow(
                subject_id=str(raw.get("subject_id") or front.stem),
                front=front, side=side,
                height_cm=float(raw["height_cm"]),
                sex=str(raw.get("sex") or "other").lower(),
                gt=gt,
            ))
    if not rows:
        sys.exit(f"{csv_path}: no subjects found")
    return rows


# ── Pretty printing ─────────────────────────────────────────────────────────

def _print_subject_table(subject: SubjectRow, preds: dict[str, float]) -> None:
    """Webui-style table: id label (UK) → predicted vs GT vs error."""
    print(f"\n── {subject.subject_id} (sex={subject.sex}, h={subject.height_cm} cm) ──")
    print(f"{'Показник':<58} {'Прогноз':>10} {'GT':>10} {'Δ см':>10}")
    for mid, label_uk, _src in CANONICAL_MEASUREMENTS:
        pv = preds.get(mid)
        gv = subject.gt.get(mid)
        pred_str = f"{pv:>10.1f}" if pv is not None else f"{'—':>10}"
        gt_str = f"{gv:>10.1f}" if gv is not None else f"{'—':>10}"
        if pv is not None and gv is not None:
            diff_str = f"{pv - gv:>+10.1f}"
        else:
            diff_str = f"{'—':>10}"
        print(f"{label_uk:<58} {pred_str} {gt_str} {diff_str}")


def _print_combo_summary(stats: ComboStats, n_subjects: int) -> None:
    print(f"\n=== {stats.label} — per-measurement error ===")
    print(f"{'Показник':<58} {'n':>4} {'MAE':>8} {'RMSE':>8} {'bias':>9}")
    for mid in CANONICAL_IDS:
        vals = stats.errors[mid]
        label = LABEL_BY_ID[mid]
        if not vals:
            print(f"{label:<58} {0:>4} {'—':>8} {'—':>8} {'—':>9}")
            continue
        abs_arr = np.array([abs(v) for v in vals])
        rmse = math.sqrt(float(np.mean(np.square(abs_arr))))
        print(f"{label:<58} {len(vals):>4} {float(np.mean(abs_arr)):>7.2f}  "
              f"{rmse:>7.2f} {float(np.mean(vals)):>+8.2f}")


def _aggregate_stats(stats: ComboStats) -> dict[str, float]:
    abs_errs: list[float] = []
    sq_errs: list[float] = []
    for vals in stats.errors.values():
        for v in vals:
            abs_errs.append(abs(v))
            sq_errs.append(v * v)
    if not abs_errs:
        return {"mae": float("nan"), "rmse": float("nan"), "n": 0}
    return {
        "mae": float(np.mean(abs_errs)),
        "rmse": math.sqrt(float(np.mean(sq_errs))),
        "n": len(abs_errs),
    }


def _print_grid_ranking(combo_stats: list[ComboStats]) -> None:
    print("\n=== Grid ranking (by overall MAE, lower is better) ===")
    rows = [(s.label, _aggregate_stats(s)) for s in combo_stats]
    rows.sort(key=lambda r: (r[1]["mae"] if not math.isnan(r[1]["mae"]) else float("inf")))
    print(f"{'combo':<48} {'n':>5} {'MAE':>8} {'RMSE':>8}")
    for label, agg in rows:
        if agg["n"] == 0:
            print(f"{label:<48} {0:>5} {'—':>8} {'—':>8}")
        else:
            print(f"{label:<48} {agg['n']:>5} {agg['mae']:>7.2f}  {agg['rmse']:>7.2f}")

    # Per-measurement winner
    print("\n=== Per-measurement winner (lowest MAE) ===")
    print(f"{'Показник':<58} {'best combo':<32} {'MAE':>8}")
    for mid in CANONICAL_IDS:
        best_label, best_mae = None, float("inf")
        for s in combo_stats:
            vals = s.errors[mid]
            if not vals:
                continue
            mae = float(np.mean([abs(v) for v in vals]))
            if mae < best_mae:
                best_mae, best_label = mae, s.label
        label = LABEL_BY_ID[mid]
        if best_label is None:
            print(f"{label:<58} {'—':<32} {'—':>8}")
        else:
            print(f"{label:<58} {best_label:<32} {best_mae:>7.2f}")


def _write_grid_report(output: Path, combo_stats: list[ComboStats]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["combo", "measurement", "n", "mae_cm", "rmse_cm", "bias_cm"])
        for s in combo_stats:
            for mid in CANONICAL_IDS:
                vals = s.errors[mid]
                if not vals:
                    w.writerow([s.label, mid, 0, "", "", ""])
                    continue
                abs_arr = np.array([abs(v) for v in vals])
                rmse = math.sqrt(float(np.mean(np.square(abs_arr))))
                w.writerow([
                    s.label, mid, len(vals),
                    round(float(np.mean(abs_arr)), 2),
                    round(rmse, 2),
                    round(float(np.mean(vals)), 2),
                ])
        w.writerow([])
        w.writerow(["combo", "overall_n", "overall_mae_cm", "overall_rmse_cm"])
        for s in combo_stats:
            agg = _aggregate_stats(s)
            w.writerow([s.label, agg["n"],
                        round(agg["mae"], 2) if agg["n"] else "",
                        round(agg["rmse"], 2) if agg["n"] else ""])


# ── Per-combo evaluation ────────────────────────────────────────────────────

def _envelope_predictions(env: Any) -> dict[str, float]:
    return {item.id: float(item.value_cm) for item in env.measurements}


def _evaluate_combo_for_subject(
    subject: SubjectRow,
    bm_template: BodyMeasurements,
    front_kp, side_kp, front_mask, side_mask, cal,
    combo: Combo,
    regressor,
    pose_backend_used: str,
) -> dict[str, float]:
    """Run the cheap part (circumferences + envelope) for a single combo."""
    bm = copy.deepcopy(bm_template)
    bm = estimate_circumferences(bm, regressor if combo.use_regressor else None)
    bm = validate_measurements(bm)
    result = InferenceResult(
        body=bm, front_kp=front_kp, side_kp=side_kp,
        front_mask=front_mask, side_mask=side_mask, cal=cal,
        has_regressor=combo.use_regressor and regressor is not None,
        pose_backend=pose_backend_used,
    )
    envelope = body_to_envelope(
        result, subject_height_cm=subject.height_cm, sex=subject.sex,
        request_id=f"eval-{subject.subject_id}",
        apply_sex_offsets=combo.apply_sex_offsets,
    )
    return _envelope_predictions(envelope)


# ── CLI ─────────────────────────────────────────────────────────────────────

def _build_grid(available_backends: set[str], regressor_loaded: bool) -> list[Combo]:
    backends = [b for b in ("custom", "coco") if b in available_backends]
    combos: list[Combo] = []
    for pb in backends:
        for use_reg in ([True, False] if regressor_loaded else [False]):
            for apply_off in (True, False):
                label = f"{pb}+{'reg' if use_reg else 'rama'}+{'off' if apply_off else 'noff'}"
                combos.append(Combo(label, pb, use_reg, apply_off))
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PointsX measurement accuracy")
    parser.add_argument("--subjects", required=True, type=Path)
    parser.add_argument("--pose-backend", choices=("custom", "coco"), default="coco")
    parser.add_argument("--pose-model-custom", type=Path, default=Path("models/pose-cus.pt"))
    parser.add_argument("--pose-model-coco", type=Path, default=Path("models/yolo26-pose.pt"))
    parser.add_argument("--seg-model", type=Path,
                        default=Path("models/yolo12l-person-seg-extended.pt"))
    parser.add_argument("--regression-model", type=Path,
                        default=Path("models/circumference_regressor.pt"))
    parser.add_argument("--no-regressor", action="store_true",
                        help="Single-combo mode: skip regressor (Ramanujan fallback).")
    parser.add_argument("--no-sex-offsets", action="store_true",
                        help="Single-combo mode: skip _SEX_CIRCUMFERENCE_OFFSETS_CM.")
    parser.add_argument("--grid", action="store_true",
                        help="Run all combinations of pose × regressor × sex-offsets and rank them.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/eval/report.csv"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    subjects = _read_subjects(args.subjects)

    pipeline = WebuiPipeline(
        pose_custom_path=args.pose_model_custom if args.pose_model_custom.is_file() else None,
        pose_coco_path=args.pose_model_coco if args.pose_model_coco.is_file() else None,
        seg_model_path=args.seg_model,
        regression_model_path=args.regression_model if args.regression_model.is_file() else None,
        device=args.device,
    )
    available = pipeline.models.available_pose_backends()
    if not available:
        sys.exit("No pose weights loaded.")
    regressor_loaded = pipeline.regressor is not None
    if not regressor_loaded:
        logger.warning("Regressor not loaded — Ramanujan fallback only")

    combos = (_build_grid(available, regressor_loaded) if args.grid else [Combo(
        label=f"{args.pose_backend}+{'rama' if args.no_regressor or not regressor_loaded else 'reg'}"
              f"+{'noff' if args.no_sex_offsets else 'off'}",
        pose_backend=args.pose_backend,
        use_regressor=(not args.no_regressor) and regressor_loaded,
        apply_sex_offsets=not args.no_sex_offsets,
    )])
    if args.pose_backend not in available and not args.grid:
        sys.exit(f"Pose backend {args.pose_backend!r} unavailable. Loaded: {sorted(available)}")

    # Cache per-(subject, pose_backend) the expensive pose+seg+extract output,
    # then reuse it across all combos that share the same pose backend.
    needed_backends = sorted({c.pose_backend for c in combos})
    combo_stats = {c.label: ComboStats(label=c.label) for c in combos}

    for subject in subjects:
        if not subject.front.is_file() or not subject.side.is_file():
            logger.warning("%s: missing image", subject.subject_id)
            for s in combo_stats.values():
                s.n_failed += 1
            continue
        front_img = cv2.imread(str(subject.front), cv2.IMREAD_COLOR)
        side_img = cv2.imread(str(subject.side), cv2.IMREAD_COLOR)
        if front_img is None or side_img is None:
            logger.warning("%s: cv2.imread None", subject.subject_id)
            for s in combo_stats.values():
                s.n_failed += 1
            continue

        per_backend_extract: dict[str, Any] = {}
        for pb in needed_backends:
            try:
                res = pipeline.measure(front_img, side_img,
                                       subject.height_cm, pose_backend=pb)
                per_backend_extract[pb] = res
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s [%s]: pipeline failed: %s",
                               subject.subject_id, pb, exc)
                per_backend_extract[pb] = None

        for combo in combos:
            res = per_backend_extract.get(combo.pose_backend)
            if res is None:
                combo_stats[combo.label].n_failed += 1
                continue
            try:
                preds = _evaluate_combo_for_subject(
                    subject, res.body, res.front_kp, res.side_kp,
                    res.front_mask, res.side_mask, res.cal,
                    combo, pipeline.regressor, pose_backend_used=combo.pose_backend,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s [%s]: combo eval failed: %s",
                               subject.subject_id, combo.label, exc)
                combo_stats[combo.label].n_failed += 1
                continue

            # Single-combo mode: also emit the webui-style table.
            if not args.grid:
                _print_subject_table(subject, preds)

            for mid, gt_cm in subject.gt.items():
                pred_cm = preds.get(mid)
                if pred_cm is None:
                    continue
                combo_stats[combo.label].errors[mid].append(pred_cm - gt_cm)

    stats_list = list(combo_stats.values())

    if args.grid:
        for s in stats_list:
            _print_combo_summary(s, len(subjects))
        _print_grid_ranking(stats_list)
        _write_grid_report(args.output, stats_list)
        logger.info("Wrote grid report to %s", args.output)
    else:
        s = stats_list[0]
        _print_combo_summary(s, len(subjects))
        agg = _aggregate_stats(s)
        if agg["n"]:
            print(f"\nOverall: MAE={agg['mae']:.2f} cm  RMSE={agg['rmse']:.2f} cm  "
                  f"observations={agg['n']}  failed={s.n_failed}/{len(subjects)}")
        _write_grid_report(args.output, stats_list)
        logger.info("Wrote report to %s", args.output)


if __name__ == "__main__":
    main()
