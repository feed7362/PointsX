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
from webui.envelope import CANONICAL_MEASUREMENTS, DISPLAY_MEASUREMENT_IDS, body_to_envelope
from webui.inference import InferenceResult, WebuiPipeline

logger = logging.getLogger(__name__)

# All canonical IDs (used to recognise GT columns from the subjects CSV) +
# the display subset (what the eval actually surfaces to the user, mirroring
# the webui results table).
CANONICAL_IDS = [mid for mid, _label, _src in CANONICAL_MEASUREMENTS]
LABEL_BY_ID = {mid: label for mid, label, _src in CANONICAL_MEASUREMENTS}
DISPLAY_IDS = list(DISPLAY_MEASUREMENT_IDS)


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
class ErrorRow:
    """Single (combo, subject, measurement) error observation."""
    combo: str
    subject_id: str
    sex: str
    measurement_id: str
    predicted_cm: float
    gt_cm: float
    error_cm: float  # predicted - gt


@dataclass
class ComboStats:
    label: str
    errors: dict[str, list[float]] = field(default_factory=lambda: {m: [] for m in DISPLAY_IDS})
    n_failed: int = 0


# ── Measurement category mapping (for the "per Обхват / Ширина / Довжина"
#    breakdown). Built from the Ukrainian label prefixes in CANONICAL_MEASUREMENTS.
def _category_for(mid: str) -> str:
    label = LABEL_BY_ID.get(mid, mid)
    if label.startswith("Обхват"):
        return "Обхват"
    if label.startswith("Ширина"):
        return "Ширина"
    if label.startswith("Довжина"):
        return "Довжина"
    if label.startswith("Висота"):
        return "Висота"
    return "Інше"


def _agg(values: list[float]) -> tuple[float, float, float] | None:
    """Return (MAE, RMSE, bias) over a list of signed errors, or None if empty."""
    if not values:
        return None
    abs_arr = np.array([abs(v) for v in values])
    rmse = math.sqrt(float(np.mean(np.square(abs_arr))))
    return float(np.mean(abs_arr)), rmse, float(np.mean(values))


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
    for mid in DISPLAY_IDS:
        label_uk = LABEL_BY_ID[mid]
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
    for mid in DISPLAY_IDS:
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


def _print_per_subject(combo_label: str, rows: list[ErrorRow]) -> None:
    """One block per subject with row-by-row pred/gt/Δ for the chosen combo."""
    by_subj: dict[str, list[ErrorRow]] = {}
    for r in rows:
        if r.combo != combo_label:
            continue
        by_subj.setdefault(r.subject_id, []).append(r)
    if not by_subj:
        return
    print(f"\n=== {combo_label}: per-subject detail ===")
    for sid, subj_rows in by_subj.items():
        sex = subj_rows[0].sex
        print(f"\n── {sid} (sex={sex}) ──")
        print(f"{'Показник':<58} {'Прогноз':>10} {'GT':>10} {'Δ см':>10}")
        # Sort by canonical display order
        ordered = sorted(subj_rows, key=lambda r: DISPLAY_IDS.index(r.measurement_id)
                         if r.measurement_id in DISPLAY_IDS else 999)
        for r in ordered:
            label = LABEL_BY_ID[r.measurement_id]
            print(f"{label:<58} {r.predicted_cm:>10.1f} {r.gt_cm:>10.1f} "
                  f"{r.error_cm:>+10.1f}")
        # Per-subject overall MAE
        agg = _agg([r.error_cm for r in subj_rows])
        if agg:
            mae, rmse, bias = agg
            print(f"{'  → overall':<58} {'':>10} {'':>10} "
                  f"  MAE={mae:.2f}  RMSE={rmse:.2f}  bias={bias:+.2f}")


def _print_per_sex(combo_label: str, rows: list[ErrorRow]) -> None:
    """For one combo: MAE/RMSE per sex × measurement, plus per-sex overall."""
    sex_groups = sorted({r.sex for r in rows if r.combo == combo_label})
    if not sex_groups:
        return
    print(f"\n=== {combo_label}: per-sex per-measurement error ===")
    header = f"{'Показник':<58} " + " ".join(f"{s:>16}" for s in sex_groups)
    print(header)
    print(f"{'':<58} " + " ".join(f"{'MAE / bias':>16}" for _ in sex_groups))
    for mid in DISPLAY_IDS:
        cells = []
        for sex in sex_groups:
            errs = [r.error_cm for r in rows
                    if r.combo == combo_label and r.sex == sex and r.measurement_id == mid]
            agg = _agg(errs)
            if agg is None:
                cells.append(f"{'—':>16}")
            else:
                mae, _rmse, bias = agg
                cells.append(f"{mae:>6.2f} / {bias:>+6.2f} ")
        print(f"{LABEL_BY_ID[mid]:<58} " + " ".join(cells))

    # Per-sex overall
    print(f"\n{combo_label}: per-sex overall —", end=" ")
    bits = []
    for sex in sex_groups:
        errs = [r.error_cm for r in rows
                if r.combo == combo_label and r.sex == sex]
        agg = _agg(errs)
        if agg:
            mae, rmse, _ = agg
            bits.append(f"{sex}: MAE={mae:.2f} RMSE={rmse:.2f} (n={len(errs)})")
    print("  |  ".join(bits))


def _print_per_category(combo_label: str, rows: list[ErrorRow]) -> None:
    """For one combo: aggregate by Ukrainian category (Обхват / Ширина / Довжина / Висота)."""
    print(f"\n=== {combo_label}: per-category error (Обхват / Ширина / Довжина / Висота) ===")
    print(f"{'Категорія':<14} {'n':>4} {'MAE':>8} {'RMSE':>8} {'bias':>9}  members")
    cats: dict[str, list[ErrorRow]] = {}
    for r in rows:
        if r.combo != combo_label:
            continue
        cats.setdefault(_category_for(r.measurement_id), []).append(r)
    # Order: Обхват first (most measurements), then Ширина, Довжина, Висота
    for cat in ("Обхват", "Ширина", "Довжина", "Висота", "Інше"):
        cat_rows = cats.get(cat) or []
        if not cat_rows:
            continue
        members = sorted({r.measurement_id for r in cat_rows})
        agg = _agg([r.error_cm for r in cat_rows])
        if agg is None:
            continue
        mae, rmse, bias = agg
        print(f"{cat:<14} {len(cat_rows):>4} {mae:>7.2f}  {rmse:>7.2f} {bias:>+8.2f}  "
              f"{', '.join(members)}")


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
    for mid in DISPLAY_IDS:
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


def _write_grid_report(
    output: Path, combo_stats: list[ComboStats], error_rows: list[ErrorRow],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)

        # — per-combo per-measurement aggregates —
        w.writerow(["# per-combo per-measurement"])
        w.writerow(["combo", "measurement", "n", "mae_cm", "rmse_cm", "bias_cm"])
        for s in combo_stats:
            for mid in DISPLAY_IDS:
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
        w.writerow(["# per-combo overall"])
        w.writerow(["combo", "overall_n", "overall_mae_cm", "overall_rmse_cm"])
        for s in combo_stats:
            agg = _aggregate_stats(s)
            w.writerow([s.label, agg["n"],
                        round(agg["mae"], 2) if agg["n"] else "",
                        round(agg["rmse"], 2) if agg["n"] else ""])

        # — flat per-(combo, subject, measurement) rows for any pivot —
        w.writerow([])
        w.writerow(["# per-(combo, subject, measurement)"])
        w.writerow(["combo", "subject_id", "sex", "measurement",
                    "predicted_cm", "gt_cm", "error_cm"])
        for r in error_rows:
            w.writerow([r.combo, r.subject_id, r.sex, r.measurement_id,
                        r.predicted_cm, r.gt_cm, r.error_cm])

        # — per-(combo, subject) overall —
        w.writerow([])
        w.writerow(["# per-(combo, subject) overall"])
        w.writerow(["combo", "subject_id", "sex", "n", "mae_cm", "rmse_cm", "bias_cm"])
        by_cs: dict[tuple[str, str, str], list[float]] = {}
        for r in error_rows:
            by_cs.setdefault((r.combo, r.subject_id, r.sex), []).append(r.error_cm)
        for (combo, sid, sex), vals in by_cs.items():
            agg = _agg(vals)
            if agg is None:
                continue
            mae, rmse, bias = agg
            w.writerow([combo, sid, sex, len(vals),
                        round(mae, 2), round(rmse, 2), round(bias, 2)])

        # — per-(combo, sex, measurement) —
        w.writerow([])
        w.writerow(["# per-(combo, sex, measurement)"])
        w.writerow(["combo", "sex", "measurement", "n", "mae_cm", "rmse_cm", "bias_cm"])
        by_csm: dict[tuple[str, str, str], list[float]] = {}
        for r in error_rows:
            by_csm.setdefault((r.combo, r.sex, r.measurement_id), []).append(r.error_cm)
        for (combo, sex, mid), vals in by_csm.items():
            agg = _agg(vals)
            if agg is None:
                continue
            mae, rmse, bias = agg
            w.writerow([combo, sex, mid, len(vals),
                        round(mae, 2), round(rmse, 2), round(bias, 2)])

        # — per-(combo, category) —
        w.writerow([])
        w.writerow(["# per-(combo, category) — Обхват / Ширина / Довжина / Висота"])
        w.writerow(["combo", "category", "n", "mae_cm", "rmse_cm", "bias_cm"])
        by_cat: dict[tuple[str, str], list[float]] = {}
        for r in error_rows:
            by_cat.setdefault((r.combo, _category_for(r.measurement_id)), []).append(r.error_cm)
        for (combo, cat), vals in by_cat.items():
            agg = _agg(vals)
            if agg is None:
                continue
            mae, rmse, bias = agg
            w.writerow([combo, cat, len(vals),
                        round(mae, 2), round(rmse, 2), round(bias, 2)])


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
    sex_offsets_override: dict[str, dict[str, float]] | None = None,
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
        sex_offsets_override=sex_offsets_override,
    )
    return _envelope_predictions(envelope)


# ── Offset fitting (median bias minimisation) ──────────────────────────────

def _fit_median_offsets(rows: list[ErrorRow]) -> dict[str, dict[str, float]]:
    """For each (sex, mid) with observations, return -median(error) as the
    additive offset that minimises L1 (MAE) when applied to predicted_cm.

    Skip cells with no observations (no offset learned).
    """
    by_cell: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        by_cell.setdefault((r.sex, r.measurement_id), []).append(r.error_cm)
    fitted: dict[str, dict[str, float]] = {}
    for (sex, mid), errs in by_cell.items():
        med = float(np.median(errs))
        # Round to 0.5 cm — nothing in real-world bias is more precise than that
        # given the input noise, and round numbers are easier to inspect.
        offset = -round(med * 2) / 2
        if abs(offset) < 0.25:
            continue  # keep the dict tidy; sub-quarter-cm offsets aren't meaningful
        fitted.setdefault(sex, {})[mid] = offset
    return fitted


def _print_fitted_offsets(fitted: dict[str, dict[str, float]]) -> None:
    print("\n=== Fitted bias offsets (paste into envelope.py "
          "_SEX_CIRCUMFERENCE_OFFSETS_CM) ===")
    if not fitted:
        print("  (no cells had enough data to fit)")
        return
    print("_SEX_CIRCUMFERENCE_OFFSETS_CM: dict[str, dict[str, float]] = {")
    for sex in sorted(fitted):
        entries = fitted[sex]
        print(f'    "{sex}": {{')
        for mid in DISPLAY_IDS:
            if mid in entries:
                print(f'        "{mid}": {entries[mid]:+.1f},')
        print("    },")
    print("}")


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
    parser.add_argument("--fit-offsets", action="store_true",
                        help="Fit per-(sex, measurement) bias offsets that minimise L1 error "
                             "(median of pred-gt). Forces single-combo mode with sex offsets OFF "
                             "for the first pass, prints a paste-ready dict, then re-runs with "
                             "the fitted offsets to show projected MAE.")
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

    # --fit-offsets is a single-combo, two-pass workflow. The first pass runs
    # with offsets OFF to collect raw bias; the second pass replays with the
    # fitted offsets so the user sees the projected MAE.
    if args.fit_offsets and args.grid:
        sys.exit("--fit-offsets is incompatible with --grid (it implies single-combo).")

    combos = (_build_grid(available, regressor_loaded) if args.grid else [Combo(
        label=f"{args.pose_backend}+{'rama' if args.no_regressor or not regressor_loaded else 'reg'}"
              f"+{'noff' if args.no_sex_offsets or args.fit_offsets else 'off'}",
        pose_backend=args.pose_backend,
        use_regressor=(not args.no_regressor) and regressor_loaded,
        apply_sex_offsets=(not args.no_sex_offsets) and (not args.fit_offsets),
    )])
    if args.pose_backend not in available and not args.grid:
        sys.exit(f"Pose backend {args.pose_backend!r} unavailable. Loaded: {sorted(available)}")

    # Cache per-(subject, pose_backend) the expensive pose+seg+extract output
    # so we can replay cheap combos and a second offset-fitting pass without
    # re-running pose+seg+extract.
    needed_backends = sorted({c.pose_backend for c in combos})
    extracts_by_subject: dict[str, dict[str, Any]] = {}
    for subject in subjects:
        if not subject.front.is_file() or not subject.side.is_file():
            logger.warning("%s: missing image", subject.subject_id)
            extracts_by_subject[subject.subject_id] = {}
            continue
        front_img = cv2.imread(str(subject.front), cv2.IMREAD_COLOR)
        side_img = cv2.imread(str(subject.side), cv2.IMREAD_COLOR)
        if front_img is None or side_img is None:
            logger.warning("%s: cv2.imread None", subject.subject_id)
            extracts_by_subject[subject.subject_id] = {}
            continue
        per_backend: dict[str, Any] = {}
        for pb in needed_backends:
            try:
                per_backend[pb] = pipeline.measure(
                    front_img, side_img, subject.height_cm, pose_backend=pb,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s [%s]: pipeline failed: %s",
                               subject.subject_id, pb, exc)
                per_backend[pb] = None
        extracts_by_subject[subject.subject_id] = per_backend

    def _run_combos(
        combos: list[Combo],
        sex_offsets_override: dict[str, dict[str, float]] | None = None,
        emit_per_subject_table: bool = False,
    ) -> tuple[dict[str, ComboStats], list[ErrorRow]]:
        combo_stats = {c.label: ComboStats(label=c.label) for c in combos}
        error_rows: list[ErrorRow] = []
        for subject in subjects:
            extracts = extracts_by_subject.get(subject.subject_id) or {}
            if not extracts:
                for s in combo_stats.values():
                    s.n_failed += 1
                continue
            for combo in combos:
                res = extracts.get(combo.pose_backend)
                if res is None:
                    combo_stats[combo.label].n_failed += 1
                    continue
                try:
                    preds = _evaluate_combo_for_subject(
                        subject, res.body, res.front_kp, res.side_kp,
                        res.front_mask, res.side_mask, res.cal,
                        combo, pipeline.regressor,
                        pose_backend_used=combo.pose_backend,
                        sex_offsets_override=sex_offsets_override,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s [%s]: combo eval failed: %s",
                                   subject.subject_id, combo.label, exc)
                    combo_stats[combo.label].n_failed += 1
                    continue
                if emit_per_subject_table:
                    _print_subject_table(subject, preds)
                for mid in DISPLAY_IDS:
                    gt_cm = subject.gt.get(mid)
                    pred_cm = preds.get(mid)
                    if gt_cm is None or pred_cm is None:
                        continue
                    err = pred_cm - gt_cm
                    combo_stats[combo.label].errors[mid].append(err)
                    error_rows.append(ErrorRow(
                        combo=combo.label, subject_id=subject.subject_id,
                        sex=subject.sex, measurement_id=mid,
                        predicted_cm=round(pred_cm, 2), gt_cm=round(gt_cm, 2),
                        error_cm=round(err, 2),
                    ))
        return combo_stats, error_rows

    combo_stats, error_rows = _run_combos(
        combos, emit_per_subject_table=not args.grid and not args.fit_offsets,
    )
    stats_list = list(combo_stats.values())

    if args.grid:
        for s in stats_list:
            _print_combo_summary(s, len(subjects))
        # Pick the best combo (lowest overall MAE) for the deeper breakdowns —
        # printing all four for every combo would be unreadable. The full
        # per-subject / per-sex / per-category data is in the CSV report.
        ranked = sorted(stats_list, key=lambda s: (
            _aggregate_stats(s)["mae"]
            if not math.isnan(_aggregate_stats(s)["mae"]) else float("inf")
        ))
        if ranked:
            best = ranked[0].label
            _print_per_subject(best, error_rows)
            _print_per_sex(best, error_rows)
            _print_per_category(best, error_rows)
        _print_grid_ranking(stats_list)
        _write_grid_report(args.output, stats_list, error_rows)
        logger.info("Wrote grid report to %s", args.output)
    else:
        s = stats_list[0]
        if args.fit_offsets:
            print(f"\n=== Pass 1 ({s.label}, sex offsets OFF) — raw bias ===")
        _print_combo_summary(s, len(subjects))
        _print_per_subject(s.label, error_rows)
        _print_per_sex(s.label, error_rows)
        _print_per_category(s.label, error_rows)
        agg = _aggregate_stats(s)
        if agg["n"]:
            print(f"\nOverall: MAE={agg['mae']:.2f} cm  RMSE={agg['rmse']:.2f} cm  "
                  f"observations={agg['n']}  failed={s.n_failed}/{len(subjects)}")

        if args.fit_offsets:
            fitted = _fit_median_offsets(error_rows)
            _print_fitted_offsets(fitted)

            # Replay with fitted offsets — re-enable apply_sex_offsets so the
            # override path is taken.
            replay_combos = [Combo(
                label=f"{s.label.replace('+noff', '+fitted')}",
                pose_backend=combos[0].pose_backend,
                use_regressor=combos[0].use_regressor,
                apply_sex_offsets=True,
            )]
            r_stats, r_rows = _run_combos(
                replay_combos, sex_offsets_override=fitted,
            )
            r_list = list(r_stats.values())
            r_s = r_list[0]
            print(f"\n=== Pass 2 ({r_s.label}, fitted offsets applied) ===")
            _print_combo_summary(r_s, len(subjects))
            _print_per_sex(r_s.label, r_rows)
            _print_per_category(r_s.label, r_rows)
            r_agg = _aggregate_stats(r_s)
            if r_agg["n"]:
                print(f"\nProjected MAE: {r_agg['mae']:.2f} cm  "
                      f"(was {agg['mae']:.2f} cm — improvement "
                      f"{agg['mae'] - r_agg['mae']:+.2f} cm)")

            stats_list = stats_list + r_list
            error_rows = error_rows + r_rows

        _write_grid_report(args.output, stats_list, error_rows)
        logger.info("Wrote report to %s", args.output)


if __name__ == "__main__":
    main()
