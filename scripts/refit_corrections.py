"""Refit envelope/corrections.py constants with leave-one-out (LOO) evaluation.

Input: a pointsx-eval report produced WITHOUT corrections
(``pointsx-eval --no-sex-offsets --output raw.csv``), which holds raw pred/gt per
subject and measurement. Corrections are multiplicative on those raw values, so
every table below is replayed exactly, without re-running inference.

For each target cell (sex x circumference, or a sex-independent length):
  none      MAE with no correction
  current   MAE with the constants in corrections.py (fitted July on n=11)
  refit-LOO fit median(gt/pred) on the other subjects, apply to the held-out one
  refit-in  in-sample fit on everyone (what the paste-ready table is; optimistic)
Gate for keeping a cell (same rule as corrections.py): bootstrap 95 % CI of the
fitted ratio excludes 1.0 AND LOO MAE beats 'none' by > 0.2 cm.

usage: .venv/Scripts/python scripts/refit_corrections.py runs/eval/raw_gated.csv [--seed 0]
Local only: the report holds per-subject errors on real people.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from webui.envelope.corrections import (  # noqa: E402
    _GIRTH_SHIFT_PCT,
    _LENGTH_SCALES_PCT,
    _SEX_CIRCUMFERENCE_SCALES_PCT,
)

CIRC_IDS = ["chest_circumference", "waist_circumference", "hip_circumference", "thigh_circumference"]
LENGTH_IDS = ["leg_length_inner_seam", "leg_length_outer_seam", "neck_base_height",
              "chest_width_front", "back_length_to_waist", "front_length_to_waist"]
MIN_GAIN_CM = 0.2
N_BOOT = 2000


def read_raw(path: Path) -> list[tuple[str, str, str, float, float]]:
    """(subject, sex, mid, pred, gt) rows of the first combo in the per-subject section."""
    rows, header, section, combo = [], None, None, None
    for row in csv.reader(path.open(encoding="utf-8")):
        if not row:
            continue
        if row[0].startswith("#"):
            section, header = row[0], None
            continue
        if section != "# per-(combo, subject, measurement)":
            continue
        if header is None:
            header = row
            continue
        r = dict(zip(header, row))
        combo = combo or r["combo"]
        if r["combo"] != combo:
            continue
        rows.append((r["subject_id"], r["sex"], r["measurement"], float(r["predicted_cm"]), float(r["gt_cm"])))
    if combo and "noff" not in combo:
        sys.exit(f"report combo is {combo!r}: run pointsx-eval with --no-sex-offsets")
    return rows


def fit_pct(pairs: list[tuple[float, float]]) -> float:
    """pointsx-eval's rule: median(gt/pred), rounded to 0.5 %, zero below 0.25 %."""
    ratios = [gt / pred for pred, gt in pairs if pred > 0]
    if not ratios:
        return 0.0
    pct = round((float(np.median(ratios)) - 1.0) * 100.0 * 2) / 2
    return 0.0 if abs(pct) < 0.25 else pct


def apply(pred: float, pct: float) -> float:
    return pred * (1.0 + pct / 100.0)


def mae(errs: list[float]) -> float | None:
    return float(np.mean(np.abs(errs))) if errs else None


def bootstrap_ci(pairs: list[tuple[float, float]], rng: random.Random) -> tuple[float, float]:
    ratios = [gt / pred for pred, gt in pairs if pred > 0]
    meds = [float(np.median(rng.choices(ratios, k=len(ratios)))) for _ in range(N_BOOT)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def evaluate_cell(cell: dict[str, tuple[float, float]], current_pct: float, rng: random.Random) -> dict:
    """cell: subject -> (pred, gt)."""
    subjects = sorted(cell)
    pairs = [cell[s] for s in subjects]
    none = mae([p - g for p, g in pairs])
    cur = mae([apply(p, current_pct) - g for p, g in pairs])
    loo_errs = []
    for s in subjects:
        others = [cell[o] for o in subjects if o != s]
        pct = fit_pct(others) if len(others) >= 2 else 0.0
        p, g = cell[s]
        loo_errs.append(apply(p, pct) - g)
    loo = mae(loo_errs)
    full_pct = fit_pct(pairs)
    insample = mae([apply(p, full_pct) - g for p, g in pairs])
    lo, hi = bootstrap_ci(pairs, rng) if len(pairs) >= 3 else (0.0, 2.0)
    keep = (lo > 1.0 or hi < 1.0) and loo is not None and (none - loo) > MIN_GAIN_CM
    return {"n": len(pairs), "none": none, "current": cur, "loo": loo, "insample": insample,
            "current_pct": current_pct, "fit_pct": full_pct, "ci": (lo, hi), "keep": keep}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--table", choices=("ellipse", "girth"), default="ellipse",
                    help="which circumference table the report was produced against (girth = POINTSX_GIRTH_MODEL=1)")
    args = ap.parse_args()
    circ_table = _GIRTH_SHIFT_PCT if args.table == "girth" else _SEX_CIRCUMFERENCE_SCALES_PCT
    rng = random.Random(args.seed)
    rows = read_raw(args.report)

    by_sex_cell: dict[tuple[str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    by_len_cell: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for subj, sex, mid, pred, gt in rows:
        if mid in CIRC_IDS:
            by_sex_cell[(sex, mid)][subj] = (pred, gt)
        elif mid in LENGTH_IDS:
            by_len_cell[mid][subj] = (pred, gt)

    hdr = f"{'cell':44}{'n':>3} {'none':>6} {'current':>8} {'refit-LOO':>10} {'refit-in':>9}  {'cur%':>6} {'fit%':>6}  CI(ratio)      keep"
    print(f"=== per-sex circumference scales ({'_GIRTH_SHIFT_PCT' if args.table == 'girth' else '_SEX_CIRCUMFERENCE_SCALES_PCT'}) ===")
    print(hdr)
    totals: dict[str, list[float]] = defaultdict(list)
    fitted_sex: dict[str, dict[str, float]] = defaultdict(dict)
    for sex in ("female", "male"):
        for mid in CIRC_IDS:
            cell = by_sex_cell.get((sex, mid))
            if not cell:
                continue
            cur_pct = circ_table.get(sex, {}).get(mid, 0.0)
            r = evaluate_cell(cell, cur_pct, rng)
            if r["keep"]:
                fitted_sex[sex][mid] = r["fit_pct"]
            for k in ("none", "current", "loo"):
                totals[k] += [r[k]] * r["n"]
            # what ships if we keep only gated cells: LOO where kept, none otherwise
            totals["gated"] += [r["loo"] if r["keep"] else r["none"]] * r["n"]
            print(f"{sex + ' ' + mid:44}{r['n']:3d} {r['none']:6.2f} {r['current']:8.2f} {r['loo']:10.2f} "
                  f"{r['insample']:9.2f}  {r['current_pct']:+6.1f} {r['fit_pct']:+6.1f}  "
                  f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  {'yes' if r['keep'] else 'no'}")

    print("\n=== sex-independent length scales (_LENGTH_SCALES_PCT) ===")
    print(hdr)
    fitted_len: dict[str, float] = {}
    for mid in LENGTH_IDS:
        cell = by_len_cell.get(mid)
        if not cell:
            continue
        r = evaluate_cell(cell, _LENGTH_SCALES_PCT.get(mid, 0.0), rng)
        if r["keep"]:
            fitted_len[mid] = r["fit_pct"]
        for k in ("none", "current", "loo"):
            totals[k] += [r[k]] * r["n"]
        totals["gated"] += [r["loo"] if r["keep"] else r["none"]] * r["n"]
        print(f"{mid:44}{r['n']:3d} {r['none']:6.2f} {r['current']:8.2f} {r['loo']:10.2f} "
              f"{r['insample']:9.2f}  {r['current_pct']:+6.1f} {r['fit_pct']:+6.1f}  "
              f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]  {'yes' if r['keep'] else 'no'}")

    n = len(totals["none"])
    print(f"\n=== overall on the {n} corrected-target observations (obs-weighted) ===")
    print(f"  none          {np.mean(totals['none']):.2f}")
    print(f"  current       {np.mean(totals['current']):.2f}   (constants fitted July on n=11; ~honest, corpus differs)")
    print(f"  refit LOO     {np.mean(totals['loo']):.2f}   (every cell refitted, held-out)")
    print(f"  refit gated   {np.mean(totals['gated']):.2f}   (only cells passing the gate, held-out)")

    print("\n=== paste-ready (in-sample fit on gated cells; report the LOO numbers above, not in-sample) ===")
    print("_SEX_CIRCUMFERENCE_SCALES_PCT = {")
    for sex in ("female", "male"):
        print(f'    "{sex}": {{' + ", ".join(f'"{m}": {p:+.1f}' for m, p in fitted_sex.get(sex, {}).items()) + "},")
    other = {m: round((fitted_sex.get("female", {}).get(m, 0.0) + fitted_sex.get("male", {}).get(m, 0.0)) / 2 * 2) / 2
             for m in CIRC_IDS if m in fitted_sex.get("female", {}) or m in fitted_sex.get("male", {})}
    print('    "other": {' + ", ".join(f'"{m}": {p:+.1f}' for m, p in other.items()) + "},")
    print("}")
    print("_LENGTH_SCALES_PCT = {" + ", ".join(f'"{m}": {p:+.1f}' for m, p in fitted_len.items()) + "}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
