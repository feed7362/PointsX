"""Compare circumference pipelines A–D on the REAL app corpus (Supabase GT).

The BodyM harness (eval/bodym.py) answers "how good is the geometry given perfect
masks". This answers the product question: **do the same pipelines help on real
photos**, where seg/pose/calibration/clothing error is included?

Pipelines (identical in spirit to eval/pipelines.py):
    A  raw ellipse                — frozen baseline (Ramanujan on app widths)
    B  per-(sex,measure) scale    — median(gt/pred), what the app ships today
    C  gated scale                — B, but skip cells whose scale is within 5% of 1
    D  learned girth (bilinear)   — c0 + c1·fw + c2·sw + c3·fw·sw, replaces the formula

**Leave-one-out cross-validation.** With n=11 subjects, fitting and scoring on the
same data is meaningless — every pipeline would look perfect. Each subject is
scored by a model fitted on the OTHER subjects only. D has 4 free parameters per
(sex, measure) cell, so it is only fitted where the training fold has enough
samples; otherwise that cell falls back to the ellipse (reported, never faked).

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe eval/app_pipelines.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
BACKEND_SRC = REPO_ROOT.parent / "Pointx-backend" / "src"
sys.path.insert(0, str(BACKEND_SRC))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pointsx.circumference import ramanujan_ellipse_circumference  # noqa: E402

# canonical id -> (front width field, side width field) on BodyMeasurements
WIDTH_FIELDS = {
    "chest_circumference": ("torso_width_front_cm", "torso_width_side_cm"),
    "waist_circumference": ("waist_width_front_cm", "waist_width_side_cm"),
    "hip_circumference": ("hip_width_front_cm", "hip_width_side_cm"),
    "thigh_circumference": ("thigh_width_front_cm", "thigh_width_side_cm"),
}
MEASURES = tuple(WIDTH_FIELDS)
D_MIN_SAMPLES = 6  # 4 coefficients — refuse to fit a cell on less
GATE_TAU = 0.05


def extract(subjects_csv: Path, models: dict[str, Path]) -> list[dict]:
    """Run the real pipeline once per subject → widths + ellipse + GT."""
    from webui.infrastructure.inference import WebuiPipeline
    import cv2

    pipe = WebuiPipeline(
        pose_custom_path=None,
        pose_coco_path=models["pose"],
        seg_model_path=models["seg"],
        regression_model_path=None,
        device="auto",
    )
    out: list[dict] = []
    rows = list(csv.DictReader(subjects_csv.open(encoding="utf-8-sig")))
    for i, r in enumerate(rows, 1):
        front, side = cv2.imread(r["front"]), cv2.imread(r["side"])
        if front is None or side is None:
            print(f"[skip] {r['subject_id']}: unreadable images", file=sys.stderr)
            continue
        try:
            res = pipe.measure(front, side, float(r["height_cm"]), pose_backend="coco")
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {r['subject_id']}: {exc}", file=sys.stderr)
            continue
        body = res.body
        rec = {"subject_id": r["subject_id"], "sex": (r.get("sex") or "other").lower(), "cells": {}}
        for m, (ff, sf) in WIDTH_FIELDS.items():
            fw, sw = getattr(body, ff, None), getattr(body, sf, None)
            gt = r.get(m)
            if fw and sw and gt not in (None, ""):
                rec["cells"][m] = {"fw": float(fw), "sw": float(sw),
                                   "ellipse": ramanujan_ellipse_circumference(float(fw), float(sw)),
                                   "gt": float(gt)}
        if rec["cells"]:
            out.append(rec)
        print(f"  … {i}/{len(rows)} {r['subject_id']} cells={len(rec['cells'])}")
    return out


def fit(train: list[dict]) -> dict:
    """Fit B/C scales and D coefficients on a training fold."""
    ratios: dict[tuple[str, str], list[float]] = {}
    rows: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for rec in train:
        for m, c in rec["cells"].items():
            key = (rec["sex"], m)
            if c["ellipse"] > 0:
                ratios.setdefault(key, []).append(c["gt"] / c["ellipse"])
            rows.setdefault(key, []).append((c["fw"], c["sw"], c["gt"]))
    scales = {k: float(np.median(v)) for k, v in ratios.items() if v}
    girth: dict[tuple[str, str], list[float]] = {}
    for key, pts in rows.items():
        if len(pts) < D_MIN_SAMPLES:
            continue
        fw = np.array([p[0] for p in pts]); sw = np.array([p[1] for p in pts])
        gt = np.array([p[2] for p in pts])
        A = np.column_stack([np.ones_like(fw), fw, sw, fw * sw])
        girth[key] = np.linalg.lstsq(A, gt, rcond=None)[0].tolist()
    return {"scales": scales, "girth": girth}


def predict(pipeline: str, cell: dict, sex: str, m: str, p: dict) -> tuple[float, bool]:
    """Return (prediction_cm, used_fallback)."""
    key = (sex, m)
    if pipeline == "A":
        return cell["ellipse"], False
    if pipeline == "B":
        return cell["ellipse"] * p["scales"].get(key, 1.0), key not in p["scales"]
    if pipeline == "C":
        s = p["scales"].get(key, 1.0)
        return (cell["ellipse"] * s if abs(s - 1.0) >= GATE_TAU else cell["ellipse"]), key not in p["scales"]
    c = p["girth"].get(key)
    if not c:
        return cell["ellipse"], True
    fw, sw = cell["fw"], cell["sw"]
    return float(c[0] + c[1] * fw + c[2] * sw + c[3] * fw * sw), False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", default=str(REPO_ROOT / "supabase-dump" / "subjects.csv"))
    ap.add_argument("--cache", default=str(EVAL_DIR / ".cache" / "app_cells.json"))
    ap.add_argument("--refresh", action="store_true", help="re-run inference, ignore cache")
    args = ap.parse_args()

    cache = Path(args.cache)
    if cache.is_file() and not args.refresh:
        data = json.loads(cache.read_text())
        print(f"[cache] {len(data)} subjects from {cache}")
    else:
        models = {"pose": REPO_ROOT / "models" / "yolo26-pose.pt",
                  "seg": REPO_ROOT / "models" / "yolo12l-person-seg-extended.pt"}
        print("[run] extracting widths via the real pipeline …")
        data = extract(Path(args.subjects), models)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, indent=2))
        print(f"[cache] wrote {cache}")

    names = ["A", "B", "C", "D"]
    err = {n: {m: [] for m in MEASURES} for n in names}
    fallbacks = {n: 0 for n in names}

    # Leave-one-out: each subject scored by a fit on the other subjects only.
    for i, held in enumerate(data):
        p = fit([r for j, r in enumerate(data) if j != i])
        for m, cell in held["cells"].items():
            for n in names:
                pred, fb = predict(n, cell, held["sex"], m, p)
                err[n][m].append(pred - cell["gt"])
                fallbacks[n] += fb

    print(f"\n=== App corpus (n={len(data)} subjects), LEAVE-ONE-OUT — MAE cm ===")
    hdr = f"{'measure':10}" + "".join(f"{n:>8}" for n in names) + f"{'Δ D-A':>8}"
    print(hdr)
    for m in MEASURES:
        if not err["A"][m]:
            continue
        maes = {n: float(np.mean(np.abs(err[n][m]))) for n in names}
        d = maes["A"] - maes["D"]
        print(f"{m.replace('_circumference',''):10}" + "".join(f"{maes[n]:8.1f}" for n in names)
              + f"{d:+8.1f}" + ("✓" if d > 0.05 else ("✗" if d < -0.05 else "·")))
    overall = {n: float(np.mean([abs(e) for m in MEASURES for e in err[n][m]])) for n in names}
    print(f"{'OVERALL':10}" + "".join(f"{overall[n]:8.1f}" for n in names)
          + f"{overall['A'] - overall['D']:+8.1f}")
    print(f"\nfallback-to-ellipse counts (cells without a usable fit): {fallbacks}")
    print(f"D fits a cell only with ≥{D_MIN_SAMPLES} training samples "
          f"(4 free params) — thin cells fall back to A, which is why D can tie A.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
