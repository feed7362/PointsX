"""BodyM ceiling eval — flat, instance-1 (per the polygon ADR build order).

Answers ONE scientific question with real ground truth:
    "Given a correct silhouette + true height, how accurate is our
     two-widths -> Ramanujan-ellipse circumference math?"

Pipeline (silhouette-only, no keypoints — BodyM has none):
    mask + height
      -> calibrate each mask independently (height_cm / body_px)
      -> torso-band finder (THE new code: clip A-pose arms without keypoints)
      -> landmark rows from the clipped width profile (waist/hip/chest/thigh)
      -> real pointsx.ramanujan_ellipse_circumference(front_w, side_d)
    vs per-sex-mean baseline ("can't get worse").

Real production code exercised: ramanujan_ellipse_circumference + measure_width_at_y.
Everything else (calibrate, rows, arm-clip) is eval-side — see README.

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe eval/bodym.py --split testA
    .venv/Scripts/python.exe eval/bodym.py --split testB --n 200
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pipelines import PIPELINES  # noqa: E402
from pointsx.circumference import ramanujan_ellipse_circumference  # noqa: E402
from pointsx.silhouette import _find_segments, measure_width_at_y  # noqa: E402

BUCKET = "amazon-bodym"
CACHE = EVAL_DIR / ".cache"
CIRCUMFERENCES = ("chest", "waist", "hip", "thigh")
TOL_LADDER = (1.0, 3.0, 5.0)
WRONG_SIZE_CM = 5.0  # safety metric: off by more than this = silently wrong size

# ── LEGAL GUARD — read before adding any pipeline ────────────────────────────
# BodyM is licensed CC BY-NC (non-commercial). In THIS project it is used for
# EVALUATION ONLY. Rules, enforced by convention here:
#   • A (ellipse) is pure geometry — YOLO widths + a math formula, no data fit.
#   • B/C/D derive *correction constants* (scales / girth coefs) from BodyM's
#     GT via median/least-squares. These are a NON-COMMERCIAL RESEARCH BENCHMARK
#     ONLY — they must NEVER ship in the product (the app over-reads, BodyM's
#     silhouette under-reads: opposite sign, proven; they don't transfer anyway).
#   • Do NOT train an ML MODEL (regressor/NN) on BodyM — that is forbidden here.
#     A learned pipeline (E+) must fit on a LEGAL source (synthetic / app GT) and
#     use BodyM only as held-out test. Synthetic is deferred: current synthetic
#     data is fragile and needs reconsideration before it can be that source.
_LICENSE = "CC BY-NC — evaluation only; fitted constants are benchmark-only, non-shippable"


def _guard_banner() -> None:
    print(f"[guard] BodyM {_LICENSE}")


# ── S3 (public, unsigned) + tiny cache ──────────────────────────────────────
def _client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))


def _get(s3, key: str) -> bytes:
    cached = CACHE / key
    if cached.is_file():
        return cached.read_bytes()
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(body)
    return body


def _csv(s3, key: str) -> list[dict]:
    return list(csv.DictReader(_get(s3, key).decode().splitlines()))


def _mask(s3, key: str) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(_get(s3, key))).convert("L")) > 127


# ── the one piece of genuinely new code: silhouette-only torso width ─────────
def _body_extent(mask: np.ndarray) -> tuple[int, int]:
    ys = np.where(mask.any(axis=1))[0]
    return int(ys[0]), int(ys[-1])


def _body_center_x(mask: np.ndarray, y0: int, y1: int) -> float:
    """Robust horizontal body centre: median row-midpoint over the torso.

    Arms are ~symmetric so (min+max)/2 stays on the body axis even before
    clipping. Used to pick the torso segment and reject detached arm lobes.
    """
    mids = []
    for y in range(y0, y1 + 1, 3):
        cols = np.where(mask[y])[0]
        if len(cols) >= 2:
            mids.append((cols[0] + cols[-1]) / 2.0)
    return float(np.median(mids)) if mids else mask.shape[1] / 2.0


def torso_width_at(mask: np.ndarray, y: float, center_x: float, margin: int = 3) -> float | None:
    """Width of the torso segment containing center_x, i.e. arms clipped off.

    At each row, split foreground into contiguous segments (`_find_segments`,
    the same helper production uses) and keep the one straddling the body axis.
    A-pose arms that separate from the torso become their own segments and are
    dropped. Averages over [y-margin, y+margin]. Returns pixels."""
    h = mask.shape[0]
    y_int = int(round(y))
    widths = []
    for row in range(max(0, y_int - margin), min(h - 1, y_int + margin) + 1):
        cols = np.where(mask[row])[0]
        if len(cols) < 2:
            continue
        segs = [s for s in _find_segments(cols) if len(s) >= 2]
        if not segs:
            continue
        # segment straddling the body axis; else the one nearest to it
        straddling = [s for s in segs if s[0] <= center_x <= s[-1]]
        seg = straddling[0] if straddling else min(
            segs, key=lambda s: min(abs(s[0] - center_x), abs(s[-1] - center_x)))
        widths.append(seg[-1] - seg[0])
    return float(np.mean(widths)) if widths else None


def _leg_width_at(mask: np.ndarray, y: float, center_x: float, margin: int = 3) -> float | None:
    """Single-leg width below the crotch: widest sub-crotch segment on one side."""
    h = mask.shape[0]
    y_int = int(round(y))
    widths = []
    for row in range(max(0, y_int - margin), min(h - 1, y_int + margin) + 1):
        cols = np.where(mask[row])[0]
        if len(cols) < 2:
            continue
        segs = [s for s in _find_segments(cols) if len(s) >= 2]
        if not segs:
            continue
        # legs = the two largest segments; take the wider one as "a thigh"
        segs.sort(key=lambda s: s[-1] - s[0], reverse=True)
        leg = segs[0]
        widths.append(leg[-1] - leg[0])
    return float(np.mean(widths)) if widths else None


# ── predict one subject ─────────────────────────────────────────────────────
def predict(front: np.ndarray, side: np.ndarray, height_cm: float) -> dict[str, float | None]:
    fy0, fy1 = _body_extent(front)
    sy0, sy1 = _body_extent(side)
    f_len, s_len = fy1 - fy0, sy1 - sy0
    if f_len < 50 or s_len < 50:
        return {m: None for m in CIRCUMFERENCES}
    f_scale = height_cm / f_len
    s_scale = height_cm / s_len
    fcx = _body_center_x(front, fy0, fy1)
    scx = _body_center_x(side, sy0, sy1)

    # clipped front width profile over the torso band, to locate landmark rows
    prof = np.array([torso_width_at(front, fy0 + i, fcx) or 0.0 for i in range(f_len)])

    def band(a: float, b: float) -> slice:
        return slice(int(a * f_len), int(b * f_len))

    # waist = narrowest torso row; hip = widest below it (silhouette extrema work).
    waist_i = band(0.30, 0.48).start + int(np.argmin(prof[band(0.30, 0.48)]))
    hip_i = waist_i + int(np.argmax(prof[waist_i:int(0.55 * f_len)]))
    # chest = widest CLEAN torso row below the armpit. In arms-down pose the arm
    # is fused to the torso above the armpit (breadth plateau); it separates at a
    # sharp breadth cliff. Find that cliff (steepest drop), take the widest row
    # below it. NB: true bust is above the armpit and is occluded by the arm —
    # this is an underbust-ish proxy, so chest is the arm-occlusion-limited metric.
    ub = slice(int(0.22 * f_len), int(0.42 * f_len))
    armpit_i = ub.start + int(np.argmin(np.diff(prof[ub]))) + 1
    chest_lo = max(armpit_i, int(0.24 * f_len))
    chest_hi = max(chest_lo + 1, waist_i)
    chest_i = chest_lo + int(np.argmax(prof[chest_lo:chest_hi]))
    # thigh = below the crotch where the two legs separate (higher rows merge them).
    thigh_i = int(0.66 * f_len)

    rows = {"chest": chest_i, "waist": waist_i, "hip": hip_i, "thigh": thigh_i}
    # out[m] = (ellipse_circ_cm, front_w_cm, side_w_cm) | None — widths surfaced so
    # pipeline D can replace the ellipse formula with a learned width→girth map.
    out: dict[str, tuple[float, float, float] | None] = {}
    for name, fi in rows.items():
        frac = fi / f_len
        y_side = sy0 + int(frac * s_len)
        if name == "thigh":
            fw = _leg_width_at(front, fy0 + fi, fcx)
            sd = _leg_width_at(side, y_side, scx)
        else:
            fw = torso_width_at(front, fy0 + fi, fcx)
            sd = torso_width_at(side, y_side, scx)
        if not fw or not sd:
            out[name] = None
            continue
        fw_cm, sw_cm = fw * f_scale, sd * s_scale
        out[name] = (ramanujan_ellipse_circumference(fw_cm, sw_cm), fw_cm, sw_cm)
    return out


# ── split loading + per-subject prediction (shared by fit and eval) ─────────
def load_split(s3, split: str) -> tuple[dict, dict, dict]:
    meas = {r["subject_id"]: r for r in _csv(s3, f"{split}/measurements.csv")}
    hwg = {r["subject_id"]: r for r in _csv(s3, f"{split}/hwg_metadata.csv")}
    photo: dict[str, str] = {}
    for r in _csv(s3, f"{split}/subject_to_photo_map.csv"):
        photo.setdefault(r["subject_id"], r["photo_id"])
    return meas, hwg, photo


def subject_pred(s3, split, sid, hwg, photo) -> dict[str, float | None]:
    front = _mask(s3, f"{split}/mask/{photo[sid]}.png")
    side = _mask(s3, f"{split}/mask_left/{photo[sid]}.png")
    return predict(front, side, float(hwg[sid]["height_cm"]))


# ── fit train-derived params: B/C scales + D girth coefs (ONE train pass) ───
# scales: L1-optimal median(gt/circ) — the closed form pointsx-eval --fit-offsets
#   uses ("pulp but lighter", no LP), for pipelines B/C.
# girth:  bilinear least-squares circ = c0 + c1·fw + c2·sw + c3·fw·sw per
#   (sex, measure) — the learned width→girth map for pipeline D.
# Both fit on TRAIN, applied to the eval split — a real generalization test.
def fit_params(s3, n_fit: int) -> dict[str, dict]:
    meas, hwg, photo = load_split(s3, "train")
    subjects = [s for s in meas if s in photo and s in hwg]
    if n_fit:
        subjects = subjects[:n_fit]
    ratios: dict[str, dict[str, list]] = {}
    rows: dict[str, dict[str, list]] = {}  # sex -> m -> [(fw, sw, gt)]
    _guard_banner()  # fitting constants FROM BodyM → benchmark-only, non-shippable
    print(f"[fit] params on train n={len(subjects)} (scales + learned girth per sex×measure)")
    for i, sid in enumerate(subjects, 1):
        sex = hwg[sid]["gender"]
        pred = subject_pred(s3, "train", sid, hwg, photo)
        for m in CIRCUMFERENCES:
            v = pred.get(m)
            if v is None:
                continue
            circ, fw, sw = v
            gt = float(meas[sid][m])
            if circ > 0:
                ratios.setdefault(sex, {mm: [] for mm in CIRCUMFERENCES})[m].append(gt / circ)
            rows.setdefault(sex, {mm: [] for mm in CIRCUMFERENCES})[m].append((fw, sw, gt))
        if i % 200 == 0:
            print(f"  … fit {i}/{len(subjects)}")

    scales = {sex: {m: float(np.median(v)) for m, v in d.items() if v}
              for sex, d in ratios.items()}
    girth: dict[str, dict[str, list]] = {}
    for sex, d in rows.items():
        for m, pts in d.items():
            if len(pts) < 4:  # need >= #coefs for a stable bilinear fit
                continue
            fw = np.array([p[0] for p in pts]); sw = np.array([p[1] for p in pts])
            gt = np.array([p[2] for p in pts])
            A = np.column_stack([np.ones_like(fw), fw, sw, fw * sw])
            coef, *_ = np.linalg.lstsq(A, gt, rcond=None)
            girth.setdefault(sex, {})[m] = coef.tolist()

    print("\n=== fitted scales (value *= scale;  % = (scale-1)*100) ===")
    print(f"{'sex':8} " + " ".join(f"{m.split('_')[0]:>8}" for m in CIRCUMFERENCES))
    for sex in sorted(scales):
        print(f"{sex:8} " + " ".join(f"{scales[sex].get(m, 1.0):8.3f}" for m in CIRCUMFERENCES))
    print(f"[fit] learned girth coefs for sexes={sorted(girth)} "
          f"(bilinear circ=c0+c1·fw+c2·sw+c3·fw·sw)")
    return {"scales": scales, "girth": girth}


# ── metrics ─────────────────────────────────────────────────────────────────
def _stats(errs: list[float]) -> dict:
    a = np.array(errs)
    return {
        "n": int(a.size),
        "mae": float(np.abs(a).mean()),
        "rmse": float(np.sqrt((a ** 2).mean())),
        "bias": float(a.mean()),
        **{f"within_{int(t)}cm": float((np.abs(a) <= t).mean()) for t in TOL_LADDER},
        "wrong_size_rate": float((np.abs(a) > WRONG_SIZE_CM).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="testA", choices=("testA", "testB"))
    ap.add_argument("--n", type=int, default=0, help="limit eval subjects (0 = all)")
    ap.add_argument("--fit-n", type=int, default=300,
                    help="train subjects to fit the correction on (0 = all ~2018)")
    args = ap.parse_args()

    s3 = _client()
    split = args.split
    params = fit_params(s3, args.fit_n)  # B/C scales + D girth, fit on TRAIN
    scales = params["scales"]

    meas, hwg, photo = load_split(s3, split)
    subjects = [s for s in meas if s in photo and s in hwg]
    if args.n:
        subjects = subjects[: args.n]
    names = [n for n, _ in PIPELINES]
    base_name = names[0]  # A — the frozen reference every pipeline is scored vs
    print(f"\n[eval] split={split} subjects={len(subjects)} "
          f"pipelines={names} (scored vs {base_name})")

    # per-pipeline signed errors; buckets track the base (A) error only
    err = {n: {m: [] for m in CIRCUMFERENCES} for n in names}
    buckets: dict[str, dict[str, list]] = {}

    def bucket(key: str, m: str, e: float):
        buckets.setdefault(key, {mm: [] for mm in CIRCUMFERENCES})[m].append(e)

    for i, sid in enumerate(subjects, 1):
        sex = hwg[sid]["gender"]
        try:
            bmi = float(hwg[sid]["weight_kg"]) / (float(hwg[sid]["height_cm"]) / 100) ** 2
            bmi_band = "BMI<25" if bmi < 25 else ("BMI25-30" if bmi < 30 else "BMI>30")
        except (ValueError, ZeroDivisionError):
            bmi_band = "BMI?"
        raw = subject_pred(s3, split, sid, hwg, photo)  # base ellipse + widths, ONCE
        gts = {m: float(meas[sid][m]) for m in CIRCUMFERENCES}
        for name, fn in PIPELINES:
            pred = fn(raw, sex, params)
            for m in CIRCUMFERENCES:
                p = pred.get(m)
                if p is None:
                    continue
                err[name][m].append(p - gts[m])
        for m in CIRCUMFERENCES:  # buckets on base (A) error; raw[m]=(circ, fw, sw)
            if raw.get(m) is not None:
                bucket(f"sex={sex}", m, raw[m][0] - gts[m])
                bucket(bmi_band, m, raw[m][0] - gts[m])
        if i % 100 == 0:
            print(f"  … {i}/{len(subjects)}")

    metrics = {n: {m: _stats(err[n][m]) for m in CIRCUMFERENCES if err[n][m]} for n in names}
    overall = {n: float(np.mean([abs(e) for m in CIRCUMFERENCES for e in err[n][m]]))
               for n in names}

    # ── orchestrated comparison: every pipeline vs base A ─────────────────────
    exp = names[1:]  # experimental pipelines
    print(f"\n=== {split}: pipelines vs {base_name} — MAE cm (Δ = {base_name}−pipeline) ===")
    hdr = f"{'measure':7} {base_name:>7}"
    for n in exp:
        hdr += f" {n:>12} {'Δ':>6}"
    print(hdr)
    for m in CIRCUMFERENCES:
        if m not in metrics[base_name]:
            continue
        base_mae = metrics[base_name][m]["mae"]
        line = f"{m.split('_')[0]:7} {base_mae:7.1f}"
        for n in exp:
            mae = metrics[n][m]["mae"]
            d = base_mae - mae
            arrow = "✓" if d > 0.05 else ("✗" if d < -0.05 else "·")
            line += f" {mae:12.1f} {d:+5.1f}{arrow}"
        print(line)
    line = f"{'OVERALL':7} {overall[base_name]:7.1f}"
    for n in exp:
        d = overall[base_name] - overall[n]
        line += f" {overall[n]:12.1f} {d:+5.1f}{'✓' if d > 0.05 else ('✗' if d < -0.05 else '·')}"
    print(line)

    print(f"\n=== {base_name} MAE by bucket ===")
    print(f"{'bucket':10} " + " ".join(f"{m.split('_')[0]:>7}" for m in CIRCUMFERENCES))
    for key in sorted(buckets):
        row = buckets[key]
        cells = [f"{np.abs(row[m]).mean():7.1f}" if row[m] else f"{'—':>7}"
                 for m in CIRCUMFERENCES]
        print(f"{key:10} " + " ".join(cells))

    # ── persist run ─────────────────────────────────────────────────────────
    runs = EVAL_DIR / "reports"
    runs.mkdir(exist_ok=True)
    n_prev = len([d for d in runs.glob("run_*") if d.is_dir()])
    run_dir = runs / f"run_{n_prev + 1:03d}"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(json.dumps({
        "split": split, "subjects": len(subjects), "fit_n": args.fit_n,
        "scales": scales, "girth": params["girth"],
        "pipelines": {n: metrics[n] for n in names},
        "overall": overall,
        "buckets": {k: {m: _stats(v[m]) for m in CIRCUMFERENCES if v[m]}
                    for k, v in buckets.items()},
    }, indent=2))
    print(f"\n[done] wrote {run_dir / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
