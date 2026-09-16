"""Girth model on ANSUR II: circumference from height + the six torso widths (survey B1/3b).

Feature sets compared with 5-fold CV OLS, exact and with N(0, sigma) noise on every width at
test time (the pipeline's widths are silhouette-in-clothes, not calipers):
  ellipse     Ramanujan ellipse of the site's own breadth/depth (what production does today,
              before the per-sex constant)
  own         the site's own breadth + depth
  own+H       + stature
  all6+H      all six widths (chest/waist/hip breadth+depth) + stature   <- fusion
  all6+H+x    + the site's own breadth*depth product (pipeline-D style bilinear term)
Thigh has no breadth/depth in ANSUR: all6+H only, vs the height-only floor.
Weight is fitted with the same inputs (3c is a by-product of this model).

Run from scripts/ansur/:  ../../.venv/Scripts/python girth.py [--sigma 1.5]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from _common import FILES, col, ellipse, load

WIDTH_COLS = ["chestbreadth", "chestdepth", "waistbreadth", "waistdepth", "hipbreadth", "buttockdepth"]
SITES = {
    "chest": ("chestbreadth", "chestdepth", "chestcircumference"),
    "waist": ("waistbreadth", "waistdepth", "waistcircumference"),
    "hip": ("hipbreadth", "buttockdepth", "buttockcircumference"),
    "thigh": (None, None, "thighcircumference"),
}
K = 5


def cv(X: np.ndarray, y: np.ndarray, noisy: list[int], sigma: float, seed: int = 0) -> float:
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    rng = np.random.default_rng(seed)
    errs: list[float] = []
    for f in range(K):
        te = idx[f::K]
        tr = sorted(set(idx) - set(te))
        w, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), X[tr]], y[tr], rcond=None)
        Xt = X[te].copy()
        for c in noisy:
            Xt[:, c] += rng.normal(0.0, sigma, len(te))
        errs += list(np.abs(np.c_[np.ones(len(te)), Xt] @ w - y[te]))
    return float(np.mean(errs))


def fit(X: np.ndarray, y: np.ndarray) -> list[float]:
    w, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)
    return [float(v) for v in w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=1.5, help="width noise at test time, cm")
    ap.add_argument("--out", default="girth_coefs.json")
    args = ap.parse_args()
    coefs: dict = {"features": ["stature", *WIDTH_COLS], "sites": {}}
    for fn, sex in FILES:
        rows = load(fn)
        H = col(rows, "stature")
        Wd = {c: col(rows, c) for c in WIDTH_COLS}
        all6 = np.c_[[Wd[c] for c in WIDTH_COLS]].T
        print(f"\n=== {sex} n={len(rows)}  (noise sigma {args.sigma} cm on widths) ===")
        print(f"{'site':6} {'sd':>5} | {'ellipse':>13} | {'own':>13} | {'own+H':>13} | {'all6+H':>13} | {'all6+H+x':>13}")
        coefs["sites"].setdefault(sex, {})
        for site, (bk, dk, ck) in SITES.items():
            y = col(rows, ck)
            cells = []
            if bk:
                B, D = Wd[bk], Wd[dk]
                E = ellipse(B, D)
                for X, noisy in ((np.c_[E], []), (np.c_[B, D], [0, 1]), (np.c_[B, D, H], [0, 1])):
                    if X is not None and X.shape[1] == 1:
                        # noise on the ellipse input: perturb B, D then recompute
                        idx = list(range(len(y)))
                        random.Random(0).shuffle(idx)
                        rng = np.random.default_rng(0)
                        errs = []
                        for f in range(K):
                            te = idx[f::K]
                            tr = sorted(set(idx) - set(te))
                            w, *_ = np.linalg.lstsq(np.c_[np.ones(len(tr)), E[tr]], y[tr], rcond=None)
                            En = ellipse(B[te] + rng.normal(0, args.sigma, len(te)),
                                         D[te] + rng.normal(0, args.sigma, len(te)))
                            errs += list(np.abs(np.c_[np.ones(len(te)), En] @ w - y[te]))
                        cells.append(f"{cv(X, y, [], 0):5.2f}/{np.mean(errs):5.2f}")
                    else:
                        cells.append(f"{cv(X, y, noisy, 0):5.2f}/{cv(X, y, noisy, args.sigma):5.2f}")
            else:
                cells += ["      -     "] * 3
            X6 = np.c_[H, all6]
            noisy6 = list(range(1, 7))
            cells.append(f"{cv(X6, y, noisy6, 0):5.2f}/{cv(X6, y, noisy6, args.sigma):5.2f}")
            if bk:
                Xx = np.c_[H, all6, Wd[bk] * Wd[dk]]
                cells.append(f"{cv(Xx, y, noisy6, 0):5.2f}/{cv(Xx, y, noisy6, args.sigma):5.2f}")
            else:
                cells.append(f"H-only {cv(np.c_[H], y, [], 0):5.2f}")
            print(f"{site:6} {y.std():5.2f} | " + " | ".join(f"{c:>13}" for c in cells))
            coefs["sites"][sex][site] = fit(X6, y)
        Wt = col(rows, "weightkg")
        print(f"weight sd {Wt.std():5.2f} | H-only {cv(np.c_[H], Wt, [], 0):5.2f} | all6+H "
              f"{cv(np.c_[H, all6], Wt, noisy6, 0):5.2f}/{cv(np.c_[H, all6], Wt, noisy6, args.sigma):5.2f} kg")
        coefs["sites"][sex]["weight_kg"] = fit(np.c_[H, all6], Wt)
    json.dump(coefs, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}  (each site: [intercept, stature, {', '.join(WIDTH_COLS)}])")
    print("cells are exact/noisy CV MAE in cm")


if __name__ == "__main__":
    main()
