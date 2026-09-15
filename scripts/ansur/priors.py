"""Fit neck and back-width priors on ANSUR II (public domain, shippable).

Which inputs to use is decided by 5-fold CV MAE with realistic noise on the photo-derived inputs:
at runtime height and sex are exact, chest/waist come from the pipeline (app GT MAE ~4 cm).

Run from scripts/ansur/ (expects a.csv / m.csv, see README.md):
    ../../.venv/Scripts/python priors.py
"""
from __future__ import annotations

import random

import numpy as np

from _common import FILES, col, load

TARGETS = {
    "neck_circumference": "neckcircumferencebase",  # corpus GT matches the base girth (F 36.4 vs 37.1)
    "back_width_scapular": "interscyei",           # across the back between posterior axillary folds
}
FEATURE_SETS = {
    "mean": [],
    "H": ["stature"],
    "H+chest": ["stature", "chestcircumference"],
    "H+chest+waist": ["stature", "chestcircumference", "waistcircumference"],
}
NOISY = {"chestcircumference", "waistcircumference"}
NOISE_SIGMA_CM = (0.0, 5.0)  # MAE 4 cm ~ sigma 5 cm
K = 5


def cv(X: np.ndarray, y: np.ndarray, noisy_cols: list[int], sigma: float, seed: int = 0) -> float:
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    rng = np.random.default_rng(seed)
    errs: list[float] = []
    for f in range(K):
        te = idx[f::K]
        tr = sorted(set(idx) - set(te))
        A = np.c_[np.ones(len(tr)), X[tr]]
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        Xt = X[te].copy()
        for c in noisy_cols:
            Xt[:, c] += rng.normal(0.0, sigma, len(te))
        errs += list(np.abs(np.c_[np.ones(len(te)), Xt] @ w - y[te]))
    return float(np.mean(errs))


def main() -> None:
    for fn, sex in FILES:
        rows = load(fn)
        print(f"\n=== {sex} (n={len(rows)}) ===")
        for mid, target in TARGETS.items():
            y = col(rows, target)
            print(f"{mid} <- ANSUR {target}: mean {y.mean():.1f} sd {y.std():.1f}")
            for name, feats in FEATURE_SETS.items():
                X = np.c_[[col(rows, c) for c in feats]].T if feats else np.zeros((len(y), 0))
                noisy = [i for i, c in enumerate(feats) if c in NOISY]
                maes = [cv(X, y, noisy, s) for s in NOISE_SIGMA_CM]
                w, *_ = np.linalg.lstsq(np.c_[np.ones(len(y)), X], y, rcond=None)
                coefs = ", ".join(f"{v:+.4f}" for v in w)
                print(f"  {name:15} CV MAE exact {maes[0]:.2f}  noisy {maes[1]:.2f}   coefs [{coefs}]")


if __name__ == "__main__":
    main()
