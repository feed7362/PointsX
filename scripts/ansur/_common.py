"""Shared ANSUR II helpers (see README.md). Uses the PRODUCTION ellipse formula."""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from pointsx.circumference import ramanujan_ellipse_circumference

# site -> (breadth col, depth col, circumference col); None = no breadth/depth in ANSUR
SITES: dict[str, tuple[str | None, str | None, str]] = {
    "chest": ("chestbreadth", "chestdepth", "chestcircumference"),
    "waist": ("waistbreadth", "waistdepth", "waistcircumference"),
    "hip": ("hipbreadth", "buttockdepth", "buttockcircumference"),
    "thigh": (None, None, "thighcircumference"),
    "neck": (None, None, "neckcircumference"),
    "bicep": (None, None, "bicepscircumferenceflexed"),
}
FILES = (("a.csv", "female"), ("m.csv", "male"))


def load(fn: str) -> list[dict]:
    return list(csv.DictReader(open(fn, encoding="latin-1")))


def col(rows: list[dict], key: str) -> np.ndarray:
    """ANSUR stores mm (and weightkg as kg*10): divide by 10 -> cm / kg."""
    return np.array([float(r[key]) / 10 for r in rows])


def ellipse(breadth_cm: np.ndarray, depth_cm: np.ndarray) -> np.ndarray:
    """Production Ramanujan formula, which takes FULL widths (not semi-axes)."""
    return np.array([ramanujan_ellipse_circumference(b, d) for b, d in zip(breadth_cm, depth_cm)])


def cv_ols(X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 0) -> float:
    """k-fold cross-validated MAE of an OLS fit with intercept."""
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    folds = [idx[i::k] for i in range(k)]
    errs: list[float] = []
    for f in range(k):
        te = folds[f]
        ts = set(te)
        tr = [i for i in idx if i not in ts]
        A = np.c_[np.ones(len(tr)), X[tr]]
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        errs += list(np.abs(np.c_[np.ones(len(te)), X[te]] @ w - y[te]))
    return float(np.mean(errs))
