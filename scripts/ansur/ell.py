"""Ramanujan ellipse bias vs real circumference, and the per-site/sex scale k."""
import numpy as np
from _common import FILES, SITES, col, ellipse, load

for fn, lbl in FILES:
    rows = load(fn)
    print(f"\n=== {lbl}  n={len(rows)}")
    for name, (bk, dk, ck) in SITES.items():
        if bk is None:
            continue
        c = col(rows, ck)
        e = ellipse(col(rows, bk), col(rows, dk))
        k = float(np.median(c / e))
        print(f"  {name:6s} ellipse bias={np.mean(e - c):+6.2f} cm  MAE={np.mean(np.abs(e - c)):5.2f}  "
              f"sd={np.std(e - c):4.2f}  median c/ellipse={k:.4f}  MAE after scale k: {np.mean(np.abs(e * k - c)):5.2f}")
