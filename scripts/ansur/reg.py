"""5-fold CV MAE: height only | height+weight+BMI | ellipse | ellipse+H+W (or +hip circ for sites without breadth/depth)."""
import numpy as np
from _common import FILES, SITES, col, cv_ols, ellipse, load

for fn, lbl in FILES:
    rows = load(fn)
    H, W = col(rows, "stature"), col(rows, "weightkg")
    BMI = W / (H / 100) ** 2
    hipc = col(rows, "buttockcircumference")
    print(f"\n=== {lbl} n={len(rows)}   stature mean {H.mean():.1f}  weight mean {W.mean():.1f}kg")
    print(f"{'site':6s} {'sd(GT)':>7s} | {'H only':>8s} | {'H+W+BMI':>9s} | {'ellipse':>8s} | {'ell+H+W':>8s} | {'H+W+hipC':>9s}")
    for name, (bk, dk, ck) in SITES.items():
        y = col(rows, ck)
        f_h, f_hw = cv_ols(np.c_[H], y), cv_ols(np.c_[H, W, BMI], y)
        if bk is not None:
            B, D = col(rows, bk), col(rows, dk)
            E = ellipse(B, D)
            print(f"{name:6s} {y.std():7.2f} | {f_h:8.2f} | {f_hw:9.2f} | {cv_ols(np.c_[E], y):8.2f} | "
                  f"{cv_ols(np.c_[E, B, D, H, W, BMI], y):8.2f} |")
        else:
            print(f"{name:6s} {y.std():7.2f} | {f_h:8.2f} | {f_hw:9.2f} | {'-':>8s} | {'-':>8s} | "
                  f"{cv_ols(np.c_[H, W, BMI, hipc], y):9.2f}")
