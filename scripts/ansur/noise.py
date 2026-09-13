"""Error budget: fused (ellipse+H+W) MAE vs Gaussian width noise and vs clothing inflation."""
import numpy as np
from _common import FILES, SITES, col, cv_ols, ellipse, load

rng = np.random.default_rng(0)
NOISE_SD = (0, 0.5, 1.0, 1.5, 2.0, 3.0)
for fn, lbl in FILES:
    rows = load(fn)
    H, W = col(rows, "stature"), col(rows, "weightkg")
    BMI = W / (H / 100) ** 2
    print(f"\n=== {lbl}: fusion MAE (cm) vs sd of Gaussian noise on breadth & depth [cm]")
    print(f"{'site':6s} " + " ".join(f"sd={s:<4}" for s in NOISE_SD) + "  | no-photo H+W")
    for name in ("chest", "waist", "hip"):
        bk, dk, ck = SITES[name]
        y, B, D = col(rows, ck), col(rows, bk), col(rows, dk)
        out = []
        for sd in NOISE_SD:
            Bn, Dn = B + rng.normal(0, sd, len(B)), D + rng.normal(0, sd, len(D))
            out.append(cv_ols(np.c_[ellipse(Bn, Dn), Bn, Dn, H, W, BMI], y))
        print(f"{name:6s} " + " ".join(f"{v:6.2f} " for v in out) + f" | {cv_ols(np.c_[H, W, BMI], y):5.2f}")
    print("  clothing inflation U(1.00,1.15) on breadth & depth, no Gaussian noise:")
    for name in ("chest", "waist"):
        bk, dk, ck = SITES[name]
        y, B, D = col(rows, ck), col(rows, bk), col(rows, dk)
        f = rng.uniform(1.0, 1.15, len(B))
        E = ellipse(B * f, D * f)
        print(f"   {name:6s} fusion={cv_ols(np.c_[E, B * f, D * f, H, W, BMI], y):5.2f}   ellipse-only+scale={cv_ols(np.c_[E], y):5.2f}")
