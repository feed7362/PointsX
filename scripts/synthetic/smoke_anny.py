"""Step 2 smoke test: are Anny bodies anatomically plausible enough to be ground truth?

Samples N bodies over the phenotype space, measures each mesh with `measure_mesh` (ANSUR landmark
heights + convex-hull slice perimeter) and checks them the way the plan's gate 1 asks:

  * per-sex means within ~2 cm of ANSUR II,
  * >= 99 % of bodies passing `pointsx.gt_sanity` (the same gate that rejected the old set),
  * measurements monotonic in the weight phenotype.

usage: .venv/Scripts/python scripts/synthetic/smoke_anny.py [--n 20] [--out runs/synthetic/anny]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from measure_mesh import measure_mesh  # noqa: E402

from pointsx.gt_sanity import check_gt  # noqa: E402

# ANSUR II means (cm) for a sanity reference — scripts/ansur/
ANSUR = {
    "female": {"height_cm": 162.8, "chest_circumference": 94.7, "waist_circumference": 86.1,
               "hip_circumference": 102.1, "thigh_circumference": 61.6, "neck_circumference": 33.0},
    "male": {"height_cm": 175.6, "chest_circumference": 105.9, "waist_circumference": 89.5,
             "hip_circumference": 105.2, "thigh_circumference": 61.9, "neck_circumference": 39.8},
}
SITES = ["height_cm", "chest_circumference", "waist_circumference", "hip_circumference",
         "thigh_circumference", "neck_circumference"]


def build(model, torch, gender: float, **phen):
    pose = torch.eye(4, dtype=torch.float32)[None, None].repeat(1, model.bone_count, 1, 1)
    kwargs = {k: 0.5 for k in model.phenotype_labels}
    kwargs.update(gender=gender, **phen)
    out = model(pose_parameters=pose, phenotype_kwargs=kwargs)
    return out["vertices"].squeeze(0).detach().numpy(), model.faces.numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default=str(REPO / "runs" / "synthetic" / "anny"))
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    import anny
    import torch
    model = anny.Anny().to(dtype=torch.float32)
    rng = np.random.default_rng(0)

    bodies = []
    for i in range(args.n):
        female = i % 2 == 0
        # MakeHuman semantics (Anny inherits them): age 0 = infant, 0.5 = 25 years, 1 = 90.
        # Anything below 0.5 is a child — the first run produced a 127 cm "male".
        phen = dict(height=float(rng.uniform(0.25, 0.75)), weight=float(rng.uniform(0.25, 0.75)),
                    muscle=float(rng.uniform(0.35, 0.65)), age=float(rng.uniform(0.52, 0.72)),
                    proportions=float(rng.uniform(0.4, 0.6)))
        v, f = build(model, torch, gender=0.0 if female else 1.0, **phen)
        sex = "female" if female else "male"
        gt = measure_mesh(v, f, sex)
        gt.update(sex=sex, body_id=i, **{f"phen_{k}": round(val, 3) for k, val in phen.items()})
        bodies.append(gt)
        print(f"  body {i:02d} {sex:6} " + "  ".join(f"{k.split('_')[0]:5} {gt.get(k, float('nan')):6.1f}"
                                                     for k in SITES))
    (out_dir / "bodies.json").write_text(json.dumps(bodies, indent=1), encoding="utf-8")

    print(f"\n=== gate 1a: per-sex means vs ANSUR II (n={args.n}) ===")
    print(f"  {'measurement':26}{'anny':>8}{'ansur':>8}{'diff':>8}")
    worst = 0.0
    for sex in ("female", "male"):
        rows = [b for b in bodies if b["sex"] == sex]
        print(f"  -- {sex} (n={len(rows)})")
        for k in SITES:
            vals = [b[k] for b in rows if k in b]
            if not vals:
                print(f"  {k:26}{'—':>8}"); continue
            mean = float(np.mean(vals)); diff = mean - ANSUR[sex][k]
            worst = max(worst, abs(diff))
            print(f"  {k:26}{mean:8.1f}{ANSUR[sex][k]:8.1f}{diff:+8.1f}")

    print("\n=== gate 1b: pointsx.gt_sanity ===")
    bad = []
    for b in bodies:
        gt = {k: v for k, v in b.items() if k in check_gt.__globals__["RATIO_BOUNDS"][0][0] or k.endswith("circumference")
              or k in ("neck_base_height", "leg_length_inner_seam")}
        c = check_gt(b["height_cm"], {k: v for k, v in b.items()
                                      if isinstance(v, (int, float)) and k != "height_cm" and not k.startswith("phen_")})
        if c.exclude_subject or c.dropped:
            bad.append((b["body_id"], c.reasons))
    print(f"  {len(bodies) - len(bad)}/{len(bodies)} pass")
    for bid, why in bad[:5]:
        print(f"   body {bid}: {'; '.join(why)}")

    print("\n=== gate 1c: monotonic in the weight phenotype (female) ===")
    fem = sorted((b for b in bodies if b["sex"] == "female"), key=lambda b: b["phen_weight"])
    for k in ("waist_circumference", "hip_circumference"):
        series = [b[k] for b in fem if k in b]
        rho = float(np.corrcoef(range(len(series)), series)[0, 1]) if len(series) > 2 else float("nan")
        print(f"  {k:26} corr with weight rank {rho:+.2f}   {series[0]:.0f} -> {series[-1]:.0f} cm")

    print(f"\nworst mean deviation from ANSUR: {worst:.1f} cm   bodies written to {out_dir / 'bodies.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
