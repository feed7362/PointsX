"""Build Anny bodies that hit a real person's measurements, taken from ANSUR II.

The deleted dataset sampled shape parameters at random and got a population with a waist mean of
111.6 cm. Here every body starts from an actual ANSUR II row — a real person's stature, chest,
waist, hip, thigh and neck — and the model parameters are solved until the mesh measures that.
The joint distribution is therefore real by construction, and the ground truth is exact: the same
`measure_mesh` definitions the eval and the app use.

Solver: each control is monotone and close to linear over its range (see the smoke test), so a
secant step converges in two or three measurements. Sites are solved in order of how much they
disturb each other — stature, then girths — and the whole set is swept twice.

usage:
    .venv/Scripts/python scripts/synthetic/build_bodies.py --n 5
    .venv/Scripts/python scripts/synthetic/build_bodies.py --n 500 --out runs/synthetic/bodies
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from measure_mesh import LANDMARK_H, _hull_perimeter_cm, _pick, _slice_components  # noqa: E402
from measure_mesh import measure_mesh  # noqa: E402

ANSUR_FILES = {"female": REPO / "scripts" / "ansur" / "a.csv", "male": REPO / "scripts" / "ansur" / "m.csv"}
# ANSUR column (mm, /10 = cm) -> our canonical id
TARGETS = {
    "stature": "height_cm",
    "chestcircumference": "chest_circumference",
    "waistcircumference": "waist_circumference",
    "buttockcircumference": "hip_circumference",
    "thighcircumference": "thigh_circumference",
    "neckcircumference": "neck_circumference",
}
# canonical id -> (control, kind). "phenotype" values live in [0, 1] (extrapolated a little further),
# "local" ones in [-1, 1].
CONTROLS: dict[str, tuple[str, str]] = {
    "height_cm": ("height", "phenotype"),
    "chest_circumference": ("measure-bust-circ-incr", "local"),
    "waist_circumference": ("measure-waist-circ-incr", "local"),
    "hip_circumference": ("measure-hips-circ-incr", "local"),
    "thigh_circumference": ("measure-thigh-circ-incr", "local"),
    "neck_circumference": ("measure-neck-circ-incr", "local"),
}
SOLVE_ORDER = ["height_cm", "chest_circumference", "waist_circumference", "hip_circumference",
               "thigh_circumference", "neck_circumference"]
LIMITS = {"phenotype": (-0.5, 1.5), "local": (-2.0, 2.0)}
TOL_CM = 0.5


def ansur_targets(sex: str, n: int, seed: int) -> list[dict[str, float]]:
    rows = list(csv.DictReader(ANSUR_FILES[sex].open(encoding="latin-1")))
    picked = random.Random(seed).sample(rows, n)
    return [{mid: float(r[col]) / 10.0 for col, mid in TARGETS.items()} for r in picked]


class Body:
    """One Anny body whose controls can be nudged and re-measured cheaply."""

    def __init__(self, model, torch, sex: str):
        self.m, self.torch, self.sex = model, torch, sex
        self.pose = torch.eye(4, dtype=torch.float32)[None, None].repeat(1, model.bone_count, 1, 1)
        self.phen = {k: 0.5 for k in model.phenotype_labels}
        self.phen["gender"] = 0.0 if sex == "female" else 1.0
        self.local: dict[str, float] = {}
        self.faces = model.faces.numpy()

    def verts(self) -> np.ndarray:
        out = self.m(pose_parameters=self.pose, phenotype_kwargs=self.phen,
                     local_changes_kwargs=self.local)
        return out["vertices"].squeeze(0).detach().numpy()

    def set(self, control: str, kind: str, value: float) -> None:
        lo, hi = LIMITS[kind]
        value = float(np.clip(value, lo, hi))
        (self.phen if kind == "phenotype" else self.local)[control] = value

    def get(self, control: str, kind: str) -> float:
        return (self.phen if kind == "phenotype" else self.local).get(control, 0.5 if kind == "phenotype" else 0.0)

    def measure_one(self, mid: str) -> float:
        """One site only — a full measure_mesh is ~0.6 s, a single slice ~0.1 s."""
        import trimesh
        v = self.verts()
        mesh = trimesh.Trimesh(vertices=v.astype(np.float64), faces=self.faces, process=False)
        z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
        if mid == "height_cm":
            return (z1 - z0) * 100.0
        stature = z1 - z0
        axis_x = float(np.median(v[:, 0]))
        i = 0 if self.sex == "female" else 1
        site = {"chest_circumference": "chest", "waist_circumference": "waist", "hip_circumference": "hip",
                "thigh_circumference": "crotch", "neck_circumference": "neck"}[mid]
        z = z0 + LANDMARK_H[site][i] * stature
        if mid == "thigh_circumference":
            z -= 0.06 * stature
        if mid == "neck_circumference":
            best = None
            for zz in np.linspace(z + 0.005 * stature, z + 0.05 * stature, 8):
                c = _pick(_slice_components(mesh, float(zz)), axis_x, "torso")
                if c is not None:
                    p = _hull_perimeter_cm(c)
                    best = p if best is None or p < best else best
            return best if best is not None else float("nan")
        comp = _pick(_slice_components(mesh, z), axis_x,
                     "limb" if mid == "thigh_circumference" else "torso")
        return _hull_perimeter_cm(comp) if comp is not None else float("nan")

    def _secant(self, mid: str, control: str, kind: str, goal: float, step: float) -> None:
        x0 = self.get(control, kind)
        y0 = self.measure_one(mid)
        if not np.isfinite(y0) or abs(y0 - goal) <= TOL_CM:
            return
        self.set(control, kind, x0 + step)
        y1 = self.measure_one(mid)
        slope = (y1 - y0) / step
        if not np.isfinite(y1) or abs(slope) < 1e-6:
            self.set(control, kind, x0)
            return
        self.set(control, kind, x0 + (goal - y0) / slope)
        y2 = self.measure_one(mid)
        if np.isfinite(y2) and abs(y2 - goal) > abs(y0 - goal):
            self.set(control, kind, x0)

    def solve(self, target: dict[str, float], sweeps: int = 2) -> dict[str, float]:
        # Coarse first: the `weight` phenotype moves every girth at once and has a far wider range
        # than the per-site modifiers, which otherwise saturate on large bodies (a 110 cm waist came
        # out 12 cm short). Aim it at the waist, then let each site fine-tune.
        if "height_cm" in target:
            self._secant("height_cm", *CONTROLS["height_cm"], target["height_cm"], 0.1)
        if "waist_circumference" in target:
            self._secant("waist_circumference", "weight", "phenotype", target["waist_circumference"], 0.25)
        for _ in range(sweeps):
            for mid in SOLVE_ORDER:
                if mid not in target:
                    continue
                control, kind = CONTROLS[mid]
                goal = target[mid]
                x0 = self.get(control, kind)
                y0 = self.measure_one(mid)
                if not np.isfinite(y0):
                    break
                if abs(y0 - goal) <= TOL_CM:
                    continue
                step = 0.25 if kind == "local" else 0.1
                self.set(control, kind, x0 + step)
                y1 = self.measure_one(mid)
                slope = (y1 - y0) / step
                if not np.isfinite(y1) or abs(slope) < 1e-6:
                    self.set(control, kind, x0)
                    continue
                self.set(control, kind, x0 + (goal - y0) / slope)      # secant step
                y2 = self.measure_one(mid)
                if np.isfinite(y2) and abs(y2 - goal) > abs(y0 - goal):
                    self.set(control, kind, x0)                        # worse: keep what we had
        return measure_mesh(self.verts(), self.faces, self.sex)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="bodies per sex")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "runs" / "synthetic" / "bodies"))
    ap.add_argument("--export-obj", action="store_true", help="also write each body as .obj for Blender")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    import anny
    import torch
    model = anny.Anny(local_changes="all", extrapolate_phenotypes=True).to(dtype=torch.float32)

    results, residuals = [], {mid: [] for mid in SOLVE_ORDER}
    t0 = time.time()
    for sex in ("female", "male"):
        for i, target in enumerate(ansur_targets(sex, args.n, args.seed)):
            body = Body(model, torch, sex)
            got = body.solve(target)
            row = {"body_id": f"{sex[0]}{i:05d}", "sex": sex,
                   "phenotypes": {k: round(v, 4) for k, v in body.phen.items()},
                   "local_changes": {k: round(v, 4) for k, v in body.local.items()},
                   "target": {k: round(v, 1) for k, v in target.items()},
                   "measured": {k: v for k, v in got.items()}}
            for mid in SOLVE_ORDER:
                if mid in target and mid in got:
                    residuals[mid].append(got[mid] - target[mid])
            results.append(row)
            if args.export_obj:
                import trimesh
                trimesh.Trimesh(vertices=body.verts(), faces=body.faces, process=False).export(
                    out_dir / f"{row['body_id']}.obj")
            print(f"  {row['body_id']} " + "  ".join(
                f"{mid.split('_')[0]:5} {target[mid]:5.1f}->{got.get(mid, float('nan')):5.1f}"
                for mid in SOLVE_ORDER if mid in target))
    (out_dir / "bodies.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    n = len(results)
    print(f"\n=== residuals over {n} bodies ({time.time() - t0:.0f} s, {(time.time() - t0) / n:.1f} s/body) ===")
    print(f"  {'measurement':26}{'mean':>8}{'MAE':>8}{'max|.|':>9}{'within 1 cm':>13}")
    for mid, v in residuals.items():
        if not v:
            continue
        a = np.abs(v)
        print(f"  {mid:26}{np.mean(v):+8.2f}{a.mean():8.2f}{a.max():9.2f}{(a <= 1).mean() * 100:12.0f} %")
    print(f"\nwritten to {out_dir / 'bodies.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
