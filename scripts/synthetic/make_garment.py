"""Build a garment mesh for one body: an offset shell that Blender's cloth solver then drapes.

Buying or downloading garment assets brings a licence question with it, and a fixed garment fits
only the body it was modelled on. Here the garment starts as the body's own surface pushed outward
by an ease allowance — the amount of room a garment has over the body — which is exactly the axis
the product cares about: a tight top adds ~1 cm to every width, a loose one 4-8 cm and it hangs
rather than follows. The shell is only a starting shape; the drape comes from the cloth simulation
in blender_render.py, because a shell alone is the shrink-wrap mistake the 2026-07 review called out.

usage:
    .venv/Scripts/python scripts/synthetic/make_garment.py --body runs/synthetic/bench/f00000.obj \
        --out runs/synthetic/garments --ease 2.0        # tight
    ... --ease 6.0                                      # loose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from measure_mesh import LANDMARK_H  # noqa: E402

# Garment coverage as fractions of stature, measured from the floor.
PIECES = {
    # a top: from just below the hip up to the shoulders, sleeveless
    "top": {"z_lo": 0.44, "z_hi": 0.82, "ease_scale": 1.0},
    # trousers: from the waist down to the ankles
    "trousers": {"z_lo": 0.05, "z_hi": 0.62, "ease_scale": 1.0},
}


def offset_shell(mesh, z_lo: float, z_hi: float, ease_m: float):
    """The body's surface between two heights, pushed out along its normals by `ease_m`."""
    import trimesh
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    centroids = v[f].mean(axis=1)
    keep = (centroids[:, 2] >= z_lo) & (centroids[:, 2] <= z_hi)
    if not keep.any():
        return None
    sub = mesh.submesh([np.where(keep)[0]], append=True)
    sv = np.asarray(sub.vertices)
    normals = np.asarray(sub.vertex_normals)
    # Push outward horizontally only: a garment hangs off the body, it does not grow a taller collar.
    horiz = normals.copy()
    horiz[:, 2] = 0.0
    lengths = np.linalg.norm(horiz, axis=1, keepdims=True)
    horiz = np.divide(horiz, np.where(lengths > 1e-6, lengths, 1.0))
    return trimesh.Trimesh(vertices=sv + horiz * ease_m, faces=np.asarray(sub.faces), process=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ease", type=float, default=2.0, help="garment ease over the body, in cm")
    ap.add_argument("--sex", default="female", choices=("female", "male"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import trimesh
    mesh = trimesh.load(args.body, process=False)
    v = np.asarray(mesh.vertices)
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    stature = z1 - z0
    ease_m = args.ease / 100.0

    written = []
    for name, spec in PIECES.items():
        shell = offset_shell(mesh, z0 + spec["z_lo"] * stature, z0 + spec["z_hi"] * stature,
                             ease_m * spec["ease_scale"])
        if shell is None:
            continue
        path = args.out / f"{args.body.stem}_{name}_ease{args.ease:g}.obj"
        shell.export(path)
        written.append((name, path, len(shell.vertices)))

    hip_z = z0 + LANDMARK_H["hip"][0 if args.sex == "female" else 1] * stature
    print(f"{args.body.stem}: stature {stature * 100:.1f} cm, ease {args.ease:g} cm, hip row z={hip_z:.3f}")
    for name, path, n in written:
        print(f"  {name:9} {n:5} verts -> {path.name}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
