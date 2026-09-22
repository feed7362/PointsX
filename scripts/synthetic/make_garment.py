"""Build a garment for one body as a hanging tube, which Blender's cloth solver then drapes.

The first version pushed the body's own surface outward by a fixed ease. The silhouette ratios it
produced matched real clothing, but the shape did not: an offset shell is the body's shape plus a
constant, so it follows every curve and can never BRIDGE a concavity. Rendered, it looked like an
inflated body — exactly the shrink-wrap the 2026-07 review rejected, and a segmenter trained on it
would learn "shrink the outline by 7 %" rather than "find the body under the clothes".

Real fabric hangs. A top is suspended from the shoulders, so at the waist it cannot be narrower than
the chest above it; trousers hang from the waistband over the hip. That gives the rule used here:

    garment radius at height z = max(body radius at any height above z, within the piece) + ease

which bridges the waist by construction and still hugs the widest parts. The tube is a loft of
elliptical rings (x and y radii taken separately, so a body's depth and breadth are both respected);
drape, wrinkles and how it falls between the legs come from the cloth simulation in
blender_render.py.

usage:
    .venv/Scripts/python scripts/synthetic/make_garment.py --body runs/synthetic/bench/f00000.obj \
        --out runs/synthetic/garments --ease 2.0     # fitted
    ... --ease 6.0                                   # loose
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

# Coverage as fractions of stature, floor-relative. The top stops at the shoulders and the arms are
# excluded entirely (sleeveless) — a sleeve needs the arm's own axis, not the body's, and the arm is
# where the hardest case lives (T6: on a side view the arm overlaps the torso).
PIECES = {
    "top": {"z_lo": 0.44, "z_hi": 0.80},
    # Trousers are one tube over the hip and two below the crotch — as a single tube they render as
    # a skirt, closing the gap between the legs that the pipeline uses to find the thigh and the
    # inner seam. ANSUR crotch height is 0.48 of stature.
    "trousers": {"z_lo": 0.05, "z_hi": 0.60, "split_below": 0.48},
}
RINGS = 48          # loft resolution along the body
SEGMENTS = 64       # around the ring
# Only vertices this close to the body's axis count as torso/leg: excludes the arms, which otherwise
# set the "radius" of a chest ring and would inflate the garment into a poncho.
_TORSO_REACH = 0.105  # fraction of stature
# Same idea for a single leg: without it the ring at the crotch picks up the whole buttock, the
# running maximum carries that width down the leg, and the two trouser legs merge into a skirt.
_LEG_REACH = 0.08


def body_profile(v: np.ndarray, z_lo: float, z_hi: float, axis: np.ndarray, stature: float):
    """Per-height half-extents of the torso in x and y, ignoring the arms."""
    reach = _TORSO_REACH * stature
    zs = np.linspace(z_lo, z_hi, RINGS)
    step = (z_hi - z_lo) / max(RINGS - 1, 1)
    rx, ry, cx, cy = [], [], [], []
    for z in zs:
        band = v[(v[:, 2] >= z - step) & (v[:, 2] <= z + step)]
        if len(band):
            near = band[np.abs(band[:, 0] - axis[0]) <= reach]
            band = near if len(near) >= 8 else band
        if len(band) < 4:
            rx.append(np.nan); ry.append(np.nan); cx.append(axis[0]); cy.append(axis[1])
            continue
        # A percentile, not the extremes: a hand or elbow straying into the band would otherwise set
        # the ring's radius and the running maximum would carry that width down the whole garment
        # (measured: a hip ring reaching out to the hand made the trousers 2.95x the body's width).
        cxi = float(np.median(band[:, 0])); cyi = float(np.median(band[:, 1]))
        rx.append(float(np.percentile(np.abs(band[:, 0] - cxi), 92)))
        ry.append(float(np.percentile(np.abs(band[:, 1] - cyi), 92)))
        cx.append(cxi); cy.append(cyi)
    return zs, np.array(rx), np.array(ry), np.array(cx), np.array(cy)


def hanging_radii(radii: np.ndarray) -> np.ndarray:
    """Running maximum from the top of the piece downward: fabric cannot narrow below what holds it."""
    out = radii.copy()
    for i in range(len(out) - 2, -1, -1):          # index 0 is the lowest ring
        if np.isnan(out[i]) or (not np.isnan(out[i + 1]) and out[i + 1] > out[i]):
            out[i] = out[i + 1]
    return out


def loft(zs, rx, ry, cx, cy):
    """Triangulated tube through the elliptical rings."""
    import trimesh
    ang = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    verts, faces = [], []
    for i, z in enumerate(zs):
        if np.isnan(rx[i]) or np.isnan(ry[i]):
            return None
        verts.append(np.column_stack([cx[i] + rx[i] * np.cos(ang),
                                      cy[i] + ry[i] * np.sin(ang),
                                      np.full(SEGMENTS, z)]))
    verts = np.vstack(verts)
    for i in range(len(zs) - 1):
        a, b = i * SEGMENTS, (i + 1) * SEGMENTS
        for j in range(SEGMENTS):
            k = (j + 1) % SEGMENTS
            faces.append([a + j, a + k, b + k])
            faces.append([a + j, b + k, b + j])
    return trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--body", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--ease", type=float, default=2.0, help="room over the body, in cm")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import trimesh
    mesh = trimesh.load(args.body, process=False)
    v = np.asarray(mesh.vertices)
    z0, z1 = float(v[:, 2].min()), float(v[:, 2].max())
    stature = z1 - z0
    axis = np.array([float(np.median(v[:, 0])), float(np.median(v[:, 1]))])
    ease_m = args.ease / 100.0

    import trimesh as _tm
    written = []
    for name, spec in PIECES.items():
        split = spec.get("split_below")
        z_lo = z0 + spec["z_lo"] * stature
        z_hi = z0 + spec["z_hi"] * stature
        parts = []
        if split is None:
            zs, rx, ry, cx, cy = body_profile(v, z_lo, z_hi, axis, stature)
            parts.append(loft(zs, hanging_radii(rx) + ease_m, hanging_radii(ry) + ease_m, cx, cy))
        else:
            z_split = z0 + split * stature
            # hip section: one tube from the crotch up
            zs, rx, ry, cx, cy = body_profile(v, z_split, z_hi, axis, stature)
            parts.append(loft(zs, hanging_radii(rx) + ease_m, hanging_radii(ry) + ease_m, cx, cy))
            # one leg each below it, so the gap between the legs survives
            for side in (-1, 1):
                below = v[v[:, 2] <= z_split]
                leg = below[(below[:, 0] - axis[0]) * side > 0]
                if len(leg) < 100:
                    continue
                leg_axis = np.array([float(np.median(leg[:, 0])), axis[1]])
                leg = leg[np.abs(leg[:, 0] - leg_axis[0]) <= _LEG_REACH * stature]
                if len(leg) < 100:
                    continue
                lz, lrx, lry, lcx, lcy = body_profile(leg, z_lo, z_split, leg_axis, stature)
                parts.append(loft(lz, hanging_radii(lrx) + ease_m, hanging_radii(lry) + ease_m, lcx, lcy))
        parts = [p for p in parts if p is not None]
        if not parts:
            continue
        piece = _tm.util.concatenate(parts) if len(parts) > 1 else parts[0]
        path = args.out / f"{args.body.stem}_{name}_ease{args.ease:g}.obj"
        piece.export(path)
        written.append((name, path, len(piece.vertices)))

    print(f"{args.body.stem}: stature {stature * 100:.1f} cm, ease {args.ease:g} cm")
    for name, path, n in written:
        print(f"  {name:9} {n:5} verts -> {path.name}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
