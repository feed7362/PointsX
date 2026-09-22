"""Measure a human mesh the way a tape does, independent of topology.

The vendored SMPL-Anthropometry works on SMPL-X vertex ids and its face segmentation, so it cannot
score a MakeHuman/Anny mesh. This module needs only vertices + faces: it slices the mesh with a
horizontal plane, keeps the cross-section belonging to the body part being measured, and takes the
convex-hull perimeter of that slice — the same thing a tape does when pulled around a limb.

Row heights come from ANSUR II landmark heights as fractions of stature (``scripts/ansur/``), per
sex, so synthetic ground truth is defined exactly like the population the app is calibrated against:

    waist (omphalion)  0.601 / 0.601    chest  0.719 / 0.735
    buttock (hip)      0.512 / 0.505    crotch 0.480 / 0.482
    knee (mid-patella) 0.276 / 0.278    cervicale (neck base) 0.857 / 0.864

usage as a library:
    from measure_mesh import measure_mesh
    gt = measure_mesh(vertices_m, faces, sex="female")     # vertices in metres, +Z up
"""
from __future__ import annotations

import numpy as np

# fraction of stature -> (female, male)
LANDMARK_H: dict[str, tuple[float, float]] = {
    "waist": (0.601, 0.601),
    "chest": (0.719, 0.735),
    "hip": (0.512, 0.505),
    "crotch": (0.480, 0.482),
    "knee": (0.276, 0.278),
    "neck": (0.857, 0.864),
}
# a circumference is measured on the slice component nearest the body axis ("torso") or on the
# component to one side ("limb")
SITES: dict[str, tuple[str, str]] = {
    "chest_circumference": ("chest", "torso"),
    "waist_circumference": ("waist", "torso"),
    "hip_circumference": ("hip", "torso"),
    "neck_circumference": ("neck", "narrowest"),
    "thigh_circumference": ("thigh", "limb"),
    "calf_circumference": ("calf", "limb"),
}


def _slice_components(mesh, z: float) -> list[np.ndarray]:
    """Closed 2-D polylines where the horizontal plane at `z` cuts the mesh."""
    section = mesh.section(plane_origin=[0.0, 0.0, z], plane_normal=[0.0, 0.0, 1.0])
    if section is None:
        return []
    planar, _ = section.to_2D()
    return [np.asarray(d)[:, :2] for d in planar.discrete if len(d) >= 3]


def _hull_perimeter_cm(points: np.ndarray) -> float:
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    ring = points[hull.vertices]
    return float(np.linalg.norm(np.diff(np.vstack([ring, ring[:1]]), axis=0), axis=1).sum() * 100.0)


def _narrowest_torso(mesh, z_lo: float, z_hi: float, axis_x: float, steps: int = 12) -> float | None:
    """Smallest torso perimeter in a height band — how a tape finds the neck (and how
    `pointsx.silhouette` finds it in 2-D): scan, keep the minimum."""
    best = None
    for z in np.linspace(z_lo, z_hi, steps):
        comp = _pick(_slice_components(mesh, float(z)), axis_x, "torso")
        if comp is None or len(comp) < 3:
            continue
        p = _hull_perimeter_cm(comp)
        if best is None or p < best:
            best = p
    return best


# A real body cross-section is never smaller than this; anything below is an ear, a hair strand or a
# mesh artifact that would otherwise win the "nearest the axis" test (the first run measured necks of
# 2-7 cm that way).
MIN_PERIMETER_CM = 15.0


def _pick(components: list[np.ndarray], axis_x: float, where: str) -> np.ndarray | None:
    """torso = the component straddling the body axis; limb = the widest one clearly off-axis."""
    comps = [c for c in components if len(c) >= 3 and _hull_perimeter_cm(c) >= MIN_PERIMETER_CM]
    if not comps:
        return None
    if where == "torso":
        straddling = [c for c in comps if c[:, 0].min() <= axis_x <= c[:, 0].max()]
        pool = straddling or comps
        return max(pool, key=lambda c: _hull_perimeter_cm(c))
    off = [c for c in comps if abs(c.mean(axis=0)[0] - axis_x) > 0.02]
    return max(off, key=lambda c: _hull_perimeter_cm(c)) if off else None


def measure_mesh(vertices: np.ndarray, faces: np.ndarray, sex: str) -> dict[str, float]:
    """Tape-style measurements from a standing mesh.

    Args:
        vertices: (N, 3) in metres, +Z up, feet at the lowest z.
        faces: (M, 3) triangle indices.
        sex: ``"female"`` or ``"male"`` — selects the ANSUR landmark fractions.

    Returns:
        Canonical id -> cm. Sites whose slice could not be isolated are omitted.
    """
    import trimesh
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                           faces=np.asarray(faces), process=False)
    z0, z1 = float(mesh.vertices[:, 2].min()), float(mesh.vertices[:, 2].max())
    stature = z1 - z0
    axis_x = float(np.median(mesh.vertices[:, 0]))
    i = 0 if sex == "female" else 1
    h = {k: z0 + v[i] * stature for k, v in LANDMARK_H.items()}
    h["thigh"] = h["crotch"] - 0.06 * stature          # just below the crotch, as the tape is used
    h["calf"] = h["knee"] + 0.10 * (h["crotch"] - h["knee"])

    out: dict[str, float] = {"height_cm": stature * 100.0}
    for mid, (site, where) in SITES.items():
        if where == "narrowest":
            # between the neck base and the chin
            v = _narrowest_torso(mesh, h["neck"] + 0.005 * stature, h["neck"] + 0.05 * stature, axis_x)
            if v is not None:
                out[mid] = round(v, 1)
            continue
        comp = _pick(_slice_components(mesh, h[site]), axis_x, where)
        if comp is not None and len(comp) >= 3:
            out[mid] = round(_hull_perimeter_cm(comp), 1)
    # CAVEAT — these two are POPULATION FRACTIONS of stature, not measurements of this mesh, so they
    # are only valid for a body whose proportions match ANSUR. Anny bodies do not: their legs run
    # long (leg/height 0.61-0.64 against the 0.43-0.53 the pipeline expects), which made the bench
    # report a -13.6 cm "error" on the inner seam that was the ground truth's fault, not the
    # pipeline's. Find the real crotch (the lowest row where the leg slices merge) and the real neck
    # base from the mesh before trusting either of these.
    out["leg_length_inner_seam"] = round((h["crotch"] - z0) * 100.0, 1)
    out["neck_base_height"] = round((h["neck"] - z0) * 100.0, 1)
    out["height_cm"] = round(out["height_cm"], 1)
    return out
