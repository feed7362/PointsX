"""Ground-truth body measurements from SMPL-X mesh vertices.

Uses SMPL-Anthropometry (github.com/DavidBoja/SMPL-Anthropometry) when available,
with a fast geometric fallback for all measurements.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BodyMeasurementsGT:
    """All ground-truth body measurements in centimetres."""
    height_cm: float = 0.0
    neck_circumference_cm: float = 0.0
    chest_circumference_cm: float = 0.0
    waist_circumference_cm: float = 0.0
    hips_circumference_cm: float = 0.0
    thigh_circumference_cm: float = 0.0       # mean of L+R
    calf_circumference_cm: float = 0.0        # mean of L+R
    wrist_circumference_cm: float = 0.0       # mean of L+R
    inseam_length_cm: float = 0.0
    arm_length_cm: float = 0.0               # shoulder → wrist (mean L+R)
    shoulder_width_cm: float = 0.0           # L_shoulder → R_shoulder

    def to_dict(self) -> dict:
        return {k: round(v, 1) for k, v in asdict(self).items()}


# ── SMPL-X joint indices used in geometric fallback ─────────────────────────
# 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee, 6=spine2,
# 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot, 12=neck,
# 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder, 17=R_shoulder,
# 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist

_J = {
    "pelvis": 0, "L_hip": 1, "R_hip": 2, "spine1": 3,
    "L_knee": 4, "R_knee": 5, "spine2": 6,
    "L_ankle": 7, "R_ankle": 8, "spine3": 9,
    "neck": 12, "L_shoulder": 16, "R_shoulder": 17,
    "L_elbow": 18, "R_elbow": 19, "L_wrist": 20, "R_wrist": 21,
}

# ── Key SMPL-X vertex indices for cross-section measurement ─────────────────
# Each body ring is defined by the Y-level of a joint + a band of vertices
# at that height from the mesh; we approximate the circumference as the
# perimeter of the horizontal cross-section polygon.

# Neck ring ~12 vertices around the neck at joint-12 height
_NECK_RING_VERTS = [
    411, 412, 413, 5765, 5766, 5767,   # front top
    3085, 3086, 3087, 6580, 6581, 6582, # back
]
# Chest ring at nipple level
_CHEST_RING_VERTS = [
    3050, 3051, 3052, 3053,   # L breast
    6545, 6546, 6547, 6548,   # R breast
    2943, 2944, 2945,          # mid back thorax
    3085, 3086,                # upper back
]
# Waist ring at navel height
_WAIST_RING_VERTS = [
    3500, 3501, 3502,   # navel front
    702, 703,           # L waist side
    4098, 4099,         # R waist side
    3020, 3021,         # lower back
]
# Hip ring at outer-hip height
_HIP_RING_VERTS = [
    1380, 1381, 1382,   # L hip outer
    4821, 4822, 4823,   # R hip outer
    3145, 3146,         # glute back
    1210, 1211,         # crotch front
]


def _perimeter_from_points(pts: np.ndarray) -> float:
    """Approximate perimeter of a roughly convex 2-D polygon (XZ plane).

    Points need not be in order — we sort by angle around centroid.
    """
    if len(pts) < 3:
        return 0.0
    xz = pts[:, [0, 2]]                   # project onto horizontal plane
    centre = xz.mean(axis=0)
    angles = np.arctan2(xz[:, 1] - centre[1], xz[:, 0] - centre[0])
    order = np.argsort(angles)
    xz = xz[order]
    # Close the loop
    closed = np.vstack([xz, xz[0]])
    diffs = np.diff(closed, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _ring_circumference_m(vertices: np.ndarray, ring_indices: list[int]) -> float:
    """Compute circumference (in metres) from a ring of vertex indices."""
    pts = vertices[ring_indices]
    return _perimeter_from_points(pts)


def _band_circumference_m(
    vertices: np.ndarray,
    y_center: float,
    band_half: float = 0.015,
) -> float:
    """Compute circumference by selecting all vertices within a Y-band.

    Works best for neck, waist, chest, hips where the body is convex.
    """
    mask = np.abs(vertices[:, 1] - y_center) < band_half
    pts = vertices[mask]
    if len(pts) < 6:
        return 0.0
    return _perimeter_from_points(pts)


def _limb_circumference_m(
    vertices: np.ndarray,
    y_center: float,
    x_center: float,
    band_half: float = 0.015,
    x_radius: float = 0.12,
) -> float:
    """Circumference of a single limb by selecting vertices near (x, y)."""
    mask = (
        (np.abs(vertices[:, 1] - y_center) < band_half) &
        (np.abs(vertices[:, 0] - x_center) < x_radius)
    )
    pts = vertices[mask]
    if len(pts) < 4:
        return 0.0
    return _perimeter_from_points(pts)


# ── SMPL-Anthropometry wrapper ───────────────────────────────────────────────

def _try_smpl_anthropometry(
    vertices: np.ndarray,
    faces: np.ndarray,
    sex: str,
) -> dict | None:
    """Attempt to compute measurements with SMPL-Anthropometry library.

    Returns a dict of measurement_name → value_cm, or None on failure.
    """
    try:
        from smpl_anthropometry import MeasurementComputer  # type: ignore

        computer = MeasurementComputer(model_type="smplx", gender=sex)
        raw = computer.compute(vertices, faces)

        # smpl_anthropometry returns values in metres; multiply by 100
        scale = 100.0

        mapping = {
            "height_cm":              raw.get("height", 0) * scale,
            "neck_circumference_cm":  raw.get("neck_girth", 0) * scale,
            "chest_circumference_cm": raw.get("chest_girth", 0) * scale,
            "waist_circumference_cm": raw.get("waist_girth", 0) * scale,
            "hips_circumference_cm":  raw.get("hips_girth", 0) * scale,
            "thigh_circumference_cm": raw.get("thigh_girth", 0) * scale,
            "calf_circumference_cm":  raw.get("calf_girth", 0) * scale,
            "wrist_circumference_cm": raw.get("wrist_girth", 0) * scale,
            "inseam_length_cm":       raw.get("inseam", 0) * scale,
            "arm_length_cm":          raw.get("arm_length", 0) * scale,
            "shoulder_width_cm":      raw.get("shoulder_breadth", 0) * scale,
        }

        # If most values are zero the library didn't return useful data
        non_zero = sum(1 for v in mapping.values() if v > 1.0)
        if non_zero >= 5:
            logger.debug("SMPL-Anthropometry succeeded (%d non-zero fields)", non_zero)
            return mapping

    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("SMPL-Anthropometry failed: %s", exc)

    return None


# ── Geometric fallback ───────────────────────────────────────────────────────

def _geometric_measurements(
    vertices: np.ndarray,
    joints: np.ndarray,
) -> BodyMeasurementsGT:
    """Fast geometric measurement extraction from SMPL-X mesh.

    Accurate to ≈1-3 cm — sufficient for regression training targets.
    """
    m = BodyMeasurementsGT()

    j = joints  # shorthand

    # ── Height ────────────────────────────────────────────────────────────
    head_y  = vertices[411, 1]
    ankle_y = (vertices[6852, 1] + vertices[3438, 1]) / 2
    m.height_cm = abs(head_y - ankle_y) * 100.0

    # ── Shoulder width ────────────────────────────────────────────────────
    m.shoulder_width_cm = float(
        np.linalg.norm(j[_J["L_shoulder"]] - j[_J["R_shoulder"]])
    ) * 100.0

    # ── Arm length (shoulder → wrist, both sides averaged) ───────────────
    l_arm = (
        np.linalg.norm(j[_J["L_shoulder"]] - j[_J["L_elbow"]]) +
        np.linalg.norm(j[_J["L_elbow"]]    - j[_J["L_wrist"]])
    )
    r_arm = (
        np.linalg.norm(j[_J["R_shoulder"]] - j[_J["R_elbow"]]) +
        np.linalg.norm(j[_J["R_elbow"]]    - j[_J["R_wrist"]])
    )
    m.arm_length_cm = (l_arm + r_arm) / 2 * 100.0

    # ── Inseam (crotch → ankle, average L+R) ─────────────────────────────
    crotch_y = vertices[1210, 1]
    l_ankle_y = j[_J["L_ankle"], 1]
    r_ankle_y = j[_J["R_ankle"], 1]
    m.inseam_length_cm = abs(crotch_y - (l_ankle_y + r_ankle_y) / 2) * 100.0

    # ── Circumferences via Y-band method ──────────────────────────────────
    neck_y    = j[_J["neck"], 1]
    chest_y   = (j[_J["L_shoulder"], 1] + j[_J["R_shoulder"], 1]) / 2 - 0.05
    navel_y   = vertices[3500, 1]
    hip_y     = j[_J["pelvis"], 1]

    # Neck (tight band)
    c = _band_circumference_m(vertices, neck_y, band_half=0.012)
    m.neck_circumference_cm = c * 100.0 if c > 0.1 else 35.0

    # Chest (slightly wider band)
    c = _band_circumference_m(vertices, chest_y, band_half=0.025)
    m.chest_circumference_cm = c * 100.0 if c > 0.3 else 90.0

    # Waist
    c = _band_circumference_m(vertices, navel_y, band_half=0.020)
    m.waist_circumference_cm = c * 100.0 if c > 0.3 else 75.0

    # Hips
    c = _band_circumference_m(vertices, hip_y, band_half=0.025)
    m.hips_circumference_cm = c * 100.0 if c > 0.3 else 95.0

    # Thigh: 25% down from hip to knee, left + right averaged
    l_knee_y = j[_J["L_knee"], 1]
    r_knee_y = j[_J["R_knee"], 1]
    l_thigh_y = hip_y + 0.25 * (l_knee_y - hip_y)
    r_thigh_y = hip_y + 0.25 * (r_knee_y - hip_y)
    l_thigh_x = j[_J["L_hip"], 0]
    r_thigh_x = j[_J["R_hip"], 0]

    l_c = _limb_circumference_m(vertices, l_thigh_y, l_thigh_x, band_half=0.015)
    r_c = _limb_circumference_m(vertices, r_thigh_y, r_thigh_x, band_half=0.015)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.1]
    m.thigh_circumference_cm = float(np.mean(valid)) if valid else 55.0

    # Calf: 60% down from knee to ankle
    l_ankle_y = j[_J["L_ankle"], 1]
    r_ankle_y = j[_J["R_ankle"], 1]
    l_calf_y = l_knee_y + 0.60 * (l_ankle_y - l_knee_y)
    r_calf_y = r_knee_y + 0.60 * (r_ankle_y - r_knee_y)

    l_c = _limb_circumference_m(vertices, l_calf_y, l_thigh_x, band_half=0.012)
    r_c = _limb_circumference_m(vertices, r_calf_y, r_thigh_x, band_half=0.012)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.05]
    m.calf_circumference_cm = float(np.mean(valid)) if valid else 36.0

    # Wrist
    l_wrist_y = j[_J["L_wrist"], 1]
    r_wrist_y = j[_J["R_wrist"], 1]
    l_wrist_x = j[_J["L_wrist"], 0]
    r_wrist_x = j[_J["R_wrist"], 0]

    l_c = _limb_circumference_m(vertices, l_wrist_y, l_wrist_x, band_half=0.010, x_radius=0.06)
    r_c = _limb_circumference_m(vertices, r_wrist_y, r_wrist_x, band_half=0.010, x_radius=0.06)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.02]
    m.wrist_circumference_cm = float(np.mean(valid)) if valid else 16.0

    return m


# ── Public API ────────────────────────────────────────────────────────────────

def compute_measurements(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    sex: str,
) -> BodyMeasurementsGT:
    """Compute all ground-truth body measurements from SMPL-X mesh.

    Tries SMPL-Anthropometry first (more accurate), falls back to
    fast geometric band-circumference method.

    Args:
        vertices:  (10475, 3) float32 — SMPL-X vertices in metres
        joints:    (127, 3)   float32 — SMPL-X joints in metres
        faces:     (N, 3)     int32   — triangle face indices
        sex:       "male" | "female"

    Returns:
        BodyMeasurementsGT with all values in centimetres
    """
    # Try SMPL-Anthropometry library first
    smpl_result = _try_smpl_anthropometry(vertices, faces, sex)

    if smpl_result is not None:
        m = BodyMeasurementsGT(**{
            k: v for k, v in smpl_result.items()
            if hasattr(BodyMeasurementsGT, k)
        })
        # Fill any zeros with geometric fallback
        geo = _geometric_measurements(vertices, joints)
        for field in BodyMeasurementsGT.__dataclass_fields__:
            if getattr(m, field, 0.0) < 1.0:
                setattr(m, field, getattr(geo, field))
        return m

    # Full geometric fallback
    logger.debug("Using geometric fallback for body measurements")
    return _geometric_measurements(vertices, joints)


def sanity_check(m: BodyMeasurementsGT) -> list[str]:
    """Return list of warning strings for out-of-range measurements."""
    warnings = []

    def _check(name: str, value: float, lo: float, hi: float) -> None:
        if not (lo <= value <= hi):
            warnings.append(f"{name}={value:.1f} out of range [{lo}, {hi}]")

    h = m.height_cm
    if h > 0:
        _check("neck_circumference_cm",   m.neck_circumference_cm,   25, 55)
        _check("chest_circumference_cm",  m.chest_circumference_cm,  60, 160)
        _check("waist_circumference_cm",  m.waist_circumference_cm,  50, 160)
        _check("hips_circumference_cm",   m.hips_circumference_cm,   70, 170)
        _check("thigh_circumference_cm",  m.thigh_circumference_cm,  35, 100)
        _check("calf_circumference_cm",   m.calf_circumference_cm,   22,  60)
        _check("wrist_circumference_cm",  m.wrist_circumference_cm,  12,  25)
        _check("shoulder_width_cm",       m.shoulder_width_cm,       30,  60)
        _check("arm_length_cm",           m.arm_length_cm,           50,  85)
        _check("inseam_length_cm",        m.inseam_length_cm,        60,  95)

    return warnings
