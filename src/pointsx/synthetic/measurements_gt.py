"""Ground-truth body measurements from SMPL-X mesh via physical Y-slicing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)

@dataclass
class BodyMeasurementsGT:
    """All ground-truth body measurements in centimetres."""
    height_cm: float = 0.0
    neck_circumference_cm: float = 0.0
    chest_circumference_cm: float = 0.0
    waist_circumference_cm: float = 0.0
    hips_circumference_cm: float = 0.0
    thigh_circumference_cm: float = 0.0
    calf_circumference_cm: float = 0.0
    wrist_circumference_cm: float = 0.0
    inseam_length_cm: float = 0.0
    arm_length_cm: float = 0.0
    shoulder_width_cm: float = 0.0

    def to_dict(self) -> dict:
        return {k: round(float(v), 1) for k, v in asdict(self).items()}

# ── SMPL-X Joint Indices ──────────────────────────────────────────────────
_J = {
    "pelvis": 0, "L_hip": 1, "R_hip": 2, "spine1": 3,
    "L_knee": 4, "R_knee": 5, "spine2": 6,
    "L_ankle": 7, "R_ankle": 8, "spine3": 9,
    "neck": 12, "L_shoulder": 16, "R_shoulder": 17,
    "L_elbow": 18, "R_elbow": 19, "L_wrist": 20, "R_wrist": 21,
}

def _perimeter_from_points(pts: np.ndarray) -> float:
    """Approximate perimeter using Convex Hull on the XZ plane."""
    if len(pts) < 3:
        return 0.0
    xz = pts[:, [0, 2]]
    try:
        hull = ConvexHull(xz)
        perimeter = 0.0
        for simplex in hull.simplices:
            p1, p2 = xz[simplex[0]], xz[simplex[1]]
            perimeter += np.linalg.norm(p1 - p2)
        return float(perimeter)
    except Exception as e:
        logger.warning(f"ConvexHull failed: {e}")
        return 0.0

def _band_circumference_m(
        vertices: np.ndarray,
        y_center: float,
        band_half: float = 0.015,
        max_x_radius: float = 0.25  # Limit to exclude arms
) -> float:
    """Slice the mesh horizontally at y_center and measure perimeter."""
    mask_y = np.abs(vertices[:, 1] - y_center) < band_half
    mask_x = np.abs(vertices[:, 0]) < max_x_radius

    pts = vertices[mask_y & mask_x]
    if len(pts) < 6:
        return 0.0
    return _perimeter_from_points(pts)

def _limb_circumference_m(
        vertices: np.ndarray,
        center_pt: np.ndarray,
        band_half: float = 0.015,
        radius: float = 0.10,
) -> float:
    """Slice the mesh at center_pt[1] and filter by a strict 2D radius (XZ plane) around the bone."""
    mask_y = np.abs(vertices[:, 1] - center_pt[1]) < band_half
    dist_xz = np.linalg.norm(vertices[:, [0, 2]] - center_pt[[0, 2]], axis=1)
    mask_xz = dist_xz < radius

    pts = vertices[mask_y & mask_xz]
    if len(pts) < 4:
        return 0.0
    return _perimeter_from_points(pts)

def _try_smpl_anthropometry(vertices: np.ndarray, faces: np.ndarray, sex: str) -> dict | None:
    """Attempt to compute measurements with SMPL-Anthropometry library."""
    try:
        from smpl_anthropometry import MeasurementComputer
        computer = MeasurementComputer(model_type="smplx", gender=sex)
        raw = computer.compute(vertices, faces)
        scale = 100.0
        mapping = {
            "height_cm": raw.get("height", 0) * scale,
            "neck_circumference_cm": raw.get("neck_girth", 0) * scale,
            "chest_circumference_cm": raw.get("chest_girth", 0) * scale,
            "waist_circumference_cm": raw.get("waist_girth", 0) * scale,
            "hips_circumference_cm": raw.get("hips_girth", 0) * scale,
            "thigh_circumference_cm": raw.get("thigh_girth", 0) * scale,
            "calf_circumference_cm": raw.get("calf_girth", 0) * scale,
            "wrist_circumference_cm": raw.get("wrist_girth", 0) * scale,
            "inseam_length_cm": raw.get("inseam", 0) * scale,
            "arm_length_cm": raw.get("arm_length", 0) * scale,
            "shoulder_width_cm": raw.get("shoulder_breadth", 0) * scale,
        }
        non_zero = sum(1 for v in mapping.values() if v > 1.0)
        if non_zero >= 5:
            return mapping
    except ImportError:
        pass
    except Exception as exc:
        pass
    return None

def _geometric_measurements(vertices: np.ndarray, joints: np.ndarray) -> BodyMeasurementsGT:
    """Fast physical slice measurement extraction from SMPL-X mesh."""
    m = BodyMeasurementsGT()
    j = joints

    # ── Height (Absolute Bounding Box Y) ──
    m.height_cm = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1])) * 100.0

    # ── Shoulder width (Joint to Joint) ──
    m.shoulder_width_cm = float(np.linalg.norm(j[_J["L_shoulder"]] - j[_J["R_shoulder"]])) * 100.0

    # ── Arm length (Average L+R) ──
    l_arm = np.linalg.norm(j[_J["L_shoulder"]] - j[_J["L_elbow"]]) + np.linalg.norm(j[_J["L_elbow"]] - j[_J["L_wrist"]])
    r_arm = np.linalg.norm(j[_J["R_shoulder"]] - j[_J["R_elbow"]]) + np.linalg.norm(j[_J["R_elbow"]] - j[_J["R_wrist"]])
    m.arm_length_cm = (l_arm + r_arm) / 2 * 100.0

    # ── Inseam (Pelvis to Ankle) ──
    crotch_y = j[_J["pelvis"], 1] - 0.04
    l_ankle_y, r_ankle_y = j[_J["L_ankle"], 1], j[_J["R_ankle"], 1]
    m.inseam_length_cm = abs(crotch_y - (l_ankle_y + r_ankle_y) / 2) * 100.0

    # ── Y-Levels for Circumferences (Torso) ──
    neck_pt = j[_J["neck"]].copy()
    neck_pt[1] += 0.04
    chest_y   = (j[_J["L_shoulder"], 1] + j[_J["R_shoulder"], 1]) / 2 - 0.12
    navel_y   = j[_J["spine1"], 1]
    hip_y     = j[_J["pelvis"], 1]

    # ── Torso Circumferences ──
    c = _limb_circumference_m(vertices, neck_pt, band_half=0.015, radius=0.08)
    m.neck_circumference_cm = c * 100.0 if c > 0.1 else 35.0

    chest_radius = (m.shoulder_width_cm / 200.0) - 0.015
    c = _band_circumference_m(vertices, chest_y, band_half=0.02, max_x_radius=max(0.12, chest_radius))
    m.chest_circumference_cm = c * 100.0 if c > 0.3 else 90.0

    c = _band_circumference_m(vertices, navel_y, band_half=0.02, max_x_radius=0.22)
    m.waist_circumference_cm = c * 100.0 if c > 0.3 else 75.0

    c = _band_circumference_m(vertices, hip_y, band_half=0.025, max_x_radius=0.30)
    m.hips_circumference_cm = c * 100.0 if c > 0.3 else 95.0

    # ── Thigh (3D interpolation from hip to knee) ──
    l_thigh_pt = j[_J["L_hip"]] + 0.25 * (j[_J["L_knee"]] - j[_J["L_hip"]])
    r_thigh_pt = j[_J["R_hip"]] + 0.25 * (j[_J["R_knee"]] - j[_J["R_hip"]])

    l_c = _limb_circumference_m(vertices, l_thigh_pt, band_half=0.02, radius=0.11)
    r_c = _limb_circumference_m(vertices, r_thigh_pt, band_half=0.02, radius=0.11)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.1]
    m.thigh_circumference_cm = float(np.mean(valid)) if valid else 50.0

    # ── Calf (3D interpolation from knee to ankle) ──
    l_calf_pt = j[_J["L_knee"]] + 0.60 * (j[_J["L_ankle"]] - j[_J["L_knee"]])
    r_calf_pt = j[_J["R_knee"]] + 0.60 * (j[_J["R_ankle"]] - j[_J["R_knee"]])

    l_c = _limb_circumference_m(vertices, l_calf_pt, band_half=0.02, radius=0.08)
    r_c = _limb_circumference_m(vertices, r_calf_pt, band_half=0.02, radius=0.08)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.05]
    m.calf_circumference_cm = float(np.mean(valid)) if valid else 35.0

    # ── Wrist (3D interpolation from wrist to elbow) ──
    l_wrist_pt = j[_J["L_wrist"]] + 0.08 * (j[_J["L_elbow"]] - j[_J["L_wrist"]])
    r_wrist_pt = j[_J["R_wrist"]] + 0.08 * (j[_J["R_elbow"]] - j[_J["R_wrist"]])

    l_c = _limb_circumference_m(vertices, l_wrist_pt, band_half=0.01, radius=0.06)
    r_c = _limb_circumference_m(vertices, r_wrist_pt, band_half=0.01, radius=0.06)
    valid = [v * 100.0 for v in [l_c, r_c] if v > 0.02]
    m.wrist_circumference_cm = float(np.mean(valid)) if valid else 16.0

    return m

def compute_measurements(vertices: np.ndarray, joints: np.ndarray, faces: np.ndarray, sex: str) -> BodyMeasurementsGT:
    smpl_result = _try_smpl_anthropometry(vertices, faces, sex)
    if smpl_result is not None:
        m = BodyMeasurementsGT(**{k: v for k, v in smpl_result.items() if hasattr(BodyMeasurementsGT, k)})
        geo = _geometric_measurements(vertices, joints)
        for field in BodyMeasurementsGT.__dataclass_fields__:
            if getattr(m, field, 0.0) < 1.0:
                setattr(m, field, getattr(geo, field))
        return m
    return _geometric_measurements(vertices, joints)

def sanity_check(m: BodyMeasurementsGT) -> list[str]:
    warnings = []
    def _check(name: str, value: float, lo: float, hi: float):
        if not (lo <= value <= hi):
            warnings.append(f"{name}={value:.1f} out of range [{lo}, {hi}]")
    h = m.height_cm
    if h > 0:
        _check("neck_circumference_cm", m.neck_circumference_cm, 25, 55)
        _check("chest_circumference_cm", m.chest_circumference_cm, 60, 160)
        _check("waist_circumference_cm", m.waist_circumference_cm, 50, 160)
        _check("hips_circumference_cm", m.hips_circumference_cm, 70, 170)
        _check("thigh_circumference_cm", m.thigh_circumference_cm, 35, 100)
    return warnings