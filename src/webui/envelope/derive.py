"""Measurements the pipeline does not output directly, derived from keypoints, widths and ratios."""
from __future__ import annotations

from pointsx.circumference import ramanujan_ellipse_circumference
from pointsx.keypoints import KP, distance, is_valid, midpoint
from pointsx.schemas import BodyMeasurements, Keypoints, SilhouetteMask

# Anthropometric ratios used when a measurement isn't directly observable
_ANKLE_TO_CALF_RATIO = 0.62


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _kp_distance_cm(kp: Keypoints, a: KP, b: KP, px_per_cm: float) -> float | None:
    """Pixel distance between two keypoints, converted to cm (or None if invalid)."""
    if not is_valid(kp.confidence, a, b):
        return None
    if px_per_cm <= 0:
        return None
    return distance(kp.points, a, b) / px_per_cm


def _kp_to_midpoint_cm(
    kp: Keypoints, single: KP, mid_a: KP, mid_b: KP, px_per_cm: float,
) -> float | None:
    """Distance from a single keypoint to the midpoint of two others, in cm."""
    if not is_valid(kp.confidence, single, mid_a, mid_b):
        return None
    if px_per_cm <= 0:
        return None
    p = kp.points[int(single)]
    m = midpoint(kp.points, mid_a, mid_b)
    import numpy as np
    return float(np.linalg.norm(p - m)) / px_per_cm


def _derive_chest_circumference(bm: BodyMeasurements) -> float | None:
    """Derive chest girth from front/side torso widths via Ramanujan ellipse.

    Returns None when the input widths are outside anatomical ranges — usually a
    sign that the silhouette was corrupted (e.g. arms outstretched intersecting
    the torso slice). The regressor handles waist/hip/neck/etc. similarly by
    being bounded; chest has no regressor output, so we sanity-check here.
    """
    fw = bm.torso_width_front_cm
    sw = bm.torso_width_side_cm
    if fw is None or sw is None:
        return None
    # Plausible adult human torso (front) ≈ 22-50 cm; (side / depth) ≈ 14-38 cm
    if not (22.0 <= fw <= 50.0) or not (14.0 <= sw <= 38.0):
        return None
    circ = ramanujan_ellipse_circumference(fw, sw)
    # Final sanity cap on the resulting circumference (60-150 cm covers everyone)
    if not (60.0 <= circ <= 150.0):
        return None
    return circ


def _derive_back_length(
    bm: BodyMeasurements, side_kp: Keypoints, px_per_cm_side: float
) -> float | None:
    """Back length from C7 (UPPER_NECK proxy) to natural waist, on the side view.

    Uses a stable proportional anchor for the waist:
    ``y_waist = upper_neck_y + 0.65 × (pelvis_y − upper_neck_y)`` — anatomical
    natural waist sits ~65 % of the way down the torso. The silhouette-detected
    waist (``bm.waist_level_side_px``) was previously preferred, but on the
    eval set it drifted subject-to-subject (RMSE 7.87 cm with ±13 cm outliers)
    because the side-view torso lacks a clear narrowest point. Proportional
    keypoint anchor is far more stable.
    """
    if px_per_cm_side <= 0:
        return None
    pts = side_kp.points
    conf = side_kp.confidence
    if not is_valid(conf, KP.UPPER_NECK):
        return None
    upper_neck_y = float(pts[KP.UPPER_NECK, 1])

    # Pelvis: prefer the explicit keypoint, fall back to midpoint of hips.
    if is_valid(conf, KP.PELVIS):
        pelvis_y = float(pts[KP.PELVIS, 1])
    elif is_valid(conf, KP.LEFT_HIP, KP.RIGHT_HIP):
        pelvis_y = (float(pts[KP.LEFT_HIP, 1]) + float(pts[KP.RIGHT_HIP, 1])) / 2
    else:
        return None

    waist_y = upper_neck_y + 0.65 * (pelvis_y - upper_neck_y)
    return abs(waist_y - upper_neck_y) / px_per_cm_side


def _derive_front_length(
    bm: BodyMeasurements, front_kp: Keypoints, px_per_cm_front: float
) -> float | None:
    # Prefer persisted waist level if available.
    if bm.waist_level_front_px is not None and is_valid(front_kp.confidence, KP.UPPER_NECK):
        if px_per_cm_front <= 0:
            return None
        return abs(float(front_kp.points[KP.UPPER_NECK, 1]) - float(bm.waist_level_front_px)) / px_per_cm_front
    # Fallback: old proxy from upper neck to hip midpoint.
    return _kp_to_midpoint_cm(front_kp, KP.UPPER_NECK, KP.LEFT_HIP, KP.RIGHT_HIP, px_per_cm_front)


def _derive_neck_base_height(
    front_kp: Keypoints, px_per_cm_front: float, front_mask: SilhouetteMask | None = None,
) -> float | None:
    """Height of the neck base above the FLOOR, which is what the tape measures.

    The ankle keypoint sits above the sole, so the old neck->ankles span under-read by the ankle
    height; the pre-2026-09-22 calibration was ~10 % short of stature and happened to cancel it,
    and once calibration was fixed the gap showed up as a -18.7 cm raw bias on the app corpus.
    The silhouette's lowest row is the floor, the same anchor calibration now uses.

    Args:
        front_kp: Front-view keypoints.
        px_per_cm_front: Scale for the front view.
        front_mask: Front silhouette; without it this falls back to the ankle keypoints.
    """
    if px_per_cm_front <= 0 or not is_valid(front_kp.confidence, KP.UPPER_NECK):
        return None
    if front_mask is not None and getattr(front_mask, "mask", None) is not None:
        import numpy as np
        rows = np.where(front_mask.mask.any(axis=1))[0]
        if len(rows) >= 2:
            return float(rows[-1] - float(front_kp.points[KP.UPPER_NECK, 1])) / px_per_cm_front
    return _kp_to_midpoint_cm(front_kp, KP.UPPER_NECK, KP.LEFT_ANKLE, KP.RIGHT_ANKLE, px_per_cm_front)


def _derive_ankle(calf_circ: float | None) -> float | None:
    if calf_circ is None or calf_circ <= 0:
        return None
    return calf_circ * _ANKLE_TO_CALF_RATIO
