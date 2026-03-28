"""Pixel-to-centimeter calibration using known person height."""

from __future__ import annotations

import logging

import numpy as np

from pointsx.keypoints import KP, distance, is_valid, midpoint
from pointsx.schemas import CalibrationInfo, Keypoints

logger = logging.getLogger(__name__)

# If head_top is missing, estimate head height as ~8% of total height
HEAD_HEIGHT_RATIO = 0.08
# If ankles missing but knees visible, ankle-to-knee is ~22% of height
ANKLE_KNEE_RATIO = 0.22


def _height_pixels(kp: Keypoints) -> float | None:
    """Compute pixel height from head_top to ankles midpoint.

    Falls back to partial skeleton if some keypoints are missing.
    """
    pts, conf = kp.points, kp.confidence

    # Determine top point
    if is_valid(conf, KP.HEAD_TOP):
        top_y = pts[KP.HEAD_TOP, 1]
    elif is_valid(conf, KP.UPPER_NECK):
        # Estimate head top from neck
        neck_y = pts[KP.UPPER_NECK, 1]
        # Need an approximate total height to add head offset
        # Use neck to ankles as ~92% of height
        if is_valid(conf, KP.LEFT_ANKLE, KP.RIGHT_ANKLE):
            ankle_y = (pts[KP.LEFT_ANKLE, 1] + pts[KP.RIGHT_ANKLE, 1]) / 2
            partial_h = ankle_y - neck_y
            head_offset = partial_h * HEAD_HEIGHT_RATIO / (1 - HEAD_HEIGHT_RATIO)
            top_y = neck_y - head_offset
        else:
            return None
    else:
        return None

    # Determine bottom point
    if is_valid(conf, KP.LEFT_ANKLE) and is_valid(conf, KP.RIGHT_ANKLE):
        bottom_y = (pts[KP.LEFT_ANKLE, 1] + pts[KP.RIGHT_ANKLE, 1]) / 2
    elif is_valid(conf, KP.LEFT_ANKLE):
        bottom_y = pts[KP.LEFT_ANKLE, 1]
    elif is_valid(conf, KP.RIGHT_ANKLE):
        bottom_y = pts[KP.RIGHT_ANKLE, 1]
    elif is_valid(conf, KP.LEFT_KNEE) or is_valid(conf, KP.RIGHT_KNEE):
        # Fallback: use knee + estimated ankle distance
        knee_ys = []
        if is_valid(conf, KP.LEFT_KNEE):
            knee_ys.append(pts[KP.LEFT_KNEE, 1])
        if is_valid(conf, KP.RIGHT_KNEE):
            knee_ys.append(pts[KP.RIGHT_KNEE, 1])
        knee_y = np.mean(knee_ys)
        partial_h = knee_y - top_y
        # knee_to_top is ~(1 - ANKLE_KNEE_RATIO) of total height
        total_h = partial_h / (1 - ANKLE_KNEE_RATIO)
        return total_h
    else:
        return None

    return abs(bottom_y - top_y)


def calibrate(
    front_kp: Keypoints,
    side_kp: Keypoints,
    known_height_cm: float,
) -> CalibrationInfo:
    """Compute px_per_cm for both views using known height."""
    front_h = _height_pixels(front_kp)
    side_h = _height_pixels(side_kp)

    if front_h is None or front_h < 10:
        raise ValueError("Cannot calibrate front view: insufficient visible keypoints")
    if side_h is None or side_h < 10:
        raise ValueError("Cannot calibrate side view: insufficient visible keypoints")

    px_per_cm_front = front_h / known_height_cm
    px_per_cm_side = side_h / known_height_cm

    # Warn if views have very different scales (different camera distances)
    ratio = px_per_cm_front / px_per_cm_side
    if ratio < 0.85 or ratio > 1.15:
        logger.warning(
            "Front/side calibration differ by %.0f%% (front=%.2f, side=%.2f px/cm). "
            "Camera distances may be unequal.",
            abs(1 - ratio) * 100,
            px_per_cm_front,
            px_per_cm_side,
        )

    return CalibrationInfo(
        px_per_cm_front=px_per_cm_front,
        px_per_cm_side=px_per_cm_side,
    )
