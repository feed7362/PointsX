"""Pixel-to-centimeter calibration using known person height."""

from __future__ import annotations

import logging

import numpy as np

from pointsx.keypoints import KP, is_valid, mean_valid_y
from pointsx.schemas import CalibrationInfo, Keypoints, SilhouetteMask

logger = logging.getLogger(__name__)

# If head_top is missing, estimate head height as ~8% of total height
HEAD_HEIGHT_RATIO = 0.08
# If ankles missing but knees visible, ankle-to-knee is ~22% of height
ANKLE_KNEE_RATIO = 0.22

# The head_top -> ankle keypoint span is NOT stature: the ankle keypoint sits above the sole and
# head_top lands below the crown. Measured on the synthetic bench (rendered bodies with exact mesh
# ground truth, `scripts/synthetic/`): the span is 0.879-0.930 of the true head-to-floor extent,
# mean ~0.90, and it differs between the front and side views of the same body — so treating it as
# full stature inflated every width by 7.5-13.7 % AND skewed the ellipse's aspect ratio. Prefer the
# silhouette extent, which is head-to-floor by construction; fall back to this ratio.
KP_SPAN_TO_STATURE = 0.90
# The silhouette extent is trusted only when it is taller than the keypoint span by a believable
# margin — otherwise the mask is clipped, merged with a shadow, or includes a second person.
_MASK_SPAN_BOUNDS = (1.0, 1.35)


def _stature_pixels(kp: Keypoints, mask: SilhouetteMask | None) -> tuple[float | None, str]:
    """Pixels per the subject's full stature — head crown to floor.

    Args:
        kp: Keypoints for the view.
        mask: Silhouette for the same view, when available.

    Returns:
        (pixels, source) where source is ``"mask"`` or ``"keypoints"``; pixels is None when
        neither anchor can be established.
    """
    span = _height_pixels(kp)
    if mask is not None and getattr(mask, "mask", None) is not None:
        rows = np.where(mask.mask.any(axis=1))[0]
        if len(rows) >= 2:
            extent = float(rows[-1] - rows[0])
            lo, hi = _MASK_SPAN_BOUNDS
            if span is None or (span > 0 and lo <= extent / span <= hi):
                return extent, "mask"
            logger.debug("Silhouette extent %.0f px implausible vs keypoint span %.0f px; using keypoints",
                         extent, span)
    if span is None:
        return None, "keypoints"
    return span / KP_SPAN_TO_STATURE, "keypoints"


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
    elif (knee_y := mean_valid_y(kp, KP.LEFT_KNEE, KP.RIGHT_KNEE)) is not None:
        # Fallback: use knee + estimated ankle distance
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
    front_mask: SilhouetteMask | None = None,
    side_mask: SilhouetteMask | None = None,
) -> CalibrationInfo:
    """Compute px_per_cm for both views using known height.

    Args:
        front_kp: Front-view keypoints.
        side_kp: Side-view keypoints.
        known_height_cm: The subject's stature, without shoes.
        front_mask: Front silhouette; when given, its head-to-floor extent sets the scale
            (the keypoint span is ~10 % short of stature — see KP_SPAN_TO_STATURE).
        side_mask: Side silhouette, same role.
    """
    front_h, front_src = _stature_pixels(front_kp, front_mask)
    side_h, side_src = _stature_pixels(side_kp, side_mask)
    logger.debug("Calibration source: front=%s side=%s", front_src, side_src)

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
