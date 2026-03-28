"""Silhouette (segmentation mask) processing for width measurements."""

from __future__ import annotations

import numpy as np

from pointsx.keypoints import KP, interpolate_y, is_valid
from pointsx.schemas import Keypoints, SilhouetteMask


def measure_width_at_y(mask: np.ndarray, y: float, margin: int = 3) -> float | None:
    """Measure horizontal width of the silhouette at a given y-coordinate.

    Averages over a vertical band [y-margin, y+margin] for robustness.
    Returns width in pixels, or None if no foreground pixels found.
    """
    h, w = mask.shape
    y_int = int(round(y))
    y_min = max(0, y_int - margin)
    y_max = min(h - 1, y_int + margin)

    widths = []
    for row in range(y_min, y_max + 1):
        cols = np.where(mask[row])[0]
        if len(cols) >= 2:
            widths.append(cols[-1] - cols[0])

    if not widths:
        return None

    return float(np.mean(widths))


def measure_limb_width_at_y(
    mask: np.ndarray, y: float, x_hint: float, margin: int = 3
) -> float | None:
    """Measure width of a single limb at y, using x_hint to identify which segment.

    In front view, both legs/arms appear as separate regions. x_hint (from the
    keypoint x-coordinate) determines which contiguous segment to measure.
    """
    h, w = mask.shape
    y_int = int(round(y))
    y_min = max(0, y_int - margin)
    y_max = min(h - 1, y_int + margin)

    widths = []
    for row in range(y_min, y_max + 1):
        cols = np.where(mask[row])[0]
        if len(cols) < 2:
            continue

        # Find contiguous segments
        segments = _find_segments(cols)
        # Pick segment closest to x_hint
        best_seg = min(segments, key=lambda s: abs((s[0] + s[-1]) / 2 - x_hint))
        widths.append(best_seg[-1] - best_seg[0])

    if not widths:
        return None

    return float(np.mean(widths))


def _find_segments(cols: np.ndarray) -> list[np.ndarray]:
    """Split sorted column indices into contiguous segments."""
    diffs = np.diff(cols)
    split_points = np.where(diffs > 3)[0] + 1  # gap > 3px = new segment
    return np.split(cols, split_points)


def extract_all_widths(
    front_mask: SilhouetteMask,
    side_mask: SilhouetteMask,
    front_kp: Keypoints,
    side_kp: Keypoints,
) -> dict[str, tuple[float | None, float | None]]:
    """Extract body widths at key y-coordinates from both views.

    Returns dict mapping body part to (front_width_px, side_width_px).
    """
    f_pts, f_conf = front_kp.points, front_kp.confidence
    s_pts, s_conf = side_kp.points, side_kp.confidence
    f_mask = front_mask.mask
    s_mask = side_mask.mask

    widths: dict[str, tuple[float | None, float | None]] = {}

    # Head: midpoint between head_top and upper_neck
    if is_valid(f_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        y_head_f = (f_pts[KP.HEAD_TOP, 1] + f_pts[KP.UPPER_NECK, 1]) / 2
        widths["head"] = (measure_width_at_y(f_mask, y_head_f), None)
    if is_valid(s_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        y_head_s = (s_pts[KP.HEAD_TOP, 1] + s_pts[KP.UPPER_NECK, 1]) / 2
        existing = widths.get("head", (None, None))
        widths["head"] = (existing[0], measure_width_at_y(s_mask, y_head_s))

    # Neck: at upper_neck y
    if is_valid(f_conf, KP.UPPER_NECK):
        y_neck_f = f_pts[KP.UPPER_NECK, 1]
        widths["neck"] = (measure_width_at_y(f_mask, y_neck_f), None)
    if is_valid(s_conf, KP.UPPER_NECK):
        y_neck_s = s_pts[KP.UPPER_NECK, 1]
        existing = widths.get("neck", (None, None))
        widths["neck"] = (existing[0], measure_width_at_y(s_mask, y_neck_s))

    # Chest / torso: at thorax y
    if is_valid(f_conf, KP.THORAX):
        y_chest_f = f_pts[KP.THORAX, 1]
        widths["torso"] = (measure_width_at_y(f_mask, y_chest_f), None)
    if is_valid(s_conf, KP.THORAX):
        y_chest_s = s_pts[KP.THORAX, 1]
        existing = widths.get("torso", (None, None))
        widths["torso"] = (existing[0], measure_width_at_y(s_mask, y_chest_s))

    # Waist: interpolated 60% between thorax and pelvis
    if is_valid(f_conf, KP.THORAX, KP.PELVIS):
        y_waist_f = interpolate_y(f_pts, KP.THORAX, KP.PELVIS, 0.6)
        widths["waist"] = (measure_width_at_y(f_mask, y_waist_f), None)
    if is_valid(s_conf, KP.THORAX, KP.PELVIS):
        y_waist_s = interpolate_y(s_pts, KP.THORAX, KP.PELVIS, 0.6)
        existing = widths.get("waist", (None, None))
        widths["waist"] = (existing[0], measure_width_at_y(s_mask, y_waist_s))

    # Hip: at pelvis / midpoint of hips
    if is_valid(f_conf, KP.LEFT_HIP, KP.RIGHT_HIP):
        y_hip_f = (f_pts[KP.LEFT_HIP, 1] + f_pts[KP.RIGHT_HIP, 1]) / 2
        widths["hip"] = (measure_width_at_y(f_mask, y_hip_f), None)
    elif is_valid(f_conf, KP.PELVIS):
        y_hip_f = f_pts[KP.PELVIS, 1]
        widths["hip"] = (measure_width_at_y(f_mask, y_hip_f), None)

    if is_valid(s_conf, KP.LEFT_HIP, KP.RIGHT_HIP):
        y_hip_s = (s_pts[KP.LEFT_HIP, 1] + s_pts[KP.RIGHT_HIP, 1]) / 2
        existing = widths.get("hip", (None, None))
        widths["hip"] = (existing[0], measure_width_at_y(s_mask, y_hip_s))
    elif is_valid(s_conf, KP.PELVIS):
        y_hip_s = s_pts[KP.PELVIS, 1]
        existing = widths.get("hip", (None, None))
        widths["hip"] = (existing[0], measure_width_at_y(s_mask, y_hip_s))

    # Thigh: 25% between hip and knee (front: single-limb, side: full width)
    for side_name, kp_hip, kp_knee in [
        ("right", KP.RIGHT_HIP, KP.RIGHT_KNEE),
        ("left", KP.LEFT_HIP, KP.LEFT_KNEE),
    ]:
        if is_valid(f_conf, kp_hip, kp_knee):
            y_thigh_f = interpolate_y(f_pts, kp_hip, kp_knee, 0.25)
            x_hint = f_pts[kp_knee, 0]
            front_w = measure_limb_width_at_y(f_mask, y_thigh_f, x_hint)
            key = f"thigh_{side_name}"
            widths[key] = (front_w, None)

    # Side thigh: use average of hip/knee midpoint
    if is_valid(s_conf, KP.RIGHT_HIP, KP.RIGHT_KNEE):
        y_thigh_s = interpolate_y(s_pts, KP.RIGHT_HIP, KP.RIGHT_KNEE, 0.25)
        side_w = measure_width_at_y(s_mask, y_thigh_s)
        for key in ["thigh_right", "thigh_left"]:
            existing = widths.get(key, (None, None))
            widths[key] = (existing[0], side_w)

    # Calf: 60% between knee and ankle
    for side_name, kp_knee, kp_ankle in [
        ("right", KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        ("left", KP.LEFT_KNEE, KP.LEFT_ANKLE),
    ]:
        if is_valid(f_conf, kp_knee, kp_ankle):
            y_calf_f = interpolate_y(f_pts, kp_knee, kp_ankle, 0.6)
            x_hint = f_pts[kp_ankle, 0]
            front_w = measure_limb_width_at_y(f_mask, y_calf_f, x_hint)
            key = f"calf_{side_name}"
            widths[key] = (front_w, None)

    if is_valid(s_conf, KP.RIGHT_KNEE, KP.RIGHT_ANKLE):
        y_calf_s = interpolate_y(s_pts, KP.RIGHT_KNEE, KP.RIGHT_ANKLE, 0.6)
        side_w = measure_width_at_y(s_mask, y_calf_s)
        for key in ["calf_right", "calf_left"]:
            existing = widths.get(key, (None, None))
            widths[key] = (existing[0], side_w)

    # Wrist
    for side_name, kp_wrist in [
        ("right", KP.RIGHT_WRIST),
        ("left", KP.LEFT_WRIST),
    ]:
        if is_valid(f_conf, kp_wrist):
            y_wrist_f = f_pts[kp_wrist, 1]
            x_hint = f_pts[kp_wrist, 0]
            front_w = measure_limb_width_at_y(f_mask, y_wrist_f, x_hint)
            key = f"wrist_{side_name}"
            widths[key] = (front_w, None)

    if is_valid(s_conf, KP.RIGHT_WRIST):
        y_wrist_s = s_pts[KP.RIGHT_WRIST, 1]
        side_w = measure_width_at_y(s_mask, y_wrist_s)
        for key in ["wrist_right", "wrist_left"]:
            existing = widths.get(key, (None, None))
            widths[key] = (existing[0], side_w)

    return widths
