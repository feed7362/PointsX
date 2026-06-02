"""Silhouette (segmentation mask) processing for width measurements."""

from __future__ import annotations

import numpy as np

from pointsx.keypoints import KP, distance, interpolate_y, is_valid
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


def measure_width_in_band_at_y(
    mask: np.ndarray,
    y: float,
    x_min: float,
    x_max: float,
    margin: int = 3,
) -> float | None:
    """Measure horizontal width inside an x-band [x_min, x_max] at a given y.

    Used for body-girth slices (torso / waist / hip / neck) on the front view
    so outstretched arms can't contaminate the measurement: the band is set
    around the shoulders/hips, and any silhouette pixels outside it (i.e. arms
    extended laterally) are ignored. Falls back to the unclipped width when
    nothing is found inside the band — that way pure-frontal A-pose photos
    where the arms are inside the band keep working unchanged.
    """
    h, w = mask.shape
    y_int = int(round(y))
    y_min = max(0, y_int - margin)
    y_max = min(h - 1, y_int + margin)

    x_lo = max(0, int(round(x_min)))
    x_hi = min(w - 1, int(round(x_max)))
    if x_hi <= x_lo:
        return measure_width_at_y(mask, y, margin)

    widths = []
    for row in range(y_min, y_max + 1):
        # Pick the cols that are simultaneously foreground AND inside the band.
        row_mask = mask[row, x_lo : x_hi + 1]
        cols = np.where(row_mask)[0]
        if len(cols) >= 2:
            # Cols are relative to the band, but the width is just (last-first).
            widths.append(cols[-1] - cols[0])

    if not widths:
        # No foreground inside band at this y — fall back to the unclipped read
        # so we don't drop a measurement entirely.
        return measure_width_at_y(mask, y, margin)

    return float(np.mean(widths))


def _continuous_row_width(mask_row: np.ndarray) -> float | None:
    """Width of the longest continuous foreground segment on one row."""
    cols = np.where(mask_row)[0]
    if len(cols) < 2:
        return None
    segments = _find_segments(cols)
    best = max(segments, key=lambda s: (s[-1] - s[0]))
    return float(best[-1] - best[0]) if len(best) >= 2 else None


def _continuous_width_at_y(
    mask: np.ndarray,
    y: float,
    margin: int = 3,
    x_band: tuple[float, float] | None = None,
) -> float | None:
    """Longest continuous width at y, averaged over a small vertical band."""
    h, w = mask.shape
    y_int = int(round(y))
    y_min = max(0, y_int - margin)
    y_max = min(h - 1, y_int + margin)

    if x_band is not None:
        x_lo = max(0, int(round(x_band[0])))
        x_hi = min(w - 1, int(round(x_band[1])))
        if x_hi <= x_lo:
            x_band = None

    widths: list[float] = []
    for row in range(y_min, y_max + 1):
        if x_band is None:
            row_mask = mask[row]
        else:
            row_mask = mask[row, x_lo : x_hi + 1]
        w_row = _continuous_row_width(row_mask)
        if w_row is not None:
            widths.append(w_row)
    if not widths:
        return None
    return float(np.mean(widths))


def _extreme_continuous_width_between_y(
    mask: np.ndarray,
    y0: float,
    y1: float,
    *,
    prefer: str,
    x_band: tuple[float, float] | None = None,
) -> float | None:
    """Min or max continuous width across all rows between y0 and y1."""
    lo = int(round(min(y0, y1)))
    hi = int(round(max(y0, y1)))
    vals: list[float] = []
    for yi in range(lo, hi + 1):
        w = _continuous_width_at_y(mask, float(yi), margin=0, x_band=x_band)
        if w is not None:
            vals.append(w)
    if not vals:
        return None
    return float(min(vals) if prefer == "min" else max(vals))


def _extreme_continuous_width_and_y_between_y(
    mask: np.ndarray,
    y0: float,
    y1: float,
    *,
    prefer: str,
    x_band: tuple[float, float] | None = None,
) -> tuple[float | None, float | None]:
    """Min/max continuous width and selected y-row."""
    lo = int(round(min(y0, y1)))
    hi = int(round(max(y0, y1)))
    best_w: float | None = None
    best_y: float | None = None
    for yi in range(lo, hi + 1):
        w = _continuous_width_at_y(mask, float(yi), margin=0, x_band=x_band)
        if w is None:
            continue
        if best_w is None:
            best_w = float(w)
            best_y = float(yi)
            continue
        if (prefer == "min" and w < best_w) or (prefer == "max" and w > best_w):
            best_w = float(w)
            best_y = float(yi)
    return best_w, best_y


def _side_torso_band(side_kp: Keypoints, mask_w: int) -> tuple[float, float] | None:
    """Compute the [x_min, x_max] band that bounds the side-view torso depth.

    On a side view we cannot use shoulders as a reference (left and right
    shoulder project onto roughly the same x). Instead we use keypoints that
    sit on the body's central column — UPPER_NECK, THORAX, PELVIS, midpoint of
    the hips — and pad symmetrically by a fraction of the body's pixel height.
    Arms extending laterally toward / away from the camera land outside this
    band, so the silhouette slice we keep is the body's actual front-to-back
    depth.
    """
    pts, conf = side_kp.points, side_kp.confidence

    spine_xs: list[float] = []
    for kp_idx in (KP.UPPER_NECK, KP.THORAX, KP.PELVIS):
        if is_valid(conf, kp_idx):
            spine_xs.append(float(pts[kp_idx, 0]))
    if is_valid(conf, KP.LEFT_HIP, KP.RIGHT_HIP):
        spine_xs.append(float((pts[KP.LEFT_HIP, 0] + pts[KP.RIGHT_HIP, 0]) / 2))

    if not spine_xs:
        return None

    center_x = float(np.mean(spine_xs))

    # Half-band width: 18 % of body pixel-height — typical adult torso depth is
    # ~14 % of body height, plus a margin so we don't shave the silhouette.
    body_h_px: float | None = None
    head_y = pts[KP.HEAD_TOP, 1] if is_valid(conf, KP.HEAD_TOP) else None
    ankle_ys: list[float] = []
    if is_valid(conf, KP.LEFT_ANKLE):
        ankle_ys.append(float(pts[KP.LEFT_ANKLE, 1]))
    if is_valid(conf, KP.RIGHT_ANKLE):
        ankle_ys.append(float(pts[KP.RIGHT_ANKLE, 1]))
    if head_y is not None and ankle_ys:
        body_h_px = abs(float(np.mean(ankle_ys)) - float(head_y))

    if body_h_px is None or body_h_px <= 0:
        # Fall back to 20 % of mask width, which is a generous-but-safe default.
        half_band = 0.10 * mask_w
    else:
        half_band = 0.18 * body_h_px

    return (
        max(0.0, center_x - half_band),
        min(float(mask_w - 1), center_x + half_band),
    )


def _front_torso_band(front_kp: Keypoints, mask_w: int) -> tuple[float, float] | None:
    """Compute the [x_min, x_max] band that bounds the front-view torso.

    Uses shoulders and hips when available (the body's lateral envelope is the
    wider of those two), padded by ~10 % of body width on each side so the band
    is generous enough to keep the torso silhouette but narrow enough to exclude
    arms outstretched far to the sides.

    Returns None when keypoints are insufficient — caller should fall back to
    unclipped width measurement.
    """
    pts, conf = front_kp.points, front_kp.confidence

    xs: list[float] = []
    if is_valid(conf, KP.LEFT_SHOULDER):
        xs.append(float(pts[KP.LEFT_SHOULDER, 0]))
    if is_valid(conf, KP.RIGHT_SHOULDER):
        xs.append(float(pts[KP.RIGHT_SHOULDER, 0]))
    if is_valid(conf, KP.LEFT_HIP):
        xs.append(float(pts[KP.LEFT_HIP, 0]))
    if is_valid(conf, KP.RIGHT_HIP):
        xs.append(float(pts[KP.RIGHT_HIP, 0]))

    if len(xs) < 2:
        return None

    x_lo = min(xs)
    x_hi = max(xs)
    body_w = x_hi - x_lo
    if body_w <= 0:
        return None

    pad = 0.10 * body_w  # 10 % padding so the band hugs the torso closely
    return (max(0.0, x_lo - pad), min(float(mask_w - 1), x_hi + pad))


def measure_limb_width_at_y(
    mask: np.ndarray,
    y: float,
    x_hint: float,
    margin: int = 3,
    x_band: tuple[float, float] | None = None,
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
        if x_band is not None and len(cols) > 0:
            x_lo = max(0, int(round(x_band[0])))
            x_hi = min(w - 1, int(round(x_band[1])))
            cols = cols[(cols >= x_lo) & (cols <= x_hi)]
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
) -> tuple[
    dict[str, tuple[float | None, float | None]],
    dict[str, tuple[float | None, float | None]],
]:
    """Extract body widths at key y-coordinates from both views.

    Returns:
      - widths: body part -> (front_width_px, side_width_px)
      - selected_y: body part -> (front_y_px, side_y_px) for selected lines
    """
    f_pts, f_conf = front_kp.points, front_kp.confidence
    s_pts, s_conf = side_kp.points, side_kp.confidence
    f_mask = front_mask.mask
    s_mask = side_mask.mask

    # Pose-aware bands for body-girth slices: keeps outstretched / forward arms
    # from contaminating torso / waist / hip / neck width slices. Each band
    # falls back to unclipped width when keypoints are insufficient.
    f_band = _front_torso_band(front_kp, f_mask.shape[1])
    if f_band is not None:
        x_lo_f, x_hi_f = f_band

        def _front_torso_width(mask: np.ndarray, y: float) -> float | None:
            return measure_width_in_band_at_y(mask, y, x_lo_f, x_hi_f)
    else:
        def _front_torso_width(mask: np.ndarray, y: float) -> float | None:
            return measure_width_at_y(mask, y)

    s_band = _side_torso_band(side_kp, s_mask.shape[1])
    if s_band is not None:
        x_lo_s, x_hi_s = s_band

        def _side_torso_width(mask: np.ndarray, y: float) -> float | None:
            return measure_width_in_band_at_y(mask, y, x_lo_s, x_hi_s)
    else:
        def _side_torso_width(mask: np.ndarray, y: float) -> float | None:
            return measure_width_at_y(mask, y)

    widths: dict[str, tuple[float | None, float | None]] = {}
    selected_y: dict[str, tuple[float | None, float | None]] = {}

    # Head: midpoint between head_top and upper_neck
    if is_valid(f_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        y_head_f = (f_pts[KP.HEAD_TOP, 1] + f_pts[KP.UPPER_NECK, 1]) / 2
        widths["head"] = (_front_torso_width(f_mask, y_head_f), None)
    if is_valid(s_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        y_head_s = (s_pts[KP.HEAD_TOP, 1] + s_pts[KP.UPPER_NECK, 1]) / 2
        existing = widths.get("head", (None, None))
        widths["head"] = (existing[0], _side_torso_width(s_mask, y_head_s))

    # Neck: shortest continuous line from a line below nose down to upper_neck.
    # Start line: nose_y + |nose_y - head_top_y|.
    if is_valid(f_conf, KP.UPPER_NECK):
        y_upper_neck = float(f_pts[KP.UPPER_NECK, 1])
        y_start = None
        if (
            front_kp.nose_xy is not None
            and (front_kp.nose_conf or 0.0) >= 0.2
            and is_valid(f_conf, KP.HEAD_TOP)
        ):
            y_nose = float(front_kp.nose_xy[1])
            y_head_top = float(f_pts[KP.HEAD_TOP, 1])
            y_start = y_nose + abs(y_nose - y_head_top)
        elif front_kp.nose_xy is not None and (front_kp.nose_conf or 0.0) >= 0.2:
            y_start = float(front_kp.nose_xy[1])
        elif is_valid(f_conf, KP.HEAD_TOP):
            y_start = float(f_pts[KP.HEAD_TOP, 1])
        if y_start is not None:
            neck_f = _extreme_continuous_width_between_y(
                f_mask, y_start, y_upper_neck, prefer="min", x_band=f_band
            )
            widths["neck"] = (neck_f, None)
    if is_valid(s_conf, KP.UPPER_NECK):
        y_upper_neck = float(s_pts[KP.UPPER_NECK, 1])
        y_start = None
        if (
            side_kp.nose_xy is not None
            and (side_kp.nose_conf or 0.0) >= 0.2
            and is_valid(s_conf, KP.HEAD_TOP)
        ):
            y_nose = float(side_kp.nose_xy[1])
            y_head_top = float(s_pts[KP.HEAD_TOP, 1])
            y_start = y_nose + abs(y_nose - y_head_top)
        elif side_kp.nose_xy is not None and (side_kp.nose_conf or 0.0) >= 0.2:
            y_start = float(side_kp.nose_xy[1])
        elif is_valid(s_conf, KP.HEAD_TOP):
            y_start = float(s_pts[KP.HEAD_TOP, 1])
        if y_start is not None:
            neck_s = _extreme_continuous_width_between_y(
                s_mask, y_start, y_upper_neck, prefer="min", x_band=s_band
            )
            existing = widths.get("neck", (None, None))
            widths["neck"] = (existing[0], neck_s)

    # Torso width:
    # - front: distance between shoulder points
    # - side: continuous silhouette width at midpoint between upper_neck and elbow
    if is_valid(f_conf, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER):
        widths["torso"] = (distance(f_pts, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER), None)
    if is_valid(s_conf, KP.UPPER_NECK) and (is_valid(s_conf, KP.RIGHT_ELBOW) or is_valid(s_conf, KP.LEFT_ELBOW)):
        elbow_y = (
            float(s_pts[KP.RIGHT_ELBOW, 1])
            if is_valid(s_conf, KP.RIGHT_ELBOW)
            else float(s_pts[KP.LEFT_ELBOW, 1])
        )
        y_torso_side = 0.5 * (float(s_pts[KP.UPPER_NECK, 1]) + elbow_y)
        torso_s = _continuous_width_at_y(s_mask, y_torso_side, margin=3, x_band=s_band)
        existing = widths.get("torso", (None, None))
        widths["torso"] = (
            existing[0],
            torso_s,
        )

    # Waist: shortest continuous line from pelvis to pelvis + 0.5*(upper_neck - pelvis).
    if is_valid(f_conf, KP.PELVIS, KP.UPPER_NECK):
        y_pelvis = f_pts[KP.PELVIS, 1]
        y_mid = y_pelvis + 0.4 * (f_pts[KP.UPPER_NECK, 1] - y_pelvis)
        waist_f, waist_f_y = _extreme_continuous_width_and_y_between_y(
            f_mask, y_pelvis, y_mid, prefer="min", x_band=f_band
        )
        widths["waist"] = (waist_f, None)
        selected_y["waist"] = (waist_f_y, None)
    if is_valid(s_conf, KP.PELVIS, KP.UPPER_NECK):
        y_pelvis = s_pts[KP.PELVIS, 1]
        y_mid = y_pelvis + 0.4 * (s_pts[KP.UPPER_NECK, 1] - y_pelvis)
        waist_s, waist_s_y = _extreme_continuous_width_and_y_between_y(
            s_mask, y_pelvis, y_mid, prefer="min", x_band=s_band
        )
        existing = widths.get("waist", (None, None))
        widths["waist"] = (existing[0], waist_s)
        existing_y = selected_y.get("waist", (None, None))
        selected_y["waist"] = (existing_y[0], waist_s_y)

    # Hip: largest continuous line from pelvis to knee.
    if is_valid(f_conf, KP.PELVIS):
        y_pelvis = f_pts[KP.PELVIS, 1]
        knee_ys = []
        if is_valid(f_conf, KP.LEFT_KNEE):
            knee_ys.append(float(f_pts[KP.LEFT_KNEE, 1]))
        if is_valid(f_conf, KP.RIGHT_KNEE):
            knee_ys.append(float(f_pts[KP.RIGHT_KNEE, 1]))
        if knee_ys:
            y_knee = float(np.mean(knee_ys))
            y_start = y_pelvis + 0.05 * (y_knee - y_pelvis)
            hip_f = _extreme_continuous_width_between_y(
                f_mask, y_start, y_knee, prefer="max", x_band=f_band
            )
            widths["hip"] = (hip_f, None)

    if is_valid(s_conf, KP.PELVIS, KP.RIGHT_KNEE):
        y_pelvis = s_pts[KP.PELVIS, 1]
        y_knee = s_pts[KP.RIGHT_KNEE, 1]
        y_start = y_pelvis + 0.05 * (y_knee - y_pelvis)
        hip_s = _extreme_continuous_width_between_y(
            s_mask, y_start, y_knee, prefer="max", x_band=s_band
        )
        existing = widths.get("hip", (None, None))
        widths["hip"] = (existing[0], hip_s)

    # Thigh:
    # - front: at the level where center gap between legs begins; measure one-leg continuous segment
    # - side: at 30% below pelvis (toward knee/ankle)
    if is_valid(f_conf, KP.PELVIS):
        pelvis_x = int(round(float(f_pts[KP.PELVIS, 0])))
        pelvis_y = int(round(float(f_pts[KP.PELVIS, 1])))
        pelvis_x = int(np.clip(pelvis_x, 0, f_mask.shape[1] - 1))
        pelvis_y = int(np.clip(pelvis_y, 0, f_mask.shape[0] - 1))
        thigh_split_x = (
            int(round(float(f_pts[KP.UPPER_NECK, 0])))
            if is_valid(f_conf, KP.UPPER_NECK)
            else pelvis_x
        )
        thigh_split_x = int(np.clip(thigh_split_x, 0, f_mask.shape[1] - 1))
        ankle_ys = []
        if is_valid(f_conf, KP.LEFT_ANKLE):
            ankle_ys.append(float(f_pts[KP.LEFT_ANKLE, 1]))
        if is_valid(f_conf, KP.RIGHT_ANKLE):
            ankle_ys.append(float(f_pts[KP.RIGHT_ANKLE, 1]))
        knee_ys = []
        if is_valid(f_conf, KP.LEFT_KNEE):
            knee_ys.append(float(f_pts[KP.LEFT_KNEE, 1]))
        if is_valid(f_conf, KP.RIGHT_KNEE):
            knee_ys.append(float(f_pts[KP.RIGHT_KNEE, 1]))
        if ankle_ys:
            y_bottom = int(round(np.mean(ankle_ys)))
        elif knee_ys:
            y_bottom = int(round(np.mean(knee_ys)))
        else:
            y_bottom = f_mask.shape[0] - 1
        y_bottom = int(np.clip(y_bottom, pelvis_y + 1, f_mask.shape[0] - 1))
        y_thigh_f = pelvis_y
        for y in range(pelvis_y, y_bottom + 1):
            if not f_mask[y, pelvis_x]:
                y_thigh_f = y
                break
        for side_name, kp_knee in [("right", KP.RIGHT_KNEE), ("left", KP.LEFT_KNEE)]:
            if is_valid(f_conf, kp_knee):
                x_hint = f_pts[kp_knee, 0]
            elif side_name == "right" and is_valid(f_conf, KP.RIGHT_HIP):
                x_hint = f_pts[KP.RIGHT_HIP, 0]
            elif side_name == "left" and is_valid(f_conf, KP.LEFT_HIP):
                x_hint = f_pts[KP.LEFT_HIP, 0]
            else:
                continue
            if side_name == "right":
                x_band = (0.0, thigh_split_x)
            else:
                x_band = (thigh_split_x, float(f_mask.shape[1] - 1))
            front_w = measure_limb_width_at_y(f_mask, y_thigh_f, x_hint, x_band=x_band)
            widths[f"thigh_{side_name}"] = (front_w, None)

    if is_valid(s_conf, KP.PELVIS):
        y_pelvis_s = float(s_pts[KP.PELVIS, 1])
        if is_valid(s_conf, KP.RIGHT_KNEE):
            y_ref = float(s_pts[KP.RIGHT_KNEE, 1])
        elif is_valid(s_conf, KP.RIGHT_ANKLE):
            y_ref = float(s_pts[KP.RIGHT_ANKLE, 1])
        else:
            y_ref = y_pelvis_s
        y_thigh_s = y_pelvis_s + 0.5 * (y_ref - y_pelvis_s)
        side_w = _side_torso_width(s_mask, y_thigh_s)
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
        side_w = _side_torso_width(s_mask, y_calf_s)
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
        side_w = _side_torso_width(s_mask, y_wrist_s)
        for key in ["wrist_right", "wrist_left"]:
            existing = widths.get(key, (None, None))
            widths[key] = (existing[0], side_w)

    return widths, selected_y
