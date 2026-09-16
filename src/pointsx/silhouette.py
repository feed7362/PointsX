"""Silhouette (segmentation mask) processing for width measurements."""

from __future__ import annotations

import numpy as np

from pointsx.keypoints import KP, NOSE_MIN_CONFIDENCE, distance, interpolate_y, is_valid, mean_valid_y
from pointsx.schemas import Keypoints, SilhouetteMask

# Upper-leg fraction caps — keep hip/thigh lines out of the knee region.
_HIP_SEARCH_MAX_FRAC = 0.20
_THIGH_GAP_SCAN_MAX_FRAC = 0.35
_THIGH_FALLBACK_FRAC = 0.25


def _pelvis_and_knee_y(kp: Keypoints) -> tuple[float, float] | None:
    """Return (y_pelvis, y_knee) when both are available from keypoints."""
    if not is_valid(kp.confidence, KP.PELVIS):
        return None
    y_knee = mean_valid_y(kp, KP.LEFT_KNEE, KP.RIGHT_KNEE)
    if y_knee is None:
        return None
    y_pelvis = float(kp.points[KP.PELVIS, 1])
    if y_knee <= y_pelvis + 1.0:
        return None
    return y_pelvis, y_knee


def hip_search_y_range(kp: Keypoints) -> tuple[float, float] | None:
    """Anatomical y-range for hip-width search (pelvis .. upper thigh only).

    Anchors on left/right hip keypoints when present; never searches below
    ``_HIP_SEARCH_MAX_FRAC`` of the pelvis-to-knee span.
    """
    span = _pelvis_and_knee_y(kp)
    if span is None:
        return None
    y_pelvis, y_knee = span
    leg_span = y_knee - y_pelvis
    y_hard_max = y_pelvis + _HIP_SEARCH_MAX_FRAC * leg_span

    y_hip = mean_valid_y(kp, KP.LEFT_HIP, KP.RIGHT_HIP)
    if y_hip is not None:
        band = 0.06 * leg_span
        y_start = max(y_pelvis + 0.02 * leg_span, y_hip - band)
        y_end = min(y_hard_max, y_hip + band)
    else:
        y_start = y_pelvis + 0.05 * leg_span
        y_end = y_pelvis + 0.16 * leg_span

    if y_end <= y_start:
        y_end = min(y_hard_max, y_start + 0.05 * leg_span)
    return y_start, y_end


def front_thigh_y_level(kp: Keypoints, mask: np.ndarray) -> float | None:
    """Front-view thigh line y: leg-gap row, capped to upper thigh (not knee)."""
    pts, conf = kp.points, kp.confidence
    if not is_valid(conf, KP.PELVIS):
        return None

    pelvis_x = int(round(float(pts[KP.PELVIS, 0])))
    pelvis_y = int(round(float(pts[KP.PELVIS, 1])))
    h, w = mask.shape
    pelvis_x = int(np.clip(pelvis_x, 0, w - 1))
    pelvis_y = int(np.clip(pelvis_y, 0, h - 1))

    span = _pelvis_and_knee_y(kp)
    if span is not None:
        y_pelvis_f, y_knee = span
        y_scan_max = int(round(y_pelvis_f + _THIGH_GAP_SCAN_MAX_FRAC * (y_knee - y_pelvis_f)))
    else:
        y_scan_max = min(h - 1, pelvis_y + int(0.15 * h))
    y_scan_max = int(np.clip(y_scan_max, pelvis_y + 1, h - 1))

    y_thigh = pelvis_y
    for y in range(pelvis_y, y_scan_max + 1):
        if not mask[y, pelvis_x]:
            y_thigh = y
            break

    if y_thigh == pelvis_y and span is not None:
        y_pelvis_f, y_knee = span
        y_hip = mean_valid_y(kp, KP.LEFT_HIP, KP.RIGHT_HIP)
        if y_hip is not None:
            y_thigh = int(round(y_hip + 0.12 * (y_knee - y_hip)))
        else:
            y_thigh = int(round(y_pelvis_f + _THIGH_FALLBACK_FRAC * (y_knee - y_pelvis_f)))
        y_thigh = int(np.clip(y_thigh, pelvis_y + 1, y_scan_max))

    return float(y_thigh)


# FRONT-view waist row, as a fraction of the pelvis->upper-neck span. Only the
# front is anchored this way; the side keeps its min-search (see extract_all_widths).
WAIST_FRONT_PELVIS_NECK_FRACTION = 0.25

# SIDE-view waist: narrowest continuous row searched between the pelvis and this
# fraction of the pelvis->upper-neck span. Also the fallback waist row when the
# search finds nothing (measurements.py).
WAIST_SIDE_SEARCH_TOP_FRACTION = 0.4


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
    ankle_y = mean_valid_y(side_kp, KP.LEFT_ANKLE, KP.RIGHT_ANKLE)
    if head_y is not None and ankle_y is not None:
        body_h_px = abs(ankle_y - float(head_y))

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


def _band_width(mask: np.ndarray, y: float, band: tuple[float, float] | None) -> float | None:
    """Silhouette span at row ``y``, clipped to the torso ``band`` when one exists.

    Pose-aware bands keep outstretched / forward arms from contaminating
    torso / hip / neck slices; without keypoints for a band we fall back to the
    unclipped span.
    """
    if band is not None:
        return measure_width_in_band_at_y(mask, y, band[0], band[1])
    return measure_width_at_y(mask, y)


def _neck_scan_start_y(kp: Keypoints) -> float | None:
    """Top row of the neck search window (the same rule for both views).

    Prefer a row one nose-to-crown distance *below* the nose (i.e. the chin),
    then the nose itself, then the head top. None when nothing usable exists.
    """
    pts, conf = kp.points, kp.confidence
    nose_ok = kp.nose_xy is not None and (kp.nose_conf or 0.0) >= NOSE_MIN_CONFIDENCE
    if nose_ok and is_valid(conf, KP.HEAD_TOP):
        y_nose = float(kp.nose_xy[1])
        return y_nose + abs(y_nose - float(pts[KP.HEAD_TOP, 1]))
    if nose_ok:
        return float(kp.nose_xy[1])
    if is_valid(conf, KP.HEAD_TOP):
        return float(pts[KP.HEAD_TOP, 1])
    return None


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

    Front and side are collected into separate per-view dicts and zipped at the
    end, so every key present in either view appears in the result as
    ``(front_px | None, side_px | None)``.

    Returns:
      - widths: body part -> (front_width_px, side_width_px)
      - selected_y: body part -> (front_y_px, side_y_px) for searched rows
    """
    f_pts, f_conf = front_kp.points, front_kp.confidence
    s_pts, s_conf = side_kp.points, side_kp.confidence
    f_mask = front_mask.mask
    s_mask = side_mask.mask

    f_band = _front_torso_band(front_kp, f_mask.shape[1])
    s_band = _side_torso_band(side_kp, s_mask.shape[1])

    front: dict[str, float | None] = {}
    side: dict[str, float | None] = {}
    front_y: dict[str, float | None] = {}
    side_y: dict[str, float | None] = {}

    # Head: midpoint between head_top and upper_neck
    if is_valid(f_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        front["head"] = _band_width(f_mask, (f_pts[KP.HEAD_TOP, 1] + f_pts[KP.UPPER_NECK, 1]) / 2, f_band)
    if is_valid(s_conf, KP.HEAD_TOP, KP.UPPER_NECK):
        side["head"] = _band_width(s_mask, (s_pts[KP.HEAD_TOP, 1] + s_pts[KP.UPPER_NECK, 1]) / 2, s_band)

    # Neck: narrowest continuous row between the chin line and upper_neck.
    for kp, mask, band, out in ((front_kp, f_mask, f_band, front), (side_kp, s_mask, s_band, side)):
        if not is_valid(kp.confidence, KP.UPPER_NECK):
            continue
        y_start = _neck_scan_start_y(kp)
        if y_start is not None:
            out["neck"] = _extreme_continuous_width_and_y_between_y(
                mask, y_start, float(kp.points[KP.UPPER_NECK, 1]), prefer="min", x_band=band
            )[0]

    # Torso width:
    # - front: distance between shoulder points
    # - side: continuous silhouette width at midpoint between upper_neck and elbow
    if is_valid(f_conf, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER):
        front["torso"] = distance(f_pts, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER)
    if is_valid(s_conf, KP.UPPER_NECK) and (is_valid(s_conf, KP.RIGHT_ELBOW) or is_valid(s_conf, KP.LEFT_ELBOW)):
        elbow_y = (
            float(s_pts[KP.RIGHT_ELBOW, 1])
            if is_valid(s_conf, KP.RIGHT_ELBOW)
            else float(s_pts[KP.LEFT_ELBOW, 1])
        )
        y_torso_side = 0.5 * (float(s_pts[KP.UPPER_NECK, 1]) + elbow_y)
        side["torso"] = _continuous_width_at_y(s_mask, y_torso_side, margin=3, x_band=s_band)

    # Waist — the two views deliberately use DIFFERENT methods, because the same
    # method is not reliable in both:
    #
    #   FRONT: fixed anatomical row. Arms hang beside the waist and a loose top
    #     bridges the taper, so the silhouette minimum is set by clothing/pose,
    #     not anatomy. Measured against tape GT: min-search corr +0.77,
    #     fixed 0.25 corr +0.89 (0.20 and 0.30 both +0.88 — broad optimum).
    #   SIDE: keep the min-search (below). In profile there are no arms crossing
    #     the torso, so the narrowest row genuinely IS the waist, and searching
    #     adapts to where each person's waist actually sits. Measured: min-search
    #     corr +0.84 beats every fixed fraction (best +0.79).
    #
    # Combination verified end-to-end (leave-one-out, per-fold refit constant):
    #   front=min + side=min  MAE 8.92   ->   front=0.25 + side=min  MAE 8.17
    # Both views still target the same anatomy; only the way they locate it
    # differs.
    if is_valid(f_conf, KP.PELVIS, KP.UPPER_NECK):
        y_pelvis = float(f_pts[KP.PELVIS, 1])
        y_waist_f = y_pelvis + WAIST_FRONT_PELVIS_NECK_FRACTION * (float(f_pts[KP.UPPER_NECK, 1]) - y_pelvis)
        front["waist"] = _continuous_width_at_y(f_mask, y_waist_f, margin=3, x_band=f_band)
        front_y["waist"] = y_waist_f
    if is_valid(s_conf, KP.PELVIS, KP.UPPER_NECK):
        y_pelvis = s_pts[KP.PELVIS, 1]
        y_top = y_pelvis + WAIST_SIDE_SEARCH_TOP_FRACTION * (s_pts[KP.UPPER_NECK, 1] - y_pelvis)
        side["waist"], side_y["waist"] = _extreme_continuous_width_and_y_between_y(
            s_mask, y_pelvis, y_top, prefer="min", x_band=s_band
        )

    # Hip: largest continuous width in upper thigh (anchored on hip keypoints).
    for kp, mask, band, out in ((front_kp, f_mask, f_band, front), (side_kp, s_mask, s_band, side)):
        hip_range = hip_search_y_range(kp)
        if hip_range is not None:
            out["hip"] = _extreme_continuous_width_and_y_between_y(
                mask, hip_range[0], hip_range[1], prefer="max", x_band=band
            )[0]

    # Thigh:
    # - front: at the level where center gap between legs begins; measure one-leg continuous segment
    # - side: at 50% between pelvis and knee (one depth, shared by both legs)
    if is_valid(f_conf, KP.PELVIS):
        w_img = f_mask.shape[1]
        pelvis_x = int(np.clip(int(round(float(f_pts[KP.PELVIS, 0]))), 0, w_img - 1))
        thigh_split_x = (
            int(round(float(f_pts[KP.UPPER_NECK, 0])))
            if is_valid(f_conf, KP.UPPER_NECK)
            else pelvis_x
        )
        thigh_split_x = int(np.clip(thigh_split_x, 0, w_img - 1))
        # front_thigh_y_level only returns None without a PELVIS, which is guarded above.
        y_thigh_f = int(round(front_thigh_y_level(front_kp, f_mask)))
        for side_name, kp_knee, kp_hip, x_band in (
            ("right", KP.RIGHT_KNEE, KP.RIGHT_HIP, (0.0, float(thigh_split_x))),
            ("left", KP.LEFT_KNEE, KP.LEFT_HIP, (float(thigh_split_x), float(w_img - 1))),
        ):
            if is_valid(f_conf, kp_knee):
                x_hint = f_pts[kp_knee, 0]
            elif is_valid(f_conf, kp_hip):
                x_hint = f_pts[kp_hip, 0]
            else:
                continue
            front[f"thigh_{side_name}"] = measure_limb_width_at_y(f_mask, y_thigh_f, x_hint, x_band=x_band)

    if is_valid(s_conf, KP.PELVIS):
        y_pelvis_s = float(s_pts[KP.PELVIS, 1])
        if is_valid(s_conf, KP.RIGHT_KNEE):
            y_ref = float(s_pts[KP.RIGHT_KNEE, 1])
        elif is_valid(s_conf, KP.RIGHT_ANKLE):
            y_ref = float(s_pts[KP.RIGHT_ANKLE, 1])
        else:
            y_ref = y_pelvis_s
        # NOTE: deliberately NOT the 0.10 hip->knee fraction that fits the FRONT
        # view. The side view is anchored to the PELVIS keypoint, so 0.10 lands on
        # the buttocks and reads a depth of 23-38 cm — anatomically impossible for
        # a thigh. It scored a marginally better corrected MAE (4.7 vs 5.7) only
        # because buttock depth correlates with thigh size, while the RAW ellipse
        # error doubled (13 -> 27 cm), i.e. the constant was hiding mismatched
        # inputs. 0.5 keeps the slice on the thigh itself.
        side["thigh_right"] = side["thigh_left"] = _band_width(
            s_mask, y_pelvis_s + 0.5 * (y_ref - y_pelvis_s), s_band
        )

    # Calf: 60% between knee and ankle
    for side_name, kp_knee, kp_ankle in (
        ("right", KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        ("left", KP.LEFT_KNEE, KP.LEFT_ANKLE),
    ):
        if is_valid(f_conf, kp_knee, kp_ankle):
            y_calf_f = interpolate_y(f_pts, kp_knee, kp_ankle, 0.6)
            front[f"calf_{side_name}"] = measure_limb_width_at_y(f_mask, y_calf_f, f_pts[kp_ankle, 0])
    if is_valid(s_conf, KP.RIGHT_KNEE, KP.RIGHT_ANKLE):
        y_calf_s = interpolate_y(s_pts, KP.RIGHT_KNEE, KP.RIGHT_ANKLE, 0.6)
        side["calf_right"] = side["calf_left"] = _band_width(s_mask, y_calf_s, s_band)

    # Wrist
    for side_name, kp_wrist in (("right", KP.RIGHT_WRIST), ("left", KP.LEFT_WRIST)):
        if is_valid(f_conf, kp_wrist):
            front[f"wrist_{side_name}"] = measure_limb_width_at_y(
                f_mask, f_pts[kp_wrist, 1], f_pts[kp_wrist, 0]
            )
    if is_valid(s_conf, KP.RIGHT_WRIST):
        side["wrist_right"] = side["wrist_left"] = _band_width(s_mask, s_pts[KP.RIGHT_WRIST, 1], s_band)

    widths = {key: (front.get(key), side.get(key)) for key in front.keys() | side.keys()}
    selected_y = {key: (front_y.get(key), side_y.get(key)) for key in front_y.keys() | side_y.keys()}
    return widths, selected_y
