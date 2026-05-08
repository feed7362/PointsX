"""Core measurement extraction: combines pose keypoints with silhouette widths."""

from __future__ import annotations

import numpy as np

from pointsx.keypoints import KP, distance, is_valid, midpoint
from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints, SilhouetteMask
from pointsx.silhouette import extract_all_widths


def _avg(*values: float | None) -> float | None:
    """Average of non-None values."""
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def _split_segments(cols: np.ndarray) -> list[np.ndarray]:
    if len(cols) == 0:
        return []
    diffs = np.diff(cols)
    split_points = np.where(diffs > 3)[0] + 1
    return np.split(cols, split_points)


def _left_arm_contour_path(
    mask: np.ndarray,
    pts: np.ndarray,
    conf: np.ndarray,
) -> list[tuple[int, int]]:
    """Front-view left-arm contour path from shoulder level to wrist level."""
    if not is_valid(conf, KP.LEFT_SHOULDER, KP.LEFT_WRIST):
        return []

    h, w = mask.shape
    y0 = int(round(float(pts[KP.LEFT_SHOULDER, 1])))
    y1 = int(round(float(pts[KP.LEFT_WRIST, 1])))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x_shoulder = int(np.clip(int(round(float(pts[KP.LEFT_SHOULDER, 0]))), 0, w - 1))
    y_wrist = y1

    x_mid_img = 0.5 * (w - 1)
    # Start: at shoulder y, then move up along left edge until segment corner.
    max_jump = 18.0
    cols0 = np.where(mask[y0])[0]
    if len(cols0) < 2:
        return []
    segments0 = [s for s in _split_segments(cols0) if len(s) >= 2]
    left_half0 = [s for s in segments0 if 0.5 * (float(s[0]) + float(s[-1])) <= x_mid_img]
    if not left_half0:
        return []
    containing0 = [s for s in left_half0 if int(s[0]) <= x_shoulder <= int(s[-1])]
    seg0 = (
        min(containing0, key=lambda s: abs(((s[0] + s[-1]) * 0.5) - x_shoulder))
        if containing0
        else min(left_half0, key=lambda s: abs(((s[0] + s[-1]) * 0.5) - x_shoulder))
    )
    x_corner = int(seg0[0])
    y_corner = y0

    for y in range(y0 - 1, -1, -1):
        cols = np.where(mask[y])[0]
        if len(cols) < 2:
            break
        segments = [s for s in _split_segments(cols) if len(s) >= 2]
        left_half_segments = [s for s in segments if 0.5 * (float(s[0]) + float(s[-1])) <= x_mid_img]
        if not left_half_segments:
            break
        # Keep search anchored to the shoulder column to avoid drifting into head/torso.
        containing = [s for s in left_half_segments if int(s[0]) <= x_shoulder <= int(s[-1])]
        if not containing:
            break
        candidates = [s for s in containing if abs(float(s[0]) - x_corner) <= max_jump]
        if not candidates:
            break
        seg = min(candidates, key=lambda s: abs(float(s[0]) - x_corner))
        x_corner = int(seg[0])
        y_corner = y

    y_start = y_corner
    x_track = float(x_corner)

    step = 1 if y_wrist >= y_start else -1
    path: list[tuple[int, int]] = [(int(round(x_track)), y_start)]
    for y in range(y_start, y_wrist + step, step):
        cols = np.where(mask[y])[0]
        if len(cols) < 2:
            continue
        segments = [s for s in _split_segments(cols) if len(s) >= 2]
        if not segments:
            continue
        left_half_segments = [s for s in segments if 0.5 * (float(s[0]) + float(s[-1])) <= x_mid_img]
        candidates = [s for s in left_half_segments if abs(((s[0] + s[-1]) * 0.5) - x_track) <= max_jump]
        pool = candidates if candidates else left_half_segments
        if not pool:
            continue
        seg = min(pool, key=lambda s: abs(((s[0] + s[-1]) * 0.5) - x_track))
        # Use only the left edge of the selected arm segment.
        x = int(seg[0])
        path.append((x, y))
        x_track = float(x)

    # End: left-most intersection of wrist y-line with the selected segment.
    cols_w = np.where(mask[y_wrist])[0]
    if len(cols_w) >= 2:
        segments_w = [s for s in _split_segments(cols_w) if len(s) >= 2]
        if segments_w:
            left_half_segments_w = [s for s in segments_w if 0.5 * (float(s[0]) + float(s[-1])) <= x_mid_img]
            candidates_w = [s for s in left_half_segments_w if abs(((s[0] + s[-1]) * 0.5) - x_track) <= max_jump]
            pool_w = candidates_w if candidates_w else left_half_segments_w
            if not pool_w:
                return path
            seg_w = min(pool_w, key=lambda s: abs(((s[0] + s[-1]) * 0.5) - x_track))
            x_end = int(seg_w[0])
            if not path or path[-1] != (x_end, y_wrist):
                path.append((x_end, y_wrist))
    return path


def extract_measurements(
    front_kp: Keypoints,
    side_kp: Keypoints,
    front_mask: SilhouetteMask,
    side_mask: SilhouetteMask,
    cal: CalibrationInfo,
) -> BodyMeasurements:
    """Extract all body measurements from pose + segmentation data."""
    m = BodyMeasurements()
    f_pts, f_conf = front_kp.points, front_kp.confidence
    s_pts, s_conf = side_kp.points, side_kp.confidence
    pf = cal.px_per_cm_front
    ps = cal.px_per_cm_side

    # ── Height ──
    if is_valid(f_conf, KP.HEAD_TOP, KP.LEFT_ANKLE, KP.RIGHT_ANKLE):
        ankle_mid = midpoint(f_pts, KP.LEFT_ANKLE, KP.RIGHT_ANKLE)
        h_px = abs(f_pts[KP.HEAD_TOP, 1] - ankle_mid[1])
        m.height_cm = h_px / pf

    # ── Shoulder width ──
    if is_valid(f_conf, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER):
        m.shoulder_width_cm = distance(f_pts, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER) / pf

    # ── Shoulder slope width (separate): average neck→shoulder slanted lengths ──
    slope_vals = []
    if is_valid(f_conf, KP.UPPER_NECK, KP.LEFT_SHOULDER):
        slope_vals.append(distance(f_pts, KP.UPPER_NECK, KP.LEFT_SHOULDER) / pf)
    if is_valid(f_conf, KP.UPPER_NECK, KP.RIGHT_SHOULDER):
        slope_vals.append(distance(f_pts, KP.UPPER_NECK, KP.RIGHT_SHOULDER) / pf)
    slope_base = _avg(*slope_vals)
    m.shoulder_slope_width_cm = (slope_base * 0.8) if slope_base is not None else None

    # ── Arm length (front only, left arm): contour length along segmentation ──
    left_arm_path = _left_arm_contour_path(front_mask.mask, f_pts, f_conf)
    if len(left_arm_path) >= 2:
        arm_px = 0.0
        for i in range(1, len(left_arm_path)):
            x0, y0 = left_arm_path[i - 1]
            x1, y1 = left_arm_path[i]
            arm_px += float(np.hypot(x1 - x0, y1 - y0))
        m.arm_length_cm = arm_px / pf

    # ── Arm length from neck ──
    if m.arm_length_cm is not None:
        if is_valid(f_conf, KP.UPPER_NECK, KP.LEFT_SHOULDER):
            neck_to_shoulder = distance(f_pts, KP.UPPER_NECK, KP.LEFT_SHOULDER) / pf
            m.arm_length_from_neck_cm = neck_to_shoulder + m.arm_length_cm

    # ── Leg length outer (side): straight line from 25% above pelvis to bottom segmentation end ──
    if is_valid(s_conf, KP.PELVIS, KP.THORAX):
        sm = side_mask.mask
        h_s, w_s = sm.shape
        pelvis_y = float(s_pts[KP.PELVIS, 1])
        thorax_y = float(s_pts[KP.THORAX, 1])
        y_start_f = pelvis_y + 0.20 * (thorax_y - pelvis_y)
        y_start = int(np.clip(int(round(y_start_f)), 0, h_s - 1))
        torso_x = float(s_pts[KP.PELVIS, 0])

        cols_start = np.where(sm[y_start])[0]
        ys_fg = np.where(sm.any(axis=1))[0]
        if len(cols_start) >= 2 and len(ys_fg) > 0:
            y_end = int(ys_fg[-1])
            cols_end = np.where(sm[y_end])[0]
            if len(cols_end) >= 2:
                x_start = int(cols_start[0] if abs(cols_start[0] - torso_x) > abs(cols_start[-1] - torso_x) else cols_start[-1])
                x_end = int(cols_end[0] if abs(cols_end[0] - torso_x) > abs(cols_end[-1] - torso_x) else cols_end[-1])
                m.leg_length_outer_cm = float(np.hypot(x_end - x_start, y_end - y_start)) / ps

    # ── Leg length inner (front): single vertical line with static x ──
    if is_valid(f_conf, KP.PELVIS):
        fm = front_mask.mask
        h, w = fm.shape
        pelvis_x = int(np.clip(int(round(float(f_pts[KP.PELVIS, 0]))), 0, w - 1))
        pelvis_y = int(np.clip(int(round(float(f_pts[KP.PELVIS, 1]))), 0, h - 1))
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
            y_ankle = int(round(np.mean(ankle_ys)))
        elif knee_ys:
            y_ankle = int(round(np.mean(knee_ys)))
        else:
            y_ankle = h - 1
        y_ankle = int(np.clip(y_ankle, pelvis_y + 1, h - 1))
        # Keep previous start anchor.
        y_start = int(round(y_ankle + 0.8 * (pelvis_y - y_ankle)))
        y_start = int(np.clip(y_start, pelvis_y, y_ankle - 1))

        cols_top = np.where(fm[y_start])[0]
        if len(cols_top) >= 2:
            # Keep one inner-leg line (left side), x stays static.
            top_cands = cols_top[cols_top < pelvis_x]
            if len(top_cands) > 0:
                x_line = int(top_cands.max())
                ys_fg = np.where(fm.any(axis=1))[0]
                ys_fg = ys_fg[ys_fg >= y_start]
                if len(ys_fg) > 0:
                    y_bottom = int(ys_fg[-1])
                    m.leg_length_inner_cm = float(abs(y_bottom - y_start)) / pf

    # ── Silhouette widths ──
    widths, selected_y = extract_all_widths(front_mask, side_mask, front_kp, side_kp)

    def _to_cm(part: str) -> tuple[float | None, float | None]:
        if part not in widths:
            return None, None
        fw, sw = widths[part]
        return (
            fw / pf if fw is not None else None,
            sw / ps if sw is not None else None,
        )

    m.head_width_front_cm, m.head_depth_side_cm = _to_cm("head")
    m.neck_width_front_cm, m.neck_width_side_cm = _to_cm("neck")
    m.torso_width_front_cm, m.torso_width_side_cm = _to_cm("torso")
    m.waist_width_front_cm, m.waist_width_side_cm = _to_cm("waist")
    m.hip_width_front_cm, m.hip_width_side_cm = _to_cm("hip")

    # Persist actual selected waist y-level from width-search.
    wy = selected_y.get("waist")
    if wy is not None:
        m.waist_level_front_px, m.waist_level_side_px = wy
    # Fallback if width-search had no valid row.
    if m.waist_level_front_px is None and is_valid(f_conf, KP.PELVIS, KP.UPPER_NECK):
        py = float(f_pts[KP.PELVIS, 1])
        ny = float(f_pts[KP.UPPER_NECK, 1])
        m.waist_level_front_px = py + 0.4 * (ny - py)
    if m.waist_level_side_px is None and is_valid(s_conf, KP.PELVIS, KP.UPPER_NECK):
        py = float(s_pts[KP.PELVIS, 1])
        ny = float(s_pts[KP.UPPER_NECK, 1])
        m.waist_level_side_px = py + 0.4 * (ny - py)

    # Thigh: average left and right
    t_r_f, t_r_s = _to_cm("thigh_right")
    t_l_f, t_l_s = _to_cm("thigh_left")
    m.thigh_width_front_cm = _avg(t_r_f, t_l_f)
    m.thigh_width_side_cm = _avg(t_r_s, t_l_s)

    # Calf: average left and right
    c_r_f, c_r_s = _to_cm("calf_right")
    c_l_f, c_l_s = _to_cm("calf_left")
    m.calf_width_front_cm = _avg(c_r_f, c_l_f)
    m.calf_width_side_cm = _avg(c_r_s, c_l_s)

    # Wrist: average left and right
    w_r_f, w_r_s = _to_cm("wrist_right")
    w_l_f, w_l_s = _to_cm("wrist_left")
    m.wrist_width_front_cm = _avg(w_r_f, w_l_f)
    m.wrist_width_side_cm = _avg(w_r_s, w_l_s)

    return m
