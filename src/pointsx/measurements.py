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

    # ── Arm length (use side view to reduce foreshortening) ──
    for kp_s, kp_e, kp_w, conf in [
        (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST, s_conf),
        (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST, s_conf),
        (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST, f_conf),
        (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST, f_conf),
    ]:
        pts = s_pts if conf is s_conf else f_pts
        px_cm = ps if conf is s_conf else pf
        if is_valid(conf, kp_s, kp_e, kp_w):
            arm_px = distance(pts, kp_s, kp_e) + distance(pts, kp_e, kp_w)
            m.arm_length_cm = arm_px / px_cm
            break

    # ── Arm length from neck ──
    if m.arm_length_cm is not None:
        for kp_neck, kp_s, conf in [
            (KP.UPPER_NECK, KP.RIGHT_SHOULDER, s_conf),
            (KP.UPPER_NECK, KP.LEFT_SHOULDER, s_conf),
            (KP.UPPER_NECK, KP.RIGHT_SHOULDER, f_conf),
            (KP.UPPER_NECK, KP.LEFT_SHOULDER, f_conf),
        ]:
            pts = s_pts if conf is s_conf else f_pts
            px_cm = ps if conf is s_conf else pf
            if is_valid(conf, kp_neck, kp_s):
                neck_to_shoulder = distance(pts, kp_neck, kp_s) / px_cm
                m.arm_length_from_neck_cm = neck_to_shoulder + m.arm_length_cm
                break

    # ── Leg length outer (side pose: hip → knee → ankle) ──
    leg_outer_vals = []
    for kp_h, kp_k, kp_a in [
        (KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        (KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE),
    ]:
        if is_valid(s_conf, kp_h, kp_k, kp_a):
            leg_px = distance(s_pts, kp_h, kp_k) + distance(s_pts, kp_k, kp_a)
            leg_outer_vals.append(leg_px / ps)
    m.leg_length_outer_cm = _avg(*leg_outer_vals)

    # ── Leg length inner (front): ankle level -> 80% toward pelvis along inner mask ──
    if is_valid(f_conf, KP.PELVIS):
        fm = front_mask.mask
        h, w = fm.shape
        pelvis_x = int(round(float(f_pts[KP.PELVIS, 0])))
        pelvis_y = int(round(float(f_pts[KP.PELVIS, 1])))
        pelvis_x = int(np.clip(pelvis_x, 0, w - 1))
        pelvis_y = int(np.clip(pelvis_y, 0, h - 1))

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
        # End at 80% from ankle toward pelvis.
        y_end = int(round(y_ankle + 0.8 * (pelvis_y - y_ankle)))
        y_end = int(np.clip(y_end, pelvis_y, y_ankle - 1))

        def inner_path_len(side: str) -> float | None:
            pts_path = []
            for y in range(y_end, y_ankle + 1):
                cols = np.where(fm[y])[0]
                if len(cols) < 2:
                    continue
                if side == "left":
                    cands = cols[cols < pelvis_x]
                    if len(cands) == 0:
                        continue
                    x = int(cands.max())  # inner edge of left leg (closest to gap)
                else:
                    cands = cols[cols > pelvis_x]
                    if len(cands) == 0:
                        continue
                    x = int(cands.min())  # inner edge of right leg (closest to gap)
                pts_path.append((x, y))
            if len(pts_path) < 2:
                return None
            d = 0.0
            for i in range(1, len(pts_path)):
                x0, y0 = pts_path[i - 1]
                x1, y1 = pts_path[i]
                d += float(np.hypot(x1 - x0, y1 - y0))
            return d

        left_len = inner_path_len("left")
        right_len = inner_path_len("right")
        side_len = _avg(left_len, right_len)
        if side_len is not None:
            m.leg_length_inner_cm = side_len / pf

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
