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

    # ── Leg length outer (hip → knee → ankle) ──
    leg_outer_vals = []
    for kp_h, kp_k, kp_a in [
        (KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        (KP.LEFT_HIP, KP.LEFT_KNEE, KP.LEFT_ANKLE),
    ]:
        if is_valid(f_conf, kp_h, kp_k, kp_a):
            leg_px = distance(f_pts, kp_h, kp_k) + distance(f_pts, kp_k, kp_a)
            leg_outer_vals.append(leg_px / pf)
    m.leg_length_outer_cm = _avg(*leg_outer_vals)

    # ── Leg length inner (pelvis → knee → ankle) ──
    leg_inner_vals = []
    for kp_k, kp_a in [
        (KP.RIGHT_KNEE, KP.RIGHT_ANKLE),
        (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    ]:
        if is_valid(f_conf, KP.PELVIS, kp_k, kp_a):
            leg_px = distance(f_pts, KP.PELVIS, kp_k) + distance(f_pts, kp_k, kp_a)
            leg_inner_vals.append(leg_px / pf)
    m.leg_length_inner_cm = _avg(*leg_inner_vals)

    # ── Silhouette widths ──
    widths = extract_all_widths(front_mask, side_mask, front_kp, side_kp)

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
