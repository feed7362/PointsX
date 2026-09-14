"""Measurement overlay: where each measurement is taken, one function per section.

Sections run in a fixed order over one _Canvas; label placement depends on that
order (labels are nudged apart vertically), so keep _SECTIONS stable.
"""
from __future__ import annotations

from functools import partial

import cv2
import numpy as np

from pointsx.keypoints import KP, is_valid
from pointsx.schemas import Keypoints, SilhouetteMask
from pointsx.silhouette import front_thigh_y_level, hip_search_y_range
from webui.visualize.mask_spans import _extreme_span_between_y, _mask_span_x
from webui.visualize.primitives import _draw_seg_overlay, _put_label


class _Canvas:
    """Overlay image plus the label rows already taken."""

    def __init__(self, out: np.ndarray) -> None:
        self.out = out
        self._used_label_y: list[int] = []

    def label(self, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        h = self.out.shape[0]
        y_clamped = int(np.clip(y, 14, h - 6))
        for dy in (0, -14, 14, -28, 28, -42, 42):
            yc = int(np.clip(y_clamped + dy, 14, h - 6))
            if all(abs(yc - yy) >= 12 for yy in self._used_label_y):
                self._used_label_y.append(yc)
                _put_label(self.out, text, x, yc, color)
                return
        self._used_label_y.append(y_clamped)
        _put_label(self.out, text, x, y_clamped, color)


def _pt(kp: Keypoints, i: KP) -> tuple[int, int]:
    pts = kp.points
    return int(round(float(pts[int(i), 0]))), int(round(float(pts[int(i), 1])))


def _left_arm_contour_path(kp: Keypoints, mask: SilhouetteMask, view: str) -> list[tuple[int, int]]:
    """Left-arm outer contour from the shoulder corner down to the wrist row (front view)."""
    pts = kp.points
    conf = kp.confidence
    if not (view == "front" and is_valid(conf, KP.LEFT_SHOULDER, KP.LEFT_WRIST)):
        return []
    fm = mask.mask
    h, w = fm.shape
    y0 = int(round(float(pts[int(KP.LEFT_SHOULDER), 1])))
    y1 = int(round(float(pts[int(KP.LEFT_WRIST), 1])))
    y0 = int(np.clip(y0, 0, h - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    x_shoulder = int(np.clip(int(round(float(pts[int(KP.LEFT_SHOULDER), 0]))), 0, w - 1))
    y_wrist = y1

    x_mid_img = 0.5 * (w - 1)
    # Start: at shoulder y, then move up along left edge until segment corner.
    max_jump = 18.0
    cols0 = np.where(fm[y0])[0]
    if len(cols0) < 2:
        return []
    diffs0 = np.diff(cols0)
    split_points0 = np.where(diffs0 > 3)[0] + 1
    segments0 = [s for s in np.split(cols0, split_points0) if len(s) >= 2]
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
        cols = np.where(fm[y])[0]
        if len(cols) < 2:
            break
        diffs = np.diff(cols)
        split_points = np.where(diffs > 3)[0] + 1
        segments = [s for s in np.split(cols, split_points) if len(s) >= 2]
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
        cols = np.where(fm[y])[0]
        if len(cols) < 2:
            continue
        diffs = np.diff(cols)
        split_points = np.where(diffs > 3)[0] + 1
        segments = [s for s in np.split(cols, split_points) if len(s) >= 2]
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
    cols_w = np.where(fm[y_wrist])[0]
    if len(cols_w) >= 2:
        diffs_w = np.diff(cols_w)
        split_points_w = np.where(diffs_w > 3)[0] + 1
        segments_w = [s for s in np.split(cols_w, split_points_w) if len(s) >= 2]
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


def _draw_torso_width(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Torso line: shoulder to shoulder (front) or silhouette span at mid-torso (side)."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    p = partial(_pt, kp)
    put_label_smart = c.label
    # Linear distances from keypoints.
    if view == "front" and is_valid(conf, KP.LEFT_SHOULDER, KP.RIGHT_SHOULDER):
        a, b = p(KP.LEFT_SHOULDER), p(KP.RIGHT_SHOULDER)
        cv2.line(out, a, b, (255, 0, 255), 2, cv2.LINE_AA)
        put_label_smart("тулуб", min(a[0], b[0]), min(a[1], b[1]) - 8, (255, 0, 255))
    elif view == "side" and is_valid(conf, KP.UPPER_NECK) and (is_valid(conf, KP.RIGHT_ELBOW) or is_valid(conf, KP.LEFT_ELBOW)):
        elbow_y = (
            float(pts[int(KP.RIGHT_ELBOW), 1])
            if is_valid(conf, KP.RIGHT_ELBOW)
            else float(pts[int(KP.LEFT_ELBOW), 1])
        )
        y_torso = 0.5 * (float(pts[int(KP.UPPER_NECK), 1]) + elbow_y)
        span = _mask_span_x(mask.mask, y_torso)
        if span is not None:
            x0, x1 = span
            yi = int(round(y_torso))
            cv2.line(out, (x0, yi), (x1, yi), (255, 0, 255), 2, cv2.LINE_AA)
            put_label_smart("тулуб", x1 + 6, yi - 2, (255, 0, 255))


def _draw_shoulder_slope(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Shoulder slope: upper neck to left shoulder (front)."""
    out = c.out
    conf = kp.confidence
    p = partial(_pt, kp)
    put_label_smart = c.label
    # shoulder_slope: upper_neck -> shoulder(s)
    slope_color = (200, 80, 255)
    if view == "front" and is_valid(conf, KP.UPPER_NECK, KP.LEFT_SHOULDER):
        n, ls = p(KP.UPPER_NECK), p(KP.LEFT_SHOULDER)
        cv2.line(out, n, ls, slope_color, 2, cv2.LINE_AA)
    if view == "front" and is_valid(conf, KP.UPPER_NECK, KP.LEFT_SHOULDER):
        ls = p(KP.LEFT_SHOULDER)
        put_label_smart("плечовий скат", ls[0] + 8, ls[1] - 10, slope_color)


def _draw_arm_length(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Arm length along the left arm contour, or shoulder→wrist as fallback (front)."""
    out = c.out
    conf = kp.confidence
    p = partial(_pt, kp)
    put_label_smart = c.label
    if view == "front":
        arm_color = (80, 220, 80)
        arm_path = _left_arm_contour_path(kp, mask, view)
        if len(arm_path) >= 2:
            for i in range(1, len(arm_path)):
                cv2.line(out, arm_path[i - 1], arm_path[i], arm_color, 2, cv2.LINE_AA)
            lx, ly = arm_path[-1]
            put_label_smart("довжина руки", lx + 6, ly, arm_color)
        elif is_valid(conf, KP.LEFT_SHOULDER, KP.LEFT_WRIST):
            # Fallback matches measurement fallback in extraction.
            a, b = p(KP.LEFT_SHOULDER), p(KP.LEFT_WRIST)
            cv2.line(out, a, b, arm_color, 2, cv2.LINE_AA)
            put_label_smart("довжина руки", b[0] + 6, b[1], arm_color)


def _draw_outer_leg(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Outer leg seam: vertical line from above the pelvis to the silhouette bottom (side)."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    put_label_smart = c.label
    # leg_outer: side straight line from 25% above pelvis to bottom segmentation end.
    if view == "side" and is_valid(conf, KP.PELVIS, KP.THORAX):
        sm = mask.mask
        h_s, w_s = sm.shape
        pelvis_y = float(pts[int(KP.PELVIS), 1])
        thorax_y = float(pts[int(KP.THORAX), 1])
        y_start = int(np.clip(int(round(pelvis_y + 0.25 * (thorax_y - pelvis_y))), 0, h_s - 1))
        torso_x = float(pts[int(KP.PELVIS), 0])
        cols_start = np.where(sm[y_start])[0]
        ys_fg = np.where(sm.any(axis=1))[0]
        if len(cols_start) >= 2 and len(ys_fg) > 0:
            y_end = int(ys_fg[-1])
            x_line = cols_start[0] if abs(cols_start[0] - torso_x) > abs(cols_start[-1] - torso_x) else cols_start[-1]
            x_line = int(np.clip(int(x_line), 0, w_s - 1))
            cv2.line(out, (x_line, y_start), (x_line, y_end), (255, 80, 80), 2, cv2.LINE_AA)
            put_label_smart("нога зовнішня", x_line + 6, y_end, (255, 80, 80))


def _draw_inner_leg(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Inner leg seam: vertical line below the crotch (front)."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    put_label_smart = c.label
    # leg_inner: one front vertical line with static x.
    if view == "front" and is_valid(conf, KP.PELVIS):
        fm = mask.mask
        h, w = fm.shape
        pelvis_x = int(round(float(pts[int(KP.PELVIS), 0])))
        pelvis_y = int(round(float(pts[int(KP.PELVIS), 1])))
        pelvis_x = int(np.clip(pelvis_x, 0, w - 1))
        pelvis_y = int(np.clip(pelvis_y, 0, h - 1))
        ankle_ys = []
        if is_valid(conf, KP.LEFT_ANKLE):
            ankle_ys.append(float(pts[int(KP.LEFT_ANKLE), 1]))
        if is_valid(conf, KP.RIGHT_ANKLE):
            ankle_ys.append(float(pts[int(KP.RIGHT_ANKLE), 1]))
        knee_ys = []
        if is_valid(conf, KP.LEFT_KNEE):
            knee_ys.append(float(pts[int(KP.LEFT_KNEE), 1]))
        if is_valid(conf, KP.RIGHT_KNEE):
            knee_ys.append(float(pts[int(KP.RIGHT_KNEE), 1]))
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
            top_cands = cols_top[cols_top < pelvis_x]
            if len(top_cands) > 0:
                x_line = int(top_cands.max())
                ys_fg = np.where(fm.any(axis=1))[0]
                ys_fg = ys_fg[ys_fg >= y_start]
                if len(ys_fg) > 0:
                    y_bottom = int(ys_fg[-1])
                    cv2.line(out, (x_line, y_start), (x_line, y_bottom), (180, 120, 255), 2, cv2.LINE_AA)
                    put_label_smart("нога внутрішня", x_line + 6, max(14, y_start - 8), (180, 120, 255))


def _draw_thigh_width(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Thigh width at the extraction row (front: right leg segment; side: full span)."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    put_label_smart = c.label
    # Thigh width visualization (mirrors extraction rule).
    thigh_color = (80, 120, 255)
    if view == "front" and is_valid(conf, KP.PELVIS):
        fm = mask.mask
        h, w = fm.shape
        pelvis_x = int(round(float(pts[int(KP.PELVIS), 0])))
        pelvis_y = int(round(float(pts[int(KP.PELVIS), 1])))
        pelvis_x = int(np.clip(pelvis_x, 0, w - 1))
        pelvis_y = int(np.clip(pelvis_y, 0, h - 1))
        x_split = (
            int(round(float(pts[int(KP.UPPER_NECK), 0])))
            if is_valid(conf, KP.UPPER_NECK)
            else pelvis_x
        )
        x_split = int(np.clip(x_split, 0, w - 1))
        y_thigh_raw = front_thigh_y_level(kp, fm)
        y_thigh = int(round(y_thigh_raw)) if y_thigh_raw is not None else pelvis_y
        if is_valid(conf, KP.RIGHT_KNEE):
            xr = int(round(float(pts[int(KP.RIGHT_KNEE), 0])))
            cols = np.where(fm[int(y_thigh)])[0]
            if len(cols) >= 2:
                diffs = np.diff(cols)
                split_points = np.where(diffs > 3)[0] + 1
                segments = np.split(cols, split_points)
                left_candidates = []
                for s in segments:
                    if len(s) < 2:
                        continue
                    sx0, sx1 = int(s[0]), int(s[-1])
                    if sx0 >= x_split:
                        continue
                    cx1 = min(sx1, x_split)
                    if cx1 - sx0 >= 1:
                        left_candidates.append((sx0, cx1))
                if left_candidates:
                    best = min(left_candidates, key=lambda p: abs(((p[0] + p[1]) * 0.5) - xr))
                    x0, x1 = int(best[0]), int(best[1])
                    cv2.line(out, (x0, y_thigh), (x1, y_thigh), thigh_color, 2, cv2.LINE_AA)
                    put_label_smart("стегно", x1 + 6, y_thigh - 2, thigh_color)
    elif view == "side" and is_valid(conf, KP.PELVIS):
        y_p = float(pts[int(KP.PELVIS), 1])
        y_ref = y_p
        if is_valid(conf, KP.RIGHT_KNEE):
            y_ref = float(pts[int(KP.RIGHT_KNEE), 1])
        elif is_valid(conf, KP.RIGHT_ANKLE):
            y_ref = float(pts[int(KP.RIGHT_ANKLE), 1])
        y_thigh = y_p + 0.5 * (y_ref - y_p)
        s = _mask_span_x(mask.mask, y_thigh)
        if s is not None:
            x0, x1 = s
            yi = int(round(y_thigh))
            cv2.line(out, (x0, yi), (x1, yi), thigh_color, 2, cv2.LINE_AA)
            put_label_smart("стегно", x1 + 6, yi - 2, thigh_color)


def _draw_neck_width(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Neck: narrowest silhouette span between the head and the upper neck."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    put_label_smart = c.label
    # Continuous silhouette lines for neck/waist/hip.
    width_color = (255, 80, 80) if view == "front" else (80, 80, 255)
    if is_valid(conf, KP.UPPER_NECK):
        y_upper_neck = float(pts[int(KP.UPPER_NECK), 1])
        y_start = None
        if kp.nose_xy is not None and (kp.nose_conf or 0.0) >= 0.2 and is_valid(conf, KP.HEAD_TOP):
            y_nose = float(kp.nose_xy[1])
            y_head_top = float(pts[int(KP.HEAD_TOP), 1])
            y_start = y_nose + abs(y_nose - y_head_top)
        elif kp.nose_xy is not None and (kp.nose_conf or 0.0) >= 0.2:
            y_start = float(kp.nose_xy[1])
        elif is_valid(conf, KP.HEAD_TOP):
            y_start = float(pts[int(KP.HEAD_TOP), 1])
        neck = _extreme_span_between_y(mask.mask, y_start, y_upper_neck, "min") if y_start is not None else None
        if neck is not None:
            x0, x1, yi = neck
            cv2.line(out, (x0, yi), (x1, yi), width_color, 2, cv2.LINE_AA)
            put_label_smart("шия", x1 + 6, yi - 2, width_color)


def _draw_waist_width(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Waist: narrowest span between the pelvis and 40 % of the way to the neck."""
    out = c.out
    pts = kp.points
    conf = kp.confidence
    put_label_smart = c.label
    width_color = (255, 80, 80) if view == "front" else (80, 80, 255)
    if is_valid(conf, KP.PELVIS, KP.UPPER_NECK):
        y_p = float(pts[int(KP.PELVIS), 1])
        y_mid = y_p + 0.4 * (float(pts[int(KP.UPPER_NECK), 1]) - y_p)
        waist = _extreme_span_between_y(mask.mask, y_p, y_mid, "min")
        if waist is not None:
            x0, x1, yi = waist
            cv2.line(out, (x0, yi), (x1, yi), width_color, 2, cv2.LINE_AA)
            put_label_smart("талія", x1 + 6, yi - 2, width_color)


def _draw_hip_width(c: _Canvas, kp: Keypoints, mask: SilhouetteMask, view: str) -> None:
    """Hips: widest span inside the hip search range."""
    out = c.out
    put_label_smart = c.label
    width_color = (255, 80, 80) if view == "front" else (80, 80, 255)
    hip_range = hip_search_y_range(kp)
    if hip_range is not None:
        y_start, y_end = hip_range
        hip = _extreme_span_between_y(mask.mask, y_start, y_end, "max")
        if hip is not None:
            x0, x1, yi = hip
            cv2.line(out, (x0, yi), (x1, yi), width_color, 2, cv2.LINE_AA)
            put_label_smart("стегна", x1 + 6, yi - 2, width_color)


_SECTIONS = (
    _draw_torso_width,
    _draw_shoulder_slope,
    _draw_arm_length,
    _draw_outer_leg,
    _draw_inner_leg,
    _draw_thigh_width,
    _draw_neck_width,
    _draw_waist_width,
    _draw_hip_width,
)


def _draw_measure_lines(bgr: np.ndarray, kp: Keypoints, mask: SilhouetteMask, view: str) -> np.ndarray:
    """Overlay key geometric lines to show how measurements are taken."""
    c = _Canvas(_draw_seg_overlay(bgr, mask))
    for draw in _SECTIONS:
        draw(c, kp, mask, view)
    return c.out
