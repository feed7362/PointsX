"""Encode pipeline debug images (pose + segmentation) as base64 PNG for the web UI."""

from __future__ import annotations

import base64
from dataclasses import fields as dataclass_fields

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pointsx.keypoints import KP, SKELETON, interpolate_y, is_valid
from pointsx.schemas import BodyMeasurements, Keypoints, SilhouetteMask

from webui.inference import InferenceResult

__all__ = ["pipeline_visualizations_b64"]

_MIN_LINE_CONF = 0.22
_MIN_POINT_CONF = 0.18
_SEG_COLOR = (64, 180, 255)  # BGR
_SEG_ALPHA = 0.38
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
)
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached
    for path in _FONT_CANDIDATES:
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[size] = font
            return font
        except Exception:
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _draw_text_unicode(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color_bgr: tuple[int, int, int],
    *,
    font_size: int,
    outline_bgr: tuple[int, int, int] | None = None,
) -> None:
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    font = _get_font(font_size)
    fill = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    if outline_bgr is not None:
        outline = (int(outline_bgr[2]), int(outline_bgr[1]), int(outline_bgr[0]))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)
    img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _draw_pose(bgr: np.ndarray, kp: Keypoints) -> np.ndarray:
    out = bgr.copy()
    pts = kp.points
    conf = kp.confidence
    for a, b in SKELETON:
        ia, ib = int(a), int(b)
        if conf[ia] >= _MIN_LINE_CONF and conf[ib] >= _MIN_LINE_CONF:
            p0 = (int(round(pts[ia][0])), int(round(pts[ia][1])))
            p1 = (int(round(pts[ib][0])), int(round(pts[ib][1])))
            cv2.line(out, p0, p1, (0, 220, 130), 2, cv2.LINE_AA)
    for i in range(len(pts)):
        if conf[i] >= _MIN_POINT_CONF:
            p = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.circle(out, p, 4, (0, 140, 255), -1, cv2.LINE_AA)
    return out


def _draw_seg_overlay(bgr: np.ndarray, mask: SilhouetteMask) -> np.ndarray:
    base = bgr.astype(np.float32)
    m = mask.mask.astype(np.float32)
    col = np.array(_SEG_COLOR, dtype=np.float32).reshape(1, 1, 3)
    overlay = base.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            m > 0.5,
            base[:, :, c] * (1.0 - _SEG_ALPHA) + col[0, 0, c] * _SEG_ALPHA,
            base[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def _draw_body_measurements_table(bm: BodyMeasurements) -> np.ndarray:
    """Render all BodyMeasurements values into a standalone debug PNG."""
    rows: list[str] = []
    for f in dataclass_fields(BodyMeasurements):
        name = f.name
        if name in ("confidence", "warnings"):
            continue
        value = getattr(bm, name)
        if value is None:
            val_txt = "n/a"
        else:
            val_txt = f"{float(value):.1f} cm"
        rows.append(f"{name}: {val_txt}")

    header = "BodyMeasurements (cm)"
    line_h = 26
    pad = 18
    width = 1040
    height = pad * 2 + line_h * (1 + len(rows))
    img = np.full((height, width, 3), 250, dtype=np.uint8)

    _draw_text_unicode(img, header, pad, pad, (20, 20, 20), font_size=24)
    y = pad + line_h + 6
    for line in rows:
        _draw_text_unicode(img, line, pad, y, (40, 40, 40), font_size=18)
        y += line_h
    return img


def _mask_span_x(mask: np.ndarray, y: float) -> tuple[int, int] | None:
    h, _w = mask.shape
    yi = int(round(y))
    if yi < 0 or yi >= h:
        return None
    cols = np.where(mask[yi])[0]
    if len(cols) < 2:
        return None
    diffs = np.diff(cols)
    split_points = np.where(diffs > 3)[0] + 1
    segments = np.split(cols, split_points)
    best = max(segments, key=lambda s: (s[-1] - s[0]))
    if len(best) < 2:
        return None
    return int(best[0]), int(best[-1])


def _extreme_span_between_y(
    mask: np.ndarray, y0: float, y1: float, mode: str
) -> tuple[int, int, int] | None:
    """Return (x0, x1, y) for min/max continuous span between two y values."""
    lo = int(round(min(y0, y1)))
    hi = int(round(max(y0, y1)))
    best: tuple[int, int, int] | None = None
    for yi in range(lo, hi + 1):
        span = _mask_span_x(mask, float(yi))
        if span is None:
            continue
        x0, x1 = span
        w = x1 - x0
        if best is None:
            best = (x0, x1, yi)
            continue
        bw = best[1] - best[0]
        if (mode == "min" and w < bw) or (mode == "max" and w > bw):
            best = (x0, x1, yi)
    return best


def _put_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    _draw_text_unicode(img, text, x, y - 12, color, font_size=18, outline_bgr=(255, 255, 255))


def _draw_measure_lines(bgr: np.ndarray, kp: Keypoints, mask: SilhouetteMask, view: str) -> np.ndarray:
    """Overlay key geometric lines to show how measurements are taken."""
    out = _draw_seg_overlay(bgr, mask)
    pts = kp.points
    conf = kp.confidence
    used_label_y: list[int] = []

    def p(i: KP) -> tuple[int, int]:
        return int(round(float(pts[int(i), 0]))), int(round(float(pts[int(i), 1])))

    def put_label_smart(text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        h = out.shape[0]
        y_clamped = int(np.clip(y, 14, h - 6))
        for dy in (0, -14, 14, -28, 28, -42, 42):
            yc = int(np.clip(y_clamped + dy, 14, h - 6))
            if all(abs(yc - yy) >= 12 for yy in used_label_y):
                used_label_y.append(yc)
                _put_label(out, text, x, yc, color)
                return
        used_label_y.append(y_clamped)
        _put_label(out, text, x, y_clamped, color)

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

    # shoulder_slope: upper_neck -> shoulder(s)
    slope_color = (200, 80, 255)
    if view == "front" and is_valid(conf, KP.UPPER_NECK, KP.LEFT_SHOULDER):
        n, ls = p(KP.UPPER_NECK), p(KP.LEFT_SHOULDER)
        cv2.line(out, n, ls, slope_color, 2, cv2.LINE_AA)
    if view == "front" and is_valid(conf, KP.UPPER_NECK, KP.RIGHT_SHOULDER):
        n, rs = p(KP.UPPER_NECK), p(KP.RIGHT_SHOULDER)
        cv2.line(out, n, rs, slope_color, 2, cv2.LINE_AA)
    if view == "front" and is_valid(conf, KP.UPPER_NECK) and (is_valid(conf, KP.LEFT_SHOULDER) or is_valid(conf, KP.RIGHT_SHOULDER)):
        n = p(KP.UPPER_NECK)
        put_label_smart("плечовий скат", n[0] + 8, n[1] - 10, slope_color)

    if is_valid(conf, KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW, KP.RIGHT_WRIST):
        a, b, c = p(KP.RIGHT_SHOULDER), p(KP.RIGHT_ELBOW), p(KP.RIGHT_WRIST)
        cv2.line(out, a, b, (80, 220, 80), 2, cv2.LINE_AA)
        cv2.line(out, b, c, (80, 220, 80), 2, cv2.LINE_AA)
        put_label_smart("довжина руки", c[0] + 6, c[1], (80, 220, 80))

    # leg_outer now comes from side pose only.
    if view == "side" and is_valid(conf, KP.RIGHT_HIP, KP.RIGHT_KNEE, KP.RIGHT_ANKLE):
        a, b, c = p(KP.RIGHT_HIP), p(KP.RIGHT_KNEE), p(KP.RIGHT_ANKLE)
        cv2.line(out, a, b, (255, 80, 80), 2, cv2.LINE_AA)
        cv2.line(out, b, c, (255, 80, 80), 2, cv2.LINE_AA)
        put_label_smart("нога зовнішня", c[0] + 6, c[1], (255, 80, 80))

    # leg_inner now comes from front mask path (ankle level -> 80% toward pelvis).
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

        if ankle_ys or knee_ys:
            y_ankle = int(round(np.mean(ankle_ys))) if ankle_ys else int(round(np.mean(knee_ys)))
            y_ankle = int(np.clip(y_ankle, pelvis_y + 1, h - 1))
            y_end = int(round(y_ankle + 0.8 * (pelvis_y - y_ankle)))
            y_end = int(np.clip(y_end, pelvis_y, y_ankle - 1))

            # Draw both inner contour paths; the computed metric averages both.
            for side in ("left", "right"):
                pts_path: list[tuple[int, int]] = []
                for y in range(y_end, y_ankle + 1):
                    cols = np.where(fm[y])[0]
                    if len(cols) < 2:
                        continue
                    if side == "left":
                        cands = cols[cols < pelvis_x]
                        if len(cands) == 0:
                            continue
                        x = int(cands.max())
                    else:
                        cands = cols[cols > pelvis_x]
                        if len(cands) == 0:
                            continue
                        x = int(cands.min())
                    pts_path.append((x, y))
                if len(pts_path) >= 2:
                    for i in range(1, len(pts_path)):
                        cv2.line(out, pts_path[i - 1], pts_path[i], (180, 120, 255), 2, cv2.LINE_AA)
            put_label_smart("нога внутрішня", pelvis_x + 8, max(14, y_end - 8), (180, 120, 255))

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
        y_bottom = h - 1
        if is_valid(conf, KP.LEFT_ANKLE) or is_valid(conf, KP.RIGHT_ANKLE):
            ankle_ys = []
            if is_valid(conf, KP.LEFT_ANKLE):
                ankle_ys.append(float(pts[int(KP.LEFT_ANKLE), 1]))
            if is_valid(conf, KP.RIGHT_ANKLE):
                ankle_ys.append(float(pts[int(KP.RIGHT_ANKLE), 1]))
            if ankle_ys:
                y_bottom = int(round(np.mean(ankle_ys)))
        y_bottom = int(np.clip(y_bottom, pelvis_y + 1, h - 1))
        y_thigh = pelvis_y
        for y in range(pelvis_y, y_bottom + 1):
            if not fm[y, pelvis_x]:
                y_thigh = y
                break
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

    if is_valid(conf, KP.PELVIS, KP.UPPER_NECK):
        y_p = float(pts[int(KP.PELVIS), 1])
        y_mid = y_p + 0.4 * (float(pts[int(KP.UPPER_NECK), 1]) - y_p)
        waist = _extreme_span_between_y(mask.mask, y_p, y_mid, "min")
        if waist is not None:
            x0, x1, yi = waist
            cv2.line(out, (x0, yi), (x1, yi), width_color, 2, cv2.LINE_AA)
            put_label_smart("талія", x1 + 6, yi - 2, width_color)

    if is_valid(conf, KP.PELVIS):
        y_p = float(pts[int(KP.PELVIS), 1])
        if view == "front":
            knees = []
            if is_valid(conf, KP.LEFT_KNEE):
                knees.append(float(pts[int(KP.LEFT_KNEE), 1]))
            if is_valid(conf, KP.RIGHT_KNEE):
                knees.append(float(pts[int(KP.RIGHT_KNEE), 1]))
            y_k = float(np.mean(knees)) if knees else None
        else:
            y_k = float(pts[int(KP.RIGHT_KNEE), 1]) if is_valid(conf, KP.RIGHT_KNEE) else None
        if y_k is not None:
            y_start = y_p + 0.05 * (y_k - y_p)
            hip = _extreme_span_between_y(mask.mask, y_start, y_k, "max")
            if hip is not None:
                x0, x1, yi = hip
                cv2.line(out, (x0, yi), (x1, yi), width_color, 2, cv2.LINE_AA)
                put_label_smart("стегна", x1 + 6, yi - 2, width_color)

    return out


def pipeline_visualizations_b64(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    result: InferenceResult,
) -> dict[str, str]:
    """Four PNGs (base64): pose-only and seg-only per view."""
    fm = result.front_mask
    sm = result.side_mask
    fk = result.front_kp
    sk = result.side_kp

    return {
        "viz_front_pose_png_b64": _png_b64(_draw_pose(front_bgr, fk)),
        "viz_front_seg_png_b64": _png_b64(_draw_seg_overlay(front_bgr, fm)),
        "viz_front_measures_png_b64": _png_b64(_draw_measure_lines(front_bgr, fk, fm, "front")),
        "viz_side_pose_png_b64": _png_b64(_draw_pose(side_bgr, sk)),
        "viz_side_seg_png_b64": _png_b64(_draw_seg_overlay(side_bgr, sm)),
        "viz_side_measures_png_b64": _png_b64(_draw_measure_lines(side_bgr, sk, sm, "side")),
        "viz_body_measures_png_b64": _png_b64(_draw_body_measurements_table(result.body)),
    }
