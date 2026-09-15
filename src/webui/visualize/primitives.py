"""Drawing primitives: fonts with Cyrillic support, text, pose skeleton, segmentation tint, PNG, values table."""
from __future__ import annotations

import base64
from dataclasses import fields as dataclass_fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pointsx.keypoints import SKELETON
from pointsx.schemas import BodyMeasurements, Keypoints, SilhouetteMask

_MIN_LINE_CONF = 0.22
_MIN_POINT_CONF = 0.18
_SEG_COLOR = (64, 180, 255)  # BGR
_SEG_ALPHA = 0.38
_FONT_CANDIDATES = (
    # MacOS Supplemental & Standard Fonts
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Helvetica.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
    # Linux Standard DejaVu & Liberation Fonts
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    # Windows Standard Arial Font
    "C:\\Windows\\Fonts\\arial.ttf",
)
_FONT_CACHE: dict[int, ImageFont.ImageFont] = {}


def _get_font(size: int) -> ImageFont.ImageFont:
    cached = _FONT_CACHE.get(size)
    if cached is not None:
        return cached

    # Try bundled font first to ensure consistent Cyrillic support across all deployment platforms
    bundled_path = Path(__file__).resolve().parents[1] / "fonts" / "DejaVuSans.ttf"  # src/webui/fonts
    paths = [str(bundled_path)] + list(_FONT_CANDIDATES)

    for path in paths:
        try:
            font = ImageFont.truetype(path, size=size)
            _FONT_CACHE[size] = font
            return font
        except Exception:
            continue
    # PIL's default is a BITMAP font with no Cyrillic coverage, so every
    # Ukrainian label degrades to tofu squares. This used to fail silently —
    # warn loudly and name the fix.
    import logging

    logging.getLogger(__name__).warning(
        "No TrueType font found (tried %d paths) — falling back to PIL's bitmap "
        "default, which has NO Cyrillic glyphs, so overlay labels will render as "
        "squares. Install fonts-dejavu-core in the image.",
        len(paths),
    )
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


def _put_label(img: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    _draw_text_unicode(img, text, x, y - 12, color, font_size=18, outline_bgr=(255, 255, 255))
