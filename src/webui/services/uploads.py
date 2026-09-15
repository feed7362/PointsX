"""Validation and decoding of uploaded photos."""
from __future__ import annotations

from typing import Any

from webui.config import DISALLOWED_CONTENT_PREFIXES, MAX_UPLOAD_BYTES
from webui.errors import AppError

UPLOAD_LABEL_UK = {"front": "Анфас", "side": "Профіль"}


def looks_like_raster_image(data: bytes) -> bool:
    """JPEG, PNG or WebP by magic bytes."""
    if len(data) < 12:
        return False
    if data[:2] == b"\xff\xd8":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False


def detect_image_extension(data: bytes) -> str:
    """Map raw image magic bytes to a filesystem extension."""
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "jpg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "bin"


def decode_upload(data: bytes, content_type: str | None, label: str) -> Any:
    """Validate upload bytes and decode to a BGR ndarray (cv2 convention).

    Checks run in a fixed order (empty → size → content type → magic bytes →
    decode); the first failure raises ``AppError(400)`` with a Ukrainian message.
    OpenCV is imported lazily so the Vercel function never loads it.
    """
    uk = UPLOAD_LABEL_UK.get(label, label)
    if len(data) == 0:
        raise AppError(400, f"{uk}: файл порожній.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise AppError(400, f"{uk}: файл завеликий (ліміт {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ).")
    ct = content_type or ""
    if any(ct.startswith(p) for p in DISALLOWED_CONTENT_PREFIXES):
        raise AppError(400, f"{uk}: недопустимий тип вмісту ({ct!r}). Очікується зображення.")
    if not looks_like_raster_image(data):
        raise AppError(400, f"{uk}: очікується JPEG, PNG або WebP.")

    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise AppError(400, f"{uk}: не вдалося розпізнати зображення.")
    return img
