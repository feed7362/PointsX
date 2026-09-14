"""webui.services — exercised directly, without HTTP (plan rule: services are testable on their own)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from webui.errors import AppError  # noqa: E402
from webui.services import dataset_capture, proxy, uploads  # noqa: E402
from webui.services.mock import build_mock_measurement_envelope  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\n"


def _png(width=10, height=12) -> bytes:
    ok, buf = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return buf.tobytes()


def test_extension_and_raster_sniffing():
    assert uploads.detect_image_extension(b"\xff\xd8rest") == "jpg"
    assert uploads.detect_image_extension(PNG + b"x" * 4) == "png"
    assert uploads.detect_image_extension(b"RIFF1234WEBPxx") == "webp"
    assert uploads.detect_image_extension(b"GIF89a") == "bin"
    assert uploads.looks_like_raster_image(PNG + b"1234") is True
    assert uploads.looks_like_raster_image(PNG) is False  # shorter than 12 bytes


def test_decode_upload_ok_and_errors():
    img = uploads.decode_upload(_png(), "image/png", "side")
    assert img.shape == (12, 10, 3)
    with pytest.raises(AppError) as err:
        uploads.decode_upload(PNG + b"\x00" * 64, "image/png", "side")
    assert err.value.status_code == 400 and err.value.detail == "Профіль: не вдалося розпізнати зображення."
    with pytest.raises(AppError) as err:
        uploads.decode_upload(b"", None, "unknown-label")
    assert err.value.detail == "unknown-label: файл порожній."


def test_unique_stem_avoids_collisions(tmp_path):
    first = dataset_capture.unique_stem(tmp_path)
    (tmp_path / f"a{first}.png").write_bytes(b"x")
    second = dataset_capture.unique_stem(tmp_path)
    assert second != first and dataset_capture.pair_stem_exists(tmp_path, first)


def test_save_capture_pair_success_and_warning(tmp_path):
    stem, warning = asyncio.run(dataset_capture.save_capture_pair(tmp_path / "ds", _png(), b"\xff\xd8" + b"0" * 20))
    assert warning is None
    assert sorted(p.name for p in (tmp_path / "ds").iterdir()) == [f"a{stem}.png", f"p{stem}.jpg"]

    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    stem, warning = asyncio.run(dataset_capture.save_capture_pair(blocker, b"1", b"2"))
    assert stem is None and warning.startswith("Не вдалося зберегти знімки у папку датасету")


def test_mock_envelope_scales_with_height_and_sex():
    tall = build_mock_measurement_envelope(210.0, "male")
    short = build_mock_measurement_envelope(150.0, "female")
    assert len(tall.measurements) == 18
    assert tall.measurements[0].value_cm > short.measurements[0].value_cm


def test_proxy_errors_on_unreachable_upstream():
    part = ("f.jpg", b"\xff\xd8", "image/jpeg")
    with pytest.raises(AppError) as err:
        asyncio.run(proxy.forward_measure("http://127.0.0.1:9", height_cm=170, sex="female",
                                          pose_backend="coco", front=part, side=part))
    assert err.value.status_code == 502
    body, status = asyncio.run(proxy.ping_space_health("http://127.0.0.1:9/"))
    assert status == 502 and body["target"] == "http://127.0.0.1:9/api/health" and "error" in body
