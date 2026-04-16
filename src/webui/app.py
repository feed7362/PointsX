"""FastAPI app: static capture UI + mock body-measurement endpoint."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DISALLOWED_CONTENT_PREFIXES = ("text/", "video/", "audio/")


def _looks_like_raster_image(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:2] == b"\xff\xd8":
        return True
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return True
    return False

# Canonical IDs for the 18 outputs (stable API); map later to pipeline / schema fields.
MEASUREMENT_SPECS: list[tuple[str, str, float]] = [
    ("chest_circumference", "Обхват грудей", 0.52),
    ("waist_circumference", "Обхват талії", 0.38),
    ("hip_circumference", "Обхват стегон", 0.54),
    ("neck_circumference", "Обхват шиї", 0.22),
    ("neck_base_height", "Висота точки основи шиї", 0.08),
    ("shoulder_slope_width", "Ширина плечового ската", 0.13),
    ("back_width_scapular", "Ширина спини (між лопатками)", 0.18),
    ("chest_width_front", "Ширина грудей (між пахвами спереду)", 0.24),
    ("back_length_to_waist", "Довжина спини до талії (по хребту)", 0.26),
    ("front_length_to_waist", "Довжина переду до талії (через найвищу точку грудей)", 0.28),
    ("arm_length_shoulder_to_wrist", "Довжина руки (від плеча до зап'ястя)", 0.36),
    ("upper_arm_circumference", "Обхват плеча (біцепс)", 0.15),
    ("wrist_circumference", "Обхват зап'ястя", 0.065),
    ("leg_length_inner_seam", "Довжина ноги по внутрішньому шву", 0.45),
    ("leg_length_outer_seam", "Довжина ноги по зовнішньому шву", 0.52),
    ("thigh_circumference", "Обхват стегна", 0.28),
    ("calf_circumference", "Обхват гомілки (литки)", 0.19),
    ("ankle_circumference", "Обхват щиколотки", 0.09),
]


class MeasurementItem(BaseModel):
    id: str
    label_uk: str
    value_cm: float = Field(..., description="Mock value in centimeters, one decimal")
    confidence: float = Field(..., ge=0.0, le=1.0)


class MockMeasureResponse(BaseModel):
    height_cm: float
    sex: Literal["male", "female", "other"]
    measurements: list[MeasurementItem]


def _sex_scale(sex: Literal["male", "female", "other"]) -> float:
    if sex == "male":
        return 1.0
    if sex == "female":
        return 0.97
    return 0.985


def build_mock_measurements(height_cm: float, sex: Literal["male", "female", "other"]) -> list[MeasurementItem]:
    """Deterministic mock lengths/circumferences from height and sex (not real anthropometry)."""
    scale = _sex_scale(sex)
    h = height_cm
    out: list[MeasurementItem] = []
    for i, (mid, label_uk, ratio) in enumerate(MEASUREMENT_SPECS):
        base = h * ratio * scale
        tweak = 1.0 + 0.01 * math.sin(i + h * 0.1)
        value = round(base * tweak, 1)
        conf = round(0.75 + 0.04 * math.cos(i * 0.7), 2)
        out.append(MeasurementItem(id=mid, label_uk=label_uk, value_cm=value, confidence=min(0.95, max(0.5, conf))))
    return out


app = FastAPI(title="PointsX WebUI (mock)", version="0.1.0")


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="Missing static index")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _validate_image_upload(upload: UploadFile, label: str) -> None:
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"{label}: empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail=f"{label}: file too large (max {MAX_UPLOAD_BYTES} bytes)")
    ct = upload.content_type or ""
    if any(ct.startswith(p) for p in DISALLOWED_CONTENT_PREFIXES):
        raise HTTPException(status_code=400, detail=f"{label}: invalid Content-Type {ct!r}")
    if not _looks_like_raster_image(data):
        raise HTTPException(status_code=400, detail=f"{label}: body is not a JPEG, PNG, or WebP image")


@app.post("/api/measure/mock", response_model=MockMeasureResponse)
async def measure_mock(
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
) -> MockMeasureResponse:
    """Return mock measurements; images are validated but not analyzed (swap-in for MeasurementPipeline)."""
    await _validate_image_upload(front, "front")
    await _validate_image_upload(side, "side")
    measurements = build_mock_measurements(height_cm, sex)
    return MockMeasureResponse(height_cm=height_cm, sex=sex, measurements=measurements)
