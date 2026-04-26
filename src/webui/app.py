"""FastAPI app: static capture UI + mock body-measurement endpoint."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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
    ("chest_circumference",          "Обхват грудей",                                        0.52),
    ("waist_circumference",          "Обхват талії",                                         0.38),
    ("hip_circumference",            "Обхват стегон",                                        0.54),
    ("neck_circumference",           "Обхват шиї",                                           0.22),
    ("neck_base_height",             "Висота точки основи шиї",                              0.08),
    ("shoulder_slope_width",         "Ширина плечового ската",                               0.13),
    ("back_width_scapular",          "Ширина спини (між лопатками)",                         0.18),
    ("chest_width_front",            "Ширина грудей (між пахвами спереду)",                  0.24),
    ("back_length_to_waist",         "Довжина спини до талії (по хребту)",                   0.26),
    ("front_length_to_waist",        "Довжина переду до талії (через найвищу точку грудей)", 0.28),
    ("arm_length_shoulder_to_wrist", "Довжина руки (від плеча до зап'ястя)",                 0.36),
    ("upper_arm_circumference",      "Обхват плеча (біцепс)",                                0.15),
    ("wrist_circumference",          "Обхват зап'ястя",                                      0.065),
    ("leg_length_inner_seam",        "Довжина ноги по внутрішньому шву",                     0.45),
    ("leg_length_outer_seam",        "Довжина ноги по зовнішньому шву",                      0.52),
    ("thigh_circumference",          "Обхват стегна",                                        0.28),
    ("calf_circumference",           "Обхват гомілки (литки)",                               0.19),
    ("ankle_circumference",          "Обхват щиколотки",                                     0.09),
]

# Which view each measurement is primarily derived from (for v2 envelope)
MEASUREMENT_SOURCE: dict[str, str] = {
    "chest_circumference":          "fused",
    "waist_circumference":          "fused",
    "hip_circumference":            "fused",
    "neck_circumference":           "front",
    "neck_base_height":             "front",
    "shoulder_slope_width":         "front",
    "back_width_scapular":          "side",
    "chest_width_front":            "front",
    "back_length_to_waist":         "side",
    "front_length_to_waist":        "side",
    "arm_length_shoulder_to_wrist": "side",
    "upper_arm_circumference":      "fused",
    "wrist_circumference":          "fused",
    "leg_length_inner_seam":        "side",
    "leg_length_outer_seam":        "side",
    "thigh_circumference":          "fused",
    "calf_circumference":           "side",
    "ankle_circumference":          "fused",
}


# ---------------------------------------------------------------------------
# Pydantic models — v2 MeasurementEnvelope
# ---------------------------------------------------------------------------

class MeasurementItem(BaseModel):
    id: str
    label_uk: str
    value_cm: float = Field(..., description="Body measurement in centimetres, one decimal")
    uncertainty_cm: float = Field(..., ge=0.0, description="1σ estimate from the pipeline")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Literal["front", "side", "fused", "manual"] = "fused"
    quality_flags: list[str] = Field(default_factory=list)


class PipelineInfo(BaseModel):
    source: Literal["mock", "mediapipe", "regression"] = "mock"
    model_version: str = "mock-0.1.0"
    unit_system: Literal["metric"] = "metric"


class SubjectInfo(BaseModel):
    height_cm: float = Field(..., ge=100, le=250)
    sex: Literal["male", "female", "other"]
    age_band: Literal["adult", "teen", "child"] | None = None
    posture_flags: list[str] = Field(default_factory=list)


class CaptureQuality(BaseModel):
    quality: float = Field(..., ge=0.0, le=1.0)
    pose_ok: bool = True
    occlusions: list[str] = Field(default_factory=list)


class CaptureInfo(BaseModel):
    front: CaptureQuality
    side: CaptureQuality


class MeasurementEnvelope(BaseModel):
    """Schema v2 — consumed by the frontend sizing + pattern engine."""
    schema_id: str = Field("pointsx.measurement.envelope", alias="schema")
    schema_version: int = 2
    request_id: str
    created_at: str
    pipeline: PipelineInfo
    subject: SubjectInfo
    capture: CaptureInfo
    measurements: list[MeasurementItem]
    derived: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Mock measurement builder
# ---------------------------------------------------------------------------

def _sex_scale(sex: Literal["male", "female", "other"]) -> float:
    if sex == "male":
        return 1.0
    if sex == "female":
        return 0.97
    return 0.985


def build_mock_measurements(
    height_cm: float,
    sex: Literal["male", "female", "other"],
) -> list[MeasurementItem]:
    """Deterministic mock measurements from height and sex (not real anthropometry)."""
    scale = _sex_scale(sex)
    h = height_cm
    out: list[MeasurementItem] = []
    for i, (mid, label_uk, ratio) in enumerate(MEASUREMENT_SPECS):
        base = h * ratio * scale
        tweak = 1.0 + 0.01 * math.sin(i + h * 0.1)
        value = round(base * tweak, 1)
        conf = round(0.75 + 0.04 * math.cos(i * 0.7), 2)
        conf = min(0.95, max(0.5, conf))
        uncertainty = round((1.0 - conf) * value * 0.05, 2)
        out.append(MeasurementItem(
            id=mid,
            label_uk=label_uk,
            value_cm=value,
            uncertainty_cm=uncertainty,
            confidence=conf,
            source=MEASUREMENT_SOURCE.get(mid, "fused"),
            quality_flags=[],
        ))
    return out


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PointsX WebUI", version="0.2.0")
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


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="Missing static index")
    return FileResponse(index_path)


@app.post("/api/measure/mock", response_model=MeasurementEnvelope)
async def measure_mock(
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
) -> MeasurementEnvelope:
    """
    Return MeasurementEnvelope v2; images are validated but not analysed
    (drop-in for the real MeasurementPipeline).
    """
    await _validate_image_upload(front, "front")
    await _validate_image_upload(side, "side")

    measurements = build_mock_measurements(height_cm, sex)
    now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    request_id = str(uuid.uuid4())

    return MeasurementEnvelope(
        **{
            "schema": "pointsx.measurement.envelope",
            "schema_version": 2,
            "request_id": request_id,
            "created_at": now,
            "pipeline": PipelineInfo(),
            "subject": SubjectInfo(height_cm=height_cm, sex=sex),
            "capture": CaptureInfo(
                front=CaptureQuality(quality=0.85, pose_ok=True),
                side=CaptureQuality(quality=0.82, pose_ok=True),
            ),
            "measurements": measurements,
            "derived": {},
            "warnings": [],
        }
    )
