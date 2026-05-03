"""FastAPI app: static capture UI + real body-measurement endpoint.

Configuration (environment variables, all optional):
    POINTSX_POSE_MODEL        path to YOLO11n-pose .pt
                              default: runs/pose/best.pt (your fine-tuned weights)
    POINTSX_SEG_MODEL         path to YOLO11n-seg .pt
                              default: runs/seg/best.pt
    POINTSX_REGRESSION_MODEL  path to circumference_regressor.pt
                              default: models/circumference_regressor.pt if present;
                              set to an empty string to force the Ramanujan ellipse
                              fallback instead.
    POINTSX_DEVICE            "auto" | "cpu" | "cuda" | "0" | …  (default: "auto")

If model loading fails, the server still starts; `/api/measure` returns 503 until
the issue is fixed.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Pydantic models — v2 MeasurementEnvelope (kept here because envelope.py imports them)
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
    source: Literal["mock", "mediapipe", "regression"] = "regression"
    model_version: str = "regression-0.1"
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


def build_mock_measurement_envelope(
    height_cm: float,
    sex: Literal["male", "female", "other"],
) -> MeasurementEnvelope:
    """Deterministic demo envelope for the «без фото» UI button (no ML)."""
    from webui.envelope import CANONICAL_MEASUREMENTS

    h_scale = height_cm / 175.0
    if sex == "female":
        sex_scale = 0.94
    elif sex == "male":
        sex_scale = 1.0
    else:
        sex_scale = 0.97

    base_cm: dict[str, float] = {
        "chest_circumference": 102.0,
        "waist_circumference": 86.0,
        "hip_circumference": 100.0,
        "neck_circumference": 39.0,
        "neck_base_height": 148.0,
        "shoulder_slope_width": 46.0,
        "back_width_scapular": 38.0,
        "chest_width_front": 34.0,
        "back_length_to_waist": 44.0,
        "front_length_to_waist": 42.0,
        "arm_length_shoulder_to_wrist": 60.0,
        "upper_arm_circumference": 30.0,
        "wrist_circumference": 17.0,
        "leg_length_inner_seam": 78.0,
        "leg_length_outer_seam": 102.0,
        "thigh_circumference": 58.0,
        "calf_circumference": 38.0,
        "ankle_circumference": 24.0,
    }

    measurements: list[MeasurementItem] = []
    for mid, label_uk, src in CANONICAL_MEASUREMENTS:
        raw = base_cm.get(mid, 50.0) * h_scale * sex_scale
        val = round(max(1.0, raw), 1)
        measurements.append(
            MeasurementItem(
                id=mid,
                label_uk=label_uk,
                value_cm=val,
                uncertainty_cm=round(max(0.5, val * 0.04), 1),
                confidence=0.55,
                source=src,
                quality_flags=["mock"],
            )
        )

    capture = CaptureInfo(
        front=CaptureQuality(quality=0.55, pose_ok=True, occlusions=[]),
        side=CaptureQuality(quality=0.55, pose_ok=True, occlusions=[]),
    )
    return MeasurementEnvelope(
        request_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pipeline=PipelineInfo(source="mock", model_version="mock-0.1", unit_system="metric"),
        subject=SubjectInfo(height_cm=height_cm, sex=sex, age_band="adult", posture_flags=[]),
        capture=capture,
        measurements=measurements,
        derived={},
        warnings=["Тестовий режим: зображення й моделі не використовувалися."],
    )


# ---------------------------------------------------------------------------
# Pipeline lifespan — load models once at startup
# ---------------------------------------------------------------------------

def _resolve_path(env_var: str, default: str) -> str:
    raw = os.environ.get(env_var, default).strip()
    return raw or default


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the WebuiPipeline once, store on app.state.pipeline.

    Failures are logged but do not crash the server — the endpoint will return
    503 until env vars are corrected and the server is restarted.
    """
    pose_path = _resolve_path("POINTSX_POSE_MODEL", "runs/pose/best.pt")
    seg_path  = _resolve_path("POINTSX_SEG_MODEL",  "runs/seg/best.pt")
    # Auto-load the regressor when present; users can override via env var or
    # disable it explicitly with POINTSX_REGRESSION_MODEL="" (empty string).
    reg_default = "models/circumference_regressor.pt"
    reg_raw = os.environ.get("POINTSX_REGRESSION_MODEL")
    if reg_raw is None:
        reg_path = reg_default if Path(reg_default).exists() else None
    else:
        reg_path = reg_raw.strip() or None
    device    = _resolve_path("POINTSX_DEVICE", "auto")

    app.state.pipeline = None
    app.state.pipeline_load_error = None

    try:
        from webui.inference import WebuiPipeline  # local import to avoid heavy deps at module load

        app.state.pipeline = WebuiPipeline(
            pose_model_path=pose_path,
            seg_model_path=seg_path,
            regression_model_path=reg_path,
            device=device,
        )
        logger.info(
            "Pipeline loaded — pose=%s seg=%s regressor=%s device=%s",
            pose_path, seg_path, reg_path or "<ellipse-fallback>", device,
        )
    except Exception as exc:  # noqa: BLE001 — we want the server to keep running
        app.state.pipeline_load_error = str(exc)
        logger.error(
            "Failed to load WebuiPipeline (endpoint will return 503): %s", exc,
        )

    yield

    app.state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PointsX WebUI", version="0.3.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _validate_and_decode(upload: UploadFile, label: str) -> np.ndarray:
    """Validate upload bytes and decode to a BGR ndarray (cv2 convention)."""
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"{label}: empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{label}: file too large (max {MAX_UPLOAD_BYTES} bytes)",
        )
    ct = upload.content_type or ""
    if any(ct.startswith(p) for p in DISALLOWED_CONTENT_PREFIXES):
        raise HTTPException(status_code=400, detail=f"{label}: invalid Content-Type {ct!r}")
    if not _looks_like_raster_image(data):
        raise HTTPException(status_code=400, detail=f"{label}: body is not a JPEG, PNG, or WebP image")

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail=f"{label}: failed to decode image")
    return img


@app.get("/")
async def index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        raise HTTPException(status_code=500, detail="Missing static index")
    return FileResponse(index_path)


@app.post("/api/measure", response_model=MeasurementEnvelope)
async def measure(
    request: Request,
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
) -> MeasurementEnvelope:
    """Run pose + seg + (optional) regression on the supplied photo pair.

    Returns a `MeasurementEnvelope` with up to 18 canonical body measurements.
    Measurements that the pipeline cannot derive are simply omitted; the
    frontend size engine tolerates a small number of missing values.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        err = getattr(request.app.state, "pipeline_load_error", None) or "pipeline not initialised"
        raise HTTPException(
            status_code=503,
            detail=f"Pipeline unavailable: {err}",
        )

    front_img = await _validate_and_decode(front, "front")
    side_img  = await _validate_and_decode(side,  "side")

    try:
        result = pipeline.measure(front_img, side_img, height_cm)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface unexpected errors to client
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    from webui.envelope import body_to_envelope

    envelope = body_to_envelope(
        result=result,
        subject_height_cm=height_cm,
        sex=sex,
        request_id=str(uuid.uuid4()),
        front_bgr=front_img,
        side_bgr=side_img,
    )
    return envelope


@app.post("/api/measure/mock", response_model=MeasurementEnvelope)
async def measure_mock(
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
) -> MeasurementEnvelope:
    """Same JSON contract as `/api/measure`, without images or ML (UI test button)."""
    return build_mock_measurement_envelope(height_cm, sex)
