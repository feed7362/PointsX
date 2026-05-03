"""FastAPI app: static capture UI + real body-measurement endpoint.

Configuration (environment variables, all optional):
    POINTSX_POSE_MODEL        path to YOLO11n-pose .pt (default: models/yolo11n-pose.pt)
    POINTSX_SEG_MODEL         path to YOLO11n-seg .pt  (default: models/yolo11n-seg.pt)
    POINTSX_REGRESSION_MODEL  path to circumference_regressor.pt (optional; falls back
                              to the Ramanujan ellipse approximation if unset/missing)
    POINTSX_DEVICE            "auto" | "cpu" | "cuda" | "0" | …  (default: "auto")
    POINTSX_TTS_VOICE         Ukrainian neural voice for ``/api/tts`` (default: uk-UA-PolinaNeural)
    POINTSX_TTS_DISABLE       ``1``/``true`` to disable server TTS (browser speech fallback only)

If model loading fails, the server still starts; `/api/measure` returns 503 until
the issue is fixed.

Speech hints use ``POST /api/tts`` (edge-tts, needs internet). If ``uv sync`` fails
(for example Torch wheels on some platforms), install TTS separately:
``.venv/bin/python -m pip install edge-tts`` then restart ``pointsx-web``.
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
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DISALLOWED_CONTENT_PREFIXES = ("text/", "video/", "audio/")

_UPLOAD_LABEL_UK = {"front": "Анфас", "side": "Профіль"}

_PIPELINE_VALUE_ERROR_UK = {
    "No person detected in front image": (
        "На знімку анфасу не виявлено людину. Переконайтеся, що фігура повністю в кадрі "
        "та поза відповідає вимогам."
    ),
    "No person detected in side image": (
        "На знімку профілю не виявлено людину. Переконайтеся, що фігура повністю в кадрі "
        "та поза відповідає вимогам."
    ),
    "No body silhouette detected in front image": (
        "На анфасі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No body silhouette detected in side image": (
        "На профілі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No segmentation mask for front image": (
        "На анфасі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "No segmentation mask for side image": (
        "На профілі не вдалося виділити силует тіла. Спробуйте інше освітлення або фон."
    ),
    "Cannot calibrate front view: insufficient visible keypoints": (
        "Недостатньо видимих ключових точок на анфасі для калібровки за зростом. "
        "Переконайтеся, що ступні та голова в кадрі."
    ),
    "Cannot calibrate side view: insufficient visible keypoints": (
        "Недостатньо видимих ключових точок на профілі для калібровки за зростом. "
        "Переконайтеся, що ступні та голова в кадрі."
    ),
    "Invalid sex for measurement pipeline": "Некоректне значення статі для пайплайну.",
}


def _pipeline_value_error_detail(message: str) -> str:
    return _PIPELINE_VALUE_ERROR_UK.get(
        message.strip(),
        f"Не вдалося обробити знімки: {message}",
    )


def _validation_errors_to_uk(errors: list[Any]) -> str:
    if not errors:
        return "Некоректні дані форми."
    parts: list[str] = []
    field_labels = {
        "height_cm": "Зріст (см)",
        "sex": "Стать",
        "front": "Фото анфасу",
        "side": "Фото профілю",
    }
    for item in errors:
        if not isinstance(item, dict):
            continue
        loc = tuple(item.get("loc") or ())
        field_key = str(loc[-1]) if loc else "form"
        label = field_labels.get(field_key, field_key)
        err_type = str(item.get("type") or "")
        msg_en = str(item.get("msg") or "")
        ctx = item.get("ctx")
        if not isinstance(ctx, dict):
            ctx = {}

        if err_type == "missing":
            parts.append(f"{label}: значення не передано.")
        elif err_type in ("float_parsing", "decimal_parsing", "int_parsing"):
            parts.append(f"{label}: потрібне число.")
        elif err_type == "greater_than_equal":
            ge = ctx.get("ge")
            parts.append(f"{label}: занадто мале значення (мінімум {ge}).")
        elif err_type == "less_than_equal":
            le = ctx.get("le")
            parts.append(f"{label}: занадто велике значення (максимум {le}).")
        elif err_type in ("literal_error", "enum"):
            parts.append(f"{label}: недопустиме значення.")
        else:
            parts.append(f"{label}: {msg_en}")
    return " ".join(parts) if parts else "Некоректні дані форми."


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

class TtsRequest(BaseModel):
    """Short Ukrainian phrase for pose hints / countdown (synthesized via edge-tts)."""

    text: str = Field(..., min_length=1, max_length=600)


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
    pose_path = _resolve_path("POINTSX_POSE_MODEL", "models/yolo11n-pose.pt")
    seg_path  = _resolve_path("POINTSX_SEG_MODEL",  "models/yolo11n-seg.pt")
    reg_path  = os.environ.get("POINTSX_REGRESSION_MODEL", "").strip() or None
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


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message = _validation_errors_to_uk(list(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": message})


async def _validate_and_decode(upload: UploadFile, label: str) -> np.ndarray:
    """Validate upload bytes and decode to a BGR ndarray (cv2 convention)."""
    uk = _UPLOAD_LABEL_UK.get(label, label)
    data = await upload.read()
    if len(data) == 0:
        raise HTTPException(status_code=400, detail=f"{uk}: файл порожній.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"{uk}: файл завеликий (ліміт {MAX_UPLOAD_BYTES // (1024 * 1024)} МБ).",
        )
    ct = upload.content_type or ""
    if any(ct.startswith(p) for p in DISALLOWED_CONTENT_PREFIXES):
        raise HTTPException(
            status_code=400,
            detail=f"{uk}: недопустимий тип вмісту ({ct!r}). Очікується зображення.",
        )
    if not _looks_like_raster_image(data):
        raise HTTPException(
            status_code=400,
            detail=f"{uk}: очікується JPEG, PNG або WebP.",
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail=f"{uk}: не вдалося розпізнати зображення.")
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
            detail=(
                "Неможливо виконати замір: моделі не завантажені на сервері. "
                "Перевірте шляхи до ваг і журнал сервера. "
                f"Технічні деталі: {err}"
            ),
        )

    front_img = await _validate_and_decode(front, "front")
    side_img  = await _validate_and_decode(side,  "side")

    try:
        result = pipeline.measure(front_img, side_img, height_cm)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=_pipeline_value_error_detail(str(exc)),
        ) from exc
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Помилка під час обчислення мірок: {exc}",
        ) from exc

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


@app.post("/api/tts")
async def tts_synthesize(body: TtsRequest) -> Response:
    """Synthesize Ukrainian speech (MP3) using a lightweight neural Edge voice."""
    from webui import tts as tts_mod

    if tts_mod.tts_disabled():
        raise HTTPException(status_code=503, detail="Синтез мовлення вимкнено на сервері.")

    try:
        mp3 = await tts_mod.synthesize_uk_speech_mp3(body.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ImportError as exc:
        logger.warning("TTS unavailable — install edge-tts in the server environment: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Пакет edge-tts не встановлено в середовищі сервера. "
                "Встановіть: `.venv/bin/python -m pip install edge-tts` і перезапустіть pointsx-web. "
                "Підказки спробують голос браузера."
            ),
        ) from exc
    except Exception as exc:
        logger.warning("TTS synthesis failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Не вдалося синтезувати мовлення. Перевірте доступ до інтернету.",
        ) from exc

    return Response(content=mp3, media_type="audio/mpeg")
