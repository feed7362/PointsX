"""FastAPI app: static capture UI + real body-measurement endpoint.

Configuration (environment variables, all optional):
    POINTSX_POSE_MODEL_CUSTOM path to 16-keypoint (LV-MHP) pose .pt
                              default: models/pose-cus.pt
    POINTSX_POSE_MODEL_COCO   path to COCO-17 pose .pt (mapped to 16 internally)
                              default: models/yolo26-pose.pt
                              (we keep the heavier pose model — calibration
                              accuracy depends on HEAD_TOP/ankle stability.
                              Auto-downloaded + bucket-mirrored on first boot.)
    POINTSX_POSE_MODEL        legacy: if set, overrides POINTSX_POSE_MODEL_CUSTOM only
    POINTSX_SEG_MODEL         path to YOLO segmentation .pt
                              default: models/yolo12l-person-seg-extended.pt
                              (heavier but more stable masks — accuracy matters
                              for body-width measurements; warm-cached in
                              LOCAL_DATA_DIR after the first boot)
    POINTSX_REGRESSION_MODEL  path to regression .pt
                              default: models/circumference_regressor.pt if present;
                              set to an empty string to force the Ramanujan ellipse
                              fallback instead.
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

import asyncio
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DISALLOWED_CONTENT_PREFIXES = ("text/", "video/", "audio/")

_DATASET_INDEX_RE = re.compile(r"^[ap](\d+)\.")
_dataset_lock = asyncio.Lock()

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
        "pose_backend": "Модель пози",
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


def _build_visualization_only_envelope(
    *,
    preview_result: Any,
    height_cm: float,
    sex: Literal["male", "female", "other"],
    pose_backend: Literal["custom", "coco"],
    warning: str,
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
) -> MeasurementEnvelope:
    """Return a valid envelope with debug visualizations when calibration fails."""
    from webui.visualize import pipeline_visualizations_b64

    derived: dict[str, Any] = {}
    try:
        derived = pipeline_visualizations_b64(front_bgr, side_bgr, preview_result)
    except Exception:
        logger.exception("Failed to generate visualization-only debug images")

    return MeasurementEnvelope(
        schema="pointsx.measurement.envelope",
        schema_version=2,
        request_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        pipeline=PipelineInfo(
            source="mediapipe",
            model_version="visualization-only",
            unit_system="metric",
            pose_backend=pose_backend,
        ),
        subject=SubjectInfo(
            height_cm=height_cm,
            sex=sex,
            posture_flags=[],
        ),
        capture=CaptureInfo(
            front=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
            side=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
        ),
        measurements=[],
        derived=derived,
        warnings=[warning],
    )


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


def _detect_image_extension(data: bytes) -> str:
    """Map raw image magic bytes to a filesystem extension."""
    if len(data) >= 2 and data[:2] == b"\xff\xd8":
        return "jpg"
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "bin"


def _next_dataset_index(directory: Path) -> int:
    """Pick the next free `i` such that no `a{i}.*` or `p{i}.*` exists yet."""
    if not directory.is_dir():
        return 1
    max_idx = 0
    for entry in directory.iterdir():
        if not entry.is_file():
            continue
        match = _DATASET_INDEX_RE.match(entry.name)
        if not match:
            continue
        try:
            idx = int(match.group(1))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1


async def _save_capture_pair_to_dataset(
    front_bytes: bytes,
    side_bytes: bytes,
) -> int | None:
    """Persist the (front, side) image pair as ``a{i}.ext`` / ``p{i}.ext``.

    Failures are logged but never raised — saving the dataset is a best-effort
    side effect of measurement and must not break the user-facing request.
    """
    try:
        async with _dataset_lock:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            idx = _next_dataset_index(DATASET_DIR)
            front_path = DATASET_DIR / f"a{idx}.{_detect_image_extension(front_bytes)}"
            side_path = DATASET_DIR / f"p{idx}.{_detect_image_extension(side_bytes)}"
            front_path.write_bytes(front_bytes)
            side_path.write_bytes(side_bytes)
            logger.info("Saved capture pair to dataset: %s, %s", front_path, side_path)
            return idx
    except Exception:
        logger.exception("Failed to save capture pair to dataset folder %s", DATASET_DIR)
        return None


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
    pose_backend: Literal["custom", "coco"] | None = None


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
    pose_custom = _resolve_path("POINTSX_POSE_MODEL_CUSTOM", "models/pose-cus.pt")
    pose_coco = _resolve_path("POINTSX_POSE_MODEL_COCO", "models/yolo26-pose.pt")
    legacy_pose = os.environ.get("POINTSX_POSE_MODEL")
    if legacy_pose is not None and str(legacy_pose).strip():
        pose_custom = str(legacy_pose).strip()
        logger.info("POINTSX_POSE_MODEL set — using as custom pose path (legacy override).")
    seg_path = _resolve_path("POINTSX_SEG_MODEL", "models/yolo12l-person-seg-extended.pt")
    # Regressor disabled by default — currently it's known to produce outliers
    # on real photos (e.g. negative-cm hips/thighs on certain subjects), and
    # the per-sex bias scales in envelope.py were fit against the Ramanujan
    # ellipse output, not the regressor's. To re-enable, set the env var:
    #   POINTSX_REGRESSION_MODEL=models/circumference_regressor.pt
    reg_raw = os.environ.get("POINTSX_REGRESSION_MODEL")
    reg_path = reg_raw.strip() if (reg_raw and reg_raw.strip()) else None
    device    = _resolve_path("POINTSX_DEVICE", "auto")

    # ── Resolve model weights ─────────────────────────────────────────────────
    # Three sources in priority order:
    #   1. LOCAL_DATA_DIR/models/<name>   ← HF Storage Bucket mounted at /data
    #   2. HF Hub model repo (env: HF_MODELS_REPO)
    #   3. S3 bucket under MODELS_S3_KEY_PREFIX (env: MODELS_S3_KEY_PREFIX)
    # The download is idempotent: files already on disk stay, so warm restarts
    # don't re-download anything.
    try:
        from pathlib import Path as _Path

        hf_repo = (os.environ.get("HF_MODELS_REPO") or "").strip()
        hf_revision = (os.environ.get("HF_MODELS_REVISION") or "main").strip()
        hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)

        def _pull(local: str | None) -> None:
            if not local:
                return
            p = _Path(local)
            if p.is_file() and p.stat().st_size > 0:
                return

            # First: HF Storage Bucket mounted at LOCAL_DATA_DIR/models/.
            try:
                from webui import storage as _storage_check
                bucket_path = _storage_check.local_model_path(p.name)
                if bucket_path is not None:
                    p.parent.mkdir(parents=True, exist_ok=True)
                    # Symlink if possible (saves disk + matches mount semantics),
                    # else copy. Falls back to copy on Windows without privilege.
                    try:
                        if p.exists() or p.is_symlink():
                            p.unlink()
                        p.symlink_to(bucket_path)
                        logger.warning(
                            "Bucket-mount linked — file=%s → %s",
                            p.name, bucket_path,
                        )
                    except (OSError, NotImplementedError):
                        import shutil as _sh
                        _sh.copy2(bucket_path, p)
                        logger.warning(
                            "Bucket-mount copied — file=%s ← %s",
                            p.name, bucket_path,
                        )
                    return
            except Exception:  # noqa: BLE001
                pass

            # Second: HF Hub model repo (free, unlimited public).
            if hf_repo:
                try:
                    from huggingface_hub import hf_hub_download
                    p.parent.mkdir(parents=True, exist_ok=True)
                    downloaded = hf_hub_download(
                        repo_id=hf_repo,
                        filename=p.name,
                        revision=hf_revision,
                        token=hf_token,
                        local_dir=str(p.parent),
                    )
                    logger.warning(
                        "HF Hub download OK — repo=%s file=%s → %s (size=%d bytes)",
                        hf_repo, p.name, downloaded, _Path(downloaded).stat().st_size,
                    )
                    return
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "HF Hub download FAILED — repo=%s file=%s err=%s. Will try S3 next.",
                        hf_repo, p.name, exc,
                    )

            # Fallback: S3 bucket (when archival is configured).
            from webui import storage as _storage
            if not _storage.is_enabled():
                return
            models_prefix = (os.environ.get("MODELS_S3_KEY_PREFIX") or "models/").lstrip("/")
            if not models_prefix.endswith("/"):
                models_prefix += "/"
            _storage.download_to_path(models_prefix + p.name, local)

        _pull(pose_coco)
        _pull(seg_path)
        if reg_path:
            _pull(reg_path)
    except Exception:  # noqa: BLE001
        logger.exception("Weight pre-fetch raised — pipeline will try local paths.")

    app.state.pipeline = None
    app.state.pipeline_load_error = None

    try:
        from webui.inference import WebuiPipeline  # local import to avoid heavy deps at module load

        app.state.pipeline = WebuiPipeline(
            pose_custom_path=pose_custom,
            pose_coco_path=pose_coco,
            seg_model_path=seg_path,
            regression_model_path=reg_path,
            device=device,
        )
        avail = app.state.pipeline.models.available_pose_backends()
        if not avail:
            app.state.pipeline = None
            raise RuntimeError(
                "No pose weights loaded: need at least one of custom (.pt) or COCO (.pt) checkpoints."
            )
        logger.info(
            "Pipeline loaded — pose_custom=%s pose_coco=%s pose_backends=%s seg=%s regressor=%s device=%s",
            pose_custom,
            pose_coco,
            sorted(avail),
            seg_path,
            reg_path or "<ellipse-fallback>",
            device,
        )

        # Warm up each model with a dummy forward pass so the very first
        # /api/measure request isn't ~2× slower than the warm rate. Disable
        # by setting POINTSX_WARMUP_DISABLE=1 if startup time is more
        # precious than first-request latency (e.g. autoscale-on-demand).
        warmup_off = (os.environ.get("POINTSX_WARMUP_DISABLE") or "").strip().lower()
        if warmup_off not in ("1", "true", "yes", "on"):
            try:
                timings = app.state.pipeline.warmup()
                pretty = ", ".join(f"{k}={v:.2f}s" for k, v in timings.items())
                logger.warning(
                    "Pipeline warmed up — %s. First /api/measure will run at "
                    "steady-state speed (no JIT penalty).",
                    pretty or "no models warmed",
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup raised — first request may be slow: %s", exc)
    except Exception as exc:  # noqa: BLE001 — we want the server to keep running
        app.state.pipeline_load_error = str(exc)
        logger.error(
            "Failed to load WebuiPipeline (endpoint will return 503): %s", exc,
        )

    # Probe storage on startup so the operator immediately knows whether
    # archival is configured. Forces the lazy config to resolve and writes
    # a single WARNING-level line either way.
    try:
        from webui import storage as _storage_probe

        if _storage_probe.is_enabled():
            logger.warning("Storage probe: archival is ENABLED on startup")
        else:
            logger.warning("Storage probe: archival is DISABLED on startup")
    except Exception:  # noqa: BLE001
        logger.exception("Storage probe failed unexpectedly")

    yield

    app.state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="PointsX WebUI", version="0.3.0", lifespan=lifespan)

# Static mount is conditional — this backend repo doesn't carry a static/
# subdir (the SPA lives in the Pointx-frontend repo on Vercel). When the
# folder is present (single-container dev mode), we still serve it so a
# developer can hit / directly without standing up Vercel.
if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.info("STATIC_DIR not present (%s) — backend runs API-only.", STATIC_DIR)

# ── CORS for the demo deployment ───────────────────────────────────────────
# The static SPA lives on Vercel; the inference API lives on Hugging Face
# Spaces. CORS_ALLOW_ORIGINS env var is a comma-separated list of allowed
# origins. ``*`` allows any origin (fine for an open scientific demo).
from fastapi.middleware.cors import CORSMiddleware

_cors_raw = os.environ.get("CORS_ALLOW_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message = _validation_errors_to_uk(list(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": message})


def _downscale_for_inference(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize a phone-camera photo so its longest side is <= max_side px.

    YOLO runs at imgsz=640 internally anyway — passing a 4000×3000 photo
    only buys CPU time on its built-in resize step (~0.5–1.5 s per
    image on free CPU). Keeping max_side at 1280 leaves headroom for
    silhouette quality at the limbs without throwing away signal.
    """
    if img is None or img.size == 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


async def _validate_and_decode(
    upload: UploadFile, label: str
) -> tuple[np.ndarray, bytes]:
    """Validate upload bytes and decode to a BGR ndarray (cv2 convention).

    Returns the decoded image alongside the raw bytes so callers can persist
    the original payload (e.g. into a dataset folder) without re-reading the
    upload stream.
    """
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
    return img, data


@app.get("/")
async def index() -> Any:
    """Serve the bundled SPA when present (dev convenience); otherwise the
    backend is API-only and / returns a small JSON banner instead.
    """
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse({
            "service": "pointx-backend",
            "status": "ok",
            "api": ["/api/measure", "/api/measure/mock", "/api/tts", "/api/health"],
            "frontend": "see Pointx-frontend (Vercel deployment)",
        })
    return FileResponse(index_path)


@app.get("/api/health")
async def health(request: Request) -> JSONResponse:
    """Cheap liveness + readiness probe.

    Returns 200 with a small JSON snapshot. Useful for:
      • HF Spaces "container healthy" detection
      • Vercel / monitoring keep-alive pings (avoids the 48 h sleep)
      • Quick smoke test before submitting a measurement

    Reports pipeline-loaded state separately so a caller can distinguish
    "Space is up" from "Space is up AND ready to measure".
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    err = getattr(request.app.state, "pipeline_load_error", None)
    backends: list[str] = []
    if pipeline is not None:
        try:
            backends = sorted(pipeline.models.available_pose_backends())
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse({
        "service": "pointx-backend",
        "status": "ok",
        "pipeline_ready": pipeline is not None,
        "pose_backends": backends,
        "pipeline_load_error": err,
    })


@app.post("/api/measure", response_model=MeasurementEnvelope)
async def measure(
    request: Request,
    background_tasks: BackgroundTasks,
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    pose_backend: Literal["custom", "coco"] = Form("coco"),
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

    avail = pipeline.models.available_pose_backends()
    if pose_backend not in avail:
        need = "pose-cus.pt (16 точок)" if pose_backend == "custom" else "yolo26-pose.pt (COCO 17)"
        raise HTTPException(
            status_code=503,
            detail=(
                f"Обрана модель пози ({pose_backend}) недоступна: відсутній файл ваг для {need}. "
                "Перевірте POINTSX_POSE_MODEL_CUSTOM / POINTSX_POSE_MODEL_COCO або оберіть інший режим."
            ),
        )

    from webui._timing import Timings
    tm = Timings()

    with tm("decode"):
        front_img, front_bytes = await _validate_and_decode(front, "front")
        side_img,  side_bytes  = await _validate_and_decode(side,  "side")

    # Pre-downscale large phone-camera shots so YOLO's resize step isn't
    # the bottleneck. The model runs at imgsz=640 internally anyway —
    # passing a 4000×3000 image only costs CPU on the resize step.
    with tm("downscale"):
        front_img = _downscale_for_inference(front_img)
        side_img = _downscale_for_inference(side_img)

    await _save_capture_pair_to_dataset(front_bytes, side_bytes)

    # One request_id per call, used for both the envelope and the archive prefix.
    request_id = str(uuid.uuid4())

    def _archive(envelope_obj, *, outcome: str) -> None:
        """Persist photos + envelope to whichever store(s) are configured.

        Order of operations (latency-aware):
          1. Local filesystem (LOCAL_DATA_DIR) — runs INLINE, ~50 ms.
             Cheap, useful to have the files on disk before the response
             returns in case anything inspects them right away.
          2. S3-compatible bucket (ELK_*/S3_*/R2_*/...) — scheduled as a
             FastAPI BackgroundTask. Runs AFTER the HTTP response is sent
             so the network round-trip to the bucket doesn't sit on the
             critical path. Failures are still logged.

        Either path is independent; both can be enabled simultaneously.
        """
        import sys
        try:
            from webui import storage
            args = dict(
                request_id=request_id,
                front_bytes=front_bytes,
                front_content_type=(front.content_type or "image/jpeg"),
                side_bytes=side_bytes,
                side_content_type=(side.content_type or "image/jpeg"),
                envelope_json=envelope_obj.model_dump(mode="json", by_alias=True),
                metadata={
                    "height_cm": str(height_cm),
                    "sex": sex,
                    "pose_backend": pose_backend,
                    "outcome": outcome,
                    "created_at": envelope_obj.created_at,
                },
            )

            local_ok = storage.archive_measurement_local(**args)
            s3_enabled = storage.is_enabled()
            s3_scheduled = False
            if s3_enabled:
                def _bg_s3_upload(_args=args, _outcome=outcome, _rid=request_id):
                    import sys as _sys
                    try:
                        ok = storage.archive_measurement(**_args)
                        print(
                            f"[archive hook bg] outcome={_outcome} request_id={_rid} s3_ok={ok}",
                            file=_sys.stderr,
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[archive hook bg] raised: {type(exc).__name__}: {exc}",
                            file=_sys.stderr,
                            flush=True,
                        )
                background_tasks.add_task(_bg_s3_upload)
                s3_scheduled = True

            print(
                f"[archive hook] outcome={outcome} request_id={request_id} "
                f"local={local_ok} s3_scheduled={s3_scheduled}",
                file=sys.stderr,
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[archive hook] raised: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            logger.exception("S3 archive raised — measurement response is unaffected.")

    import asyncio
    try:
        # Pipeline is sync (PyTorch + numpy). asyncio.to_thread keeps the
        # event loop responsive so /api/tts and health checks can still
        # answer while this request crunches pose+seg.
        result = await asyncio.to_thread(
            pipeline.measure,
            front_img, side_img, height_cm,
            pose_backend=pose_backend, timings=tm,
        )
    except ValueError as exc:
        err_text = str(exc).strip()
        if err_text.startswith("Cannot calibrate "):
            preview = await asyncio.to_thread(
                pipeline.preview,
                front_img, side_img, pose_backend=pose_backend,
            )
            envelope = _build_visualization_only_envelope(
                preview_result=preview,
                height_cm=height_cm,
                sex=sex,
                pose_backend=pose_backend,
                warning=_pipeline_value_error_detail(err_text),
                front_bgr=front_img,
                side_bgr=side_img,
            )
            envelope.request_id = request_id
            logger.info(
                "Full model output envelope: %s",
                envelope.model_dump(mode="json", by_alias=True),
            )
            _archive(envelope, outcome="calibration_failed")
            return envelope
        raise HTTPException(
            status_code=400,
            detail=_pipeline_value_error_detail(err_text),
        ) from exc
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise HTTPException(
            status_code=500,
            detail=f"Помилка під час обчислення мірок: {exc}",
        ) from exc

    from webui.envelope import body_to_envelope

    # `with_viz` query param gates the heavy base64-PNG render. Default
    # ON for backward-compat; set ?with_viz=0 in the frontend to shave
    # ~1-2 s off the response when overlays aren't needed.
    with_viz_raw = request.query_params.get("with_viz", "1").strip().lower()
    with_viz = with_viz_raw not in ("0", "false", "no", "off")

    with tm("envelope"):
        envelope = body_to_envelope(
            result=result,
            subject_height_cm=height_cm,
            sex=sex,
            request_id=request_id,
            front_bgr=front_img if with_viz else None,
            side_bgr=side_img if with_viz else None,
        )

    logger.warning("Measurement timings — %s", tm.format())

    _archive(envelope, outcome="ok")

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
