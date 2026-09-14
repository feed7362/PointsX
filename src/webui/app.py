"""FastAPI app: static capture UI + real body-measurement endpoint.

One app, two deployments:
    HF Space (Docker)  loads the models and runs /api/measure locally.
    Vercel             POINTSX_VERCEL=1 + POINTSX_INFERENCE_ENDPOINT → proxy mode:
                       no models, /api/measure is forwarded to the Space; /api/tts,
                       /api/measure/mock and the static UI are served locally. The
                       Vercel venv has no torch/opencv, so heavy imports stay lazy.

Configuration (environment variables, all optional):
    POINTSX_POSE_MODEL_CUSTOM path to 16-keypoint (LV-MHP) pose .pt
                              default: models/pose-cus.pt
    POINTSX_POSE_MODEL_COCO   path to COCO-17 pose .pt (mapped to 16 internally)
                              default: models/yolo26-pose.pt
                              (we keep the heavier pose model — calibration
                              accuracy depends on HEAD_TOP/ankle stability.)
    POINTSX_POSE_MODEL        legacy: if set, overrides POINTSX_POSE_MODEL_CUSTOM only
    POINTSX_SEG_MODEL         path to YOLO segmentation .pt
                              default: models/yolo12l-person-seg-extended.pt
    POINTSX_USE_REGRESSOR     ``1`` to use the circumference regressor instead of the
                              Ramanujan ellipse (default: off — the correction tables
                              in envelope.py were fitted against the ellipse)
    POINTSX_REGRESSION_MODEL  regressor .pt used when POINTSX_USE_REGRESSOR=1
                              default: models/reg.pt
    POINTSX_DEVICE            "auto" | "cpu" | "cuda" | "0" | …  (default: "auto")
    POINTSX_WARMUP_DISABLE    ``1`` to skip the startup dummy forward pass
    POINTSX_DATASET_DIR       path to save captured image pairs (default: dataset)
    POINTSX_TTS_VOICE         Ukrainian neural voice for ``/api/tts`` (default: uk-UA-PolinaNeural)
    POINTSX_TTS_DISABLE       ``1``/``true`` to disable server TTS (browser speech fallback only)
    POINTSX_VERCEL            set by api/index.py on Vercel (skips the static mount)
    POINTSX_INFERENCE_ENDPOINT  base URL of the HF Space; with POINTSX_VERCEL → proxy mode
    HF_MODELS_REPO / HF_MODELS_REVISION / HF_TOKEN  model repo for weight pre-fetch
    CORS_ALLOW_ORIGINS        comma-separated origins (default: ``*``)

If model loading fails, the server still starts; `/api/measure` returns 503 and
`/api/health` reports ``pipeline_ready: false`` until the issue is fixed.

Speech hints use ``POST /api/tts`` (edge-tts, needs internet). If ``uv sync`` fails
(for example Torch wheels on some platforms), install TTS separately:
``.venv/bin/python -m pip install edge-tts`` then restart ``pointsx-web``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from webui.config import (
    DISALLOWED_CONTENT_PREFIXES,
    MAX_UPLOAD_BYTES,
    STATIC_DIR,
    Settings,
    get_settings,
)
from webui.errors import pipeline_value_error_detail as _pipeline_value_error_detail
from webui.errors import validation_errors_to_uk as _validation_errors_to_uk
from webui.schemas import (
    CaptureInfo,
    CaptureQuality,
    MeasurementEnvelope,
    MeasurementItem,
    PipelineInfo,
    SubjectInfo,
    TtsRequest,
)

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
# Proxy mode (Vercel): /api/measure is forwarded to the HF Space; mock + TTS stay local.
_INFERENCE_ENDPOINT = _SETTINGS.inference_endpoint
_IS_VERCEL_PROXY_MODE = _SETTINGS.proxy_mode
DATASET_DIR = _SETTINGS.dataset_dir

_dataset_lock = asyncio.Lock()

_UPLOAD_LABEL_UK = {"front": "Анфас", "side": "Профіль"}


def _build_visualization_only_envelope(
    *,
    preview_result: Any,
    height_cm: float,
    sex: Literal["male", "female", "other"],
    pose_backend: Literal["custom", "coco"],
    warning: str,
    front_bgr: Any,
    side_bgr: Any,
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


def _dataset_pair_stem_exists(directory: Path, stem: str) -> bool:
    """True if any ``a{stem}.*`` or ``p{stem}.*`` file already exists."""
    for prefix in ("a", "p"):
        if any(directory.glob(f"{prefix}{stem}.*")):
            return True
    return False


def _unique_dataset_stem(directory: Path) -> str:
    """UTC timestamp stem for a capture pair; suffix ``_N`` if a collision exists."""
    now = datetime.now(timezone.utc)
    base = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    stem = base
    n = 0
    while directory.is_dir() and _dataset_pair_stem_exists(directory, stem):
        n += 1
        stem = f"{base}_{n}"
    return stem


async def _save_capture_pair_to_dataset(
    front_bytes: bytes,
    side_bytes: bytes,
) -> tuple[str | None, str | None]:
    """Persist the (front, side) image pair as ``a{timestamp}.ext`` / ``p{timestamp}.ext``.

    On failure, logs and returns a Ukrainian warning string for the API
    ``warnings`` list (measurement flow still succeeds).
    """
    try:
        async with _dataset_lock:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            stem = _unique_dataset_stem(DATASET_DIR)
            front_path = DATASET_DIR / f"a{stem}.{_detect_image_extension(front_bytes)}"
            side_path = DATASET_DIR / f"p{stem}.{_detect_image_extension(side_bytes)}"
            front_path.write_bytes(front_bytes)
            side_path.write_bytes(side_bytes)
            logger.info("Saved capture pair to dataset: %s, %s", front_path, side_path)
            return stem, None
    except Exception as exc:  # noqa: BLE001 — best-effort persistence
        logger.exception("Failed to save capture pair to dataset folder %s", DATASET_DIR)
        detail = str(exc).strip() or type(exc).__name__
        msg = (
            "Не вдалося зберегти знімки у папку датасету "
            f"({DATASET_DIR}): {detail}"
        )
        return None, msg


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

def _prefetch_weights(paths: list[str | None]) -> None:
    """Make sure each weight file exists locally before the models load.

    Sources in priority order (idempotent — files already on disk stay, so
    warm restarts download nothing):
      1. LOCAL_DATA_DIR/models/<name>   ← HF Storage Bucket mounted at /data
      2. HF Hub model repo (env: HF_MODELS_REPO)
      3. S3 bucket under MODELS_S3_KEY_PREFIX (when archival is configured)
    """
    hf_repo = (os.environ.get("HF_MODELS_REPO") or "").strip()
    hf_revision = (os.environ.get("HF_MODELS_REVISION") or "main").strip()
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)

    def _pull(local: str | None) -> None:
        if not local:
            return
        p = Path(local)
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
                    logger.warning("Bucket-mount linked — file=%s → %s", p.name, bucket_path)
                except (OSError, NotImplementedError):
                    import shutil as _sh
                    _sh.copy2(bucket_path, p)
                    logger.warning("Bucket-mount copied — file=%s ← %s", p.name, bucket_path)
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
                    hf_repo, p.name, downloaded, Path(downloaded).stat().st_size,
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

    try:
        for path in paths:
            _pull(path)
    except Exception:  # noqa: BLE001
        logger.exception("Weight pre-fetch raised — pipeline will try local paths.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the WebuiPipeline once, store on app.state.pipeline.

    In Vercel proxy mode (POINTSX_VERCEL + POINTSX_INFERENCE_ENDPOINT set),
    weight pre-fetch, model loading, warmup and the storage probe are all
    skipped — /api/measure is proxied to the remote inference backend instead.

    Failures are logged but do not crash the server — the endpoint will return
    503 until env vars are corrected and the server is restarted.
    """
    app.state.pipeline = None
    app.state.pipeline_load_error = None

    if _IS_VERCEL_PROXY_MODE:
        logger.info(
            "Vercel proxy mode: /api/measure will be forwarded to %s",
            _INFERENCE_ENDPOINT,
        )
        yield
        return

    cfg = Settings.from_env()
    pose_custom = cfg.pose_custom_path
    if cfg.pose_custom_from_legacy_env:
        logger.info("POINTSX_POSE_MODEL set — using as custom pose path (legacy override).")
    pose_coco = cfg.pose_coco_path
    seg_path = cfg.seg_model_path
    reg_path = cfg.regression_model_path
    device = cfg.device

    _prefetch_weights([pose_coco, seg_path, reg_path])

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
        # /api/measure request isn't ~2× slower than the warm rate.
        if not cfg.warmup_disable:
            try:
                timings = app.state.pipeline.warmup()
                pretty = ", ".join(f"{k}={v:.2f}s" for k, v in timings.items())
                logger.warning("Pipeline warmed up — %s.", pretty or "no models warmed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup raised — first request may be slow: %s", exc)
    except Exception as exc:  # noqa: BLE001 — we want the server to keep running
        app.state.pipeline_load_error = str(exc)
        logger.error(
            "Failed to load WebuiPipeline (endpoint will return 503): %s", exc,
        )

    # Probe storage on startup so the operator immediately knows whether
    # archival is configured.
    try:
        from webui import storage as _storage_probe

        state = "ENABLED" if _storage_probe.is_enabled() else "DISABLED"
        logger.warning("Storage probe: archival is %s on startup", state)
    except Exception:  # noqa: BLE001
        logger.exception("Storage probe failed unexpectedly")

    yield

    app.state.pipeline = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FitMeasure AI WebUI", version="0.3.0", lifespan=lifespan)

# On Vercel, static files are served directly via rewrite rules — skip the mount.
# The HF Space image does not ship static/ (API-only), so the mount is conditional too.
if STATIC_DIR.is_dir() and not _SETTINGS.vercel:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
else:
    logger.info("Static mount skipped (dir present=%s) — API-only.", STATIC_DIR.is_dir())

# CORS_ALLOW_ORIGINS: comma-separated list; ``*`` allows any origin.
_cors_origins = list(_SETTINGS.cors_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _camera_permissions_policy(request: Request, call_next):  # noqa: ANN001
    """Allow in-page camera on this origin (required for some mobile browsers)."""
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    message = _validation_errors_to_uk(list(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": message})


async def _validate_and_decode(
    upload: UploadFile, label: str
) -> tuple[Any, bytes]:
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

    import cv2
    import numpy as np

    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail=f"{uk}: не вдалося розпізнати зображення.")
    return img, data


@app.get("/")
async def index() -> Any:
    """Serve the bundled SPA when present; the API-only Space returns a JSON banner."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return JSONResponse({
            "service": "pointx-backend",
            "status": "ok",
            "api": ["/api/measure", "/api/measure/mock", "/api/tts", "/api/health"],
        })
    return FileResponse(index_path)


@app.get("/dataset.html")
async def dataset() -> FileResponse:
    dataset_path = STATIC_DIR / "dataset.html"
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="Missing dataset page")
    return FileResponse(dataset_path)


@app.get("/api/health")
async def health(request: Request) -> JSONResponse:
    """Cheap liveness + readiness probe.

    Reports pipeline-loaded state separately so a caller (HF container health,
    post-deploy smoke test) can distinguish "Space is up" from "Space is up AND
    ready to measure".
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
        "proxy_mode": _IS_VERCEL_PROXY_MODE,
        "pipeline_ready": pipeline is not None,
        "pose_backends": backends,
        "pipeline_load_error": err,
    })


@app.get("/api/keepalive")
async def keepalive(request: Request) -> JSONResponse:
    """Ping the HF Space so the free tier never sleeps (called by the Vercel cron in vercel.json).

    When the CRON_SECRET env var is set (Vercel sends it as a Bearer token on cron
    calls), other callers get 401. A timeout still counts: the request itself
    wakes a sleeping Space, which then needs a few minutes to boot.
    """
    secret = (os.environ.get("CRON_SECRET") or "").strip()
    if secret and request.headers.get("authorization") != f"Bearer {secret}":
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if not _IS_VERCEL_PROXY_MODE:
        return JSONResponse({"target": "self", "ok": True})

    import httpx

    url = f"{_INFERENCE_ENDPOINT.rstrip('/')}/api/health"  # type: ignore[union-attr]
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            upstream = await client.get(url)
    except httpx.TimeoutException:
        return JSONResponse({"target": url, "waking": True}, status_code=202)
    except httpx.RequestError as exc:
        logger.error("Keepalive ping to %s failed: %s", url, exc)
        return JSONResponse({"target": url, "error": str(exc)}, status_code=502)

    ready = False
    if upstream.status_code == 200:
        try:
            ready = bool(upstream.json().get("pipeline_ready"))
        except ValueError:
            pass
    return JSONResponse(
        {"target": url, "status": upstream.status_code, "pipeline_ready": ready},
        status_code=200 if ready else 502,
    )


async def _proxy_measure_to_hf(
    height_cm: float,
    sex: str,
    pose_backend: str,
    front: UploadFile,
    side: UploadFile,
) -> Response:
    """Forward the parsed multipart /api/measure request parameters to the HF Space backend.

    The upstream response (JSON or error) is returned verbatim to the client.
    Uses httpx with a generous timeout for heavy CPU inference.
    """
    import httpx

    target_url = f"{_INFERENCE_ENDPOINT.rstrip('/')}/api/measure"  # type: ignore[union-attr]
    front_bytes = await front.read()
    side_bytes = await side.read()

    files = {
        "front": (front.filename or "front.jpg", front_bytes, front.content_type or "image/jpeg"),
        "side": (side.filename or "side.jpg", side_bytes, side.content_type or "image/jpeg"),
    }
    data = {
        "height_cm": str(height_cm),
        "sex": sex,
        "pose_backend": pose_backend,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.post(target_url, data=data, files=files)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/json"),
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Час очікування відповіді від сервера інференсу вичерпано. Спробуйте ще раз.",
        )
    except httpx.RequestError as exc:
        logger.error("Proxy request to HF Space failed: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"Не вдалося зʼєднатися з сервером інференсу: {exc}",
        )


def _with_warning(envelope: MeasurementEnvelope, warning: str | None) -> MeasurementEnvelope:
    if not warning:
        return envelope
    return envelope.model_copy(update={"warnings": [*envelope.warnings, warning]})


@app.post("/api/measure", response_model=MeasurementEnvelope)
async def measure(
    request: Request,
    background_tasks: BackgroundTasks,
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    pose_backend: Literal["custom", "coco"] = Form("coco"),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
) -> Any:
    """Run pose + seg + (optional) regression on the supplied photo pair.

    In Vercel proxy mode, the request is forwarded transparently to the
    HuggingFace Space inference backend (POINTSX_INFERENCE_ENDPOINT).

    Returns a `MeasurementEnvelope` with up to 18 canonical body measurements.
    Measurements that the pipeline cannot derive are simply omitted; the
    frontend size engine tolerates a small number of missing values.
    """
    # --- Vercel proxy mode: forward to the HF Space ---
    if _IS_VERCEL_PROXY_MODE:
        return await _proxy_measure_to_hf(
            height_cm=height_cm,
            sex=sex,
            pose_backend=pose_backend,
            front=front,
            side=side,
        )

    # --- Local / self-hosted mode: run models directly ---
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

    # The pipeline downscales internally too (idempotent); doing it here keeps the
    # overlay images in the same pixel frame as the keypoints and masks.
    from pointsx.pipeline import downscale_for_inference

    with tm("downscale"):
        front_img = downscale_for_inference(front_img)
        side_img = downscale_for_inference(side_img)

    _, dataset_save_warning = await _save_capture_pair_to_dataset(front_bytes, side_bytes)

    # One request_id per call, used for both the envelope and the archive prefix.
    request_id = str(uuid.uuid4())

    def _archive(envelope_obj: MeasurementEnvelope, *, outcome: str) -> None:
        """Persist photos + envelope to whichever store(s) are configured.

        1. Local filesystem (LOCAL_DATA_DIR) — inline, ~50 ms.
        2. S3-compatible bucket — FastAPI BackgroundTask, runs after the
           response is sent so the bucket round-trip stays off the critical path.
        Failures are logged and never affect the response.
        """
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
            s3_scheduled = False
            if storage.is_enabled():
                def _bg_s3_upload(_args=args, _outcome=outcome, _rid=request_id):
                    try:
                        ok = storage.archive_measurement(**_args)
                        logger.warning("Archive (bg): outcome=%s request_id=%s s3_ok=%s", _outcome, _rid, ok)
                    except Exception:  # noqa: BLE001
                        logger.exception("Archive (bg) raised for request_id=%s", _rid)
                background_tasks.add_task(_bg_s3_upload)
                s3_scheduled = True

            logger.warning(
                "Archive: outcome=%s request_id=%s local=%s s3_scheduled=%s",
                outcome, request_id, local_ok, s3_scheduled,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Archive raised — measurement response is unaffected.")

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
            envelope = _with_warning(envelope, dataset_save_warning)
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
    # ON for backward-compat; ?with_viz=0 shaves ~1-2 s off the response.
    with_viz = request.query_params.get("with_viz", "1").strip().lower() not in ("0", "false", "no", "off")

    with tm("envelope"):
        envelope = body_to_envelope(
            result=result,
            subject_height_cm=height_cm,
            sex=sex,
            request_id=request_id,
            front_bgr=front_img if with_viz else None,
            side_bgr=side_img if with_viz else None,
        )
    envelope = _with_warning(envelope, dataset_save_warning)

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
    """Synthesize speech (MP3) using a lightweight neural Edge voice."""
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
