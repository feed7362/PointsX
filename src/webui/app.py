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
from typing import Any, Literal

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from webui.config import (
    STATIC_DIR,
    Settings,
    get_settings,
)
from webui.errors import AppError
from webui.errors import pipeline_value_error_detail as _pipeline_value_error_detail
from webui.errors import validation_errors_to_uk as _validation_errors_to_uk
from webui.infrastructure.weights import prefetch_weights
from webui.schemas import (
    CaptureInfo,
    CaptureQuality,
    MeasurementEnvelope,
    PipelineInfo,
    SubjectInfo,
    TtsRequest,
)
from webui.services.dataset_capture import save_capture_pair
from webui.services.mock import build_mock_measurement_envelope
from webui.services.proxy import forward_measure, ping_space_health
from webui.services.uploads import decode_upload

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()
# Proxy mode (Vercel): /api/measure is forwarded to the HF Space; mock + TTS stay local.
_INFERENCE_ENDPOINT = _SETTINGS.inference_endpoint
_IS_VERCEL_PROXY_MODE = _SETTINGS.proxy_mode
DATASET_DIR = _SETTINGS.dataset_dir

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


# ---------------------------------------------------------------------------
# Pipeline lifespan — load models once at startup
# ---------------------------------------------------------------------------

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

    prefetch_weights([pose_coco, seg_path, reg_path])

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


@app.exception_handler(AppError)
async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


async def _validate_and_decode(
    upload: UploadFile, label: str
) -> tuple[Any, bytes]:
    """Read an upload and decode it; returns ``(bgr_image, raw_bytes)`` (validation: services.uploads)."""
    data = await upload.read()
    return decode_upload(data, upload.content_type, label), data


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

    body, status = await ping_space_health(_INFERENCE_ENDPOINT)  # type: ignore[arg-type]
    return JSONResponse(body, status_code=status)

async def _proxy_measure_to_hf(
    height_cm: float,
    sex: str,
    pose_backend: str,
    front: UploadFile,
    side: UploadFile,
) -> Response:
    """Forward /api/measure to the HF Space; the upstream response is returned verbatim."""
    front_part = (front.filename or "front.jpg", await front.read(), front.content_type or "image/jpeg")
    side_part = (side.filename or "side.jpg", await side.read(), side.content_type or "image/jpeg")
    status, content, media_type = await forward_measure(
        _INFERENCE_ENDPOINT,  # type: ignore[arg-type]
        height_cm=height_cm,
        sex=sex,
        pose_backend=pose_backend,
        front=front_part,
        side=side_part,
    )
    return Response(content=content, status_code=status, media_type=media_type)


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

    _, dataset_save_warning = await save_capture_pair(DATASET_DIR, front_bytes, side_bytes)

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
