"""Startup/shutdown: load the measurement pipeline once (skipped entirely in Vercel proxy mode)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from webui.config import Settings, get_settings
from webui.infrastructure.weights import prefetch_weights

logger = logging.getLogger(__name__)


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

    settings = get_settings()
    if settings.proxy_mode:
        logger.info(
            "Vercel proxy mode: /api/measure will be forwarded to %s",
            settings.inference_endpoint,
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
        from webui.infrastructure.inference import WebuiPipeline  # local import to avoid heavy deps at module load

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
        from webui.infrastructure import storage as _storage_probe

        state = "ENABLED" if _storage_probe.is_enabled() else "DISABLED"
        logger.warning("Storage probe: archival is %s on startup", state)
    except Exception:  # noqa: BLE001
        logger.exception("Storage probe failed unexpectedly")

    yield

    app.state.pipeline = None
