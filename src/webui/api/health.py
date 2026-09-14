"""Liveness/readiness and the keepalive ping."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from webui.api import dependencies as deps
from webui.config import Settings
from webui.services.proxy import ping_space_health

router = APIRouter()


@router.get("/api/health")
async def health(
    pipeline: Any = Depends(deps.pipeline),
    load_error: str | None = Depends(deps.pipeline_load_error),
    settings: Settings = Depends(deps.settings),
) -> JSONResponse:
    """Cheap liveness + readiness probe.

    Reports pipeline-loaded state separately so a caller (HF container health,
    post-deploy smoke test) can distinguish "Space is up" from "Space is up AND
    ready to measure".
    """
    backends: list[str] = []
    if pipeline is not None:
        try:
            backends = sorted(pipeline.models.available_pose_backends())
        except Exception:  # noqa: BLE001
            pass
    return JSONResponse({
        "service": "pointx-backend",
        "status": "ok",
        "proxy_mode": settings.proxy_mode,
        "pipeline_ready": pipeline is not None,
        "pose_backends": backends,
        "pipeline_load_error": load_error,
    })


@router.get("/api/keepalive")
async def keepalive(request: Request, settings: Settings = Depends(deps.settings)) -> JSONResponse:
    """Ping the HF Space so the free tier never sleeps (called by the Vercel cron in vercel.json).

    When the CRON_SECRET env var is set (Vercel sends it as a Bearer token on cron
    calls), other callers get 401. Read per request so the secret can be rotated
    without a redeploy.
    """
    secret = (os.environ.get("CRON_SECRET") or "").strip()
    if secret and request.headers.get("authorization") != f"Bearer {secret}":
        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    if not settings.proxy_mode:
        return JSONResponse({"target": "self", "ok": True})

    body, status = await ping_space_health(settings.inference_endpoint)  # type: ignore[arg-type]
    return JSONResponse(body, status_code=status)
