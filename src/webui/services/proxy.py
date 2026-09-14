"""Vercel proxy mode: forward measurements to the HF Space and keep it awake."""
from __future__ import annotations

import logging
from typing import Any

from webui.errors import AppError

logger = logging.getLogger(__name__)

UploadPart = tuple[str, bytes, str]  # (filename, bytes, content_type)


async def forward_measure(
    endpoint: str,
    *,
    height_cm: float,
    sex: str,
    pose_backend: str,
    front: UploadPart,
    side: UploadPart,
) -> tuple[int, bytes, str]:
    """POST the parsed /api/measure form to ``{endpoint}/api/measure``.

    Returns the upstream ``(status_code, body, media_type)`` so the caller can
    pass it through verbatim. Timeouts → ``AppError(504)``, connection errors →
    ``AppError(502)``. The generous timeout covers slow CPU inference.
    """
    import httpx

    target_url = f"{endpoint.rstrip('/')}/api/measure"
    files = {"front": front, "side": side}
    data = {
        "height_cm": str(height_cm),
        "sex": sex,
        "pose_backend": pose_backend,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            upstream = await client.post(target_url, data=data, files=files)
        return upstream.status_code, upstream.content, upstream.headers.get("content-type", "application/json")
    except httpx.TimeoutException:
        raise AppError(504, "Час очікування відповіді від сервера інференсу вичерпано. Спробуйте ще раз.")
    except httpx.RequestError as exc:
        logger.error("Proxy request to HF Space failed: %s", exc)
        raise AppError(502, f"Не вдалося зʼєднатися з сервером інференсу: {exc}")


async def ping_space_health(endpoint: str) -> tuple[dict[str, Any], int]:
    """GET ``{endpoint}/api/health`` for the keepalive cron; returns ``(body, status_code)``.

    200 when the Space answers with ``pipeline_ready``; 202 on timeout (the
    request itself wakes a sleeping Space, which then needs minutes to boot);
    502 on connection errors or a Space that is up but not ready.
    """
    import httpx

    url = f"{endpoint.rstrip('/')}/api/health"
    try:
        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            upstream = await client.get(url)
    except httpx.TimeoutException:
        return {"target": url, "waking": True}, 202
    except httpx.RequestError as exc:
        logger.error("Keepalive ping to %s failed: %s", url, exc)
        return {"target": url, "error": str(exc)}, 502

    ready = False
    if upstream.status_code == 200:
        try:
            ready = bool(upstream.json().get("pipeline_ready"))
        except ValueError:
            pass
    return {"target": url, "status": upstream.status_code, "pipeline_ready": ready}, (200 if ready else 502)
