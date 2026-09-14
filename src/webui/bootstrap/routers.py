"""Routes and the optional static mount."""
from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from webui.api import ROUTERS
from webui.config import STATIC_DIR, Settings

logger = logging.getLogger(__name__)


def include_routers(app: FastAPI, settings: Settings) -> None:
    # On Vercel, static files are served directly via vercel.json routes — skip the mount.
    # The HF Space image does not ship static/ (API-only), so the mount is conditional too.
    if STATIC_DIR.is_dir() and not settings.vercel:
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    else:
        logger.info("Static mount skipped (dir present=%s) — API-only.", STATIC_DIR.is_dir())

    for router in ROUTERS:
        app.include_router(router)
