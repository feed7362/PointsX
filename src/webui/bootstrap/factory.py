"""Application factory — the only place the FastAPI app is assembled."""
from __future__ import annotations

from fastapi import FastAPI

from webui.bootstrap.exceptions import register_exception_handlers
from webui.bootstrap.lifespan import lifespan
from webui.bootstrap.middleware import add_middlewares
from webui.bootstrap.routers import include_routers
from webui.config import get_settings


def create_app(use_lifespan: bool = True) -> FastAPI:
    """Build the app. ``use_lifespan=False`` skips model loading (tests, tooling)."""
    settings = get_settings()
    app = FastAPI(title="FitMeasure AI WebUI", version="0.3.0", lifespan=lifespan if use_lifespan else None)
    include_routers(app, settings)
    add_middlewares(app, settings)
    register_exception_handlers(app)
    return app
