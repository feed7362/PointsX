"""FastAPI dependencies shared by the routers."""
from __future__ import annotations

from typing import Any

from fastapi import Request

from webui.config import Settings, get_settings


def settings() -> Settings:
    """Process-wide settings snapshot (see webui.config.get_settings)."""
    return get_settings()


def pipeline(request: Request) -> Any:
    """The WebuiPipeline loaded by the lifespan, or None. Never raises: form validation must win (422 before 503)."""
    return getattr(request.app.state, "pipeline", None)


def pipeline_load_error(request: Request) -> str | None:
    return getattr(request.app.state, "pipeline_load_error", None)
