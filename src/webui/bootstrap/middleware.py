"""HTTP middleware: CORS and the camera Permissions-Policy header."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from webui.config import Settings


async def camera_permissions_policy(request: Request, call_next):  # noqa: ANN001
    """Allow in-page camera on this origin (required for some mobile browsers)."""
    response = await call_next(request)
    response.headers["Permissions-Policy"] = "camera=(self)"
    return response


def add_middlewares(app: FastAPI, settings: Settings) -> None:
    # Order matters: the middleware added last runs outermost, so the camera header
    # is also set on CORS preflight responses.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
    app.middleware("http")(camera_permissions_policy)
