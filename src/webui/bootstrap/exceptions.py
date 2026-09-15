"""Exception handlers: every error leaves the app as ``{"detail": <Ukrainian text>}``."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from webui.errors import AppError, validation_errors_to_uk


async def request_validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    message = validation_errors_to_uk(list(exc.errors()))
    return JSONResponse(status_code=422, content={"detail": message})


async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
    app.add_exception_handler(AppError, app_error_handler)
