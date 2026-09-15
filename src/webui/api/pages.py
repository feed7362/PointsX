"""Static HTML pages served by the app itself (locally; Vercel serves them via vercel.json routes)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from webui.config import STATIC_DIR

router = APIRouter()


@router.get("/")
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


@router.get("/dataset.html")
async def dataset() -> FileResponse:
    dataset_path = STATIC_DIR / "dataset.html"
    if not dataset_path.is_file():
        raise HTTPException(status_code=404, detail="Missing dataset page")
    return FileResponse(dataset_path)
