"""POST /api/measure (real or proxied to the HF Space) and /api/measure/mock."""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Request, UploadFile
from fastapi.responses import Response

from webui.api import dependencies as deps
from webui.config import Settings
from webui.schemas import MeasurementEnvelope
from webui.services.measurement import PhotoUpload, run_measurement
from webui.services.mock import build_mock_measurement_envelope
from webui.services.proxy import forward_measure

router = APIRouter()


@router.post("/api/measure", response_model=MeasurementEnvelope)
async def measure(
    request: Request,
    background_tasks: BackgroundTasks,
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
    pose_backend: Literal["custom", "coco"] = Form("coco"),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
    settings: Settings = Depends(deps.settings),
    pipeline: Any = Depends(deps.pipeline),
    load_error: str | None = Depends(deps.pipeline_load_error),
) -> Any:
    """Run pose + seg + (optional) regression on the supplied photo pair.

    In Vercel proxy mode, the request is forwarded transparently to the
    HuggingFace Space inference backend (POINTSX_INFERENCE_ENDPOINT).

    Returns a `MeasurementEnvelope` with up to 18 canonical body measurements.
    Measurements that the pipeline cannot derive are simply omitted; the
    frontend size engine tolerates a small number of missing values.
    """
    if settings.proxy_mode:
        front_part = (front.filename or "front.jpg", await front.read(), front.content_type or "image/jpeg")
        side_part = (side.filename or "side.jpg", await side.read(), side.content_type or "image/jpeg")
        status, content, media_type = await forward_measure(
            settings.inference_endpoint,  # type: ignore[arg-type]
            height_cm=height_cm,
            sex=sex,
            pose_backend=pose_backend,
            front=front_part,
            side=side_part,
        )
        return Response(content=content, status_code=status, media_type=media_type)

    # `with_viz` query param gates the heavy base64-PNG render. Default
    # ON for backward-compat; ?with_viz=0 shaves ~1-2 s off the response.
    with_viz = request.query_params.get("with_viz", "1").strip().lower() not in ("0", "false", "no", "off")
    return await run_measurement(
        pipeline=pipeline,
        pipeline_load_error=load_error,
        dataset_dir=settings.dataset_dir,
        front=PhotoUpload(await front.read(), front.content_type),
        side=PhotoUpload(await side.read(), side.content_type),
        height_cm=height_cm,
        sex=sex,
        pose_backend=pose_backend,
        with_viz=with_viz,
        schedule_background=background_tasks.add_task,
    )


@router.post("/api/measure/mock", response_model=MeasurementEnvelope)
async def measure_mock(
    height_cm: float = Form(..., ge=100, le=250),
    sex: Literal["male", "female", "other"] = Form(...),
) -> MeasurementEnvelope:
    """Same JSON contract as `/api/measure`, without images or ML (UI test button)."""
    return build_mock_measurement_envelope(height_cm, sex)
