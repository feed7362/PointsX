"""Local measurement flow: check pipeline → decode → downscale → save pair → run → envelope → archive.

FastAPI-free: the route reads the uploads and passes a scheduler for background
work (``BackgroundTasks.add_task``). Heavy imports (OpenCV via pointsx.pipeline,
visualization, envelope) stay inside functions so the Vercel function can import
this module without them.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from webui.errors import AppError, pipeline_value_error_detail
from webui.schemas import CaptureInfo, CaptureQuality, MeasurementEnvelope, PipelineInfo, SubjectInfo
from webui.services.dataset_capture import save_capture_pair
from webui.services.uploads import decode_upload

logger = logging.getLogger(__name__)

Sex = Literal["male", "female", "other"]
PoseBackend = Literal["custom", "coco"]
ScheduleTask = Callable[[Callable[[], None]], None]


@dataclass(frozen=True)
class PhotoUpload:
    data: bytes
    content_type: str | None


def build_visualization_only_envelope(
    *,
    preview_result: Any,
    height_cm: float,
    sex: Sex,
    pose_backend: PoseBackend,
    warning: str,
    front_bgr: Any,
    side_bgr: Any,
) -> MeasurementEnvelope:
    """Return a valid envelope with debug visualizations when calibration fails."""
    from webui.visualize import pipeline_visualizations_b64

    derived: dict[str, Any] = {}
    try:
        derived = pipeline_visualizations_b64(front_bgr, side_bgr, preview_result)
    except Exception:
        logger.exception("Failed to generate visualization-only debug images")

    return MeasurementEnvelope(
        schema="pointsx.measurement.envelope",
        schema_version=2,
        request_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        pipeline=PipelineInfo(
            source="mediapipe",
            model_version="visualization-only",
            unit_system="metric",
            pose_backend=pose_backend,
        ),
        subject=SubjectInfo(
            height_cm=height_cm,
            sex=sex,
            posture_flags=[],
        ),
        capture=CaptureInfo(
            front=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
            side=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
        ),
        measurements=[],
        derived=derived,
        warnings=[warning],
    )


def with_warning(envelope: MeasurementEnvelope, warning: str | None) -> MeasurementEnvelope:
    if not warning:
        return envelope
    return envelope.model_copy(update={"warnings": [*envelope.warnings, warning]})


def require_pipeline(pipeline: Any, load_error: str | None, pose_backend: PoseBackend) -> Any:
    """The loaded pipeline, or ``AppError(503)`` if it or the requested pose backend is unavailable."""
    if pipeline is None:
        err = load_error or "pipeline not initialised"
        raise AppError(
            503,
            "Неможливо виконати замір: моделі не завантажені на сервері. "
            "Перевірте шляхи до ваг і журнал сервера. "
            f"Технічні деталі: {err}",
        )
    avail = pipeline.models.available_pose_backends()
    if pose_backend not in avail:
        need = "pose-cus.pt (16 точок)" if pose_backend == "custom" else "yolo26-pose.pt (COCO 17)"
        raise AppError(
            503,
            f"Обрана модель пози ({pose_backend}) недоступна: відсутній файл ваг для {need}. "
            "Перевірте POINTSX_POSE_MODEL_CUSTOM / POINTSX_POSE_MODEL_COCO або оберіть інший режим.",
        )
    return pipeline


def archive_measurement(
    envelope: MeasurementEnvelope,
    *,
    outcome: str,
    request_id: str,
    front: PhotoUpload,
    side: PhotoUpload,
    height_cm: float,
    sex: Sex,
    pose_backend: PoseBackend,
    schedule_background: ScheduleTask,
) -> None:
    """Persist photos + envelope to whichever store(s) are configured.

    1. Local filesystem (LOCAL_DATA_DIR) — inline, ~50 ms.
    2. S3-compatible bucket — scheduled as background work, runs after the
       response is sent so the bucket round-trip stays off the critical path.
    Failures are logged and never affect the response.
    """
    try:
        from webui import storage
        args = dict(
            request_id=request_id,
            front_bytes=front.data,
            front_content_type=(front.content_type or "image/jpeg"),
            side_bytes=side.data,
            side_content_type=(side.content_type or "image/jpeg"),
            envelope_json=envelope.model_dump(mode="json", by_alias=True),
            metadata={
                "height_cm": str(height_cm),
                "sex": sex,
                "pose_backend": pose_backend,
                "outcome": outcome,
                "created_at": envelope.created_at,
            },
        )

        local_ok = storage.archive_measurement_local(**args)
        s3_scheduled = False
        if storage.is_enabled():
            def _bg_s3_upload(_args=args, _outcome=outcome, _rid=request_id):
                try:
                    ok = storage.archive_measurement(**_args)
                    logger.warning("Archive (bg): outcome=%s request_id=%s s3_ok=%s", _outcome, _rid, ok)
                except Exception:  # noqa: BLE001
                    logger.exception("Archive (bg) raised for request_id=%s", _rid)
            schedule_background(_bg_s3_upload)
            s3_scheduled = True

        logger.warning(
            "Archive: outcome=%s request_id=%s local=%s s3_scheduled=%s",
            outcome, request_id, local_ok, s3_scheduled,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Archive raised — measurement response is unaffected.")


async def run_measurement(
    *,
    pipeline: Any,
    pipeline_load_error: str | None,
    dataset_dir: Path,
    front: PhotoUpload,
    side: PhotoUpload,
    height_cm: float,
    sex: Sex,
    pose_backend: PoseBackend,
    with_viz: bool,
    schedule_background: ScheduleTask,
) -> MeasurementEnvelope:
    """Run pose + seg + (optional) regression on the photo pair and build the envelope.

    Raises ``AppError`` 503 (models/backend unavailable), 400 (bad upload or a
    pipeline ValueError), 500 (unexpected pipeline failure). A calibration
    failure is not an error: it returns a visualization-only envelope.
    """
    pipeline = require_pipeline(pipeline, pipeline_load_error, pose_backend)

    from webui._timing import Timings
    tm = Timings()

    with tm("decode"):
        front_img = decode_upload(front.data, front.content_type, "front")
        side_img = decode_upload(side.data, side.content_type, "side")

    # The pipeline downscales internally too (idempotent); doing it here keeps the
    # overlay images in the same pixel frame as the keypoints and masks.
    from pointsx.pipeline import downscale_for_inference

    with tm("downscale"):
        front_img = downscale_for_inference(front_img)
        side_img = downscale_for_inference(side_img)

    _, dataset_save_warning = await save_capture_pair(dataset_dir, front.data, side.data)

    # One request_id per call, used for both the envelope and the archive prefix.
    request_id = str(uuid.uuid4())

    def _archive(envelope_obj: MeasurementEnvelope, *, outcome: str) -> None:
        archive_measurement(
            envelope_obj, outcome=outcome, request_id=request_id, front=front, side=side,
            height_cm=height_cm, sex=sex, pose_backend=pose_backend, schedule_background=schedule_background,
        )

    try:
        # Pipeline is sync (PyTorch + numpy). asyncio.to_thread keeps the
        # event loop responsive so /api/tts and health checks can still
        # answer while this request crunches pose+seg.
        result = await asyncio.to_thread(
            pipeline.measure,
            front_img, side_img, height_cm,
            pose_backend=pose_backend, timings=tm,
        )
    except ValueError as exc:
        err_text = str(exc).strip()
        if err_text.startswith("Cannot calibrate "):
            preview = await asyncio.to_thread(
                pipeline.preview,
                front_img, side_img, pose_backend=pose_backend,
            )
            envelope = build_visualization_only_envelope(
                preview_result=preview,
                height_cm=height_cm,
                sex=sex,
                pose_backend=pose_backend,
                warning=pipeline_value_error_detail(err_text),
                front_bgr=front_img,
                side_bgr=side_img,
            )
            envelope.request_id = request_id
            envelope = with_warning(envelope, dataset_save_warning)
            _archive(envelope, outcome="calibration_failed")
            return envelope
        raise AppError(400, pipeline_value_error_detail(err_text)) from exc
    except Exception as exc:
        logger.exception("Pipeline failed")
        raise AppError(500, f"Помилка під час обчислення мірок: {exc}") from exc

    from webui.envelope import body_to_envelope

    with tm("envelope"):
        envelope = body_to_envelope(
            result=result,
            subject_height_cm=height_cm,
            sex=sex,
            request_id=request_id,
            front_bgr=front_img if with_viz else None,
            side_bgr=side_img if with_viz else None,
        )
    envelope = with_warning(envelope, dataset_save_warning)

    logger.warning("Measurement timings — %s", tm.format())

    _archive(envelope, outcome="ok")

    return envelope
