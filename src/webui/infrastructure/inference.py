"""WebUI-local inference wrapper.

Composes the existing `pointsx` public surface (BodyModels + calibrate +
extract_measurements + estimate_circumferences + validate_measurements) so the
endpoint can return a `BodyMeasurements` *plus* the keypoints and calibration
needed to derive the missing 7 envelope IDs.

We deliberately do NOT use `pointsx.pipeline.MeasurementPipeline` because it
returns only the final `BodyMeasurements` and we need the intermediates here.
This module never modifies anything inside `src/pointsx/`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pointsx.calibration import calibrate
from pointsx.circumference import estimate_circumferences
from pointsx.measurements import extract_measurements
from pointsx.models import BodyModels, PoseBackend
from pointsx.pipeline import MeasurementPipeline, downscale_for_inference
from pointsx.postprocess import validate_measurements
from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints, SilhouetteMask
from webui._timing import Timings

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Everything the webui needs to build a MeasurementEnvelope."""
    body: BodyMeasurements
    front_kp: Keypoints
    side_kp: Keypoints
    front_mask: SilhouetteMask
    side_mask: SilhouetteMask
    cal: CalibrationInfo
    has_regressor: bool
    pose_backend: str


# Shared with the CLI orchestrator so seg-mask selection and regressor loading
# cannot drift between the two entry points.
_reference_point = MeasurementPipeline._keypoint_reference
_load_regressor = MeasurementPipeline._load_regression_model


class WebuiPipeline:
    """Slim, side-effect-free wrapper around pointsx's inference surface."""

    def __init__(
        self,
        pose_custom_path: str | Path | None,
        pose_coco_path: str | Path | None,
        seg_model_path: str | Path,
        regression_model_path: str | Path | None = None,
        img_size: int = 640,
        device: str = "auto",
    ) -> None:
        self.models = BodyModels(
            pose_custom_path=pose_custom_path,
            pose_coco_path=pose_coco_path,
            seg_model_path=seg_model_path,
            img_size=img_size,
            device=device,
        )
        self.regressor = None
        if regression_model_path:
            reg_path = Path(regression_model_path)
            if reg_path.exists():
                self.regressor = _load_regressor(reg_path)
                logger.info("Loaded regression model from %s", reg_path)
            else:
                logger.warning(
                    "Regression model path %s does not exist; falling back to ellipse approximation",
                    reg_path,
                )

    def warmup(self) -> dict[str, float]:
        """Run one dummy forward pass through each model.

        PyTorch + Ultralytics defer JIT compilation, NMS kernel setup, and
        memory-pool allocation to the first call. Warming up at startup
        moves that ~3-5 s/model penalty from "first user request" to
        "container boot", so the first real measurement isn't 2× slower
        than the steady-state rate.

        Sequential (not parallel) so peak memory stays bounded — important
        on free CPU tiers where parallel model init can OOM.

        Returns wall-clock timing per stage for the logs.
        """
        import time
        timings: dict[str, float] = {}

        # 640×640 mid-grey BGR image — enough pixels for the pose/seg
        # heads to run through their full code paths but cheap to compute.
        # Grey (not pure black) reduces the chance of degenerate behaviour
        # in conv layers (anti-flat-input).
        dummy = np.full((self.models.img_size, self.models.img_size, 3),
                        128, dtype=np.uint8)

        for backend in sorted(self.models.available_pose_backends()):
            t0 = time.perf_counter()
            try:
                self.models.predict_pose(dummy, view="front", pose_backend=backend)
                timings[f"pose:{backend}"] = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warmup pose:%s failed: %s", backend, exc)

        t0 = time.perf_counter()
        try:
            # Segmentation gets a synthetic reference point at image centre.
            ref = (self.models.img_size / 2.0, self.models.img_size / 2.0)
            self.models.predict_segmentation(dummy, view="front", reference_point=ref)
            timings["seg"] = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup seg failed: %s", exc)

        return timings

    def measure(
        self,
        front_img: np.ndarray,
        side_img: np.ndarray,
        height_cm: float,
        *,
        pose_backend: PoseBackend = "coco",
        timings: Timings | None = None,
    ) -> InferenceResult:
        """Run the full pose+seg+regression pipeline on a pair of images.

        When a ``Timings`` instance is supplied, each blocking phase is
        bracketed by ``with timings(phase_name):`` so the caller gets a
        wall-clock breakdown without instrumenting every line.
        """
        tm = timings if timings is not None else Timings()

        front_kp, side_kp, front_mask, side_mask = self._predict_pose_and_masks(
            front_img, side_img, pose_backend=pose_backend, timings=tm,
        )

        with tm("calibrate"):
            cal = calibrate(front_kp, side_kp, height_cm, front_mask=front_mask, side_mask=side_mask)
        with tm("extract"):
            bm = extract_measurements(front_kp, side_kp, front_mask, side_mask, cal)
        with tm("circumferences"):
            bm = estimate_circumferences(bm, self.regressor)
        with tm("validate"):
            bm = validate_measurements(bm)

        return InferenceResult(
            body=bm,
            front_kp=front_kp,
            side_kp=side_kp,
            front_mask=front_mask,
            side_mask=side_mask,
            cal=cal,
            has_regressor=self.regressor is not None,
            pose_backend=pose_backend,
        )

    def preview(
        self,
        front_img: np.ndarray,
        side_img: np.ndarray,
        *,
        pose_backend: PoseBackend = "coco",
    ) -> InferenceResult:
        """Run pose+seg only (no calibration/measurements), for debug visualizations."""
        front_kp, side_kp, front_mask, side_mask = self._predict_pose_and_masks(
            front_img, side_img, pose_backend=pose_backend
        )
        return InferenceResult(
            body=BodyMeasurements(),
            front_kp=front_kp,
            side_kp=side_kp,
            front_mask=front_mask,
            side_mask=side_mask,
            cal=CalibrationInfo(px_per_cm_front=1.0, px_per_cm_side=1.0),
            has_regressor=False,
            pose_backend=pose_backend,
        )

    def _predict_pose_and_masks(
        self,
        front_img: np.ndarray,
        side_img: np.ndarray,
        *,
        pose_backend: PoseBackend,
        timings: Timings | None = None,
    ) -> tuple[Keypoints, Keypoints, SilhouetteMask, SilhouetteMask]:
        """Common pose+seg stage shared by full measure and preview modes.

        Inputs are downscaled here (idempotent), so every caller measures on
        the same image size as production. Callers that render overlays must
        pass ``downscale_for_inference(img)`` to the visualizer too, or the
        keypoints will not line up.
        """
        tm = timings if timings is not None else Timings()
        front_img = downscale_for_inference(front_img)
        side_img = downscale_for_inference(side_img)

        with tm("pose_front"):
            front_kp = self.models.predict_pose(front_img, view="front", pose_backend=pose_backend)
        if front_kp is None:
            raise ValueError("No person detected in front image")
        with tm("pose_side"):
            side_kp = self.models.predict_pose(side_img, view="side", pose_backend=pose_backend)
        if side_kp is None:
            raise ValueError("No person detected in side image")

        with tm("seg_front"):
            front_mask = self.models.predict_segmentation(
                front_img, view="front", reference_point=_reference_point(front_kp)
            )
        if front_mask is None:
            raise ValueError("No body silhouette detected in front image")
        with tm("seg_side"):
            side_mask = self.models.predict_segmentation(
                side_img, view="side", reference_point=_reference_point(side_kp)
            )
        if side_mask is None:
            raise ValueError("No body silhouette detected in side image")

        return front_kp, side_kp, front_mask, side_mask
