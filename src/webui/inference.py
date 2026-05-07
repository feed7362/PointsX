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
from pointsx.keypoints import MIN_CONFIDENCE
from pointsx.circumference import estimate_circumferences
from pointsx.measurements import extract_measurements
from pointsx.models import BodyModels, PoseBackend
from pointsx.postprocess import validate_measurements
from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints, SilhouetteMask

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


def _reference_point(kp: Keypoints) -> tuple[float, float]:
    """Subject center for seg-mask selection (mirrors MeasurementPipeline)."""
    valid = kp.confidence >= MIN_CONFIDENCE
    if np.any(valid):
        center = kp.points[valid].mean(axis=0)
    else:
        center = kp.points.mean(axis=0)
    return float(center[0]), float(center[1])


def _load_regressor(path: str | Path):
    """Load the trained CircumferenceRegressor from a .pt file."""
    import torch

    from pointsx.regression.model import CircumferenceRegressor

    model = CircumferenceRegressor()
    model.load_state_dict(
        torch.load(str(path), map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


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

    def measure(
        self,
        front_img: np.ndarray,
        side_img: np.ndarray,
        height_cm: float,
        *,
        pose_backend: PoseBackend = "custom",
    ) -> InferenceResult:
        """Run the full pose+seg+regression pipeline on a pair of images."""
        front_kp = self.models.predict_pose(front_img, view="front", pose_backend=pose_backend)
        if front_kp is None:
            raise ValueError("No person detected in front image")
        side_kp = self.models.predict_pose(side_img, view="side", pose_backend=pose_backend)
        if side_kp is None:
            raise ValueError("No person detected in side image")

        front_mask = self.models.predict_segmentation(
            front_img, view="front", reference_point=_reference_point(front_kp)
        )
        if front_mask is None:
            raise ValueError("No body silhouette detected in front image")
        side_mask = self.models.predict_segmentation(
            side_img, view="side", reference_point=_reference_point(side_kp)
        )
        if side_mask is None:
            raise ValueError("No body silhouette detected in side image")

        cal = calibrate(front_kp, side_kp, height_cm)
        bm = extract_measurements(front_kp, side_kp, front_mask, side_mask, cal)
        bm = estimate_circumferences(bm, regression_model=None)
        bm = validate_measurements(bm)

        return InferenceResult(
            body=bm,
            front_kp=front_kp,
            side_kp=side_kp,
            front_mask=front_mask,
            side_mask=side_mask,
            cal=cal,
            has_regressor=False,
            pose_backend=pose_backend,
        )
