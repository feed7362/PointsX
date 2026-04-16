"""Top-level pipeline orchestrator for body measurement extraction."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from pointsx.calibration import calibrate
from pointsx.circumference import estimate_circumferences
from pointsx.measurements import extract_measurements
from pointsx.models import BodyModels
from pointsx.postprocess import validate_measurements
from pointsx.schemas import BodyMeasurements, Keypoints

logger = logging.getLogger(__name__)


class MeasurementPipeline:
    """Extract body measurements from front + side photos.

    Usage:
        pipeline = MeasurementPipeline()
        result = pipeline("front.jpg", "side.jpg", height_cm=175.0)
        print(result.to_dict())
    """

    def __init__(
        self,
        pose_model_path: str | Path = "models/yolo11n-pose.pt",
        seg_model_path: str | Path = "models/yolo11n-seg.pt",
        regression_model_path: str | Path | None = None,
        img_size: int = 640,
        device: str = "auto",
    ):
        self._models = BodyModels(
            pose_model_path=pose_model_path,
            seg_model_path=seg_model_path,
            img_size=img_size,
            device=device,
        )
        self._regression_model = None
        if regression_model_path and Path(regression_model_path).exists():
            self._regression_model = self._load_regression_model(regression_model_path)
            logger.info("Loaded regression model from %s", regression_model_path)

    def __call__(
        self,
        front_image: str | Path | np.ndarray,
        side_image: str | Path | np.ndarray,
        height_cm: float,
    ) -> BodyMeasurements:
        """Run the full measurement pipeline.

        Args:
            front_image: Front-view photo (path or BGR ndarray).
            side_image: Side/profile-view photo (path or BGR ndarray).
            height_cm: Known height of the person in centimeters.

        Returns:
            BodyMeasurements with all extracted values.
        """
        front_img = self._load_image(front_image)
        side_img = self._load_image(side_image)

        logger.info("Running pose estimation...")
        front_kp = self._models.predict_pose(front_img, view="front")
        side_kp = self._models.predict_pose(side_img, view="side")

        if front_kp is None:
            raise ValueError("No person detected in front image")
        if side_kp is None:
            raise ValueError("No person detected in side image")

        logger.info("Running segmentation...")
        front_ref = self._keypoint_reference(front_kp)
        side_ref = self._keypoint_reference(side_kp)
        front_mask = self._models.predict_segmentation(front_img, view="front", reference_point=front_ref)
        side_mask = self._models.predict_segmentation(side_img, view="side", reference_point=side_ref)

        if front_mask is None:
            raise ValueError("No segmentation mask for front image")
        if side_mask is None:
            raise ValueError("No segmentation mask for side image")

        logger.info("Calibrating...")
        cal = calibrate(front_kp, side_kp, height_cm)

        logger.info("Extracting measurements...")
        measurements = extract_measurements(
            front_kp, side_kp, front_mask, side_mask, cal
        )

        logger.info("Estimating circumferences...")
        measurements = estimate_circumferences(measurements, self._regression_model)

        logger.info("Validating...")
        measurements = validate_measurements(measurements)

        return measurements

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> np.ndarray:
        """Load image from path or pass through ndarray."""
        if isinstance(image, np.ndarray):
            return image
        path = str(image)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return img

    @staticmethod
    def _load_regression_model(path: str | Path):
        """Load trained regression model."""
        import torch
        from pointsx.regression.model import CircumferenceRegressor

        model = CircumferenceRegressor()
        model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
        model.eval()
        return model

    @staticmethod
    def _keypoint_reference(kp: Keypoints) -> tuple[float, float]:
        """Estimate subject center from valid keypoints."""
        valid = kp.confidence >= 0.3
        if np.any(valid):
            center = kp.points[valid].mean(axis=0)
        else:
            center = kp.points.mean(axis=0)
        return float(center[0]), float(center[1])
