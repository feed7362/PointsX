"""Data structures for the PointsX body measurement pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Keypoints:
    """Pose keypoints for a single person in one view."""

    points: np.ndarray  # (16, 2) pixel coordinates [x, y]
    confidence: np.ndarray  # (16,) per-keypoint confidence 0..1
    view: str  # "front" or "side"


@dataclass
class SilhouetteMask:
    """Binary segmentation mask for a single person."""

    mask: np.ndarray  # (H, W) bool array
    contour: np.ndarray | None  # (N, 2) outer contour points [x, y], or None
    view: str  # "front" or "side"


@dataclass
class CalibrationInfo:
    """Pixel-to-cm conversion factors for each view."""

    px_per_cm_front: float
    px_per_cm_side: float


@dataclass
class BodyMeasurements:
    """All extracted body measurements in centimeters."""

    # Height
    height_cm: float | None = None

    # Head
    head_width_front_cm: float | None = None
    head_depth_side_cm: float | None = None

    # Neck
    neck_width_front_cm: float | None = None
    neck_width_side_cm: float | None = None
    neck_circumference_cm: float | None = None

    # Shoulders
    shoulder_width_cm: float | None = None

    # Chest / torso
    torso_width_front_cm: float | None = None
    torso_width_side_cm: float | None = None

    # Waist
    waist_width_front_cm: float | None = None
    waist_width_side_cm: float | None = None
    waist_circumference_cm: float | None = None

    # Hips
    hip_width_front_cm: float | None = None
    hip_width_side_cm: float | None = None
    hip_circumference_cm: float | None = None

    # Thigh
    thigh_width_front_cm: float | None = None
    thigh_width_side_cm: float | None = None
    thigh_circumference_cm: float | None = None

    # Calf
    calf_width_front_cm: float | None = None
    calf_width_side_cm: float | None = None
    calf_circumference_cm: float | None = None

    # Wrist
    wrist_width_front_cm: float | None = None
    wrist_width_side_cm: float | None = None
    wrist_circumference_cm: float | None = None

    # Leg lengths
    leg_length_outer_cm: float | None = None
    leg_length_inner_cm: float | None = None

    # Arm lengths
    arm_length_cm: float | None = None
    arm_length_from_neck_cm: float | None = None

    # Confidence scores per measurement (0.0 - 1.0)
    confidence: dict[str, float] = field(default_factory=dict)

    # Validation warnings
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return all measurements as a flat dict (excluding None values)."""
        result = {}
        for fname in self.__dataclass_fields__:
            if fname in ("confidence", "warnings"):
                continue
            val = getattr(self, fname)
            if val is not None:
                result[fname] = round(val, 1)
        return result
