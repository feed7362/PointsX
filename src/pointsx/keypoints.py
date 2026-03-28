"""Keypoint definitions and geometric helpers for the 16-point LV-MHP-v2 skeleton."""

from __future__ import annotations

from enum import IntEnum

import numpy as np


class KP(IntEnum):
    """Keypoint indices for the 16-point LV-MHP-v2 body skeleton."""

    RIGHT_ANKLE = 0
    RIGHT_KNEE = 1
    RIGHT_HIP = 2
    LEFT_HIP = 3
    LEFT_KNEE = 4
    LEFT_ANKLE = 5
    PELVIS = 6
    THORAX = 7
    UPPER_NECK = 8
    HEAD_TOP = 9
    RIGHT_WRIST = 10
    RIGHT_ELBOW = 11
    RIGHT_SHOULDER = 12
    LEFT_SHOULDER = 13
    LEFT_ELBOW = 14
    LEFT_WRIST = 15


NUM_KEYPOINTS = 16

# Skeleton connections for visualization
SKELETON = [
    (KP.RIGHT_ANKLE, KP.RIGHT_KNEE),
    (KP.RIGHT_KNEE, KP.RIGHT_HIP),
    (KP.LEFT_HIP, KP.LEFT_KNEE),
    (KP.LEFT_KNEE, KP.LEFT_ANKLE),
    (KP.RIGHT_HIP, KP.PELVIS),
    (KP.LEFT_HIP, KP.PELVIS),
    (KP.PELVIS, KP.THORAX),
    (KP.THORAX, KP.UPPER_NECK),
    (KP.UPPER_NECK, KP.HEAD_TOP),
    (KP.THORAX, KP.RIGHT_SHOULDER),
    (KP.RIGHT_SHOULDER, KP.RIGHT_ELBOW),
    (KP.RIGHT_ELBOW, KP.RIGHT_WRIST),
    (KP.THORAX, KP.LEFT_SHOULDER),
    (KP.LEFT_SHOULDER, KP.LEFT_ELBOW),
    (KP.LEFT_ELBOW, KP.LEFT_WRIST),
]

# Left-right flip pairs for augmentation
FLIP_IDX = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

# Keypoint names (human-readable)
KP_NAMES = [
    "Right Ankle", "Right Knee", "Right Hip",
    "Left Hip", "Left Knee", "Left Ankle",
    "Pelvis", "Thorax", "Upper Neck", "Head Top",
    "Right Wrist", "Right Elbow", "Right Shoulder",
    "Left Shoulder", "Left Elbow", "Left Wrist",
]

# Minimum confidence to consider a keypoint valid
MIN_CONFIDENCE = 0.3


def point(kps: np.ndarray, idx: KP) -> np.ndarray:
    """Get (x, y) for a single keypoint."""
    return kps[int(idx)]


def midpoint(kps: np.ndarray, a: KP, b: KP) -> np.ndarray:
    """Get the midpoint between two keypoints."""
    return (kps[int(a)] + kps[int(b)]) / 2.0


def distance(kps: np.ndarray, a: KP, b: KP) -> float:
    """Euclidean distance between two keypoints in pixels."""
    return float(np.linalg.norm(kps[int(a)] - kps[int(b)]))


def is_valid(confidence: np.ndarray, *indices: KP) -> bool:
    """Check if all specified keypoints have sufficient confidence."""
    return all(confidence[int(i)] >= MIN_CONFIDENCE for i in indices)


def interpolate_y(kps: np.ndarray, top: KP, bottom: KP, ratio: float) -> float:
    """Get the y-coordinate at a given ratio between two keypoints.

    ratio=0.0 → top keypoint y, ratio=1.0 → bottom keypoint y.
    """
    y_top = kps[int(top), 1]
    y_bottom = kps[int(bottom), 1]
    return y_top + ratio * (y_bottom - y_top)
