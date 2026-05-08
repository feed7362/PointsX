"""Map Ultralytics COCO-17 person keypoints to PointsX 16-point LV-MHP layout."""

from __future__ import annotations

import numpy as np

from pointsx.keypoints import KP
from pointsx.schemas import Keypoints

# COCO person keypoint indices (Ultralytics / OpenPose convention)
COCO_NOSE = 0
COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_ELBOW = 7
COCO_RIGHT_ELBOW = 8
COCO_LEFT_WRIST = 9
COCO_RIGHT_WRIST = 10
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12
COCO_LEFT_KNEE = 13
COCO_RIGHT_KNEE = 14
COCO_LEFT_ANKLE = 15
COCO_RIGHT_ANKLE = 16

# Place upper_neck slightly lower (closer to thorax) and keep head_top behavior unchanged.
# Image y decreases upward.
_UPPER_NECK_NOSE_BLEND = 0.27  # 10% lower vs previous 0.30 neck offset
_HEAD_TOP_NECK_REFERENCE_BLEND = 0.30  # preserve current head_top vertical behavior
_HEAD_TOP_K_FRONT = 0.75  # keep front head_top lower
_HEAD_TOP_K_SIDE = 1.10  # restore side head_top to previous behavior


def coco17_to_lv_mhp16(xy: np.ndarray, conf: np.ndarray, view: str) -> Keypoints:
    """Convert a single person's COCO-17 pose to PointsX 16-keypoint order.

    Args:
        xy: (17, 2) pixel coordinates
        conf: (17,) confidence in [0, 1]
        view: "front" or "side" for Keypoints metadata

    Raises:
        ValueError: if keypoint count is not 17
    """
    if xy.shape[0] != 17 or conf.shape[0] != 17:
        raise ValueError(
            f"COCO pose expected 17 keypoints, got xy.shape={xy.shape}, conf.shape={conf.shape}"
        )

    out_xy = np.zeros((16, 2), dtype=np.float64)
    out_cf = np.zeros(16, dtype=np.float64)

    # Direct limb mapping (COCO → LV-MHP indices)
    out_xy[KP.RIGHT_ANKLE] = xy[COCO_RIGHT_ANKLE]
    out_cf[KP.RIGHT_ANKLE] = conf[COCO_RIGHT_ANKLE]
    out_xy[KP.RIGHT_KNEE] = xy[COCO_RIGHT_KNEE]
    out_cf[KP.RIGHT_KNEE] = conf[COCO_RIGHT_KNEE]
    out_xy[KP.RIGHT_HIP] = xy[COCO_RIGHT_HIP]
    out_cf[KP.RIGHT_HIP] = conf[COCO_RIGHT_HIP]
    out_xy[KP.LEFT_HIP] = xy[COCO_LEFT_HIP]
    out_cf[KP.LEFT_HIP] = conf[COCO_LEFT_HIP]
    out_xy[KP.LEFT_KNEE] = xy[COCO_LEFT_KNEE]
    out_cf[KP.LEFT_KNEE] = conf[COCO_LEFT_KNEE]
    out_xy[KP.LEFT_ANKLE] = xy[COCO_LEFT_ANKLE]
    out_cf[KP.LEFT_ANKLE] = conf[COCO_LEFT_ANKLE]

    out_xy[KP.RIGHT_WRIST] = xy[COCO_RIGHT_WRIST]
    out_cf[KP.RIGHT_WRIST] = conf[COCO_RIGHT_WRIST]
    out_xy[KP.RIGHT_ELBOW] = xy[COCO_RIGHT_ELBOW]
    out_cf[KP.RIGHT_ELBOW] = conf[COCO_RIGHT_ELBOW]
    out_xy[KP.RIGHT_SHOULDER] = xy[COCO_RIGHT_SHOULDER]
    out_cf[KP.RIGHT_SHOULDER] = conf[COCO_RIGHT_SHOULDER]
    out_xy[KP.LEFT_SHOULDER] = xy[COCO_LEFT_SHOULDER]
    out_cf[KP.LEFT_SHOULDER] = conf[COCO_LEFT_SHOULDER]
    out_xy[KP.LEFT_ELBOW] = xy[COCO_LEFT_ELBOW]
    out_cf[KP.LEFT_ELBOW] = conf[COCO_LEFT_ELBOW]
    out_xy[KP.LEFT_WRIST] = xy[COCO_LEFT_WRIST]
    out_cf[KP.LEFT_WRIST] = conf[COCO_LEFT_WRIST]

    # Derived: pelvis, thorax, neck proxy, head top
    l_hip, r_hip = xy[COCO_LEFT_HIP], xy[COCO_RIGHT_HIP]
    c_lh, c_rh = conf[COCO_LEFT_HIP], conf[COCO_RIGHT_HIP]
    out_xy[KP.PELVIS] = (l_hip + r_hip) * 0.5
    out_cf[KP.PELVIS] = float(min(c_lh, c_rh))

    l_sh, r_sh = xy[COCO_LEFT_SHOULDER], xy[COCO_RIGHT_SHOULDER]
    c_ls, c_rs = conf[COCO_LEFT_SHOULDER], conf[COCO_RIGHT_SHOULDER]
    thorax = (l_sh + r_sh) * 0.5
    out_xy[KP.THORAX] = thorax
    out_cf[KP.THORAX] = float(min(c_ls, c_rs))

    nose = xy[COCO_NOSE]
    c_n = conf[COCO_NOSE]
    # Keep torso centerline for x; use nose only to refine vertical placement.
    upper_neck = np.array(
        [thorax[0], thorax[1] + _UPPER_NECK_NOSE_BLEND * (nose[1] - thorax[1])],
        dtype=np.float64,
    )
    c_neck = min(out_cf[KP.THORAX], c_n)
    out_xy[KP.UPPER_NECK] = upper_neck
    out_cf[KP.UPPER_NECK] = float(c_neck)

    # Extend vertically using the previous neck reference to keep head_top unchanged.
    upper_neck_for_head_top = np.array(
        [thorax[0], thorax[1] + _HEAD_TOP_NECK_REFERENCE_BLEND * (nose[1] - thorax[1])],
        dtype=np.float64,
    )
    head_top_k = _HEAD_TOP_K_SIDE if view == "side" else _HEAD_TOP_K_FRONT
    y_head_top = nose[1] + head_top_k * (nose[1] - upper_neck_for_head_top[1])
    out_xy[KP.HEAD_TOP] = np.array([upper_neck[0], y_head_top], dtype=np.float64)
    out_cf[KP.HEAD_TOP] = float(min(c_n, c_neck))

    return Keypoints(
        points=out_xy.astype(np.float32),
        confidence=out_cf.astype(np.float32),
        view=view,
        nose_xy=nose.astype(np.float32),
        nose_conf=float(c_n),
    )
