"""Encode pipeline debug images (pose + segmentation) as base64 PNG for the web UI."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from pointsx.keypoints import SKELETON
from pointsx.schemas import Keypoints, SilhouetteMask

from webui.inference import InferenceResult

__all__ = ["pipeline_visualizations_b64"]

_MIN_LINE_CONF = 0.22
_MIN_POINT_CONF = 0.18
_SEG_COLOR = (64, 180, 255)  # BGR
_SEG_ALPHA = 0.38


def _draw_pose(bgr: np.ndarray, kp: Keypoints) -> np.ndarray:
    out = bgr.copy()
    pts = kp.points
    conf = kp.confidence
    for a, b in SKELETON:
        ia, ib = int(a), int(b)
        if conf[ia] >= _MIN_LINE_CONF and conf[ib] >= _MIN_LINE_CONF:
            p0 = (int(round(pts[ia][0])), int(round(pts[ia][1])))
            p1 = (int(round(pts[ib][0])), int(round(pts[ib][1])))
            cv2.line(out, p0, p1, (0, 220, 130), 2, cv2.LINE_AA)
    for i in range(len(pts)):
        if conf[i] >= _MIN_POINT_CONF:
            p = (int(round(pts[i][0])), int(round(pts[i][1])))
            cv2.circle(out, p, 4, (0, 140, 255), -1, cv2.LINE_AA)
    return out


def _draw_seg_overlay(bgr: np.ndarray, mask: SilhouetteMask) -> np.ndarray:
    base = bgr.astype(np.float32)
    m = mask.mask.astype(np.float32)
    col = np.array(_SEG_COLOR, dtype=np.float32).reshape(1, 1, 3)
    overlay = base.copy()
    for c in range(3):
        overlay[:, :, c] = np.where(
            m > 0.5,
            base[:, :, c] * (1.0 - _SEG_ALPHA) + col[0, 0, c] * _SEG_ALPHA,
            base[:, :, c],
        )
    return np.clip(overlay, 0, 255).astype(np.uint8)


def _png_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


def pipeline_visualizations_b64(
    front_bgr: np.ndarray,
    side_bgr: np.ndarray,
    result: InferenceResult,
) -> dict[str, str]:
    """Four PNGs (base64): pose-only and seg-only per view."""
    fm = result.front_mask
    sm = result.side_mask
    fk = result.front_kp
    sk = result.side_kp

    return {
        "viz_front_pose_png_b64": _png_b64(_draw_pose(front_bgr, fk)),
        "viz_front_seg_png_b64": _png_b64(_draw_seg_overlay(front_bgr, fm)),
        "viz_side_pose_png_b64": _png_b64(_draw_pose(side_bgr, sk)),
        "viz_side_seg_png_b64": _png_b64(_draw_seg_overlay(side_bgr, sm)),
    }
