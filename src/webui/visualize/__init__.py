"""Encode pipeline debug images (pose + segmentation + measurement overlays) as base64 PNG for the web UI.

    primitives.py     fonts, Unicode text, pose skeleton, segmentation tint, PNG encoding, values table
    mask_spans.py     horizontal silhouette spans used to place width lines
    measure_lines.py  one function per measurement overlay
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from webui.visualize.measure_lines import _draw_measure_lines
from webui.visualize.primitives import _draw_body_measurements_table, _draw_pose, _draw_seg_overlay, _png_b64

if TYPE_CHECKING:
    from webui.infrastructure.inference import InferenceResult

__all__ = ["pipeline_visualizations_b64"]


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
        "viz_front_measures_png_b64": _png_b64(_draw_measure_lines(front_bgr, fk, fm, "front")),
        "viz_side_pose_png_b64": _png_b64(_draw_pose(side_bgr, sk)),
        "viz_side_seg_png_b64": _png_b64(_draw_seg_overlay(side_bgr, sm)),
        "viz_side_measures_png_b64": _png_b64(_draw_measure_lines(side_bgr, sk, sm, "side")),
        "viz_body_measures_png_b64": _png_b64(_draw_body_measurements_table(result.body)),
    }
