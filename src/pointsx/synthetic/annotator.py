"""Project 3D body landmarks to 2D image coordinates and write YOLO labels.

Handles:
  - 3D → 2D projection using Blender camera matrices
  - Visibility classification (2=visible, 1=occluded, 0=out-of-frame)
  - Depth-buffer occlusion check (uses z-buffer passed from Blender)
  - YOLO pose label format output
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Minimum visible keypoints required to write a label
MIN_VISIBLE_KP = 5

# Bounding-box padding factor around the visible keypoints
BBOX_PAD = 0.15  # 15% of the tight bbox extent
BBOX_MIN_PX = 50  # absolute minimum side length in pixels


def classify_visibility(
        coords_px: np.ndarray,
        depth: np.ndarray,
        depth_buffer: np.ndarray | None,
        img_w: int,
        img_h: int,
        occlusion_threshold: float = 0.02,
) -> np.ndarray:
    """Classify each landmark as visible (2), occluded (1), or off-frame (0).

    Args:
        coords_px:          (25, 2) pixel coords
        depth:              (25,)   camera depths
        depth_buffer:       (H, W)  float32 z-buffer from Blender Z-pass, or None
        img_w, img_h:       image size
        occlusion_threshold: metres — landmark is occluded if scene depth is
                             this much closer than the landmark depth

    Returns:
        visibility: (25,) uint8 with values 0 / 1 / 2
    """
    vis = np.full(len(coords_px), 2, dtype=np.uint8)

    for i, (xy, d) in enumerate(zip(coords_px, depth)):
        x, y = xy
        # Off-frame or behind camera
        if d <= 0 or x < 0 or x >= img_w or y < 0 or y >= img_h:
            vis[i] = 0
            continue

        # Depth buffer occlusion check
        if depth_buffer is not None:
            xi, yi = int(round(x)), int(round(y))
            xi = max(0, min(img_w - 1, xi))
            yi = max(0, min(img_h - 1, yi))
            scene_depth = float(depth_buffer[yi, xi])
            # scene_depth is distance from camera; d is also distance
            if scene_depth < d - occlusion_threshold:
                vis[i] = 1  # occluded

    return vis


def build_yolo_label(
        coords_px: np.ndarray,
        visibility: np.ndarray,
        img_w: int,
        img_h: int,
) -> str | None:
    """Build YOLO pose label string for a single image.

    Format:
        class cx cy w h  x0 y0 v0  x1 y1 v1 ... x24 y24 v24
    All coordinates are normalised to [0, 1].

    Returns None if fewer than MIN_VISIBLE_KP keypoints are visible (v >= 1).
    """
    visible_mask = visibility >= 1
    if visible_mask.sum() < MIN_VISIBLE_KP:
        logger.debug("Only %d visible keypoints — skipping", visible_mask.sum())
        return None

    # Tight bbox from visible keypoints
    vis_pts = coords_px[visible_mask]
    x_min = vis_pts[:, 0].min()
    x_max = vis_pts[:, 0].max()
    y_min = vis_pts[:, 1].min()
    y_max = vis_pts[:, 1].max()

    # Padding
    pad_x = max((x_max - x_min) * BBOX_PAD, BBOX_MIN_PX / 2)
    pad_y = max((y_max - y_min) * BBOX_PAD, BBOX_MIN_PX / 2)
    x_min = max(0, x_min - pad_x)
    x_max = min(img_w - 1, x_max + pad_x)
    y_min = max(0, y_min - pad_y)
    y_max = min(img_h - 1, y_max + pad_y)

    # Normalise bbox to [0, 1]
    cx = ((x_min + x_max) / 2) / img_w
    cy = ((y_min + y_max) / 2) / img_h
    bw = (x_max - x_min) / img_w
    bh = (y_max - y_min) / img_h

    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]

    for i, (xy, v) in enumerate(zip(coords_px, visibility)):
        x_n = float(np.clip(xy[0] / img_w, 0.0, 1.0))
        y_n = float(np.clip(xy[1] / img_h, 0.0, 1.0))
        parts.append(f"{x_n:.6f} {y_n:.6f} {int(v)}")

    return " ".join(parts)


def write_yolo_label(label: str, path: Path) -> None:
    """Write a single YOLO label string to a .txt file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(label + "\n")
