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
BBOX_PAD = 0.15      # 15% of the tight bbox extent
BBOX_MIN_PX = 50     # absolute minimum side length in pixels


def project_landmarks_to_2d(
    landmarks_3d: list[np.ndarray],
    camera_matrix: np.ndarray,
    view_matrix: np.ndarray,
    img_w: int,
    img_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D world coordinates to 2D image pixels.

    Args:
        landmarks_3d:  list of 25 arrays shape (3,) — world XYZ in metres
        camera_matrix: (3, 3) intrinsic matrix (from Blender render params)
        view_matrix:   (4, 4) world-to-camera extrinsic matrix
        img_w, img_h:  image dimensions in pixels

    Returns:
        coords_px:  (25, 2) float32 — pixel (x, y) per landmark (may be outside frame)
        depth:      (25,)   float32 — camera-space depth (positive = in front)
    """
    pts = np.array(landmarks_3d, dtype=np.float64)   # (25, 3)

    # ── World → Camera space ─────────────────────────────────────────────
    # Blender's camera looks down -Z in camera space; Y is up.
    # view_matrix: column-major 4×4.  Apply: cam_pt = view_matrix @ [x,y,z,1]
    ones = np.ones((len(pts), 1), dtype=np.float64)
    pts_h = np.hstack([pts, ones])                    # (25, 4)
    cam_pts = (view_matrix @ pts_h.T).T               # (25, 4)
    cam_xyz = cam_pts[:, :3]                           # (25, 3)

    # In Blender camera coords: camera looks down -Z
    # depth = -Z component (positive means in front of camera)
    depth = -cam_xyz[:, 2].astype(np.float32)

    # ── Camera → Image (perspective division) ────────────────────────────
    # NDC: divide by -Z (Blender convention)
    eps = 1e-8
    z = -cam_xyz[:, 2:3]                              # positive denominator
    z = np.where(np.abs(z) < eps, eps, z)
    ndc_xy = cam_xyz[:, :2] / z                       # (25, 2)

    # Camera intrinsics: [fx, 0, cx; 0, fy, cy; 0,0,1]
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    px_x = fx * ndc_xy[:, 0] + cx
    px_y = fy * ndc_xy[:, 1] + cy

    # Blender renders with Y=0 at bottom; flip for image convention
    px_y = img_h - px_y

    coords_px = np.stack([px_x, px_y], axis=1).astype(np.float32)  # (25, 2)
    return coords_px, depth


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


# ── Camera matrix helpers (called by pipeline / blender_render) ──────────────

def blender_camera_matrix(
    focal_length_mm: float,
    sensor_width_mm: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Build a 3×3 pinhole camera intrinsic matrix from Blender parameters.

    Args:
        focal_length_mm: camera focal length in millimetres (e.g. 50)
        sensor_width_mm: sensor width in mm (default Blender: 36 mm)
        img_w, img_h:    render resolution

    Returns:
        K: (3, 3) float64 intrinsic matrix
    """
    # Pixel focal length
    f_px = (focal_length_mm / sensor_width_mm) * img_w

    cx = img_w / 2.0
    cy = img_h / 2.0

    K = np.array([
        [f_px, 0.0,  cx],
        [0.0,  f_px, cy],
        [0.0,  0.0,  1.0],
    ], dtype=np.float64)
    return K


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert XYZ Euler angles (radians) to a 3×3 rotation matrix.

    Uses Blender's XYZ extrinsic convention.
    """
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def build_view_matrix(
    camera_location: tuple[float, float, float],
    camera_rotation_euler: tuple[float, float, float],
) -> np.ndarray:
    """Build a 4×4 world-to-camera view matrix from Blender camera transform.

    Args:
        camera_location:       (x, y, z) camera world position in metres
        camera_rotation_euler: (rx, ry, rz) Blender XYZ Euler angles in radians

    Returns:
        view_matrix: (4, 4) float64
    """
    R = euler_to_rotation_matrix(*camera_rotation_euler)
    t = np.array(camera_location, dtype=np.float64)

    # View matrix: world → camera  =  [R | -R @ t]
    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R
    Rt[:3,  3] = -R @ t

    return Rt
