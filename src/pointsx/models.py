"""YOLO model wrappers for pose estimation and segmentation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from pointsx.schemas import Keypoints, SilhouetteMask


class BodyModels:
    """Loads and runs YOLO pose + segmentation models."""

    def __init__(
        self,
        pose_model_path: str | Path = "models/yolo11n-pose.pt",
        seg_model_path: str | Path = "models/yolo11n-seg.pt",
        img_size: int = 640,
        device: str = "auto",
    ):
        self.img_size = img_size
        self.device = _resolve_device(device)
        self._pose = YOLO(str(pose_model_path))
        self._seg = YOLO(str(seg_model_path))

    def predict_pose(self, image: np.ndarray, view: str) -> Keypoints | None:
        """Run pose estimation. Returns keypoints for the largest detected person."""
        results = self._pose(image, imgsz=self.img_size, verbose=False, device=self.device)
        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            return None

        best_idx = 0
        if r.boxes is not None and len(r.boxes) > 0:
            cls = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else None
            if cls is not None:
                candidate_indices = [i for i, c in enumerate(cls) if int(c) == 0]
            else:
                candidate_indices = list(range(len(r.boxes)))

            if candidate_indices:
                boxes_xywh = r.boxes.xywh[candidate_indices].cpu().numpy()
                areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
                centers = boxes_xywh[:, :2]
                rel_idx = _select_primary_person_index(areas, centers, None)
                best_idx = int(candidate_indices[rel_idx])

        pts = r.keypoints.xy[best_idx].cpu().numpy()  # (K, 2)
        conf = r.keypoints.conf[best_idx].cpu().numpy()  # (K,)

        return Keypoints(points=pts, confidence=conf, view=view)

    def predict_segmentation(
        self,
        image: np.ndarray,
        view: str,
        reference_point: tuple[float, float] | None = None,
    ) -> SilhouetteMask | None:
        """Run segmentation and return the best-matching person mask."""
        results = self._seg(image, imgsz=self.img_size, verbose=False, device=self.device)
        r = results[0]

        if r.masks is None or len(r.masks) == 0 or r.boxes is None or len(r.boxes) == 0:
            return None

        h, w = image.shape[:2]

        # Filter for person class (class 0 in COCO)
        person_indices = [i for i, cls in enumerate(r.boxes.cls) if int(cls) == 0]

        if not person_indices:
            return None

        boxes_xywh = r.boxes.xywh[person_indices].cpu().numpy()
        areas = boxes_xywh[:, 2] * boxes_xywh[:, 3]
        centers = boxes_xywh[:, :2]
        rel_idx = _select_primary_person_index(areas, centers, reference_point)
        best_idx = person_indices[rel_idx]

        mask = r.masks.data[best_idx].cpu().numpy()  # (mask_h, mask_w)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        selected = mask > 0.5

        # Extract outer contour
        contour = None
        mask_uint8 = selected.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour = largest.reshape(-1, 2)  # (N, 2)

        return SilhouetteMask(mask=selected, contour=contour, view=view)


def _resolve_device(device: str) -> str:
    """Resolve runtime device from a user-facing option."""
    if device == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return device


def _select_primary_person_index(
    areas: np.ndarray,
    centers: np.ndarray,
    reference_point: tuple[float, float] | None,
) -> int:
    """Pick one person index from candidate detections."""
    if len(areas) == 1:
        return 0
    if reference_point is None:
        return int(np.argmax(areas))

    reference = np.array(reference_point, dtype=np.float32)
    dists = np.linalg.norm(centers - reference, axis=1)
    area_bonus = areas / (areas.max() + 1e-6)
    score = dists - 0.1 * area_bonus
    return int(np.argmin(score))
