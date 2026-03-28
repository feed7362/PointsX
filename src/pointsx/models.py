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
    ):
        self.img_size = img_size
        self._pose = YOLO(str(pose_model_path))
        self._seg = YOLO(str(seg_model_path))

    def predict_pose(self, image: np.ndarray, view: str) -> Keypoints | None:
        """Run pose estimation. Returns keypoints for the largest detected person."""
        results = self._pose(image, imgsz=self.img_size, verbose=False, device="cpu")
        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            return None

        # Select person with largest bounding box area
        if r.boxes is not None and len(r.boxes) > 1:
            areas = r.boxes.xywh[:, 2] * r.boxes.xywh[:, 3]
            best_idx = int(areas.argmax())
        else:
            best_idx = 0

        pts = r.keypoints.xy[best_idx].cpu().numpy()  # (K, 2)
        conf = r.keypoints.conf[best_idx].cpu().numpy()  # (K,)

        return Keypoints(points=pts, confidence=conf, view=view)

    def predict_segmentation(self, image: np.ndarray, view: str) -> SilhouetteMask | None:
        """Run segmentation. Returns combined binary mask of all person detections."""
        results = self._seg(image, imgsz=self.img_size, verbose=False, device="cpu")
        r = results[0]

        if r.masks is None or len(r.masks) == 0:
            return None

        h, w = image.shape[:2]

        # Filter for person class (class 0 in COCO)
        person_indices = [
            i for i, cls in enumerate(r.boxes.cls) if int(cls) == 0
        ]

        if not person_indices:
            return None

        # Combine all person masks
        combined = np.zeros((h, w), dtype=bool)
        for idx in person_indices:
            mask = r.masks.data[idx].cpu().numpy()  # (mask_h, mask_w)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            combined |= mask > 0.5

        # Extract outer contour
        contour = None
        mask_uint8 = combined.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            contour = largest.reshape(-1, 2)  # (N, 2)

        return SilhouetteMask(mask=combined, contour=contour, view=view)
