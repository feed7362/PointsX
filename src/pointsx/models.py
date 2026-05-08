"""YOLO model wrappers for pose estimation and segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from ultralytics import YOLO

from pointsx.keypoints import MIN_CONFIDENCE
from pointsx.pose_coco import coco17_to_lv_mhp16
from pointsx.schemas import Keypoints, SilhouetteMask

logger = logging.getLogger(__name__)


def _adjust_custom_keypoint_confidence(conf: np.ndarray) -> np.ndarray:
    """Spread soft confidences from custom LV-MHP-style heads into [MIN_CONFIDENCE, 1].

    YOLO custom pose often peaks well below COCO-style scores; geometry is still
    usable but ``is_valid`` would skip most chains. Only rescale when the peak is
    low; leave strong models unchanged.
    """
    c = np.asarray(conf, dtype=np.float32).copy()
    pos = c > 1e-6
    if not np.any(pos):
        return c
    peak = float(np.max(c[pos]))
    if peak >= 0.38:
        return c
    floor = float(MIN_CONFIDENCE)
    c[pos] = floor + (1.0 - floor) * np.clip(c[pos] / peak, 0.0, 1.0)
    return c

PoseBackend = Literal["custom", "coco"]


def _ensure_yolo_weights(path: Path) -> Path | None:
    """If `path` doesn't exist, try Ultralytics' built-in auto-download by basename.

    Ultralytics' YOLO() resolves a bare filename (e.g. ``yolo26-pose.pt``)
    against its release assets and downloads to cwd on first use. We invoke it
    that way, then move the resulting file into `path` so subsequent runs are
    offline. Returns the final path if the file ends up on disk, else None.
    """
    if path.is_file():
        return path
    name = path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Auto-downloading YOLO weights %s via Ultralytics…", name)
    try:
        YOLO(name)  # triggers download into cwd (or HF cache, depending on version)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-download of %s failed: %s", name, exc)
        return None

    cwd_candidate = Path.cwd() / name
    if cwd_candidate.is_file() and cwd_candidate.resolve() != path.resolve():
        try:
            cwd_candidate.replace(path)
        except OSError as exc:
            logger.warning("Could not move %s to %s: %s", cwd_candidate, path, exc)
            return cwd_candidate if cwd_candidate.is_file() else None
    return path if path.is_file() else None


class BodyModels:
    """Loads and runs YOLO pose (custom 16-pt + COCO-17) + segmentation."""

    def __init__(
        self,
        pose_custom_path: str | Path | None = "models/pose-cus.pt",
        pose_coco_path: str | Path | None = "models/yolo26-pose.pt",
        seg_model_path: str | Path = "models/yolo12l-person-seg-extended.pt",
        img_size: int = 640,
        device: str = "auto",
    ):
        self.img_size = img_size
        self.device = _resolve_device(device)
        self._pose_custom_path = str(pose_custom_path) if pose_custom_path else ""
        self._pose_coco_path = str(pose_coco_path) if pose_coco_path else ""

        self._pose_custom: YOLO | None = None
        self._pose_coco: YOLO | None = None

        if pose_custom_path and Path(pose_custom_path).is_file():
            self._pose_custom = YOLO(str(pose_custom_path))
        else:
            logger.warning("Custom pose weights not found (%s); backend 'custom' disabled", pose_custom_path)

        if pose_coco_path:
            resolved = _ensure_yolo_weights(Path(pose_coco_path))
            if resolved is not None:
                self._pose_coco = YOLO(str(resolved))
            else:
                logger.warning(
                    "COCO pose weights not found (%s) and auto-download failed; "
                    "backend 'coco' disabled",
                    pose_coco_path,
                )

        self._seg = YOLO(str(seg_model_path))

    def available_pose_backends(self) -> set[PoseBackend]:
        """Backends with loaded weights."""
        out: set[PoseBackend] = set()
        if self._pose_custom is not None:
            out.add("custom")
        if self._pose_coco is not None:
            out.add("coco")
        return out

    def predict_pose(
        self,
        image: np.ndarray,
        view: str,
        *,
        pose_backend: PoseBackend = "custom",
    ) -> Keypoints | None:
        """Run pose estimation. Returns keypoints for the largest detected person."""
        yolo = self._pose_custom if pose_backend == "custom" else self._pose_coco
        if yolo is None:
            raise ValueError(
                f"Pose backend {pose_backend!r} is not available (weights missing on server)"
            )

        results = yolo(image, imgsz=self.img_size, verbose=False, device=self.device)
        r = results[0]

        if r.keypoints is None or len(r.keypoints) == 0:
            return None

        best_idx = _best_person_keypoint_index(r)

        pts = r.keypoints.xy[best_idx].cpu().numpy()  # (K, 2)
        conf = r.keypoints.conf[best_idx].cpu().numpy()  # (K,)

        if pose_backend == "custom":
            if pts.shape[0] != 16:
                raise ValueError(
                    f"Custom pose model returned {pts.shape[0]} keypoints (expected 16). "
                    "Use COCO backend if your checkpoint outputs 17 points."
                )
            conf = _adjust_custom_keypoint_confidence(conf)
            return Keypoints(points=pts, confidence=conf, view=view)

        if pts.shape[0] != 17:
            raise ValueError(
                f"COCO pose model returned {pts.shape[0]} keypoints (expected 17). "
                "Use custom backend for a native 16-point model."
            )
        return coco17_to_lv_mhp16(pts, conf, view)

    def predict_segmentation(
        self,
        image: np.ndarray,
        view: str,
        reference_point: tuple[float, float] | None = None,
        conf: float = 0.1,
    ) -> SilhouetteMask | None:
        """Run segmentation and return the best-matching person mask.

        ``conf`` is intentionally low (0.10) so we don't drop side-view bodies
        when there's a synthetic→photo domain gap. Increase to 0.25 for real-world
        photos where false positives are more expensive.
        """
        results = self._seg(image, imgsz=self.img_size, verbose=False,
                            device=self.device, conf=conf)
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


def _best_person_keypoint_index(r) -> int:
    """Index into keypoints/boxes for the primary person (class 0, largest if ambiguous)."""
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
    return best_idx


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
