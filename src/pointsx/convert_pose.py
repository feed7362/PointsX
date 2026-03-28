"""Convert LV-MHP-v2 pose annotations (.mat) to YOLO pose format."""

import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from tqdm import tqdm

# LV-MHP-v2 keypoint indices
# 0-15: body keypoints, 16-17: face bbox, 18-19: instance bbox
NUM_BODY_KPS = 16
BBOX_TL_IDX = 18  # instance bbox top-left
BBOX_BR_IDX = 19  # instance bbox bottom-right

# Visibility remapping: LV-MHP-v2 -> YOLO
# LV-MHP: 0=visible, 1=occluded, 2=not labeled
# YOLO:   0=not labeled, 1=occluded, 2=visible
VIS_MAP = {0: 2, 1: 1, 2: 0}

# Left-right flip indices for data augmentation
# Pairs: (L-ankle,R-ankle), (L-knee,R-knee), (L-hip,R-hip),
#        (L-wrist,R-wrist), (L-elbow,R-elbow), (L-shoulder,R-shoulder)
FLIP_IDX = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]

KEYPOINT_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "left_hip", "left_knee", "left_ankle",
    "pelvis", "thorax", "upper_neck", "head_top",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
]


def convert_mat_to_yolo(mat_path: Path, img_path: Path) -> list[str]:
    """Convert a single .mat pose annotation to YOLO format lines.

    Returns list of YOLO format strings, one per person.
    """
    img = Image.open(img_path)
    img_w, img_h = img.size

    data = loadmat(str(mat_path))
    person_keys = sorted(k for k in data.keys() if k.startswith("person_"))

    lines = []
    for key in person_keys:
        kps = data[key]  # (20, 3) float32: [x, y, vis]

        # Extract instance bbox from keypoints 18-19
        tl = kps[BBOX_TL_IDX]  # top-left [x, y, vis]
        br = kps[BBOX_BR_IDX]  # bottom-right [x, y, vis]

        # Skip if bbox is invalid
        if tl[0] < 0 and tl[1] < 0 and br[0] < 0 and br[1] < 0:
            continue

        # Compute normalized bbox center and size
        x1, y1 = max(0, tl[0]), max(0, tl[1])
        x2, y2 = min(img_w, br[0]), min(img_h, br[1])

        if x2 <= x1 or y2 <= y1:
            continue

        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h

        # Build keypoint string (body keypoints 0-15 only)
        kp_parts = []
        for i in range(NUM_BODY_KPS):
            x, y, v = kps[i]
            v_int = int(v)
            yolo_vis = VIS_MAP.get(v_int, 0)

            if x < 0 or y < 0 or yolo_vis == 0:
                kp_parts.extend([0.0, 0.0, 0])
            else:
                kp_parts.extend([x / img_w, y / img_h, yolo_vis])

        kp_str = " ".join(
            f"{v:.6f}" if isinstance(v, float) else str(v) for v in kp_parts
        )
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kp_str}")

    return lines


def convert_split(
    src_root: Path, dst_root: Path, split: str
) -> tuple[int, int]:
    """Convert a dataset split. Returns (num_images, num_persons)."""
    src_images = src_root / split / "images"
    src_poses = src_root / split / "pose_annos"
    dst_images = dst_root / split / "images"
    dst_labels = dst_root / split / "labels"

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    mat_files = sorted(src_poses.glob("*.mat"))
    total_persons = 0

    for mat_path in tqdm(mat_files, desc=f"Converting {split}"):
        stem = mat_path.stem
        img_path = src_images / f"{stem}.jpg"

        if not img_path.exists():
            continue

        lines = convert_mat_to_yolo(mat_path, img_path)
        total_persons += len(lines)

        # Write label file
        label_path = dst_labels / f"{stem}.txt"
        label_path.write_text("\n".join(lines) + "\n" if lines else "")

        # Create hard link for image (works on Windows without admin)
        link_path = dst_images / f"{stem}.jpg"
        if not link_path.exists():
            os.link(img_path.resolve(), link_path)

    return len(mat_files), total_persons


def write_dataset_yaml(dst_root: Path) -> None:
    """Write the YOLO dataset configuration file."""
    yaml_content = f"""# LV-MHP-v2 Pose Dataset (YOLO format)
# 16 body keypoints from LV-MHP-v2

path: {dst_root.resolve().as_posix()}
train: train/images
val: val/images

# Keypoint shape: [num_keypoints, dims] (dims=3 means x, y, visibility)
kpt_shape: [16, 3]

# Left-right flip indices for augmentation
flip_idx: {FLIP_IDX}

names:
  0: person
"""
    yaml_path = dst_root / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"Dataset YAML written to: {yaml_path}")


def main():
    project_root = Path(__file__).resolve().parents[2]
    src_root = project_root / "data" / "LV-MHP-v2"
    dst_root = project_root / "data" / "LV-MHP-v2-pose"

    print(f"Source: {src_root}")
    print(f"Destination: {dst_root}")

    for split in ["train", "val"]:
        n_images, n_persons = convert_split(src_root, dst_root, split)
        print(f"{split}: {n_images} images, {n_persons} person annotations")

    write_dataset_yaml(dst_root)
    print("Done!")


if __name__ == "__main__":
    main()
