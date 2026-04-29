"""25 custom body measurement landmark definitions mapped to SMPL-X joints/vertices.

Each landmark is either:
  - Joint (J): direct SMPL-X joint index → output.joints[idx]
  - Vertex (V): SMPL-X vertex index → output.vertices[idx]
  - VertexMean (M): mean of multiple vertex indices

Indices verified against SMPL-X topology and SMPL-Anthropometry tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LMType(Enum):
    JOINT = "joint"
    VERTEX = "vertex"
    VERTEX_MEAN = "vertex_mean"


@dataclass
class LandmarkDef:
    idx: int | list[int]  # joint or vertex index / indices
    lm_type: LMType
    name: str
    description: str


# ── SMPL-X Joint indices (body joints 0-21, then extras) ──────────────────
# 0=pelvis, 1=L_hip, 2=R_hip, 3=spine1, 4=L_knee, 5=R_knee,
# 6=spine2, 7=L_ankle, 8=R_ankle, 9=spine3, 10=L_foot, 11=R_foot,
# 12=neck, 13=L_collar, 14=R_collar, 15=head, 16=L_shoulder, 17=R_shoulder,
# 18=L_elbow, 19=R_elbow, 20=L_wrist, 21=R_wrist

# ── SMPL-X Vertex indices (10475 total) ────────────────────────────────────
# Key anatomical vertices (confirmed via SMPL-Anthropometry + visual inspection):
#   Head top:      411
#   Chin:         8152
#   L_nipple:     3050  R_nipple: 6545
#   Navel:        3500
#   Crotch:       1210
#   L_armpit:     1850  R_armpit: 5250  (approx, refine with SMPL-Anthropometry)
#   L_waist:       702  R_waist:  4098
#   L_hip_outer: 1380   R_hip_outer: 4821
#   Mid_back:     2943  (posterior spine at thorax height)
#   L_sacrum:     3020  (posterior at hip height)
#   Glute_max:    3145  (max posterior vertex at hip height)


LANDMARKS: list[LandmarkDef] = [
    LandmarkDef(15, LMType.JOINT, "head_top", "Голова (суглоб) / Head"),
    LandmarkDef(12, LMType.JOINT, "chin", "Шия-перед / Front Neck"),
    LandmarkDef(12, LMType.JOINT, "back_of_neck", "Задня частина шиї / Back of neck"),
    LandmarkDef(16, LMType.JOINT, "left_shoulder", "Ліве плече / Left shoulder"),
    LandmarkDef(17, LMType.JOINT, "right_shoulder", "Праве плече / Right shoulder"),

    LandmarkDef(13, LMType.JOINT, "left_nipple", "Ліва грудь (Collar)"),
    LandmarkDef(14, LMType.JOINT, "right_nipple", "Права грудь (Collar)"),

    LandmarkDef(16, LMType.JOINT, "left_armpit", "Ліва пахва (Shoulder anchor)"),
    LandmarkDef(17, LMType.JOINT, "right_armpit", "Права пахва (Shoulder anchor)"),

    LandmarkDef(9, LMType.JOINT, "mid_back", "Середина спини (spine3)"),
    LandmarkDef(3, LMType.JOINT, "navel", "Пупок / Талія (spine1)"),

    LandmarkDef(18, LMType.JOINT, "left_elbow", "Лівий лікоть / Left elbow"),
    LandmarkDef(19, LMType.JOINT, "right_elbow", "Правий лікоть / Right elbow"),

    LandmarkDef(3, LMType.JOINT, "left_waist", "Ліва талія (spine1 anchor)"),
    LandmarkDef(3, LMType.JOINT, "right_waist", "Права талія (spine1 anchor)"),

    LandmarkDef(0, LMType.JOINT, "lower_back", "Крижі (pelvis)"),

    LandmarkDef(1, LMType.JOINT, "left_outer_hip", "Ліве стегно (L_hip)"),
    LandmarkDef(2, LMType.JOINT, "right_outer_hip", "Праве стегно (R_hip)"),

    LandmarkDef(0, LMType.JOINT, "crotch", "Пах (pelvis)"),
    LandmarkDef(0, LMType.JOINT, "glute", "Сідниці (pelvis anchor)"),

    LandmarkDef(20, LMType.JOINT, "left_wrist", "Ліве зап'ясток / Left wrist"),
    LandmarkDef(21, LMType.JOINT, "right_wrist", "Праве зап'ясток / Right wrist"),

    LandmarkDef(4, LMType.JOINT, "left_knee", "Ліве коліно / Left knee"),
    LandmarkDef(5, LMType.JOINT, "right_knee", "Праве коліно / Right knee"),

    LandmarkDef(7, LMType.JOINT, "left_ankle", "Ліва щиколотка / Left ankle"),
    LandmarkDef(8, LMType.JOINT, "right_ankle", "Права щиколотка / Right ankle"),
]

NUM_LANDMARKS = len(LANDMARKS)  # 26
assert NUM_LANDMARKS == 26, f"Expected 26 landmarks, got {NUM_LANDMARKS}"

LANDMARK_NAMES = [lm.name for lm in LANDMARKS]

# 16-keypoint order used by the runtime/inference stack.
POINTSX16_NAMES = [
    "right_ankle",
    "right_knee",
    "right_outer_hip",
    "left_outer_hip",
    "left_knee",
    "left_ankle",
    "lower_back",
    "mid_back",
    "back_of_neck",
    "head_top",
    "right_wrist",
    "right_elbow",
    "right_shoulder",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
]
POINTSX16_FLIP_IDX = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
POINTSX16_INDEX = {name: LANDMARK_NAMES.index(name) for name in POINTSX16_NAMES}

# Left-right flip index map for YOLO data augmentation
# Maps each landmark index to its mirror-image counterpart
FLIP_IDX = [
    0,  # head_top       → head_top
    1,  # chin           → chin
    2,  # back_of_neck   → back_of_neck
    4,  # left_shoulder  ↔ right_shoulder
    3,  # right_shoulder ↔ left_shoulder
    6,  # left_nipple    ↔ right_nipple
    5,  # right_nipple   ↔ left_nipple
    8,  # left_armpit    ↔ right_armpit
    7,  # right_armpit   ↔ left_armpit
    9,  # mid_back       → mid_back
    10,  # navel          → navel
    12,  # left_elbow     ↔ right_elbow
    11,  # right_elbow    ↔ left_elbow
    14,  # left_waist     ↔ right_waist
    13,  # right_waist    ↔ left_waist
    15,  # lower_back     → lower_back
    17,  # left_outer_hip ↔ right_outer_hip
    16,  # right_outer_hip ↔ left_outer_hip
    18,  # crotch         → crotch
    19,  # glute          → glute
    21,  # left_wrist     ↔ right_wrist
    20,  # right_wrist    ↔ left_wrist
    23,  # left_knee      ↔ right_knee
    22,  # right_knee     ↔ left_knee
    25,  # left_ankle     ↔ right_ankle
    24,  # right_ankle    ↔ left_ankle
]

assert len(FLIP_IDX) == NUM_LANDMARKS


def extract_landmarks(vertices, joints) -> list:
    """Extract 3D positions of all 26 landmarks from SMPL-X output.

    Args:
        vertices: numpy array (10475, 3) or torch tensor
        joints:   numpy array (N, 3) or torch tensor, N >= 22

    Returns:
        List of 26 numpy arrays shape (3,) — world coordinates [x, y, z]
    """
    import numpy as np

    if hasattr(vertices, "numpy"):
        vertices = vertices.numpy()
    if hasattr(joints, "numpy"):
        joints = joints.numpy()

    coords = []
    for lm in LANDMARKS:
        if lm.lm_type == LMType.JOINT:
            coords.append(joints[lm.idx].copy())
        elif lm.lm_type == LMType.VERTEX:
            coords.append(vertices[lm.idx].copy())
        elif lm.lm_type == LMType.VERTEX_MEAN:
            pts = np.array([vertices[i] for i in lm.idx])
            coords.append(pts.mean(axis=0))

    return coords


def select_pointsx16(landmarks_3d_by_name: dict[str, list[float] | tuple[float, float, float]]) -> list:
    """Select and order synthetic landmarks to match the 16-point runtime skeleton."""
    selected = []
    for name in POINTSX16_NAMES:
        if name not in landmarks_3d_by_name:
            raise KeyError(f"Missing required landmark for 16-point export: {name}")
        selected.append(landmarks_3d_by_name[name])
    return selected
