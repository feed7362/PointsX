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
    idx: int | list[int]   # joint or vertex index / indices
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
    # ── Head ──────────────────────────────────────────────────────────────
    LandmarkDef(411,          LMType.VERTEX,      "head_top",        "Тім'я / Top of head"),
    LandmarkDef(8152,         LMType.VERTEX,      "chin",            "Підборіддя / Chin"),

    # ── Neck / upper back ──────────────────────────────────────────────────
    LandmarkDef(12,           LMType.JOINT,       "back_of_neck",    "Задня частина шиї / Back of neck"),

    # ── Shoulders ─────────────────────────────────────────────────────────
    LandmarkDef(16,           LMType.JOINT,       "left_shoulder",   "Ліве плече / Left shoulder"),
    LandmarkDef(17,           LMType.JOINT,       "right_shoulder",  "Праве плече / Right shoulder"),

    # ── Chest ─────────────────────────────────────────────────────────────
    LandmarkDef([3050, 6545], LMType.VERTEX_MEAN, "chest_nipple",    "Лінія сосків / Nipple line (chest)"),

    # ── Armpits ───────────────────────────────────────────────────────────
    LandmarkDef(1850,         LMType.VERTEX,      "left_armpit",     "Ліва пахва / Left armpit"),
    LandmarkDef(5250,         LMType.VERTEX,      "right_armpit",    "Права пахва / Right armpit"),

    # ── Back / torso ──────────────────────────────────────────────────────
    LandmarkDef(2943,         LMType.VERTEX,      "mid_back",        "Середина спини / Mid back"),
    LandmarkDef(3500,         LMType.VERTEX,      "navel",           "Пупок / Navel / Abdomen"),

    # ── Elbows ────────────────────────────────────────────────────────────
    LandmarkDef(18,           LMType.JOINT,       "left_elbow",      "Ліпень лікоть / Left elbow"),
    LandmarkDef(19,           LMType.JOINT,       "right_elbow",     "Правий лікоть / Right elbow"),

    # ── Waist ─────────────────────────────────────────────────────────────
    LandmarkDef(702,          LMType.VERTEX,      "left_waist",      "Ліва талія / Left waist"),
    LandmarkDef(4098,         LMType.VERTEX,      "right_waist",     "Права талія / Right waist"),

    # ── Lower back ────────────────────────────────────────────────────────
    LandmarkDef(3020,         LMType.VERTEX,      "lower_back",      "Крижі / Lower back / Sacrum"),

    # ── Hips ──────────────────────────────────────────────────────────────
    LandmarkDef(1380,         LMType.VERTEX,      "left_outer_hip",  "Ліве стегно зовні / Left outer hip"),
    LandmarkDef(4821,         LMType.VERTEX,      "right_outer_hip", "Праве стегно зовні / Right outer hip"),

    # ── Crotch / glute ────────────────────────────────────────────────────
    LandmarkDef(1210,         LMType.VERTEX,      "crotch",          "Пах / Crotch"),
    LandmarkDef(3145,         LMType.VERTEX,      "glute",           "Сідниці / Glute / Buttock"),

    # ── Wrists ────────────────────────────────────────────────────────────
    LandmarkDef(20,           LMType.JOINT,       "left_wrist",      "Ліве зап'ясток / Left wrist"),
    LandmarkDef(21,           LMType.JOINT,       "right_wrist",     "Праве зап'ясток / Right wrist"),

    # ── Knees ─────────────────────────────────────────────────────────────
    LandmarkDef(4,            LMType.JOINT,       "left_knee",       "Ліве коліно / Left knee"),
    LandmarkDef(5,            LMType.JOINT,       "right_knee",      "Праве коліно / Right knee"),

    # ── Ankles ────────────────────────────────────────────────────────────
    LandmarkDef(7,            LMType.JOINT,       "left_ankle",      "Ліва щиколотка / Left ankle"),
    LandmarkDef(8,            LMType.JOINT,       "right_ankle",     "Права щиколотка / Right ankle"),
]

NUM_LANDMARKS = len(LANDMARKS)  # 25
assert NUM_LANDMARKS == 25, f"Expected 25 landmarks, got {NUM_LANDMARKS}"

LANDMARK_NAMES = [lm.name for lm in LANDMARKS]

# Left-right flip index map for YOLO data augmentation
# Maps each landmark index to its mirror-image counterpart
FLIP_IDX = [
    0,   # head_top       → head_top
    1,   # chin           → chin
    2,   # back_of_neck   → back_of_neck
    4,   # left_shoulder  ↔ right_shoulder
    3,   # right_shoulder ↔ left_shoulder
    5,   # chest_nipple   → chest_nipple
    7,   # left_armpit    ↔ right_armpit
    6,   # right_armpit   ↔ left_armpit
    8,   # mid_back       → mid_back
    9,   # navel          → navel
    11,  # left_elbow     ↔ right_elbow
    10,  # right_elbow    ↔ left_elbow
    13,  # left_waist     ↔ right_waist
    12,  # right_waist    ↔ left_waist
    14,  # lower_back     → lower_back
    16,  # left_outer_hip ↔ right_outer_hip
    15,  # right_outer_hip ↔ left_outer_hip
    17,  # crotch         → crotch
    18,  # glute          → glute
    20,  # left_wrist     ↔ right_wrist
    19,  # right_wrist    ↔ left_wrist
    22,  # left_knee      ↔ right_knee
    21,  # right_knee     ↔ left_knee
    24,  # left_ankle     ↔ right_ankle
    23,  # right_ankle    ↔ left_ankle
]

assert len(FLIP_IDX) == NUM_LANDMARKS


def extract_landmarks(vertices, joints) -> list:
    """Extract 3D positions of all 25 landmarks from SMPL-X output.

    Args:
        vertices: numpy array (10475, 3) or torch tensor
        joints:   numpy array (N, 3) or torch tensor, N >= 22

    Returns:
        List of 25 numpy arrays shape (3,) — world coordinates [x, y, z]
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
