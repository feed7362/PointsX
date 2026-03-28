"""SMPL-X body generation: sample diverse body shapes and compute landmarks + measurements."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from pointsx.synthetic.landmarks import extract_landmarks, LANDMARK_NAMES

logger = logging.getLogger(__name__)

# ── Height calibration constants (empirical from SMPL-X neutral model) ──────
# β[0] is the primary height principal component
# Linear mapping: height_cm ≈ MEAN_HEIGHT + β[0] * HEIGHT_STD
MEAN_HEIGHT_M = {"male": 1.77, "female": 1.64}
HEIGHT_STD_M = {"male": 0.065, "female": 0.060}

# BMI class → β[1] approximate range (weight PC)
# Negative β[1] = thinner, positive = heavier
BMI_BETA1_RANGE = {
    "very_thin":    (-3.0, -1.5),
    "thin":         (-1.5, -0.5),
    "normal":       (-0.5,  0.5),
    "overweight":   ( 0.5,  1.5),
    "obese":        ( 1.5,  3.0),
}

BMI_CLASSES = list(BMI_BETA1_RANGE.keys())
BMI_WEIGHTS = [0.10, 0.20, 0.40, 0.20, 0.10]  # realistic distribution


@dataclass
class BodySample:
    body_id: int
    sex: str
    target_height_cm: float
    bmi_class: str
    betas: list[float]            # shape (10,)
    body_pose: list[float]        # shape (63,) — 21 joints × 3 axis-angle
    global_orient: list[float]    # shape (3,)

    # Outputs (filled after SMPL-X forward pass)
    actual_height_cm: float = 0.0
    obj_path: str = ""
    landmarks_path: str = ""


def _height_to_beta0(target_height_m: float, sex: str) -> float:
    """Convert desired height to SMPL-X β[0] parameter."""
    return (target_height_m - MEAN_HEIGHT_M[sex]) / HEIGHT_STD_M[sex]


def _sample_bmi_beta1() -> tuple[str, float]:
    """Sample a BMI class and corresponding β[1] value."""
    bmi_class = np.random.choice(BMI_CLASSES, p=BMI_WEIGHTS)
    lo, hi = BMI_BETA1_RANGE[bmi_class]
    return bmi_class, float(np.random.uniform(lo, hi))


# ── Pose definitions ──────────────────────────────────────────────────────
def _a_pose() -> np.ndarray:
    """A-Pose: arms at 15-20° from body, feet shoulder-width. Front view ideal."""
    pose = np.zeros(63)
    # Shoulder abduction ~15-20° (joints 16=L_shoulder, 17=R_shoulder → local indices 13,14 in body_pose)
    # body_pose[j*3:(j+1)*3] = axis-angle for joint j+1 (joint 0=pelvis excluded)
    # L_shoulder = joint 16 → body_pose index (16-1)*3 = 45
    # R_shoulder = joint 17 → body_pose index (17-1)*3 = 48
    arm_angle = np.radians(np.random.uniform(15, 20))
    pose[45] = arm_angle    # L_shoulder z-axis (abduction)
    pose[48] = -arm_angle   # R_shoulder z-axis (adduction = negative)
    # Slight hip outward rotation for visibility of crotch
    pose[1] = np.radians(np.random.uniform(5, 10))   # L_hip
    pose[4] = -np.radians(np.random.uniform(5, 10))  # R_hip
    return pose


def _side_pose() -> np.ndarray:
    """Side pose: arms raised 45° forward, body upright. Profile view ideal."""
    pose = np.zeros(63)
    # Arms forward 45° (shoulder flexion)
    # L_shoulder forward = negative x rotation
    pose[45 + 1] = -np.radians(np.random.choice([40, 45, 50]))
    pose[48 + 1] = -np.radians(np.random.choice([40, 45, 50]))
    return pose


def _casual_pose() -> np.ndarray:
    """Casual pose: slight hip tilt, slouch, head down — robustness testing."""
    pose = np.zeros(63)
    # Hip tilt: shift weight to one leg
    hip_tilt = np.radians(np.random.uniform(5, 12))
    pose[0] = hip_tilt if np.random.random() > 0.5 else -hip_tilt  # spine1 lateral

    # Slight shoulder droop
    pose[45] = np.radians(np.random.uniform(5, 15))
    pose[48] = -np.radians(np.random.uniform(5, 15))

    # Head down
    neck_tilt = (14 - 1) * 3  # neck joint → body_pose index 39
    pose[neck_tilt] = np.radians(np.random.uniform(5, 15))

    return pose


POSE_GENERATORS = [_a_pose, _side_pose, _casual_pose]
POSE_NAMES = ["a_pose", "side_pose", "casual"]
POSE_WEIGHTS = [0.40, 0.40, 0.20]  # 70% useful + 30% robustness (rounded to pose pairs)


def generate_body_samples(
    n_bodies: int = 500,
    seed: int = 42,
) -> list[BodySample]:
    """Generate N unique body configurations (shape + 3 poses each)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    samples = []
    body_id = 1

    n_male = n_bodies // 2
    n_female = n_bodies - n_male
    sexes = ["male"] * n_male + ["female"] * n_female
    rng.shuffle(sexes)

    for sex in sexes:
        # Sample height: 150–200 cm
        target_height_cm = float(rng.uniform(150, 200))
        target_height_m = target_height_cm / 100.0

        # Build β parameters
        bmi_class, beta1 = _sample_bmi_beta1()
        beta0 = _height_to_beta0(target_height_m, sex)

        betas = np.zeros(10)
        betas[0] = beta0
        betas[1] = beta1
        # Proportions: short/long legs, wide/narrow shoulders, belly
        betas[2:] = rng.standard_normal(8) * 0.8

        # Global orient: minimal random jitter (person facing camera)
        global_orient = rng.standard_normal(3) * 0.05

        for pose_fn, pose_name in zip(POSE_GENERATORS, POSE_NAMES):
            # Add random noise to pose for variety
            base_pose = pose_fn()
            noise = rng.standard_normal(63) * 0.02
            body_pose = (base_pose + noise).tolist()

            samples.append(BodySample(
                body_id=body_id,
                sex=sex,
                target_height_cm=target_height_cm,
                bmi_class=bmi_class,
                betas=betas.tolist(),
                body_pose=body_pose,
                global_orient=global_orient.tolist(),
            ))
            body_id += 1

    logger.info("Generated %d body samples (%d bodies × 3 poses)", len(samples), n_bodies)
    return samples


def run_smplx_forward(sample: BodySample, model_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Run SMPL-X forward pass for one body sample.

    Returns:
        vertices: (10475, 3) float32
        joints:   (127, 3)  float32  (SMPL-X full joints)
        height_m: actual height in meters computed from mesh
    """
    import smplx

    model_path = model_dir / f"SMPLX_{sample.sex.upper()}.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            f"SMPL-X model not found: {model_path}\n"
            "Download from https://smpl-x.is.tue.mpg.de/ and place in models/smplx/"
        )

    model = smplx.create(
        str(model_dir),
        model_type="smplx",
        gender=sample.sex,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext="npz",
    )

    betas = torch.tensor([sample.betas], dtype=torch.float32)
    body_pose = torch.tensor([sample.body_pose], dtype=torch.float32)
    global_orient = torch.tensor([sample.global_orient], dtype=torch.float32)
    expression = torch.zeros(1, 10)

    with torch.no_grad():
        output = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            expression=expression,
            return_verts=True,
        )

    vertices = output.vertices[0].numpy()   # (10475, 3)
    joints = output.joints[0].numpy()       # (127, 3)

    # Compute actual height from mesh
    head_y = vertices[411, 1]               # vertex 411 = head_top
    ankle_y = (vertices[6852, 1] + vertices[3438, 1]) / 2  # L+R ankle vertices
    height_m = abs(head_y - ankle_y)

    return vertices, joints, height_m


def save_body_obj(vertices: np.ndarray, faces: np.ndarray, path: Path) -> None:
    """Save body mesh as OBJ file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def save_landmarks_json(
    sample: BodySample,
    landmarks_3d: list[np.ndarray],
    measurements: dict,
    path: Path,
) -> None:
    """Save landmark 3D coordinates + ground truth measurements to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "body_id": sample.body_id,
        "sex": sample.sex,
        "pose": _get_pose_name(sample.body_id),
        "target_height_cm": round(sample.target_height_cm, 1),
        "actual_height_cm": round(sample.actual_height_cm, 1),
        "bmi_class": sample.bmi_class,
        "measurements": {k: round(v, 1) for k, v in measurements.items()},
        "landmarks_3d": {
            name: coord.tolist()
            for name, coord in zip(LANDMARK_NAMES, landmarks_3d)
        },
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def _get_pose_name(body_id: int) -> str:
    """Infer pose name from body_id (cycles: a_pose, side_pose, casual)."""
    return POSE_NAMES[(body_id - 1) % 3]
