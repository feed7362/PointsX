"""SMPL-X body generation: sample diverse body shapes and compute landmarks + measurements."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pointsx.synthetic.landmarks import LANDMARK_NAMES


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars/arrays to Python types."""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


logger = logging.getLogger(__name__)

# ── Height calibration constants (empirical from SMPL-X neutral model) ──────
# β[0] is the primary height principal component
# Linear mapping: height_cm ≈ MEAN_HEIGHT + β[0] * HEIGHT_STD
MEAN_HEIGHT_M = {"male": 1.77, "female": 1.64}
HEIGHT_STD_M = {"male": 0.065, "female": 0.060}

# BMI class → β[1] approximate range (weight PC)
# Negative β[1] = thinner, positive = heavier
BMI_BETA1_RANGE = {
    "very_thin": (-3.0, -1.5),
    "thin": (-1.5, -0.5),
    "normal": (-0.5, 0.5),
    "overweight": (0.5, 1.5),
    "obese": (1.5, 3.0),
}

BMI_CLASSES = list(BMI_BETA1_RANGE.keys())
BMI_WEIGHTS = [0.10, 0.20, 0.40, 0.20, 0.10]  # realistic distribution


@dataclass
class BodySample:
    body_id: int
    sex: str
    target_height_cm: float
    bmi_class: str
    betas: list[float]  # shape (10,)
    body_pose: list[float]  # shape (63,) — 21 joints × 3 axis-angle
    global_orient: list[float]  # shape (3,)

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
    """A-Pose: arms hanging down ~10-25° from vertical (true A-pose).

    SMPL-X's rest pose is T-pose (arms horizontal). To reach A-pose we have to
    rotate each shoulder downward by ~65-80° from horizontal. We rotate around
    the local Z-axis (axis-angle index 2 within the joint's 3 components),
    which is the abduction/adduction axis for the shoulder joint.

      L_shoulder = joint 16 → body_pose[(16-1)*3 + 2] = body_pose[47]
      R_shoulder = joint 17 → body_pose[(17-1)*3 + 2] = body_pose[50]
    """
    pose = np.zeros(63)
    arm_drop = np.radians(np.random.uniform(65, 80))  # T-pose → A-pose
    pose[47] = -arm_drop   # L_shoulder: rotate arm DOWN
    pose[50] =  arm_drop   # R_shoulder: mirror
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
POSE_WEIGHTS = [1.0, 0.0, 0.0]  # 100% a-pose for clean silhouette-width measurements


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

        # Honor POSE_WEIGHTS — generate one body per pose with non-zero weight.
        # Each body gets a unique body_id, so 1500 a-pose bodies (n=1500, weights=[1,0,0])
        # produces exactly 1500 samples instead of 4500.
        for pose_fn, pose_name, weight in zip(POSE_GENERATORS, POSE_NAMES, POSE_WEIGHTS):
            if weight <= 0.0:
                continue
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


_SMPLX_MODEL_CACHE: dict[str, object] = {}


def _get_smplx_model(sex: str, model_dir: Path) -> object:
    """Load an SMPL-X model, caching by gender to avoid repeated disk I/O."""
    if sex not in _SMPLX_MODEL_CACHE:
        import smplx

        model_path = model_dir / f"SMPLX_{sex.upper()}.npz"
        if not model_path.exists():
            raise FileNotFoundError(
                f"SMPL-X model not found: {model_path}\n"
                "Download from https://smpl-x.is.tue.mpg.de/ and place in models/smplx/"
            )
        _SMPLX_MODEL_CACHE[sex] = smplx.create(
            str(model_path),
            model_type="smplx",
            gender=sex,
            use_face_contour=False,
            num_betas=10,
            num_expression_coeffs=10,
            ext="npz",
        )
    return _SMPLX_MODEL_CACHE[sex]


def run_smplx_forward(sample: BodySample, model_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Run SMPL-X forward pass for one body sample and enforce exact target height.

    Returns:
        vertices: (10475, 3) float32
        joints:   (127, 3)  float32
        height_m: actual height in meters (now guaranteed to match target)
    """
    model = _get_smplx_model(sample.sex, model_dir)

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

    vertices = output.vertices[0].numpy()  # (10475, 3)
    joints = output.joints[0].numpy()  # (127, 3)
    raw_height_m = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))

    target_height_m = sample.target_height_cm / 100.0

    if raw_height_m < 0.1:
        raw_height_m = 1.7

    scale_factor = target_height_m / raw_height_m

    vertices *= scale_factor
    joints *= scale_factor

    lowest_y = np.min(vertices[:, 1])
    vertices[:, 1] -= lowest_y
    joints[:, 1] -= lowest_y

    final_height_m = target_height_m

    return vertices, joints, final_height_m


def save_body_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    path: Path,
    uv_verts: np.ndarray | None = None,
    uv_faces: np.ndarray | None = None,
) -> None:
    """Save body mesh as Wavefront OBJ.

    When ``uv_verts`` and ``uv_faces`` are supplied (the SMPL-X UV layout),
    they're written as ``vt`` lines and ``f v/vt v/vt v/vt`` indices, so
    Blender / any OBJ importer applies textures correctly. Without them the
    body is exported with no UVs (textured renders end up looking blotchy).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    has_uv = (
        uv_verts is not None
        and uv_faces is not None
        and len(uv_verts) > 0
        and len(uv_faces) == len(faces)
    )

    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if has_uv:
            for uv in uv_verts:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
            for face, uvf in zip(faces, uv_faces):
                f.write(
                    f"f "
                    f"{int(face[0]) + 1}/{int(uvf[0]) + 1} "
                    f"{int(face[1]) + 1}/{int(uvf[1]) + 1} "
                    f"{int(face[2]) + 1}/{int(uvf[2]) + 1}\n"
                )
        else:
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")


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
        "target_height_cm": round(float(sample.target_height_cm), 1),
        "actual_height_cm": round(float(sample.actual_height_cm), 1),
        "bmi_class": sample.bmi_class,
        "measurements": {k: round(float(v), 1) for k, v in measurements.items()},
        "landmarks_3d": {
            name: coord.tolist()
            for name, coord in zip(LANDMARK_NAMES, landmarks_3d)
        },
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, cls=NumpyEncoder))


def _get_pose_name(body_id: int) -> str:
    """Infer pose name from body_id (cycles: a_pose, side_pose, casual)."""
    return POSE_NAMES[(body_id - 1) % 3]
