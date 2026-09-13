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

# ── Anthropometric sampling priors ─────────────────────────────────────────
# Target stature (cm): realistic adult population per sex (approx NHANES/ANSUR),
# sampled as a truncated normal — NOT uniform 150–200, which over-weights the
# extremes and forces β[0] out of its plausible range.
HEIGHT_MEAN_CM = {"male": 176.0, "female": 163.0}
HEIGHT_STD_CM = {"male": 7.0, "female": 6.5}
HEIGHT_CLIP_CM = (147.0, 200.0)

# Shape betas β[2:] are sampled from the model's own N(0,1) shape prior (clipped)
# — the correct prior by construction, replacing the ad-hoc ×0.8 under-dispersion.
SHAPE_BETA_CLIP = 2.5

# BMI class → β[1] range (the dominant corpulence axis). Stratified sampling
# keeps the obese/thin TAILS covered (BMI>30 is the worst real-eval bucket).
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


def _sample_bmi_beta1(rng: np.random.Generator) -> tuple[str, float]:
    """Sample a BMI class and corresponding β[1] value."""
    bmi_class = rng.choice(BMI_CLASSES, p=BMI_WEIGHTS)
    lo, hi = BMI_BETA1_RANGE[bmi_class]
    return bmi_class, float(rng.uniform(lo, hi))


def _sample_target_height_cm(sex: str, rng: np.random.Generator) -> float:
    """Truncated-normal adult stature for the sex."""
    lo, hi = HEIGHT_CLIP_CM
    h = rng.normal(HEIGHT_MEAN_CM[sex], HEIGHT_STD_CM[sex])
    return float(np.clip(h, lo, hi))


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
    np.random.seed(seed)  # pose generators use the legacy global RNG — seed it too

    samples = []
    body_id = 1

    n_male = n_bodies // 2
    n_female = n_bodies - n_male
    sexes = ["male"] * n_male + ["female"] * n_female
    rng.shuffle(sexes)

    for sex in sexes:
        # Realistic stature per sex (truncated normal), not uniform 150–200.
        target_height_cm = _sample_target_height_cm(sex, rng)

        # Build β parameters. β[0] (the dominant stature axis) is left as a
        # PLACEHOLDER 0 here and SOLVED at forward time to hit target_height_cm
        # given the sampled shape — so height comes from a real shape parameter
        # (correct allometry), never a mesh rescale.
        bmi_class, beta1 = _sample_bmi_beta1(rng)

        betas = np.zeros(10)
        betas[0] = 0.0  # solved in run_smplx_forward
        betas[1] = beta1  # corpulence axis (BMI-stratified)
        # Remaining proportions from the model's own N(0,1) shape prior (clipped).
        betas[2:] = np.clip(rng.standard_normal(8), -SHAPE_BETA_CLIP, SHAPE_BETA_CLIP)

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


def _forward(model, betas_vec: np.ndarray, body_pose, global_orient, expression):
    """One SMPL-X forward pass → (vertices, joints) numpy."""
    betas_t = torch.from_numpy(np.ascontiguousarray(betas_vec, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(
            betas=betas_t,
            body_pose=body_pose, global_orient=global_orient,
            expression=expression, return_verts=True,
        )
    return out.vertices[0].numpy(), out.joints[0].numpy()


def _height_m(vertices: np.ndarray) -> float:
    return float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))


# Measured dheight/dβ0 per sex (cached) — replaces the hand-guessed HEIGHT_STD.
_BETA0_SLOPE_CACHE: dict[str, float] = {}


def _beta0_height_slope(model, sex: str, body_pose, global_orient, expression) -> float:
    """Empirically measured metres of stature per unit β[0], near neutral shape."""
    if sex not in _BETA0_SLOPE_CACHE:
        base = np.zeros(10)
        lo = base.copy(); lo[0] = -2.0
        hi = base.copy(); hi[0] = +2.0
        h_lo = _height_m(_forward(model, lo, body_pose, global_orient, expression)[0])
        h_hi = _height_m(_forward(model, hi, body_pose, global_orient, expression)[0])
        _BETA0_SLOPE_CACHE[sex] = (h_hi - h_lo) / 4.0
    return _BETA0_SLOPE_CACHE[sex]


def run_smplx_forward(sample: BodySample, model_dir: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Run SMPL-X forward pass, SOLVING β[0] to hit the target height.

    β[0] (the dominant stature axis) is solved so the emergent height matches
    `target_height_cm` GIVEN the sampled shape (β[1:]), using a measured
    dheight/dβ0 slope. Height therefore comes from a genuine shape parameter, so
    circumferences keep their natural (non-linear) allometry — unlike a uniform
    mesh rescale, which forced every girth to scale linearly with height and
    corrupted the height→girth relationship the model learns. The returned height
    is the true emergent height (≈ target within ~1 cm); the GT measured from this
    mesh is self-consistent with it.

    Returns:
        vertices: (10475, 3) float32 — floor-aligned (min y = 0)
        joints:   (127, 3)  float32
        height_m: emergent height in metres

    Raises:
        ValueError: degenerate forward pass — caller skips the body, no fake height.
    """
    model = _get_smplx_model(sample.sex, model_dir)

    body_pose = torch.tensor([sample.body_pose], dtype=torch.float32)
    global_orient = torch.tensor([sample.global_orient], dtype=torch.float32)
    expression = torch.zeros(1, 10)

    betas = np.asarray(sample.betas, dtype=np.float64).copy()

    # Solve β[0] for the target height given the sampled shape.
    verts0, _ = _forward(model, betas, body_pose, global_orient, expression)  # β0=0 placeholder
    h0 = _height_m(verts0)
    slope = _beta0_height_slope(model, sample.sex, body_pose, global_orient, expression)
    if abs(slope) < 1e-4:
        raise ValueError(f"degenerate β0→height slope ({slope:.5f}) for {sample.sex}")
    betas[0] = (sample.target_height_cm / 100.0 - h0) / slope

    vertices, joints = _forward(model, betas, body_pose, global_orient, expression)
    sample.betas = betas.tolist()  # record the solved β[0]

    height_m = float(np.max(vertices[:, 1]) - np.min(vertices[:, 1]))
    if height_m < 0.1:
        raise ValueError(f"degenerate SMPL-X forward pass (height={height_m:.3f} m)")

    # Floor-align only (no scaling): feet at y=0.
    lowest_y = np.min(vertices[:, 1])
    vertices[:, 1] -= lowest_y
    joints[:, 1] -= lowest_y

    return vertices, joints, height_m


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
