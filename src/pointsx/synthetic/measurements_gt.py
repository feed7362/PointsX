"""Ground-truth body measurements from an SMPL-X mesh.

Uses the vendored SMPL-Anthropometry (MIT) via `anthropometry.runner`: the mesh
is plane-sliced through anatomical landmarks, the slice is filtered to the
relevant BODY PART by face segmentation (arms can't contaminate waist/hip), and
a convex-hull perimeter is taken — the tape-measure model. These are the vetted
definitions used across the shape-estimation literature.

DESIGN RULE — NO FAKE GT. A measurement that fails to compute is left `None` and
the body is reported as unusable for that field. We NEVER substitute a
population-average constant: a synthetic dataset whose labels are silently
constants trains a model to regress toward those constants. Callers must drop
bodies missing the core circumferences (see `has_core_measurements`).
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from pointsx.synthetic.anthropometry.runner import measure_smplx

logger = logging.getLogger(__name__)

# The four garment-critical circumferences a usable body MUST have.
CORE_FIELDS = (
    "chest_circumference_cm",
    "waist_circumference_cm",
    "hips_circumference_cm",
    "thigh_circumference_cm",
)

# Plausibility bounds (cm) — a computed value outside these is a slice failure,
# not a real body, so it is discarded (set to None), never clamped or faked.
_BOUNDS = {
    "height_cm": (140.0, 220.0),
    "neck_circumference_cm": (25.0, 60.0),
    "chest_circumference_cm": (60.0, 170.0),
    "waist_circumference_cm": (45.0, 170.0),
    "hips_circumference_cm": (65.0, 180.0),
    "thigh_circumference_cm": (30.0, 100.0),
    "calf_circumference_cm": (20.0, 60.0),
    "wrist_circumference_cm": (12.0, 24.0),
    "shoulder_width_cm": (30.0, 60.0),
    "arm_length_cm": (40.0, 95.0),
    "inseam_length_cm": (60.0, 100.0),
}


@dataclass
class BodyMeasurementsGT:
    """Ground-truth body measurements in centimetres. `None` = not measurable
    on this mesh (never a fabricated constant)."""
    height_cm: float | None = None
    neck_circumference_cm: float | None = None
    chest_circumference_cm: float | None = None
    waist_circumference_cm: float | None = None
    hips_circumference_cm: float | None = None
    thigh_circumference_cm: float | None = None
    calf_circumference_cm: float | None = None
    wrist_circumference_cm: float | None = None
    inseam_length_cm: float | None = None
    arm_length_cm: float | None = None
    shoulder_width_cm: float | None = None

    def to_dict(self) -> dict:
        """Only measured fields; None (unmeasurable) fields are omitted."""
        return {k: round(float(v), 1) for k, v in asdict(self).items() if v is not None}


def compute_measurements(
    vertices: np.ndarray,
    joints: np.ndarray,
    faces: np.ndarray,
    sex: str,  # kept for signature compatibility; SMPL-X GT is sex-agnostic here
) -> BodyMeasurementsGT:
    """Measure an SMPL-X mesh. Fields that fail or fall out of plausible range
    are left None — the body is then only as usable as its measured fields."""
    try:
        raw = measure_smplx(vertices, joints, faces)
    except Exception as exc:  # noqa: BLE001 — one bad mesh must not kill the run
        logger.warning("measure_smplx failed: %s", exc)
        return BodyMeasurementsGT()

    fields = {f.name for f in BodyMeasurementsGT.__dataclass_fields__.values()}
    kept: dict[str, float] = {}
    for field, value in raw.items():
        if field not in fields:
            continue
        lo, hi = _BOUNDS.get(field, (float("-inf"), float("inf")))
        if lo <= value <= hi:
            kept[field] = value
        else:
            logger.debug("Discarding %s=%.1f (outside [%.0f, %.0f])", field, value, lo, hi)
    return BodyMeasurementsGT(**kept)


def has_core_measurements(m: BodyMeasurementsGT) -> bool:
    """True iff all four garment-critical circumferences were measured.
    Callers should skip bodies for which this is False."""
    return all(getattr(m, f) is not None for f in CORE_FIELDS)


def sanity_check(m: BodyMeasurementsGT) -> list[str]:
    """Report which fields are missing (unmeasurable). Kept for callers that
    log per-body diagnostics; bounds are already enforced in compute."""
    return [f"{k} not measurable" for k, v in asdict(m).items() if v is None]
