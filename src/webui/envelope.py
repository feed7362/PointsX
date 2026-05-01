"""Convert a `BodyMeasurements` (+ keypoints, calibration) to a `MeasurementEnvelope`.

The envelope is the public API contract consumed by the frontend size + pattern
engines. It exposes 18 canonical measurement IDs. The `BodyMeasurements`
dataclass maps directly to 11 of them; the other 7 are derived here from
keypoints + widths (no extra ML required).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pointsx.circumference import ramanujan_ellipse_circumference
from pointsx.keypoints import KP, distance, is_valid, midpoint
from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints

from webui.inference import InferenceResult
from webui.visualize import pipeline_visualizations_b64


# ---------------------------------------------------------------------------
# Canonical 18-id table (ordered, stable). Each row is:
#   (envelope_id, label_uk, source_view)
# ---------------------------------------------------------------------------

CANONICAL_MEASUREMENTS: list[tuple[str, str, str]] = [
    ("chest_circumference",          "Обхват грудей",                                        "fused"),
    ("waist_circumference",          "Обхват талії",                                         "fused"),
    ("hip_circumference",            "Обхват стегон",                                        "fused"),
    ("neck_circumference",           "Обхват шиї",                                           "front"),
    ("neck_base_height",             "Висота точки основи шиї",                              "front"),
    ("shoulder_slope_width",         "Ширина плечового ската",                               "front"),
    ("back_width_scapular",          "Ширина спини (між лопатками)",                         "side"),
    ("chest_width_front",            "Ширина грудей (між пахвами спереду)",                  "front"),
    ("back_length_to_waist",         "Довжина спини до талії (по хребту)",                   "side"),
    ("front_length_to_waist",        "Довжина переду до талії (через найвищу точку грудей)", "side"),
    ("arm_length_shoulder_to_wrist", "Довжина руки (від плеча до зап'ястя)",                 "side"),
    ("upper_arm_circumference",      "Обхват плеча (біцепс)",                                "fused"),
    ("wrist_circumference",          "Обхват зап'ястя",                                      "fused"),
    ("leg_length_inner_seam",        "Довжина ноги по внутрішньому шву",                     "side"),
    ("leg_length_outer_seam",        "Довжина ноги по зовнішньому шву",                      "side"),
    ("thigh_circumference",          "Обхват стегна",                                        "fused"),
    ("calf_circumference",           "Обхват гомілки (литки)",                               "side"),
    ("ankle_circumference",          "Обхват щиколотки",                                     "fused"),
]

# Default per-id confidence (used when BodyMeasurements.confidence is empty)
_DEFAULT_CONFIDENCE: dict[str, float] = {
    # Direct circumferences from regressor / ellipse — high
    "waist_circumference":          0.85,
    "hip_circumference":            0.85,
    "neck_circumference":           0.80,
    "thigh_circumference":          0.80,
    "calf_circumference":           0.85,
    "wrist_circumference":          0.85,
    # Direct widths / lengths — medium-high
    "chest_width_front":            0.75,
    "shoulder_slope_width":         0.70,
    "arm_length_shoulder_to_wrist": 0.70,
    "leg_length_inner_seam":        0.70,
    "leg_length_outer_seam":        0.70,
    # Geometric derivations
    "chest_circumference":          0.70,
    "back_width_scapular":          0.55,
    "back_length_to_waist":         0.65,
    "front_length_to_waist":        0.65,
    "neck_base_height":             0.65,
    # Anthropometric ratio approximations — low
    "upper_arm_circumference":      0.40,
    "ankle_circumference":          0.40,
}

# Anthropometric ratios used when a measurement isn't directly observable
_UPPER_ARM_TO_CHEST_RATIO = 0.34   # adult average upper-arm girth ≈ 33-35% of chest girth
_ANKLE_TO_CALF_RATIO      = 0.62


# ---------------------------------------------------------------------------
# Pydantic models — re-imported from app to avoid duplication
# ---------------------------------------------------------------------------

# Imports at function-call time to avoid circular imports (app imports envelope,
# envelope would import app)
def _envelope_models():  # pragma: no cover - tiny indirection
    from webui.app import (
        CaptureInfo,
        CaptureQuality,
        MeasurementEnvelope,
        MeasurementItem,
        PipelineInfo,
        SubjectInfo,
    )
    return MeasurementEnvelope, MeasurementItem, PipelineInfo, SubjectInfo, CaptureInfo, CaptureQuality


# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _kp_distance_cm(kp: Keypoints, a: KP, b: KP, px_per_cm: float) -> float | None:
    """Pixel distance between two keypoints, converted to cm (or None if invalid)."""
    if not is_valid(kp.confidence, a, b):
        return None
    if px_per_cm <= 0:
        return None
    return distance(kp.points, a, b) / px_per_cm


def _kp_to_midpoint_cm(
    kp: Keypoints, single: KP, mid_a: KP, mid_b: KP, px_per_cm: float,
) -> float | None:
    """Distance from a single keypoint to the midpoint of two others, in cm."""
    if not is_valid(kp.confidence, single, mid_a, mid_b):
        return None
    if px_per_cm <= 0:
        return None
    p = kp.points[int(single)]
    m = midpoint(kp.points, mid_a, mid_b)
    import numpy as np
    return float(np.linalg.norm(p - m)) / px_per_cm


def _derive_chest_circumference(bm: BodyMeasurements) -> float | None:
    fw = bm.torso_width_front_cm
    sw = bm.torso_width_side_cm
    if fw is None or sw is None:
        return None
    return ramanujan_ellipse_circumference(fw, sw)


def _derive_back_length(side_kp: Keypoints, px_per_cm_side: float) -> float | None:
    # Side view: vertical span from upper neck to mid-hips.
    return _kp_to_midpoint_cm(side_kp, KP.UPPER_NECK, KP.LEFT_HIP, KP.RIGHT_HIP, px_per_cm_side)


def _derive_front_length(front_kp: Keypoints, px_per_cm_front: float) -> float | None:
    # Front view proxy for front body length to waist.
    return _kp_to_midpoint_cm(front_kp, KP.UPPER_NECK, KP.LEFT_HIP, KP.RIGHT_HIP, px_per_cm_front)


def _derive_neck_base_height(front_kp: Keypoints, px_per_cm_front: float) -> float | None:
    # Vertical span from upper neck to mid-ankles ≈ standing height minus head.
    return _kp_to_midpoint_cm(front_kp, KP.UPPER_NECK, KP.LEFT_ANKLE, KP.RIGHT_ANKLE, px_per_cm_front)


def _derive_upper_arm(chest_circ: float | None) -> float | None:
    if chest_circ is None or chest_circ <= 0:
        return None
    return chest_circ * _UPPER_ARM_TO_CHEST_RATIO


def _derive_ankle(calf_circ: float | None) -> float | None:
    if calf_circ is None or calf_circ <= 0:
        return None
    return calf_circ * _ANKLE_TO_CALF_RATIO


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def _value_for_id(
    mid: str,
    bm: BodyMeasurements,
    front_kp: Keypoints,
    side_kp: Keypoints,
    cal: CalibrationInfo,
    chest_circ_cm: float | None,
) -> tuple[float | None, list[str]]:
    """Return (value_cm, quality_flags) for a single canonical id.

    `chest_circ_cm` is precomputed once because two other IDs depend on it.
    """
    flags: list[str] = []

    # Direct mappings -------------------------------------------------------
    if mid == "waist_circumference":           return bm.waist_circumference_cm, flags
    if mid == "hip_circumference":             return bm.hip_circumference_cm, flags
    if mid == "neck_circumference":            return bm.neck_circumference_cm, flags
    if mid == "thigh_circumference":           return bm.thigh_circumference_cm, flags
    if mid == "calf_circumference":            return bm.calf_circumference_cm, flags
    if mid == "wrist_circumference":           return bm.wrist_circumference_cm, flags
    if mid == "chest_width_front":             return bm.torso_width_front_cm, flags
    if mid == "shoulder_slope_width":          return bm.shoulder_width_cm, flags
    if mid == "arm_length_shoulder_to_wrist":  return bm.arm_length_cm, flags
    if mid == "leg_length_inner_seam":         return bm.leg_length_inner_cm, flags
    if mid == "leg_length_outer_seam":         return bm.leg_length_outer_cm, flags

    # Geometric derivations -------------------------------------------------
    if mid == "chest_circumference":
        flags.append("derived")
        return chest_circ_cm, flags
    if mid == "back_width_scapular":
        flags.append("proxy")
        return bm.torso_width_side_cm, flags
    if mid == "back_length_to_waist":
        flags.append("derived")
        return _derive_back_length(side_kp, cal.px_per_cm_side), flags
    if mid == "front_length_to_waist":
        flags.append("derived")
        return _derive_front_length(front_kp, cal.px_per_cm_front), flags
    if mid == "neck_base_height":
        flags.append("derived")
        return _derive_neck_base_height(front_kp, cal.px_per_cm_front), flags

    # Anthropometric approximations -----------------------------------------
    if mid == "upper_arm_circumference":
        flags.append("approximation")
        return _derive_upper_arm(chest_circ_cm), flags
    if mid == "ankle_circumference":
        flags.append("approximation")
        return _derive_ankle(bm.calf_circumference_cm), flags

    return None, flags


def body_to_envelope(
    result: InferenceResult,
    subject_height_cm: float,
    sex: Literal["male", "female", "other"],
    request_id: str,
    front_bgr: Any | None = None,
    side_bgr: Any | None = None,
) -> Any:
    """Build a MeasurementEnvelope from a WebuiPipeline InferenceResult."""
    (
        MeasurementEnvelope,
        MeasurementItem,
        PipelineInfo,
        SubjectInfo,
        CaptureInfo,
        CaptureQuality,
    ) = _envelope_models()

    bm = result.body
    chest_circ_cm = _derive_chest_circumference(bm)

    items = []
    for mid, label_uk, source in CANONICAL_MEASUREMENTS:
        value, flags = _value_for_id(
            mid, bm, result.front_kp, result.side_kp, result.cal, chest_circ_cm,
        )
        if value is None:
            # Skip — frontend size engine tolerates missing measurements.
            continue
        # Confidence: prefer pipeline-provided, fall back to per-id default.
        conf = bm.confidence.get(mid, _DEFAULT_CONFIDENCE.get(mid, 0.5))
        conf = max(0.0, min(1.0, float(conf)))
        # Uncertainty: 5 % of value scaled by (1 − confidence).
        uncertainty = round((1.0 - conf) * float(value) * 0.05, 2)

        items.append(MeasurementItem(
            id=mid,
            label_uk=label_uk,
            value_cm=round(float(value), 1),
            uncertainty_cm=uncertainty,
            confidence=round(conf, 2),
            source=source,  # type: ignore[arg-type]
            quality_flags=flags,
        ))

    derived: dict[str, Any] = {}
    if front_bgr is not None and side_bgr is not None:
        try:
            derived = pipeline_visualizations_b64(front_bgr, side_bgr, result)
        except Exception:
            # Keep API response valid even if debug visualization generation fails.
            derived = {}

    return MeasurementEnvelope(
        schema="pointsx.measurement.envelope",
        schema_version=2,
        request_id=request_id,
        created_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        pipeline=PipelineInfo(
            source="regression" if result.has_regressor else "mediapipe",
            model_version="regression-0.1" if result.has_regressor else "ellipse-0.1",
            unit_system="metric",
        ),
        subject=SubjectInfo(
            height_cm=subject_height_cm,
            sex=sex,
            posture_flags=[],
        ),
        capture=CaptureInfo(
            front=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
            side=CaptureQuality(quality=1.0, pose_ok=True, occlusions=[]),
        ),
        measurements=items,
        derived=derived,
        warnings=list(bm.warnings),
    )
