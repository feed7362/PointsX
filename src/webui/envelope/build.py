"""Assemble a MeasurementEnvelope from a pipeline InferenceResult."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints
from webui.envelope.catalog import CANONICAL_MEASUREMENTS, _DEFAULT_CONFIDENCE, _PLAUSIBLE_RANGE_CM
from webui.envelope.corrections import (
    _LENGTH_SCALES_PCT,
    _SEX_CIRCUMFERENCE_SCALES_PCT,
    _SEX_SCALE_TARGET_IDS,
)
from webui.envelope.derive import (
    _derive_ankle,
    _derive_back_length,
    _derive_chest_circumference,
    _derive_front_length,
    _derive_neck_base_height,
    _derive_upper_arm,
)
from webui.envelope.priors import PRIOR_IDS, predict_prior
from webui.schemas import (
    CaptureInfo,
    CaptureQuality,
    MeasurementEnvelope,
    MeasurementItem,
    PipelineInfo,
    SubjectInfo,
)

if TYPE_CHECKING:
    from webui.infrastructure.inference import InferenceResult


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
    *,
    height_cm: float,
    sex: str,
    chest_for_prior: float | None,
) -> tuple[float | None, list[str]]:
    """Return (value_cm, quality_flags) for a single canonical id.

    `chest_circ_cm` is precomputed once because other IDs depend on it; `chest_for_prior` is the
    same value after the per-sex correction, i.e. what the user sees.
    """
    flags: list[str] = []

    # Population priors (ANSUR II) for what the silhouette cannot resolve -----
    if mid in PRIOR_IDS:
        flags.append("prior")
        return predict_prior(mid, sex, height_cm, chest_for_prior), flags

    # Direct mappings -------------------------------------------------------
    if mid == "waist_circumference":           return bm.waist_circumference_cm, flags
    if mid == "hip_circumference":             return bm.hip_circumference_cm, flags
    if mid == "thigh_circumference":           return bm.thigh_circumference_cm, flags
    if mid == "calf_circumference":            return bm.calf_circumference_cm, flags
    if mid == "wrist_circumference":           return bm.wrist_circumference_cm, flags
    if mid == "chest_width_front":             return bm.torso_width_front_cm, flags
    if mid == "shoulder_slope_width":          return bm.shoulder_slope_width_cm, flags
    if mid == "arm_length_shoulder_to_wrist":  return bm.arm_length_cm, flags
    if mid == "leg_length_inner_seam":         return bm.leg_length_inner_cm, flags
    if mid == "leg_length_outer_seam":         return bm.leg_length_outer_cm, flags

    # Geometric derivations -------------------------------------------------
    if mid == "chest_circumference":
        flags.append("derived")
        return chest_circ_cm, flags
    if mid == "back_length_to_waist":
        flags.append("derived")
        return _derive_back_length(bm, side_kp, cal.px_per_cm_side), flags
    if mid == "front_length_to_waist":
        flags.append("derived")
        return _derive_front_length(bm, front_kp, cal.px_per_cm_front), flags
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
    *,
    apply_sex_offsets: bool = True,
    sex_offsets_override: dict[str, dict[str, float]] | None = None,
) -> Any:
    """Build a MeasurementEnvelope from a WebuiPipeline InferenceResult.

    Args:
        apply_sex_offsets: when False, skip the per-sex multiplicative bias
            correction (the historic name is kept; today this controls
            _SEX_CIRCUMFERENCE_SCALES_PCT, not the deprecated additive table).
        sex_offsets_override: percent-scale dict ``{sex: {mid: pct}}`` that
            substitutes _SEX_CIRCUMFERENCE_SCALES_PCT for this call. Used by
            ``pointsx-eval --fit-offsets`` to A/B-test newly fitted scales.
    """

    bm = result.body
    chest_circ_cm = _derive_chest_circumference(bm)

    if apply_sex_offsets:
        scales_table = sex_offsets_override or _SEX_CIRCUMFERENCE_SCALES_PCT
        sex_scales_pct = scales_table.get(sex, {})
    else:
        sex_scales_pct = {}
    chest_for_prior = chest_circ_cm
    if chest_for_prior is not None and "chest_circumference" in sex_scales_pct:
        chest_for_prior *= 1.0 + float(sex_scales_pct["chest_circumference"]) / 100.0

    items = []
    out_of_range: list[str] = []
    for mid, label_uk, source in CANONICAL_MEASUREMENTS:
        value, flags = _value_for_id(
            mid, bm, result.front_kp, result.side_kp, result.cal, chest_circ_cm,
            height_cm=subject_height_cm, sex=sex, chest_for_prior=chest_for_prior,
        )
        if value is None:
            # Skip — frontend size engine tolerates missing measurements.
            continue
        # Apply per-sex multiplicative bias correction (whitelisted IDs only).
        if mid in _SEX_SCALE_TARGET_IDS and mid in sex_scales_pct:
            value = max(0.0, float(value) * (1.0 + float(sex_scales_pct[mid]) / 100.0))
        # Sex-INDEPENDENT correction for the non-circumference measurements.
        elif apply_sex_offsets and mid in _LENGTH_SCALES_PCT:
            value = max(0.0, float(value) * (1.0 + _LENGTH_SCALES_PCT[mid] / 100.0))

        # Sanity-range gate: never drop. Tag with `out_of_range` so callers and
        # the UI can mark it visually, but the value is still surfaced. Pydantic
        # uncertainty stays non-negative thanks to abs(value) below.
        lo, hi = _PLAUSIBLE_RANGE_CM.get(mid, (0.5, 250.0))
        if not (lo <= float(value) <= hi):
            flags = list(flags) + ["out_of_range"]
            out_of_range.append(f"{mid}={float(value):.1f}")

        # Confidence: prefer pipeline-provided, fall back to per-id default.
        conf = bm.confidence.get(mid, _DEFAULT_CONFIDENCE.get(mid, 0.5))
        conf = max(0.0, min(1.0, float(conf)))
        # Uncertainty: 5 % of |value| scaled by (1 − confidence).
        uncertainty = round((1.0 - conf) * abs(float(value)) * 0.05, 2)

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
            from webui.visualize import pipeline_visualizations_b64
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
            pose_backend=result.pose_backend,
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
        warnings=list(bm.warnings) + (
            [f"Out of plausible range: {', '.join(out_of_range)}"]
            if out_of_range else []
        ),
    )
