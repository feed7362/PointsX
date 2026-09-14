"""Convert a `BodyMeasurements` (+ keypoints, calibration) to a `MeasurementEnvelope`.

The envelope is the public API contract consumed by the frontend size + pattern
engines. It exposes 18 canonical measurement IDs. The `BodyMeasurements`
dataclass maps directly to 11 of them; the other 7 are derived here from
keypoints + widths (no extra ML required).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from pointsx.circumference import ramanujan_ellipse_circumference
from pointsx.keypoints import KP, distance, is_valid, midpoint
from pointsx.schemas import BodyMeasurements, CalibrationInfo, Keypoints
from webui.schemas import (
    CaptureInfo,
    CaptureQuality,
    MeasurementEnvelope,
    MeasurementItem,
    PipelineInfo,
    SubjectInfo,
)

if TYPE_CHECKING:
    from webui.inference import InferenceResult


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

# IDs surfaced in user-facing displays (eval table, webui results panel). The
# hidden remainder of CANONICAL_MEASUREMENTS is still computed and returned in
# the API response because the pattern engine / size charts consume it, but
# these are the only ones the user reads on screen.
#
# Order matches the webui's MEASUREMENT_MANUAL_ORDER in tailoring.js
# (after HIDDEN_MEASUREMENT_IDS is applied) — keep them in lockstep when
# editing one or the other.
DISPLAY_MEASUREMENT_IDS: list[str] = [
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "thigh_circumference",
    "neck_base_height",
    "chest_width_front",
    "shoulder_slope_width",
    "arm_length_shoulder_to_wrist",
    "leg_length_outer_seam",
    "leg_length_inner_seam",
    "back_length_to_waist",
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

# Per-sex multiplicative bias correction for the four core circumferences.
# Values are PERCENT scale-factors applied as ``value *= 1 + pct/100``. They
# only apply to these four IDs (chest/waist/hip/thigh) — other measurements
# are not bias-corrected here.
#
# Fit fresh values via ``pointsx-eval --fit-offsets`` and paste the printed dict back here when new
# ground-truth subjects arrive. Refitted 2026-07-20 on the app GT corpus (n=11
# real subjects with tape measurements, 8 F / 3 M), AFTER the thigh-width
# extraction fix. Method: median(gt / predicted), the L1-optimal multiplicative
# correction, rounded to 0.5 %.
#
# Male chest is deliberately absent (see the inline note on the female cell):
# its fitted +3.9 % improves in-sample but worsens leave-one-out — n=3
# overfitting. Leave-one-out MAE over the four circumferences: 6.50 cm with
# this table, 6.81 cm correcting all four, 11.0 cm with no correction at all.
#
# CAVEAT: the male cells rest on n=3 subjects — treat them as provisional and
# refit once the corpus has ~8+ male subjects.
_SEX_CIRCUMFERENCE_SCALES_PCT: dict[str, dict[str, float]] = {
    "female": {  # n=8
        # Chest RESTORED 2026-07-20 after a real-world report of ~10 cm
        # over-measurement. It had been gated to zero because its bootstrap CI
        # spans 1.0 — but "not significant" means UNCERTAIN, not "use zero": the
        # fitted point estimate is still the best single guess, and dropping it
        # measured worse (MAE 5.9 uncorrected vs 5.0 here; LOOCV 5.9 vs 5.6).
        # Male chest is deliberately still absent: its fitted +3.9 % improves
        # in-sample (4.6) but WORSENS leave-one-out (6.3) — n=3 overfitting.
        "chest_circumference":  -4.5,
        # Waist refitted 2026-07-29 for the new FRONT anchor (fixed 0.25 of the
        # pelvis->neck span instead of the narrowest row). MAE 5.98 -> 4.45.
        "waist_circumference": -17.5,   # %
        "hip_circumference":    -5.0,
        "thigh_circumference": -17.0,
    },
    "male": {  # n=3 — provisional
        "waist_circumference": -15.5,
        "hip_circumference":   -10.5,
        "thigh_circumference": -23.5,
    },
    # "other" averages male and female so an unknown-sex subject is biased
    # toward neither extreme.
    "other": {
        "waist_circumference": -14.0,
        "hip_circumference":    -8.0,
        "thigh_circumference": -20.0,
    },
}

# Sex-INDEPENDENT bias corrections for the non-circumference measurements.
# Fitted 2026-07-20 on the same n=11 app GT corpus, median(gt / predicted).
#
# Why these were the biggest remaining errors: the per-sex table above only ever
# covered the four circumferences, so lengths and heights carried their full
# systematic bias uncorrected. They were the WORST measurements in the pipeline
# (inner seam MAE 10.7 cm with bias +10.7 — i.e. essentially pure offset, no
# scatter), simply because nothing corrected them.
#
# Sex-INDEPENDENT on purpose: splitting these by sex measured WORSE in
# leave-one-out (5.45 vs 5.39 cm overall) — 8 female / 3 male is too thin to
# support per-sex length fits, so the split fits noise.
#
# Only measurements passing BOTH gates are listed: (a) the bootstrap 95 % CI on
# the fitted ratio excludes 1.0, and (b) leave-one-out MAE improves by >0.2 cm.
# Deliberately EXCLUDED by those gates:
#   chest_width_front     +8.9 % but CI [-1.0, +25.0] spans zero, LOOCV -0.3
#   back_length_to_waist  +6.8 % but CI [-0.7, +13.6] spans zero, LOOCV +0.7
#
# Leave-one-out overall MAE: 6.18 cm before -> 5.23 cm with this table.
_LENGTH_SCALES_PCT: dict[str, float] = {
    "leg_length_inner_seam":  -9.5,   # %   MAE 10.7 -> 6.8
    "leg_length_outer_seam":  -2.5,   #     MAE  8.7 -> 7.7
    "neck_base_height":       +4.0,   #     MAE  6.3 -> 2.8
}

# Set of IDs eligible for the multiplicative bias correction. Anything outside
# this set is left untouched by the sex-scale logic.
_SEX_SCALE_TARGET_IDS: set[str] = {
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "thigh_circumference",
}

# Anthropometric ratios used when a measurement isn't directly observable
_UPPER_ARM_TO_CHEST_RATIO = 0.34   # adult average upper-arm girth ≈ 33-35% of chest girth
_ANKLE_TO_CALF_RATIO      = 0.62

# Per-id plausible range. Measurements outside this band are dropped (None) so
# the envelope never advertises an impossible value. Real upstream model
# failures (e.g. regressor outputting negative cm for occluded limbs) would
# otherwise reach Pydantic and surface as a 500.
_PLAUSIBLE_RANGE_CM: dict[str, tuple[float, float]] = {
    "neck_circumference":           (20.0,  70.0),
    "chest_circumference":          (60.0, 160.0),
    "waist_circumference":          (50.0, 160.0),
    "hip_circumference":            (60.0, 170.0),
    "thigh_circumference":          (30.0,  90.0),
    "calf_circumference":           (20.0,  60.0),
    "wrist_circumference":          (10.0,  25.0),
    "upper_arm_circumference":      (15.0,  60.0),
    "ankle_circumference":          (15.0,  35.0),
    "shoulder_slope_width":         ( 8.0,  25.0),
    "back_width_scapular":          (15.0,  50.0),
    "chest_width_front":            (20.0,  60.0),
    "neck_base_height":             (110.0, 200.0),
    "back_length_to_waist":         (25.0,  60.0),
    "front_length_to_waist":        (25.0,  60.0),
    "arm_length_shoulder_to_wrist": (35.0,  90.0),
    "leg_length_inner_seam":        (50.0, 100.0),
    "leg_length_outer_seam":        (60.0, 115.0),
}


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
    """Derive chest girth from front/side torso widths via Ramanujan ellipse.

    Returns None when the input widths are outside anatomical ranges — usually a
    sign that the silhouette was corrupted (e.g. arms outstretched intersecting
    the torso slice). The regressor handles waist/hip/neck/etc. similarly by
    being bounded; chest has no regressor output, so we sanity-check here.
    """
    fw = bm.torso_width_front_cm
    sw = bm.torso_width_side_cm
    if fw is None or sw is None:
        return None
    # Plausible adult human torso (front) ≈ 22-50 cm; (side / depth) ≈ 14-38 cm
    if not (22.0 <= fw <= 50.0) or not (14.0 <= sw <= 38.0):
        return None
    circ = ramanujan_ellipse_circumference(fw, sw)
    # Final sanity cap on the resulting circumference (60-150 cm covers everyone)
    if not (60.0 <= circ <= 150.0):
        return None
    return circ


def _derive_back_length(
    bm: BodyMeasurements, side_kp: Keypoints, px_per_cm_side: float
) -> float | None:
    """Back length from C7 (UPPER_NECK proxy) to natural waist, on the side view.

    Uses a stable proportional anchor for the waist:
    ``y_waist = upper_neck_y + 0.65 × (pelvis_y − upper_neck_y)`` — anatomical
    natural waist sits ~65 % of the way down the torso. The silhouette-detected
    waist (``bm.waist_level_side_px``) was previously preferred, but on the
    eval set it drifted subject-to-subject (RMSE 7.87 cm with ±13 cm outliers)
    because the side-view torso lacks a clear narrowest point. Proportional
    keypoint anchor is far more stable.
    """
    if px_per_cm_side <= 0:
        return None
    pts = side_kp.points
    conf = side_kp.confidence
    if not is_valid(conf, KP.UPPER_NECK):
        return None
    upper_neck_y = float(pts[KP.UPPER_NECK, 1])

    # Pelvis: prefer the explicit keypoint, fall back to midpoint of hips.
    if is_valid(conf, KP.PELVIS):
        pelvis_y = float(pts[KP.PELVIS, 1])
    elif is_valid(conf, KP.LEFT_HIP, KP.RIGHT_HIP):
        pelvis_y = (float(pts[KP.LEFT_HIP, 1]) + float(pts[KP.RIGHT_HIP, 1])) / 2
    else:
        return None

    waist_y = upper_neck_y + 0.65 * (pelvis_y - upper_neck_y)
    return abs(waist_y - upper_neck_y) / px_per_cm_side


def _derive_front_length(
    bm: BodyMeasurements, front_kp: Keypoints, px_per_cm_front: float
) -> float | None:
    # Prefer persisted waist level if available.
    if bm.waist_level_front_px is not None and is_valid(front_kp.confidence, KP.UPPER_NECK):
        if px_per_cm_front <= 0:
            return None
        return abs(float(front_kp.points[KP.UPPER_NECK, 1]) - float(bm.waist_level_front_px)) / px_per_cm_front
    # Fallback: old proxy from upper neck to hip midpoint.
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
    if mid == "shoulder_slope_width":          return bm.shoulder_slope_width_cm, flags
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

    items = []
    out_of_range: list[str] = []
    for mid, label_uk, source in CANONICAL_MEASUREMENTS:
        value, flags = _value_for_id(
            mid, bm, result.front_kp, result.side_kp, result.cal, chest_circ_cm,
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
