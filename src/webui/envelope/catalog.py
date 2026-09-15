"""The 18 canonical envelope measurements: ids, labels, display subset, default confidence, plausible ranges."""
from __future__ import annotations

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
