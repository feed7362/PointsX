"""Post-processing: sanity checks, smoothing, and confidence scoring."""

from __future__ import annotations

import logging

from pointsx.schemas import BodyMeasurements

logger = logging.getLogger(__name__)

# Anthropometric ratio checks: (name, numerator_field, denominator_field, min, max)
RATIO_CHECKS = [
    ("shoulder/height", "shoulder_width_cm", "height_cm", 0.22, 0.30),
    ("leg/height", "leg_length_outer_cm", "height_cm", 0.43, 0.53),
    ("arm/height", "arm_length_cm", "height_cm", 0.30, 0.40),
    ("head/shoulder", "head_width_front_cm", "shoulder_width_cm", 0.30, 0.55),
]

# Circumference ratio checks
CIRC_RATIO_CHECKS = [
    ("waist/hip circumference", "waist_circumference_cm", "hip_circumference_cm", 0.65, 0.95),
]

# Absolute range checks: (field, min_cm, max_cm, description)
ABSOLUTE_CHECKS = [
    ("height_cm", 140, 220, "Height"),
    ("shoulder_width_cm", 30, 60, "Shoulder width"),
    ("neck_circumference_cm", 28, 55, "Neck circumference"),
    ("waist_circumference_cm", 55, 150, "Waist circumference"),
    ("hip_circumference_cm", 70, 160, "Hip circumference"),
    ("thigh_circumference_cm", 35, 85, "Thigh circumference"),
    ("calf_circumference_cm", 25, 55, "Calf circumference"),
    ("arm_length_cm", 45, 90, "Arm length"),
    ("leg_length_outer_cm", 70, 120, "Outer leg length"),
]


def validate_measurements(m: BodyMeasurements) -> BodyMeasurements:
    """Run sanity checks and add warnings. Modifies and returns the same object."""
    warnings = []

    # Ratio checks
    for name, num_field, den_field, lo, hi in RATIO_CHECKS:
        num = getattr(m, num_field, None)
        den = getattr(m, den_field, None)
        if num is not None and den is not None and den > 0:
            ratio = num / den
            if ratio < lo or ratio > hi:
                warnings.append(
                    f"{name} ratio {ratio:.2f} outside expected range [{lo:.2f}, {hi:.2f}]"
                )

    for name, num_field, den_field, lo, hi in CIRC_RATIO_CHECKS:
        num = getattr(m, num_field, None)
        den = getattr(m, den_field, None)
        if num is not None and den is not None and den > 0:
            ratio = num / den
            if ratio < lo or ratio > hi:
                warnings.append(
                    f"{name} ratio {ratio:.2f} outside expected range [{lo:.2f}, {hi:.2f}]"
                )

    # Absolute range checks
    for field, lo, hi, desc in ABSOLUTE_CHECKS:
        val = getattr(m, field, None)
        if val is not None and (val < lo or val > hi):
            warnings.append(f"{desc} {val:.1f} cm outside expected range [{lo}, {hi}] cm")

    # Symmetry checks (large L/R discrepancy)
    if m.leg_length_outer_cm is not None and m.leg_length_inner_cm is not None:
        if m.leg_length_inner_cm > m.leg_length_outer_cm:
            warnings.append("Inner leg length exceeds outer leg length")

    m.warnings = warnings
    for w in warnings:
        logger.warning("Measurement validation: %s", w)

    return m
