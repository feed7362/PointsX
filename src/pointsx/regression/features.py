"""Feature vector construction for the circumference regression model."""

from __future__ import annotations

import numpy as np

from pointsx.circumference import ramanujan_ellipse_circumference
from pointsx.schemas import BodyMeasurements


def build_feature_vector(m: BodyMeasurements) -> np.ndarray | None:
    """Build a 28-dimensional feature vector from body measurements.

    Features (28 total):
        [0-6]   7 front widths: head, neck, torso, waist, hip, thigh, calf (cm)
        [7-13]  7 side widths: head, neck, torso, waist, hip, thigh, calf (cm)
        [14-19] 6 ellipse circumferences: neck, waist, hip, thigh, calf, wrist (cm)
        [20-23] 4 scalar measurements: height, shoulder_width, torso_length, leg_length (cm)
        [24-27] 4 ratios: waist/hip, chest/waist, thigh/hip, shoulder/hip

    Returns None if essential measurements are missing.
    """
    # Front widths
    front_widths = [
        m.head_width_front_cm,
        m.neck_width_front_cm,
        m.torso_width_front_cm,
        m.waist_width_front_cm,
        m.hip_width_front_cm,
        m.thigh_width_front_cm,
        m.calf_width_front_cm,
    ]

    # Side widths
    side_widths = [
        m.head_depth_side_cm,
        m.neck_width_side_cm,
        m.torso_width_side_cm,
        m.waist_width_side_cm,
        m.hip_width_side_cm,
        m.thigh_width_side_cm,
        m.calf_width_side_cm,
    ]

    # Ellipse circumference estimates
    pairs = [
        (m.neck_width_front_cm, m.neck_width_side_cm),
        (m.waist_width_front_cm, m.waist_width_side_cm),
        (m.hip_width_front_cm, m.hip_width_side_cm),
        (m.thigh_width_front_cm, m.thigh_width_side_cm),
        (m.calf_width_front_cm, m.calf_width_side_cm),
        (m.wrist_width_front_cm, m.wrist_width_side_cm),
    ]
    ellipse_circs = []
    for fw, sw in pairs:
        if fw is not None and sw is not None:
            ellipse_circs.append(ramanujan_ellipse_circumference(fw, sw))
        else:
            ellipse_circs.append(0.0)

    # Scalar measurements
    height = m.height_cm or 0.0
    shoulder = m.shoulder_width_cm or 0.0
    # Torso length approximation: neck to hip
    torso_length = 0.0
    if m.height_cm and m.leg_length_outer_cm:
        torso_length = m.height_cm - m.leg_length_outer_cm
    leg_length = m.leg_length_outer_cm or 0.0

    scalars = [height, shoulder, torso_length, leg_length]

    # Dimensionless ratios (0.0 if denominator is missing)
    def _ratio(a: float | None, b: float | None) -> float:
        if a and b and b > 0:
            return a / b
        return 0.0

    ratios = [
        _ratio(m.waist_width_front_cm, m.hip_width_front_cm),
        _ratio(m.torso_width_front_cm, m.waist_width_front_cm),
        _ratio(m.thigh_width_front_cm, m.hip_width_front_cm),
        _ratio(m.shoulder_width_cm, m.hip_width_front_cm),
    ]

    # Replace None with 0.0
    front_widths = [v or 0.0 for v in front_widths]
    side_widths = [v or 0.0 for v in side_widths]

    features = front_widths + side_widths + ellipse_circs + scalars + ratios
    assert len(features) == 28

    # If too many features are zero, the prediction won't be useful
    nonzero = sum(1 for f in features if f != 0.0)
    if nonzero < 10:
        return None

    return np.array(features, dtype=np.float32)
