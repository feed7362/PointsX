"""Circumference estimation from front + side width measurements."""

from __future__ import annotations

import math

from pointsx.schemas import BodyMeasurements


def ramanujan_ellipse_circumference(width_a: float, width_b: float) -> float:
    """Estimate circumference of an ellipse using Ramanujan's approximation.

    Args:
        width_a: Full width from one projection (e.g., front view) in cm.
        width_b: Full width from other projection (e.g., side view) in cm.

    Returns:
        Estimated circumference in cm.
    """
    a = width_a / 2.0  # semi-axis
    b = width_b / 2.0  # semi-axis
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def _estimate_single(front_w: float | None, side_w: float | None) -> float | None:
    """Estimate circumference from front and side widths."""
    if front_w is not None and side_w is not None:
        return ramanujan_ellipse_circumference(front_w, side_w)
    # If only one projection available, assume circular cross-section
    if front_w is not None:
        return math.pi * front_w
    if side_w is not None:
        return math.pi * side_w
    return None


def estimate_circumferences(
    m: BodyMeasurements,
    regression_model=None,
) -> BodyMeasurements:
    """Add circumference estimates to measurements.

    Uses regression model when provided, otherwise falls back to ellipse approximation.
    Modifies and returns the same BodyMeasurements object.
    """
    if regression_model is not None:
        return _estimate_with_regression(m, regression_model)

    # Ellipse-based estimation for all circumferences
    m.neck_circumference_cm = _estimate_single(m.neck_width_front_cm, m.neck_width_side_cm)
    m.waist_circumference_cm = _estimate_single(m.waist_width_front_cm, m.waist_width_side_cm)
    m.hip_circumference_cm = _estimate_single(m.hip_width_front_cm, m.hip_width_side_cm)
    m.thigh_circumference_cm = _estimate_single(m.thigh_width_front_cm, m.thigh_width_side_cm)
    m.calf_circumference_cm = _estimate_single(m.calf_width_front_cm, m.calf_width_side_cm)
    m.wrist_circumference_cm = _estimate_single(m.wrist_width_front_cm, m.wrist_width_side_cm)

    return m


def _estimate_with_regression(m: BodyMeasurements, model) -> BodyMeasurements:
    """Use trained regression model for circumference estimation."""
    from pointsx.regression.features import build_feature_vector

    features = build_feature_vector(m)
    if features is None:
        # Not enough data for regression, fall back to ellipse
        return estimate_circumferences(m, regression_model=None)

    predictions = model.predict(features)

    m.neck_circumference_cm = float(predictions[0])
    m.waist_circumference_cm = float(predictions[1])
    m.hip_circumference_cm = float(predictions[2])
    m.thigh_circumference_cm = float(predictions[3])
    m.calf_circumference_cm = float(predictions[4])
    m.wrist_circumference_cm = float(predictions[5])

    return m
