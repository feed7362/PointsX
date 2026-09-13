"""Circumference estimation from front + side width measurements."""

from __future__ import annotations

import math

from pointsx.schemas import BodyMeasurements

# Body parts with a (front width, side width) pair on BodyMeasurements. This is
# also the regressor's output order (see _estimate_with_regression).
ELLIPSE_PARTS = ("neck", "waist", "hip", "thigh", "calf", "wrist")


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
    for part in ELLIPSE_PARTS:
        setattr(
            m,
            f"{part}_circumference_cm",
            _estimate_single(getattr(m, f"{part}_width_front_cm"), getattr(m, f"{part}_width_side_cm")),
        )

    return m


def _estimate_with_regression(m: BodyMeasurements, model) -> BodyMeasurements:
    """Use trained regression model for circumference estimation."""
    from pointsx.regression.features import build_feature_vector

    features = build_feature_vector(m)
    if features is None:
        # Not enough data for regression, fall back to ellipse
        return estimate_circumferences(m, regression_model=None)

    predictions = model.predict(features)

    for i, part in enumerate(ELLIPSE_PARTS):
        setattr(m, f"{part}_circumference_cm", float(predictions[i]))

    return m
