"""Girth model: chest/waist/hip/thigh circumference (and weight) from height + six torso widths.

Linear model per sex and site fitted on ANSUR II (US Army 2012, cleared for unlimited public
release, so shippable): ``scripts/ansur/girth.py``, OLS on all subjects (1 986 F / 4 082 M).
Inputs are the caliper breadth/depth at chest, waist and hip; at runtime they are the pipeline's
silhouette widths, which over-read (clothing, segmentation), so the output is followed by the
per-site domain shift in ``corrections._GIRTH_SHIFT_PCT`` fitted on the app GT corpus.

Why fusion over the per-site ellipse: with exact widths both are near the floor (chest 1.6 vs 1.9,
waist 1.4 vs 1.4, hip 1.1 vs 1.2 cm, 5-fold CV), but with N(0, 1.5 cm) noise on every width the
fused model is the more robust one (chest F 3.35 vs 3.66, M 3.38 vs 4.22; waist 2.90 vs 2.99).
Thigh has no breadth/depth in ANSUR: torso widths + height give 2.55 / 2.66 cm (F / M) vs 4.2 / 4.4
from height alone. Weight from the same inputs: 3.3 / 4.2 kg with noise. A bilinear
breadth*depth term was tried and hurt under noise (waist +0.6 cm), so the model is plain linear.

Enable with ``POINTSX_GIRTH_MODEL=1`` (or ``body_to_envelope(girth_model=True)``).
"""
from __future__ import annotations

import os

from pointsx.schemas import BodyMeasurements

SITES = ("chest", "waist", "hip", "thigh")
GIRTH_IDS = {
    "chest": "chest_circumference",
    "waist": "waist_circumference",
    "hip": "hip_circumference",
    "thigh": "thigh_circumference",
}

# Coefficients per sex and site: (intercept, stature, chest breadth, chest depth, waist breadth,
# waist depth, hip breadth, buttock depth); all in cm, weight in kg. Regenerate with
# ``scripts/ansur/girth.py`` (writes girth_coefs.json) and paste.
_COEFS: dict[str, dict[str, tuple[float, ...]]] = {
    "female": {
        "chest": (7.72786, -0.01866, 1.07094, 2.05434, 0.19828, 0.10446, 0.00839, 0.07984),
        "waist": (1.62735, -0.00783, 0.12412, 0.11445, 1.74210, 1.23978, -0.12916, 0.23519),
        "hip": (1.55982, 0.04172, 0.02159, 0.07618, -0.07815, 0.04401, 1.83254, 1.19650),
        "thigh": (-2.45827, 0.01397, -0.00854, 0.00917, -0.11813, 0.08628, 0.99860, 1.20943),
        "weight_kg": (-119.72030, 0.46073, 0.71127, 0.74674, -0.07916, 0.57032, 0.91814, 1.39770),
    },
    "male": {
        "chest": (5.74906, -0.03571, 1.35526, 2.01588, 0.20343, 0.11223, 0.03919, 0.21677),
        "waist": (-1.02700, -0.01626, 0.06580, 0.14233, 1.50040, 1.54470, 0.12574, 0.09631),
        "hip": (1.08852, 0.01987, 0.07072, 0.21050, -0.03980, -0.06703, 1.78024, 1.27426),
        "thigh": (0.51675, -0.02268, -0.04845, 0.25968, 0.07236, -0.27227, 0.89476, 1.38168),
        "weight_kg": (-145.04939, 0.45150, 0.86886, 1.52139, 0.05082, 0.45292, 0.98879, 1.66357),
    },
}


def girth_model_enabled(override: bool | None = None) -> bool:
    if override is not None:
        return override
    return os.environ.get("POINTSX_GIRTH_MODEL", "").strip().lower() in ("1", "true", "yes", "on")


def _widths(bm: BodyMeasurements) -> list[float] | None:
    vals = (bm.torso_width_front_cm, bm.torso_width_side_cm, bm.waist_width_front_cm, bm.waist_width_side_cm,
            bm.hip_width_front_cm, bm.hip_width_side_cm)
    if any(v is None or v <= 0 for v in vals):
        return None
    return [float(v) for v in vals]


def _predict(coefs: tuple[float, ...], height_cm: float, widths: list[float]) -> float:
    return coefs[0] + coefs[1] * height_cm + sum(c * w for c, w in zip(coefs[2:], widths))


def predict_girths(bm: BodyMeasurements, sex: str, height_cm: float) -> dict[str, float]:
    """Predict the four circumferences and weight_kg from the pipeline widths.

    Args:
        bm: Pipeline output; needs the six torso widths (chest/waist/hip front + side).
        sex: ``"male"``, ``"female"`` or ``"other"`` (mean of both models).
        height_cm: Subject height.

    Returns:
        ``{"chest": cm, "waist": cm, "hip": cm, "thigh": cm, "weight_kg": kg}``, or ``{}`` when a
        width is missing (callers fall back to the ellipse values).
    """
    widths = _widths(bm)
    if widths is None or height_cm <= 0:
        return {}
    sexes = [sex] if sex in _COEFS else ["female", "male"]
    out: dict[str, float] = {}
    for key in (*SITES, "weight_kg"):
        out[key] = sum(_predict(_COEFS[s][key], height_cm, widths) for s in sexes) / len(sexes)
    return out
