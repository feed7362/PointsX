"""Population priors for measurements the photos cannot resolve: neck and back width.

Silhouette extraction put the neck at the narrowest span under the jaw (app GT bias +24 cm) and
reported the side-view body depth as back width (bias -7 cm). Both are replaced by linear models
fitted on ANSUR II (US Army 2012, cleared for unlimited public release, so shippable), from inputs
the app has: height, sex and the pipeline's chest circumference.

Provenance: ``scripts/ansur/priors.py``, OLS per sex on all subjects (1 986 F / 4 082 M), 5-fold CV
MAE with N(0, 5 cm) noise added to chest at test time (the pipeline's chest is noisy):

    neck_circumference   <- neckcircumferencebase  H+chest F 1.25 / M 1.58, H only F 1.40 / M 1.96
    back_width_scapular  <- interscyei             H+chest F 2.04 / M 2.34, H only F 2.31 / M 2.77

The app corpus GT matches the neck BASE girth (F mean 36.4 vs ANSUR base 37.1, mid-neck 33.0).
Adding waist improved noisy CV by <0.1 cm and brings in clothing error, so it is not used.
No coefficient is fitted on app GT: those subjects are the evaluation set.
"""
from __future__ import annotations

# (intercept, per cm of height, per cm of chest circumference)
_WITH_CHEST: dict[str, dict[str, tuple[float, float, float]]] = {
    "neck_circumference": {
        "female": (10.9956, 0.0789, 0.1402),
        "male": (14.4807, 0.0366, 0.2129),
    },
    "back_width_scapular": {
        "female": (9.6913, 0.0415, 0.2205),
        "male": (12.5014, 0.0073, 0.2768),
    },
}
# (intercept, per cm of height) — used when the pipeline has no chest value
_HEIGHT_ONLY: dict[str, dict[str, tuple[float, float]]] = {
    "neck_circumference": {"female": (16.6386, 0.1257), "male": (25.2236, 0.1038)},
    "back_width_scapular": {"female": (18.5644, 0.1153), "male": (26.4684, 0.0946)},
}

PRIOR_IDS = frozenset(_WITH_CHEST)


def _predict_for_sex(mid: str, sex: str, height_cm: float, chest_cm: float | None) -> float:
    if chest_cm is not None and chest_cm > 0:
        b0, bh, bc = _WITH_CHEST[mid][sex]
        return b0 + bh * height_cm + bc * chest_cm
    b0, bh = _HEIGHT_ONLY[mid][sex]
    return b0 + bh * height_cm


def predict_prior(mid: str, sex: str, height_cm: float, chest_cm: float | None) -> float | None:
    """Predict a prior-based measurement.

    Args:
        mid: Canonical id, one of ``PRIOR_IDS``.
        sex: ``"male"``, ``"female"`` or ``"other"`` (average of both models).
        height_cm: Subject height in cm.
        chest_cm: Chest circumference as reported to the user (after corrections), or None.

    Returns:
        Value in cm, or None for an unknown id or a non-positive height.
    """
    if mid not in PRIOR_IDS or height_cm <= 0:
        return None
    if sex in ("female", "male"):
        return _predict_for_sex(mid, sex, height_cm, chest_cm)
    return (_predict_for_sex(mid, "female", height_cm, chest_cm)
            + _predict_for_sex(mid, "male", height_cm, chest_cm)) / 2
