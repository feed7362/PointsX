"""webui.envelope.priors — ANSUR II neck / back-width priors."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from webui.envelope.priors import PRIOR_IDS, predict_prior  # noqa: E402


@pytest.mark.parametrize(("mid", "sex", "height", "chest", "lo", "hi"), [
    ("neck_circumference", "female", 163.0, 95.0, 36.0, 38.5),   # ANSUR F base neck mean 37.1
    ("neck_circumference", "male", 176.0, 106.0, 42.0, 45.0),    # ANSUR M mean 43.5
    ("back_width_scapular", "female", 163.0, 95.0, 36.0, 38.5),  # ANSUR F interscye mean 37.3
    ("back_width_scapular", "male", 176.0, 106.0, 41.5, 44.5),   # ANSUR M mean 43.1
])
def test_average_body_gets_population_mean(mid, sex, height, chest, lo, hi):
    assert lo <= predict_prior(mid, sex, height, chest) <= hi


def test_bigger_chest_means_bigger_neck():
    assert predict_prior("neck_circumference", "male", 176.0, 120.0) > predict_prior(
        "neck_circumference", "male", 176.0, 95.0)


def test_other_is_the_average_of_both_models():
    f = predict_prior("neck_circumference", "female", 170.0, 100.0)
    m = predict_prior("neck_circumference", "male", 170.0, 100.0)
    assert predict_prior("neck_circumference", "other", 170.0, 100.0) == pytest.approx((f + m) / 2)


def test_height_only_fallback_without_chest():
    value = predict_prior("back_width_scapular", "female", 163.0, None)
    assert value is not None and 35.0 <= value <= 39.0


def test_unknown_id_or_bad_height():
    assert predict_prior("waist_circumference", "female", 163.0, 95.0) is None
    assert predict_prior("neck_circumference", "female", 0.0, 95.0) is None
    assert PRIOR_IDS == {"neck_circumference", "back_width_scapular"}
