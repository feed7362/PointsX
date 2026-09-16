"""webui.envelope.girth — ANSUR girth model (chest/waist/hip/thigh + weight from six torso widths)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pointsx.schemas import BodyMeasurements  # noqa: E402
from webui.envelope.girth import girth_model_enabled, predict_girths  # noqa: E402

# ANSUR female means: chest 26.9/24.7, waist 30.0/21.3, hip 35.4/23.3 (breadth/depth, cm)
F_MEAN = dict(torso_width_front_cm=26.9, torso_width_side_cm=24.7, waist_width_front_cm=30.0,
              waist_width_side_cm=21.3, hip_width_front_cm=35.4, hip_width_side_cm=23.3)


def test_average_female_widths_give_population_means():
    g = predict_girths(BodyMeasurements(**F_MEAN), "female", 162.8)
    assert set(g) == {"chest", "waist", "hip", "thigh", "weight_kg"}
    assert g["chest"] == pytest.approx(94.7, abs=2.5)   # ANSUR F chest circ mean
    assert g["waist"] == pytest.approx(86.1, abs=2.5)
    assert g["hip"] == pytest.approx(102.1, abs=2.5)
    assert g["thigh"] == pytest.approx(61.6, abs=3.0)
    assert g["weight_kg"] == pytest.approx(67.8, abs=5.0)


def test_wider_waist_means_bigger_waist():
    base = predict_girths(BodyMeasurements(**F_MEAN), "female", 165.0)
    wide = predict_girths(BodyMeasurements(**{**F_MEAN, "waist_width_front_cm": 35.0}), "female", 165.0)
    assert wide["waist"] > base["waist"] + 5


def test_other_is_mean_of_both_sexes():
    f = predict_girths(BodyMeasurements(**F_MEAN), "female", 170.0)
    m = predict_girths(BodyMeasurements(**F_MEAN), "male", 170.0)
    o = predict_girths(BodyMeasurements(**F_MEAN), "other", 170.0)
    assert o["chest"] == pytest.approx((f["chest"] + m["chest"]) / 2)


def test_missing_width_returns_empty():
    assert predict_girths(BodyMeasurements(**{**F_MEAN, "hip_width_side_cm": None}), "female", 165.0) == {}
    assert predict_girths(BodyMeasurements(**F_MEAN), "female", 0.0) == {}


def test_flag(monkeypatch):
    monkeypatch.delenv("POINTSX_GIRTH_MODEL", raising=False)
    assert girth_model_enabled() is False and girth_model_enabled(True) is True
    monkeypatch.setenv("POINTSX_GIRTH_MODEL", "1")
    assert girth_model_enabled() is True and girth_model_enabled(False) is False
