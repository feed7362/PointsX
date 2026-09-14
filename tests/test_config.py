"""webui.config.Settings — covers the lifespan settings the HTTP contract tests cannot reach."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from webui.config import DEFAULT_DATASET_DIR, Settings, env_flag  # noqa: E402

VARS = (
    "POINTSX_VERCEL", "POINTSX_INFERENCE_ENDPOINT", "POINTSX_DATASET_DIR", "CORS_ALLOW_ORIGINS",
    "POINTSX_POSE_MODEL_CUSTOM", "POINTSX_POSE_MODEL", "POINTSX_POSE_MODEL_COCO", "POINTSX_SEG_MODEL",
    "POINTSX_USE_REGRESSOR", "POINTSX_REGRESSION_MODEL", "POINTSX_DEVICE", "POINTSX_WARMUP_DISABLE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults():
    s = Settings.from_env()
    assert (s.vercel, s.inference_endpoint, s.proxy_mode) == (False, None, False)
    assert s.dataset_dir == DEFAULT_DATASET_DIR and s.cors_origins == ("*",)
    assert s.pose_custom_path == "models/pose-cus.pt" and not s.pose_custom_from_legacy_env
    assert s.pose_coco_path == "models/yolo26-pose.pt"
    assert s.seg_model_path == "models/yolo12l-person-seg-extended.pt"
    assert s.regression_model_path is None and s.device == "auto" and s.warmup_disable is False


def test_proxy_mode_needs_both_vercel_and_endpoint(monkeypatch):
    monkeypatch.setenv("POINTSX_VERCEL", "1")
    assert Settings.from_env().proxy_mode is False
    monkeypatch.setenv("POINTSX_INFERENCE_ENDPOINT", "  https://space.example  ")
    s = Settings.from_env()
    assert s.proxy_mode is True and s.inference_endpoint == "https://space.example"
    monkeypatch.delenv("POINTSX_VERCEL")
    assert Settings.from_env().proxy_mode is False


def test_regressor_is_opt_in(monkeypatch):
    monkeypatch.setenv("POINTSX_REGRESSION_MODEL", "models/custom.pt")
    assert Settings.from_env().regression_model_path is None
    monkeypatch.setenv("POINTSX_USE_REGRESSOR", "true")
    assert Settings.from_env().regression_model_path == "models/custom.pt"
    monkeypatch.setenv("POINTSX_REGRESSION_MODEL", "   ")
    assert Settings.from_env().regression_model_path == "models/reg.pt"


def test_legacy_pose_env_overrides_custom_path(monkeypatch):
    monkeypatch.setenv("POINTSX_POSE_MODEL_CUSTOM", "models/a.pt")
    monkeypatch.setenv("POINTSX_POSE_MODEL", "   ")
    s = Settings.from_env()
    assert s.pose_custom_path == "models/a.pt" and not s.pose_custom_from_legacy_env
    monkeypatch.setenv("POINTSX_POSE_MODEL", " models/legacy.pt ")
    s = Settings.from_env()
    assert s.pose_custom_path == "models/legacy.pt" and s.pose_custom_from_legacy_env


def test_blank_paths_fall_back_and_cors_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("POINTSX_SEG_MODEL", "  ")
    monkeypatch.setenv("POINTSX_DEVICE", " cpu ")
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", " https://a.example , ,https://b.example")
    monkeypatch.setenv("POINTSX_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("POINTSX_WARMUP_DISABLE", "YES")
    s = Settings.from_env()
    assert s.seg_model_path == "models/yolo12l-person-seg-extended.pt" and s.device == "cpu"
    assert s.cors_origins == ("https://a.example", "https://b.example")
    assert s.dataset_dir == tmp_path and s.warmup_disable is True


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("On", True), ("0", False), ("", False), ("nope", False)])
def test_env_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("POINTSX_WARMUP_DISABLE", raw)
    assert env_flag("POINTSX_WARMUP_DISABLE") is expected
