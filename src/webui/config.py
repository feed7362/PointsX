"""Runtime settings for the web app, read from environment variables.

One dataclass instead of scattered ``os.environ`` reads. Light on purpose (no
pydantic-settings): the Vercel function installs only requirements.txt.

Environment variables (all optional):
    POINTSX_VERCEL            set by api/index.py on Vercel (skips the static mount)
    POINTSX_INFERENCE_ENDPOINT  base URL of the HF Space; with POINTSX_VERCEL → proxy mode
    POINTSX_DATASET_DIR       where captured image pairs are saved (default: <repo>/dataset)
    CORS_ALLOW_ORIGINS        comma-separated origins (default: ``*``)
    POINTSX_POSE_MODEL_CUSTOM path to 16-keypoint (LV-MHP) pose .pt (default: models/pose-cus.pt)
    POINTSX_POSE_MODEL        legacy: if set, overrides POINTSX_POSE_MODEL_CUSTOM
    POINTSX_POSE_MODEL_COCO   path to COCO-17 pose .pt (default: models/yolo26-pose.pt)
    POINTSX_SEG_MODEL         path to segmentation .pt (default: models/yolo12l-person-seg-extended.pt)
    POINTSX_USE_REGRESSOR     ``1`` to use the circumference regressor instead of the Ramanujan ellipse
    POINTSX_REGRESSION_MODEL  regressor .pt used when POINTSX_USE_REGRESSOR=1 (default: models/reg.pt)
    POINTSX_DEVICE            "auto" | "cpu" | "cuda" | "0" | …  (default: "auto")
    POINTSX_WARMUP_DISABLE    ``1`` to skip the startup dummy forward pass

Read at call time elsewhere (so they can change without a restart):
``CRON_SECRET`` (keepalive), ``POINTSX_TTS_VOICE`` / ``POINTSX_TTS_DISABLE`` (tts),
``HF_MODELS_REPO`` / ``HF_TOKEN`` / ``MODELS_S3_KEY_PREFIX`` (weight pre-fetch), storage ``*_S3_*`` vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
DISALLOWED_CONTENT_PREFIXES = ("text/", "video/", "audio/")

_TRUTHY = ("1", "true", "yes", "on")


def env_flag(name: str) -> bool:
    """True when the variable is set to 1/true/yes/on (case-insensitive)."""
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def _path_env(name: str, default: str) -> str:
    """Env value, stripped; unset or blank falls back to ``default``."""
    return os.environ.get(name, default).strip() or default


@dataclass(frozen=True)
class Settings:
    # deployment mode
    vercel: bool
    inference_endpoint: str | None
    # web
    dataset_dir: Path
    cors_origins: tuple[str, ...]
    # models (consumed by the lifespan)
    pose_custom_path: str
    pose_custom_from_legacy_env: bool
    pose_coco_path: str
    seg_model_path: str
    regression_model_path: str | None
    device: str
    warmup_disable: bool

    @property
    def proxy_mode(self) -> bool:
        """Vercel proxy mode: no models; /api/measure is forwarded to ``inference_endpoint``."""
        return self.vercel and self.inference_endpoint is not None

    @classmethod
    def from_env(cls) -> Settings:
        legacy_pose = os.environ.get("POINTSX_POSE_MODEL")
        legacy = legacy_pose is not None and bool(legacy_pose.strip())
        return cls(
            vercel=bool(os.environ.get("POINTSX_VERCEL")),
            inference_endpoint=os.environ.get("POINTSX_INFERENCE_ENDPOINT", "").strip() or None,
            dataset_dir=Path(os.environ.get("POINTSX_DATASET_DIR", str(DEFAULT_DATASET_DIR))),
            cors_origins=tuple(o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()),
            pose_custom_path=(
                legacy_pose.strip() if legacy else _path_env("POINTSX_POSE_MODEL_CUSTOM", "models/pose-cus.pt")
            ),
            pose_custom_from_legacy_env=legacy,
            pose_coco_path=_path_env("POINTSX_POSE_MODEL_COCO", "models/yolo26-pose.pt"),
            seg_model_path=_path_env("POINTSX_SEG_MODEL", "models/yolo12l-person-seg-extended.pt"),
            # Opt-in: the regressor produced outliers on real photos, and the per-sex
            # correction tables in envelope.py were fitted against the ellipse output.
            regression_model_path=(
                _path_env("POINTSX_REGRESSION_MODEL", "models/reg.pt") if env_flag("POINTSX_USE_REGRESSOR") else None
            ),
            device=_path_env("POINTSX_DEVICE", "auto"),
            warmup_disable=env_flag("POINTSX_WARMUP_DISABLE"),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings snapshot taken on first use (process start)."""
    return Settings.from_env()
