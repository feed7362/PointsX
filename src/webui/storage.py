"""S3-compatible archival for demo / scientific data collection.

Works with any S3-compatible object store: Cloudflare R2, Backblaze B2,
AWS S3, MinIO, Wasabi, iDrive e2, Scaleway, etc. The provider's full
endpoint URL is supplied via ``S3_ENDPOINT``; no provider-specific code.

When configured (env vars set), every ``/api/measure`` call archives:

  • front photo            → ``measurements/<request_id>/front.<ext>``
  • side photo             → ``measurements/<request_id>/side.<ext>``
  • measurement envelope   → ``measurements/<request_id>/envelope.json``

Required environment variables:

  S3_ENDPOINT            full URL incl. https://
                         R2 example:    https://<acct>.r2.cloudflarestorage.com
                         B2 example:    https://s3.<region>.backblazeb2.com
                         AWS example:   https://s3.amazonaws.com
                         MinIO example: https://minio.example.com
  S3_ACCESS_KEY_ID
  S3_SECRET_ACCESS_KEY
  S3_BUCKET

Optional:

  S3_REGION              defaults to ``auto`` (R2 expects this; AWS / B2
                         want the actual region like ``us-east-1``)
  S3_PREFIX              prepended to every object key (default: empty)

If any of the required vars are missing the storage layer is silently
disabled — the measurement endpoint still works, without archiving.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class S3Config:
    endpoint: str
    access_key_id: str
    secret_access_key: str
    bucket: str
    region: str
    prefix: str  # may be empty; trailing slash always stripped


def _load_config() -> S3Config | None:
    required = (
        ("S3_ENDPOINT", os.environ.get("S3_ENDPOINT")),
        ("S3_ACCESS_KEY_ID", os.environ.get("S3_ACCESS_KEY_ID")),
        ("S3_SECRET_ACCESS_KEY", os.environ.get("S3_SECRET_ACCESS_KEY")),
        ("S3_BUCKET", os.environ.get("S3_BUCKET")),
    )
    missing = [name for name, value in required if not value]
    if missing:
        # WARNING (not INFO) so it shows under uvicorn's default config —
        # otherwise the message would be invisible and you'd think archival
        # was working when it never even tried.
        logger.warning(
            "Object-storage archival DISABLED — missing env vars: %s. "
            "Set them in .env (local) or the HF Space's Variables and secrets.",
            ", ".join(missing),
        )
        return None
    return S3Config(
        endpoint=os.environ["S3_ENDPOINT"].rstrip("/"),
        access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        bucket=os.environ["S3_BUCKET"],
        region=os.environ.get("S3_REGION", "auto"),
        prefix=(os.environ.get("S3_PREFIX") or "").rstrip("/"),
    )


class _LazyClient:
    """Defer boto3 import + client construction until first archive call."""

    _SENTINEL = object()

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config: Any = self._SENTINEL
        self._client: Any | None = None

    @property
    def config(self) -> S3Config | None:
        if self._config is self._SENTINEL:
            self._config = _load_config()
        return self._config

    def client(self) -> Any | None:
        cfg = self.config
        if cfg is None:
            return None
        with self._lock:
            if self._client is None:
                try:
                    import boto3  # noqa: WPS433 — local import keeps cold start fast
                    from botocore.config import Config as BotoConfig
                except ImportError:
                    logger.warning(
                        "boto3 not installed — object-storage archival disabled. "
                        "Add `boto3>=1.35.0` to pyproject and reinstall."
                    )
                    return None
                self._client = boto3.client(
                    "s3",
                    endpoint_url=cfg.endpoint,
                    aws_access_key_id=cfg.access_key_id,
                    aws_secret_access_key=cfg.secret_access_key,
                    region_name=cfg.region,
                    config=BotoConfig(
                        signature_version="s3v4",
                        retries={"max_attempts": 2, "mode": "standard"},
                    ),
                )
                logger.warning(
                    "Object-storage client READY — bucket=%s endpoint=%s",
                    cfg.bucket, cfg.endpoint,
                )
            return self._client


_lazy = _LazyClient()


def is_enabled() -> bool:
    return _lazy.config is not None


def _key(request_id: str, name: str) -> str:
    cfg = _lazy.config
    prefix = (cfg.prefix + "/") if (cfg and cfg.prefix) else ""
    return f"{prefix}measurements/{request_id}/{name}"


def _sanitise_request_id(request_id: str) -> str:
    rid = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in request_id)[:96]
    return rid or "anon"


def _extension_for(content_type: str) -> str:
    ct = (content_type or "").lower().strip()
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    return "bin"


def archive_measurement(
    request_id: str,
    *,
    front_bytes: bytes,
    front_content_type: str,
    side_bytes: bytes,
    side_content_type: str,
    envelope_json: dict[str, Any] | str,
    metadata: dict[str, str] | None = None,
) -> bool:
    """Upload the photos + envelope to object storage under a per-request prefix.

    Returns True on success, False on any failure (logged but never raised —
    the measurement response must not be blocked by storage hiccups).
    """
    client = _lazy.client()
    cfg = _lazy.config
    if client is None or cfg is None:
        return False

    rid = _sanitise_request_id(request_id)
    json_body = (
        envelope_json
        if isinstance(envelope_json, str)
        else json.dumps(envelope_json, ensure_ascii=False)
    )
    md = {k: str(v)[:256] for k, v in (metadata or {}).items()}

    front_ext = _extension_for(front_content_type)
    side_ext = _extension_for(side_content_type)

    uploads = [
        (_key(rid, f"front.{front_ext}"), front_bytes, front_content_type or "image/jpeg"),
        (_key(rid, f"side.{side_ext}"), side_bytes, side_content_type or "image/jpeg"),
        (_key(rid, "envelope.json"), json_body.encode("utf-8"), "application/json"),
    ]
    try:
        for key, body, content_type in uploads:
            client.put_object(
                Bucket=cfg.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
                Metadata=md or {},
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Object-storage archive FAILED for request_id=%s: %s", rid, exc)
        return False
    logger.warning(
        "Object-storage archive OK — request_id=%s key_prefix=%s",
        rid, _key(rid, "").rstrip("/"),
    )
    return True
