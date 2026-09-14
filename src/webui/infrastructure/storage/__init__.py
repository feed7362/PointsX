"""Archival storage for demo / scientific data collection.

Backends: ``supabase`` (Storage REST, preferred), ``s3`` (any S3-compatible store),
``local`` (LOCAL_DATA_DIR / HF bucket mount); ``crypto`` seals photos before upload.

S3-compatible archival:

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

from typing import Any

from webui.infrastructure.storage.crypto import ENC_ALGO
from webui.infrastructure.storage.local import archive_measurement_local, local_model_path
from webui.infrastructure.storage.s3 import S3Config, archive_to_s3, download_to_path
from webui.infrastructure.storage.s3 import _lazy as _s3_lazy
from webui.infrastructure.storage.supabase import _supabase_config, archive_to_supabase

__all__ = [
    "ENC_ALGO",
    "S3Config",
    "archive_measurement",
    "archive_measurement_local",
    "download_to_path",
    "is_enabled",
    "local_model_path",
]


def is_enabled() -> bool:
    """True when any archival backend is configured (Supabase REST or S3)."""
    return _supabase_config() is not None or _s3_lazy.config is not None


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

    Prefers the Supabase Storage REST API when SUPABASE_URL +
    SUPABASE_SERVICE_ROLE_KEY are configured, since that reuses the credentials
    the dataset flow already works with. Falls back to the S3-compatible client.
    """
    sb = _supabase_config()
    if sb is not None:
        return archive_to_supabase(
            sb, request_id,
            front_bytes=front_bytes, front_content_type=front_content_type,
            side_bytes=side_bytes, side_content_type=side_content_type,
            envelope_json=envelope_json,
        )
    return archive_to_s3(
        request_id,
        front_bytes=front_bytes, front_content_type=front_content_type,
        side_bytes=side_bytes, side_content_type=side_content_type,
        envelope_json=envelope_json, metadata=metadata,
    )
