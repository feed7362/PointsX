"""Supabase Storage REST backend (preferred when SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY are set)."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

from webui.infrastructure.storage.common import _extension_for, _sanitise_request_id
from webui.infrastructure.storage.crypto import _dataset_public_key, _seal

logger = logging.getLogger(__name__)


# ── Supabase Storage REST backend ───────────────────────────────────────────
# Preferred over the S3-compatible protocol because it reuses the credentials
# the dataset flow ALREADY uses (SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY).
#
# Supabase exposes two storage APIs with SEPARATE credential systems:
#   * Storage REST  — Authorization: Bearer <service_role JWT>   <- this one
#   * S3-compatible — AWS SigV4 with keys from Storage -> S3 Access Keys
# A project API key is rejected by the S3 protocol with a bare 403, which is
# what the archive hit. Using REST removes the second credential entirely.


@dataclass(frozen=True)
class _SupabaseConfig:
    url: str          # https://<ref>.supabase.co
    service_key: str
    bucket: str
    prefix: str


def _supabase_config() -> _SupabaseConfig | None:
    url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    bucket = (
        os.environ.get("SUPABASE_STORAGE_BUCKET")
        or os.environ.get("STORAGE_BUCKET")
        or ""
    ).strip()
    if not (url and key and bucket):
        return None
    prefix = (os.environ.get("S3_PREFIX") or os.environ.get("SUPABASE_PREFIX") or "").strip()
    return _SupabaseConfig(url=url, service_key=key, bucket=bucket, prefix=prefix.strip("/"))


def _supabase_put(cfg: _SupabaseConfig, key: str, body: bytes, content_type: str) -> None:
    """Upload one object via the Storage REST API. Raises on non-2xx."""
    import urllib.error
    import urllib.request

    path = f"{cfg.prefix}/{key}" if cfg.prefix else key
    url = f"{cfg.url}/storage/v1/object/{cfg.bucket}/{path.lstrip('/')}"
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            # supabase-js sends the key in BOTH headers, and the gateway needs
            # `apikey` for the new-format keys (sb_secret_...). Sending only
            # Authorization makes it parse the value as a legacy JWT and fail
            # with "Invalid Compact JWS".
            "apikey": cfg.service_key,
            "Authorization": f"Bearer {cfg.service_key}",
            "Content-Type": content_type or "application/octet-stream",
            "Content-Length": str(len(body)),
            # Overwrite instead of failing when a request_id is retried.
            "x-upsert": "true",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status // 100 != 2:
                raise RuntimeError(f"HTTP {resp.status}")
    except urllib.error.HTTPError as exc:  # noqa: PERF203
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc



def archive_to_supabase(
    sb: _SupabaseConfig,
    request_id: str,
    *,
    front_bytes: bytes,
    front_content_type: str,
    side_bytes: bytes,
    side_content_type: str,
    envelope_json: dict | str,
) -> bool:
    """Upload photos + envelope via Storage REST, sealed unless POINTSX_ARCHIVE_ENCRYPT=0."""
    rid = _sanitise_request_id(request_id)
    json_body = (
        envelope_json
        if isinstance(envelope_json, str)
        else json.dumps(envelope_json, ensure_ascii=False)
    )
    pub = _dataset_public_key()
    try:
        if pub:
            # Sealed, so every object is an opaque blob: same encryption as
            # dataset submissions, and octet-stream passes the bucket's MIME
            # whitelist. Decrypt with scripts/decrypt_dataset.py.
            uploads = [
                (f"measurements/{rid}/front.bin", _seal(front_bytes, pub)),
                (f"measurements/{rid}/side.bin", _seal(side_bytes, pub)),
                (f"measurements/{rid}/envelope.json.bin",
                 _seal(json_body.encode("utf-8"), pub)),
            ]
            ctype = "application/octet-stream"
        else:
            uploads = [
                (f"measurements/{rid}/front.{_extension_for(front_content_type)}", front_bytes),
                (f"measurements/{rid}/side.{_extension_for(side_content_type)}", side_bytes),
                (f"measurements/{rid}/envelope.json", json_body.encode("utf-8")),
            ]
            ctype = ""
        for key, body in uploads:
            _supabase_put(sb, key, body, ctype or (
                "application/json" if key.endswith(".json") else "image/jpeg"))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Supabase Storage archive FAILED for request_id=%s: %s", rid, exc,
        )
        return False
    logger.warning(
        "Supabase Storage archive OK — request_id=%s bucket=%s prefix=%s encrypted=%s",
        rid, sb.bucket, sb.prefix or "(none)", bool(pub),
    )
    return True
