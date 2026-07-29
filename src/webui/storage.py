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
    prefix: str             # may be empty; trailing slash always stripped
    force_path_style: bool  # most non-AWS S3-compatible providers need this


# Env-var prefixes we try, in order. Lets users paste the example .env block
# from any S3-compatible provider's docs without renaming variables:
#
#   S3_*    canonical (recommended)
#   ELK_*   ElasticLake
#   R2_*    Cloudflare R2 examples
#   B2_*    Backblaze B2 examples
#   AWS_*   AWS / boto3 native names (uses AWS_ACCESS_KEY_ID etc.)
#
# Within one .env, only one prefix should appear. The first prefix whose four
# required vars are all set wins.
_ENV_PREFIXES = ("S3_", "ELK_", "R2_", "B2_", "AWS_")


def _read_prefixed(prefix: str) -> dict[str, str | None]:
    # Most providers use the same suffix names as AWS:
    #   <PREFIX>ENDPOINT, <PREFIX>ACCESS_KEY_ID, <PREFIX>SECRET_ACCESS_KEY,
    #   <PREFIX>BUCKET, <PREFIX>REGION, <PREFIX>PREFIX, <PREFIX>FORCE_PATH_STYLE
    def _clean(name: str) -> str | None:
        """Read an env var, stripping surrounding whitespace/newlines.

        Secrets pasted into a dashboard textarea very often pick up a trailing
        newline. botocore embeds the key id verbatim in the SigV4 Authorization
        header, so a single "\n" produces:

            An HTTP Client raised an unhandled exception:
            Invalid header value b'AWS4-HMAC-SHA256 Credential=...\\n/2026...'

        — which is opaque and looks nothing like a credential problem. Strip
        here so a stray newline can never reach the signer.
        """
        raw = os.environ.get(f"{prefix}{name}")
        if raw is None:
            return None
        cleaned = raw.strip()
        if cleaned != raw:
            logger.warning(
                "Object-storage env %s%s had surrounding whitespace — stripped.",
                prefix, name,
            )
        return cleaned or None

    return {
        "endpoint": _clean("ENDPOINT"),
        "access_key_id": _clean("ACCESS_KEY_ID"),
        "secret_access_key": _clean("SECRET_ACCESS_KEY"),
        "bucket": _clean("BUCKET"),
        "region": _clean("REGION"),
        "prefix": _clean("PREFIX"),
        "force_path_style": _clean("FORCE_PATH_STYLE"),
    }


def _load_config() -> S3Config | None:
    # Pick the first prefix that has the four required values filled in.
    chosen: str | None = None
    values: dict[str, str | None] = {}
    for prefix in _ENV_PREFIXES:
        candidate = _read_prefixed(prefix)
        if all(candidate[k] for k in ("endpoint", "access_key_id", "secret_access_key", "bucket")):
            chosen = prefix
            values = candidate
            break

    if chosen is None:
        logger.warning(
            "Object-storage archival DISABLED — set one of these prefix groups "
            "in .env or the HF Space's Variables and secrets: %s (need ENDPOINT, "
            "ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET).",
            "|".join(p.rstrip("_") for p in _ENV_PREFIXES),
        )
        return None
    if chosen != "S3_":
        logger.warning(
            "Object-storage: detected %s* env-var prefix (recommended canonical name is S3_*).",
            chosen,
        )
    endpoint = values["endpoint"].rstrip("/")  # type: ignore[union-attr]
    # Heuristic: AWS S3 is the only major provider that *prefers* virtual-hosted
    # addressing; almost every other S3-compatible provider (R2, B2, MinIO,
    # Wasabi, iDrive, Scaleway, ElasticLake, …) wants path-style. The env var
    # <PREFIX>FORCE_PATH_STYLE overrides the heuristic if needed.
    force_raw = (values.get("force_path_style") or "").strip().lower()
    if force_raw in ("1", "true", "yes", "on"):
        force_path_style = True
    elif force_raw in ("0", "false", "no", "off"):
        force_path_style = False
    else:
        force_path_style = "amazonaws.com" not in endpoint.lower()

    # Supabase exposes its S3-compatible API at <project>.supabase.co/storage/v1/s3.
    # Pointing at the bare project URL yields a signed request against the REST
    # gateway, which fails in confusing ways. Auto-correct rather than fail.
    low = endpoint.lower()
    if "supabase.co" in low and "/storage/v1/s3" not in low:
        endpoint = endpoint.rstrip("/") + "/storage/v1/s3"
        logger.warning(
            "Object-storage endpoint looked like a bare Supabase project URL — "
            "using %s (the S3-compatible path).", endpoint,
        )

    return S3Config(
        endpoint=endpoint,
        access_key_id=values["access_key_id"],          # type: ignore[arg-type]
        secret_access_key=values["secret_access_key"],  # type: ignore[arg-type]
        bucket=values["bucket"],                        # type: ignore[arg-type]
        region=values.get("region") or "us-east-1",
        prefix=(values.get("prefix") or "").rstrip("/"),
        force_path_style=force_path_style,
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
                # boto3 1.36+ defaults to "flexible checksums" that send a
                # streaming-aws-chunked body and omit Content-Length — strict
                # providers (ElasticLake, some MinIO setups) reject this with
                # `MissingContentLength`. Pin checksum calculation to "when
                # required" and disable response validation; combined with the
                # explicit ContentLength on put_object below this forces a
                # regular non-chunked request.
                self._client = boto3.client(
                    "s3",
                    endpoint_url=cfg.endpoint,
                    aws_access_key_id=cfg.access_key_id,
                    aws_secret_access_key=cfg.secret_access_key,
                    region_name=cfg.region,
                    config=BotoConfig(
                        signature_version="s3v4",
                        # 5 attempts / adaptive: ElasticLake intermittently returns
                        # InternalError under load and 2 was not enough — production
                        # logs show "reached max retries: 2".
                        retries={"max_attempts": 5, "mode": "adaptive"},
                        request_checksum_calculation="when_required",
                        response_checksum_validation="when_required",
                        s3={"addressing_style": "path" if cfg.force_path_style else "virtual"},
                    ),
                )
                logger.warning(
                    "Object-storage client READY — bucket=%s endpoint=%s "
                    "region=%s addressing=%s",
                    cfg.bucket, cfg.endpoint, cfg.region,
                    "path" if cfg.force_path_style else "virtual",
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


def download_to_path(key: str, local_path) -> bool:
    """Download one object from the configured bucket to a local file.

    Idempotent / skip-existing: returns True without making a network call
    if the file already exists locally and is non-empty. Used by the boot
    flow to pull model weights from S3 on container startup, so they don't
    have to ship in the Docker image (and can be rotated without rebuilds).
    """
    from pathlib import Path as _Path

    target = _Path(local_path)
    if target.is_file() and target.stat().st_size > 0:
        return True

    client = _lazy.client()
    cfg = _lazy.config
    if client is None or cfg is None:
        logger.warning(
            "download_to_path(%s): storage not configured — model weight cannot be fetched.",
            key,
        )
        return False

    full_key = (cfg.prefix + "/" + key) if cfg.prefix else key
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        client.download_file(cfg.bucket, full_key, str(tmp))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "download_to_path FAILED — key=%s bucket=%s endpoint=%s err=%s",
            full_key, cfg.bucket, cfg.endpoint, exc,
        )
        if tmp.is_file():
            try:
                tmp.unlink()
            except OSError:
                pass
        return False
    tmp.replace(target)
    logger.warning(
        "download_to_path OK — key=%s → %s (size=%d bytes)",
        full_key, target, target.stat().st_size,
    )
    return True


# ── Local-filesystem archival (HF Storage Bucket mounted at /data) ─────────
# HF Storage Buckets attached to a Space appear at /data inside the container
# (read-write). When LOCAL_DATA_DIR is set, archive_measurement writes
# directly to disk under that path instead of (or in addition to) S3.
# Same layout as S3: <LOCAL_DATA_DIR>/measurements/<uuid>/{front.jpg,...}.

def _local_data_dir() -> "Path | None":
    from pathlib import Path as _Path
    raw = (os.environ.get("LOCAL_DATA_DIR") or "").strip()
    if not raw:
        return None
    p = _Path(raw)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("LOCAL_DATA_DIR=%s not writable: %s", raw, exc)
        return None
    return p


def archive_measurement_local(
    request_id: str,
    *,
    front_bytes: bytes,
    front_content_type: str,
    side_bytes: bytes,
    side_content_type: str,
    envelope_json: dict | str,
    metadata: dict[str, str] | None = None,
) -> bool:
    """Write photos + envelope to the local data directory (HF bucket mount).

    Mirrors the bucket key layout used by archive_measurement so analysis
    scripts can treat both stores interchangeably.
    """
    base = _local_data_dir()
    if base is None:
        return False
    rid = _sanitise_request_id(request_id)
    folder = base / "measurements" / rid
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"front.{_extension_for(front_content_type)}").write_bytes(front_bytes)
        (folder / f"side.{_extension_for(side_content_type)}").write_bytes(side_bytes)
        json_body = (
            envelope_json
            if isinstance(envelope_json, str)
            else json.dumps(envelope_json, ensure_ascii=False)
        )
        (folder / "envelope.json").write_text(json_body, encoding="utf-8")
        if metadata:
            (folder / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except OSError as exc:
        logger.warning(
            "Local archive FAILED — request_id=%s base=%s err=%s",
            rid, base, exc,
        )
        return False
    logger.warning(
        "Local archive OK — request_id=%s folder=%s",
        rid, folder,
    )
    return True


def local_model_path(name: str) -> "Path | None":
    """Look up a model file inside the bucket / LOCAL_DATA_DIR.

    Checks (in order) — first non-empty hit wins:
      <LOCAL_DATA_DIR>/models/<name>     ← preferred layout
      <LOCAL_DATA_DIR>/<name>            ← bucket-root layout (HF dashboard
                                            uploads land here unless you
                                            create a folder explicitly)
    Returns None when nothing matches; a WARN line is emitted either way
    so the operator can tell from the Logs tab what was checked.
    """
    base = _local_data_dir()
    if base is None:
        logger.warning(
            "local_model_path(%s): LOCAL_DATA_DIR is unset / unwritable — "
            "bucket lookup skipped.",
            name,
        )
        return None
    candidates = [base / "models" / name, base / name]
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            logger.warning(
                "local_model_path FOUND — name=%s → %s (size=%d bytes)",
                name, candidate, candidate.stat().st_size,
            )
            return candidate
    logger.warning(
        "local_model_path MISS — name=%s tried=%s",
        name, [str(c) for c in candidates],
    )
    return None


def _extension_for(content_type: str) -> str:
    ct = (content_type or "").lower().strip()
    if "jpeg" in ct or "jpg" in ct:
        return "jpg"
    if "png" in ct:
        return "png"
    if "webp" in ct:
        return "webp"
    return "bin"


# Error codes that are TRANSIENT on our provider but that boto3 will not retry
# on its own. `SignatureDoesNotMatch` is nominally a 403 (auth) error, so botocore
# treats it as permanent — but ElasticLake returns it intermittently with
# "signing key not available", and an immediate retry with the same credentials
# succeeds. Verified repeatedly against the live bucket.
_RETRYABLE_S3_CODES = frozenset({
    "SignatureDoesNotMatch",
    "InternalError",
    "ServiceUnavailable",
    "SlowDown",
    "RequestTimeout",
    "RequestTimeTooSkewed",
})
_PUT_RETRIES = 4
_PUT_BACKOFF_S = 0.5


def _metadata_disabled() -> bool:
    """True when POINTSX_S3_NO_METADATA is set — drop x-amz-meta-* headers."""
    return (os.environ.get("POINTSX_S3_NO_METADATA") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _error_detail(exc: Exception) -> str:
    """Extra diagnostics for an S3 error whose parsed code/message are empty.

    Providers that answer with a JSON (or empty) body instead of the S3 XML
    error document leave botocore with nothing to parse, so the exception
    stringifies to the useless "An error occurred () when calling ...: ".
    Surface the HTTP status and the raw body so the failure is actionable.
    """
    meta = getattr(exc, "response", None)
    if not isinstance(meta, dict):
        return ""
    rm = meta.get("ResponseMetadata") or {}
    bits = []
    status = rm.get("HTTPStatusCode")
    if status:
        bits.append(f"http_status={status}")
    err = meta.get("Error") or {}
    if err.get("Code"):
        bits.append(f"code={err['Code']}")
    if err.get("Message"):
        bits.append(f"msg={err['Message']}")
    hdrs = rm.get("HTTPHeaders") or {}
    for h in ("x-amz-error-code", "x-amz-error-message", "content-type"):
        if hdrs.get(h):
            bits.append(f"{h}={hdrs[h]}")
    if status in (401, 403):
        bits.append(
            "HINT: Supabase S3 needs credentials from Storage -> S3 Access Keys; "
            "a project API key (sb_secret_.../service_role JWT) is NOT accepted"
        )
    elif status == 400:
        bits.append(
            "HINT: 400 often means an unsupported feature — Supabase S3 rejects "
            "some x-amz-meta-* custom metadata; try POINTSX_S3_NO_METADATA=1"
        )
    return f"  [{'; '.join(bits)}]" if bits else ""


def _put_with_retry(client, **kwargs) -> None:
    """put_object with backoff on provider-transient errors.

    Runs inside a BackgroundTask, so sleeping here never delays the measurement
    response. Re-raises the last error once attempts are exhausted.
    """
    import time

    from botocore.exceptions import ClientError

    for attempt in range(1, _PUT_RETRIES + 1):
        try:
            client.put_object(**kwargs)
            if attempt > 1:
                logger.warning(
                    "Object-storage put succeeded on attempt %d — key=%s",
                    attempt, kwargs.get("Key"),
                )
            return
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code not in _RETRYABLE_S3_CODES or attempt == _PUT_RETRIES:
                raise
            delay = _PUT_BACKOFF_S * (2 ** (attempt - 1))
            logger.warning(
                "Object-storage put attempt %d/%d failed (%s) — retrying in %.1fs",
                attempt, _PUT_RETRIES, code, delay,
            )
            time.sleep(delay)


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
            # ContentLength explicit: some stricter S3-compatible providers
            # (ElasticLake, certain MinIO configs) reject chunked-encoded
            # uploads where boto3 omits the header. Passing len(body)
            # bypasses chunked encoding for bytes payloads.
            _put_with_retry(
                client,
                Bucket=cfg.bucket,
                Key=key,
                Body=body,
                ContentLength=len(body),
                ContentType=content_type,
                # Custom x-amz-meta-* headers are optional. Some S3-compatible
                # backends (Supabase among them) reject requests carrying them,
                # and the archive is worth more than the metadata — set
                # POINTSX_S3_NO_METADATA=1 to drop them.
                Metadata={} if _metadata_disabled() else (md or {}),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Object-storage archive FAILED for request_id=%s: %s%s",
            rid, exc, _error_detail(exc),
        )
        return False
    logger.warning(
        "Object-storage archive OK — request_id=%s key_prefix=%s",
        rid, _key(rid, "").rstrip("/"),
    )
    return True
