"""Local-filesystem archival and model lookup (HF Storage Bucket mounted at /data)."""
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

from webui.infrastructure.storage.common import _extension_for, _sanitise_request_id

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


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
