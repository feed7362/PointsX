"""Pre-fetch model weights before the pipeline loads them."""
from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def prefetch_weights(paths: list[str | None]) -> None:
    """Make sure each weight file exists locally before the models load.

    Sources in priority order (idempotent — files already on disk stay, so
    warm restarts download nothing):
      1. LOCAL_DATA_DIR/models/<name>   ← HF Storage Bucket mounted at /data
      2. HF Hub model repo (env: HF_MODELS_REPO)
      3. S3 bucket under MODELS_S3_KEY_PREFIX (when archival is configured)
    """
    hf_repo = (os.environ.get("HF_MODELS_REPO") or "").strip()
    hf_revision = (os.environ.get("HF_MODELS_REVISION") or "main").strip()
    hf_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None)

    def _pull(local: str | None) -> None:
        if not local:
            return
        p = Path(local)
        if p.is_file() and p.stat().st_size > 0:
            return

        # First: HF Storage Bucket mounted at LOCAL_DATA_DIR/models/.
        try:
            from webui.infrastructure import storage as _storage_check
            bucket_path = _storage_check.local_model_path(p.name)
            if bucket_path is not None:
                p.parent.mkdir(parents=True, exist_ok=True)
                # Symlink if possible (saves disk + matches mount semantics),
                # else copy. Falls back to copy on Windows without privilege.
                try:
                    if p.exists() or p.is_symlink():
                        p.unlink()
                    p.symlink_to(bucket_path)
                    logger.warning("Bucket-mount linked — file=%s → %s", p.name, bucket_path)
                except (OSError, NotImplementedError):
                    import shutil as _sh
                    _sh.copy2(bucket_path, p)
                    logger.warning("Bucket-mount copied — file=%s ← %s", p.name, bucket_path)
                return
        except Exception:  # noqa: BLE001
            pass

        # Second: HF Hub model repo (free, unlimited public).
        if hf_repo:
            try:
                from huggingface_hub import hf_hub_download
                p.parent.mkdir(parents=True, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=p.name,
                    revision=hf_revision,
                    token=hf_token,
                    local_dir=str(p.parent),
                )
                logger.warning(
                    "HF Hub download OK — repo=%s file=%s → %s (size=%d bytes)",
                    hf_repo, p.name, downloaded, Path(downloaded).stat().st_size,
                )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "HF Hub download FAILED — repo=%s file=%s err=%s. Will try S3 next.",
                    hf_repo, p.name, exc,
                )

        # Fallback: S3 bucket (when archival is configured).
        from webui.infrastructure import storage as _storage
        if not _storage.is_enabled():
            return
        models_prefix = (os.environ.get("MODELS_S3_KEY_PREFIX") or "models/").lstrip("/")
        if not models_prefix.endswith("/"):
            models_prefix += "/"
        _storage.download_to_path(models_prefix + p.name, local)

    try:
        for path in paths:
            _pull(path)
    except Exception:  # noqa: BLE001
        logger.exception("Weight pre-fetch raised — pipeline will try local paths.")
