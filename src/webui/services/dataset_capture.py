"""Best-effort saving of each captured (front, side) photo pair into the dataset folder."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from webui.services.uploads import detect_image_extension

logger = logging.getLogger(__name__)

_lock = asyncio.Lock()


def pair_stem_exists(directory: Path, stem: str) -> bool:
    """True if any ``a{stem}.*`` or ``p{stem}.*`` file already exists."""
    for prefix in ("a", "p"):
        if any(directory.glob(f"{prefix}{stem}.*")):
            return True
    return False


def unique_stem(directory: Path) -> str:
    """UTC timestamp stem for a capture pair; suffix ``_N`` if a collision exists."""
    now = datetime.now(timezone.utc)
    base = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    stem = base
    n = 0
    while directory.is_dir() and pair_stem_exists(directory, stem):
        n += 1
        stem = f"{base}_{n}"
    return stem


async def save_capture_pair(
    dataset_dir: Path,
    front_bytes: bytes,
    side_bytes: bytes,
) -> tuple[str | None, str | None]:
    """Persist the pair as ``a{timestamp}.ext`` / ``p{timestamp}.ext``.

    Returns ``(stem, None)`` on success. On failure, logs and returns
    ``(None, warning_uk)`` for the API ``warnings`` list — the measurement flow
    still succeeds.
    """
    try:
        async with _lock:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            stem = unique_stem(dataset_dir)
            front_path = dataset_dir / f"a{stem}.{detect_image_extension(front_bytes)}"
            side_path = dataset_dir / f"p{stem}.{detect_image_extension(side_bytes)}"
            front_path.write_bytes(front_bytes)
            side_path.write_bytes(side_bytes)
            logger.info("Saved capture pair to dataset: %s, %s", front_path, side_path)
            return stem, None
    except Exception as exc:  # noqa: BLE001 — best-effort persistence
        logger.exception("Failed to save capture pair to dataset folder %s", dataset_dir)
        detail = str(exc).strip() or type(exc).__name__
        msg = (
            "Не вдалося зберегти знімки у папку датасету "
            f"({dataset_dir}): {detail}"
        )
        return None, msg
