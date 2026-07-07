"""Decrypt libsodium sealed-box photos (reuses scripts/decrypt_dataset.py)."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from decrypt_dataset import decrypt_photo, load_private_key  # noqa: E402

__all__ = ["load_private_key_from_file", "decrypt_bytes"]


def load_private_key_from_file(keyfile: Path):
    if not keyfile.is_file():
        raise FileNotFoundError(f"Private key file not found: {keyfile}")
    key_b64 = keyfile.read_text().strip()
    return load_private_key(key_b64)


def decrypt_bytes(ciphertext: bytes, private_key) -> bytes:
    return decrypt_photo(ciphertext, private_key)
