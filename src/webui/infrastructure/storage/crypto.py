"""Sealed-box encryption for archived photos (encrypt-only: needs just the public key)."""
from __future__ import annotations

import os

# Same scheme as the dataset flow: libsodium sealed box (X25519), i.e.
# crypto_box_seal. Encrypting needs ONLY the public key, so this container can
# seal photos but can never open them — the private key stays offline. That
# makes the archive E2E-encrypted at rest exactly like dataset submissions, and
# as a side effect every object becomes application/octet-stream, which is what
# the MIME-restricted dataset bucket accepts.
ENC_ALGO = "libsodium-sealedbox-x25519"
_DEFAULT_DATASET_PUBLIC_KEY = "HxB6+jdtcSdOHKc6e/YmIH4MlQUjlJtrkwGY7sF3WW8="


def _dataset_public_key() -> str | None:
    """Base64 X25519 public key used to seal archived photos."""
    if (os.environ.get("POINTSX_ARCHIVE_ENCRYPT") or "").strip().lower() in ("0", "false", "no", "off"):
        return None
    return (os.environ.get("DATASET_PUBLIC_KEY") or _DEFAULT_DATASET_PUBLIC_KEY).strip() or None


def _seal(body: bytes, public_key_b64: str) -> bytes:
    """Encrypt with a libsodium sealed box. Raises if PyNaCl is unavailable."""
    import base64

    from nacl.public import PublicKey, SealedBox

    return SealedBox(PublicKey(base64.b64decode(public_key_b64))).encrypt(body)
