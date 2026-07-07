#!/usr/bin/env python3
"""Verify dataset private key matches the public key in frontend config."""

from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

try:
    import nacl.public
except ImportError:
    print("Install PyNaCl: pip install PyNaCl")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_JS = ROOT / "src/webui/static/js/dataset/config.js"
DEFAULT_KEY = ROOT / "keys/dataset_private_key.txt"


def read_public_from_config() -> str:
    text = CONFIG_JS.read_text()
    match = re.search(r"DATASET_PUBLIC_KEY\s*=\s*'([^']+)'", text)
    if not match:
        raise SystemExit(f"Could not find DATASET_PUBLIC_KEY in {CONFIG_JS}")
    return match.group(1)


def public_from_private_file(keyfile: Path) -> str:
    priv = nacl.public.PrivateKey(base64.b64decode(keyfile.read_text().strip()))
    return base64.b64encode(bytes(priv.public_key)).decode()


def main() -> None:
    keyfile = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_KEY
    if not keyfile.is_file():
        raise SystemExit(f"Private key not found: {keyfile}")

    config_pub = read_public_from_config()
    derived_pub = public_from_private_file(keyfile)

    print("Private key file:", keyfile)
    print("Derived public:  ", derived_pub)
    print("config.js public:", config_pub)
    print()

    if derived_pub == config_pub:
        print("OK — keys match. Decryption should work for new submissions.")
        return

    print("MISMATCH — your private key does NOT match config.js / the live site.")
    print()
    print("Photos on the site were encrypted with the config.js public key.")
    print("Your private key can only decrypt photos encrypted with:")
    print(f"  {derived_pub}")
    print()
    print("Fix for future submissions: update config.js:")
    print(f"  export const DATASET_PUBLIC_KEY = '{derived_pub}';")
    print("Then redeploy dataset.html / config.js.")
    print()
    print("Note: submissions encrypted with the OLD public key cannot be")
    print("decrypted unless you still have that old keypair's private half.")
    sys.exit(1)


if __name__ == "__main__":
    main()
