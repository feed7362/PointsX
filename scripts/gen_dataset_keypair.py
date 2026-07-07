#!/usr/bin/env python3
"""
Generate an X25519 keypair for dataset photo encryption.

The public key is embedded in the frontend (dataset/config.js).
The private key MUST be kept offline and never committed to version control.

Usage:
    python scripts/gen_dataset_keypair.py

Outputs:
    - Public key (base64) for frontend config
    - Private key (base64) to store securely offline
"""

import base64
from pathlib import Path

try:
    import nacl.public
    import nacl.encoding
except ImportError:
    print("ERROR: PyNaCl is not installed.")
    print("Install it with: pip install PyNaCl")
    exit(1)


def generate_keypair():
    """Generate X25519 keypair for sealed box encryption."""
    private_key = nacl.public.PrivateKey.generate()
    public_key = private_key.public_key
    
    # Encode keys to base64 for easy storage/transport
    private_key_b64 = base64.b64encode(bytes(private_key)).decode('ascii')
    public_key_b64 = base64.b64encode(bytes(public_key)).decode('ascii')
    
    return public_key_b64, private_key_b64


def main():
    print("=" * 70)
    print("Dataset Photo Encryption Keypair Generator")
    print("=" * 70)
    print()
    
    public_key_b64, private_key_b64 = generate_keypair()
    
    print("✓ Keypair generated successfully!")
    print()
    print("-" * 70)
    print("PUBLIC KEY (embed in frontend config):")
    print("-" * 70)
    print(f"  {public_key_b64}")
    print()
    print("Add this to: src/webui/static/js/dataset/config.js")
    print(f"  export const DATASET_PUBLIC_KEY = '{public_key_b64}';")
    print()
    print("-" * 70)
    print("PRIVATE KEY (KEEP OFFLINE - DO NOT COMMIT):")
    print("-" * 70)
    print(f"  {private_key_b64}")
    print()
    print("⚠️  CRITICAL: Save this private key securely!")
    print("   - Store it in a password manager or encrypted vault")
    print("   - Never commit it to version control")
    print("   - You need it to decrypt the dataset photos")
    print()
    
    # Optionally write to a .gitignored file
    gitignore_path = Path(__file__).parent.parent / ".gitignore"
    if gitignore_path.exists() and "dataset_private_key.txt" not in gitignore_path.read_text():
        print("Recommendation: Add 'dataset_private_key.txt' to .gitignore")
    
    save = input("Save private key to dataset_private_key.txt? (y/N): ").strip().lower()
    if save == 'y':
        key_file = Path(__file__).parent / "dataset_private_key.txt"
        key_file.write_text(f"{private_key_b64}\n")
        key_file.chmod(0o600)  # Restrict permissions
        print(f"✓ Private key saved to: {key_file}")
        print("  (File permissions set to 600 - owner read/write only)")
    
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
