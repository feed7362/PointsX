#!/usr/bin/env python3
"""
Decrypt dataset photos encrypted with libsodium sealed box.

Usage:
    # Decrypt a single file
    python scripts/decrypt_dataset.py --key <base64_private_key> --input encrypted.bin --output photo.jpg
    
    # Decrypt from a key file
    python scripts/decrypt_dataset.py --keyfile dataset_private_key.txt --input encrypted.bin --output photo.jpg
    
    # Batch decrypt a directory
    python scripts/decrypt_dataset.py --keyfile dataset_private_key.txt --batch dataset_photos/ --output decrypted/

Requirements:
    pip install PyNaCl
"""

import argparse
import base64
import sys
from pathlib import Path

try:
    import nacl.public
    import nacl.encoding
except ImportError:
    print("ERROR: PyNaCl is not installed.")
    print("Install it with: pip install PyNaCl")
    sys.exit(1)


def load_private_key(key_b64: str) -> nacl.public.PrivateKey:
    """Load private key from base64 string."""
    try:
        key_bytes = base64.b64decode(key_b64.strip())
        return nacl.public.PrivateKey(key_bytes)
    except Exception as e:
        raise ValueError(f"Invalid private key format: {e}")


def decrypt_photo(ciphertext: bytes, private_key: nacl.public.PrivateKey) -> bytes:
    """Decrypt sealed box ciphertext."""
    try:
        box = nacl.public.SealedBox(private_key)
        plaintext = box.decrypt(ciphertext)
        return plaintext
    except Exception as e:
        raise ValueError(f"Decryption failed: {e}")


def decrypt_file(input_path: Path, output_path: Path, private_key: nacl.public.PrivateKey):
    """Decrypt a single file."""
    print(f"Decrypting: {input_path.name} -> {output_path.name}")
    
    ciphertext = input_path.read_bytes()
    plaintext = decrypt_photo(ciphertext, private_key)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(plaintext)
    
    print(f"  ✓ Decrypted {len(ciphertext)} bytes -> {len(plaintext)} bytes")


def decrypt_batch(input_dir: Path, output_dir: Path, private_key: nacl.public.PrivateKey):
    """Decrypt all files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    if not input_dir.is_dir():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    encrypted_files = list(input_dir.glob("*"))
    if not encrypted_files:
        print(f"No files found in {input_dir}")
        return
    
    print(f"Found {len(encrypted_files)} files in {input_dir}")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    for input_file in encrypted_files:
        if not input_file.is_file():
            continue
        
        # Preserve filename, optionally add .jpg if no extension
        output_file = output_dir / input_file.name
        if not output_file.suffix:
            output_file = output_file.with_suffix('.jpg')
        
        try:
            decrypt_file(input_file, output_file, private_key)
            success_count += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")
            error_count += 1
    
    print()
    print(f"Batch complete: {success_count} succeeded, {error_count} failed")


def main():
    parser = argparse.ArgumentParser(
        description="Decrypt dataset photos encrypted with libsodium sealed box",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decrypt single file with key
  %(prog)s --key abc123... --input encrypted.bin --output photo.jpg
  
  # Decrypt single file with key file
  %(prog)s --keyfile dataset_private_key.txt --input encrypted.bin --output photo.jpg
  
  # Batch decrypt directory
  %(prog)s --keyfile dataset_private_key.txt --batch dataset_photos/ --output decrypted/
"""
    )
    
    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument("--key", help="Base64-encoded private key")
    key_group.add_argument("--keyfile", type=Path, help="File containing private key")
    
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--batch", type=Path, help="Batch decrypt directory")
    mode_group.add_argument("--input", type=Path, help="Input encrypted file")
    
    parser.add_argument("--output", type=Path, required=True, help="Output path (file or directory)")
    
    args = parser.parse_args()
    
    # Load private key
    if args.keyfile:
        if not args.keyfile.exists():
            print(f"ERROR: Key file not found: {args.keyfile}")
            sys.exit(1)
        key_b64 = args.keyfile.read_text().strip()
    else:
        key_b64 = args.key
    
    try:
        private_key = load_private_key(key_b64)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Decrypt
    try:
        if args.batch:
            decrypt_batch(args.batch, args.output, private_key)
        else:
            if not args.input.exists():
                print(f"ERROR: Input file not found: {args.input}")
                sys.exit(1)
            decrypt_file(args.input, args.output, private_key)
        
        print()
        print("✓ Decryption complete!")
    
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
