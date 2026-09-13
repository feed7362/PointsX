"""Download SMPLitex photo-real skin UV maps for SMPL-X.

The SMPLitex dataset (https://github.com/dancasas/SMPLitex) is gated on
Hugging Face — you need to accept the licence and pass an HF token to
download. This script supports two paths:

  1. Authenticated HF download (requires `huggingface_hub` + an access token).
  2. Manual: drop your own UV-map .png files into the target directory; this
     script does nothing if the folder already has files.

Usage:
    # Option A (auth required):
    python -m pointsx.synthetic.download_smplitex \
        --out assets/textures/smplitex --token hf_XXXX

    # Option B (manual):
    # Just put any UV-map PNGs (4K SMPL-X compatible) into
    # assets/textures/smplitex/ — the renderer picks them up automatically.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_HF_REPO = "dancasas/SMPLitex"


def download_via_hf(out_dir: Path, token: str, limit: int = 50) -> int:
    """Download via the official `huggingface_hub` API (handles auth + gated access)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed. Run:\n"
            "  pip install huggingface_hub\n"
            "Or download manually — see module docstring.",
            file=sys.stderr,
        )
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pulling SMPLitex from HF repo {_HF_REPO} (this is a gated dataset)…")
    try:
        snapshot_download(
            repo_id=_HF_REPO,
            repo_type="dataset",
            local_dir=str(out_dir),
            token=token,
            allow_patterns=["*.png", "textures/*.png", "**/*.png"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: HF download failed: {exc}", file=sys.stderr)
        print(
            "\nIf you got a 401/403, you need to:\n"
            f"  1. Visit https://huggingface.co/datasets/{_HF_REPO}\n"
            "  2. Accept the licence terms (one-time per account)\n"
            "  3. Create an access token at https://huggingface.co/settings/tokens\n"
            "  4. Re-run with --token hf_xxxxxxxx\n",
            file=sys.stderr,
        )
        return 0

    pngs = list(out_dir.rglob("*.png"))
    print(f"Got {len(pngs)} PNG file(s) in {out_dir}")
    return len(pngs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SMPLitex skin UV maps for SMPL-X")
    parser.add_argument("--out", required=True, type=Path,
                        help="Destination directory (e.g. assets/textures/smplitex)")
    parser.add_argument("--token", default=None,
                        help="HF access token. If omitted, this script does nothing — "
                             "see module docstring for the manual path.")
    parser.add_argument("--limit", type=int, default=50,
                        help="(Reserved for future per-file mode; ignored by snapshot_download.)")
    args = parser.parse_args()

    if args.out.is_dir():
        existing = list(args.out.rglob("*.png"))
        if existing:
            print(f"{args.out} already has {len(existing)} PNG(s) — nothing to do.")
            return

    if not args.token:
        print(
            "No --token provided.\n\n"
            "SMPLitex is a gated Hugging Face dataset. To use auto-download:\n"
            f"  1. Open https://huggingface.co/datasets/{_HF_REPO}\n"
            "  2. Accept the dataset licence (one-time)\n"
            "  3. Create a token at https://huggingface.co/settings/tokens\n"
            "  4. Re-run:  python -m pointsx.synthetic.download_smplitex \\\n"
            f"               --out {args.out} --token hf_xxxxxxxx\n\n"
            "Alternative — drop any 4K SMPL-X UV-map PNGs into\n"
            f"  {args.out}\n"
            "and the renderer will pick them up automatically.",
            file=sys.stderr,
        )
        return

    download_via_hf(args.out, args.token, args.limit)


if __name__ == "__main__":
    main()
