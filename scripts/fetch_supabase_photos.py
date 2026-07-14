"""One-shot: download + decrypt every dataset photo from Supabase to a local folder.

Reuses dataset-viewer/repository.py (DatasetRepository) — same Supabase client,
same NaCl decryption, same key/config checks. Nothing new reimplemented.

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe scripts/fetch_supabase_photos.py
    # -> ./supabase-dump/<submission_id>/{front,side}.jpg + meta.json

Env comes from PointsX/.env (SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
STORAGE_BUCKET, DATASET_PRIVATE_KEY_FILE). Skips photos already on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# repository.py lives in dataset-viewer/ and imports its siblings by bare name.
sys.path.insert(0, str(REPO_ROOT / "dataset-viewer"))

# The .env path for the key is "../keys/..." relative to dataset-viewer, which
# resolves outside the repo. Point at the real file before repository.py reads it.
_key = REPO_ROOT / "keys" / "dataset_private_key.txt"
if _key.is_file():
    os.environ["DATASET_PRIVATE_KEY_FILE"] = str(_key)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "supabase-dump"),
                    help="Local destination folder (default: ./supabase-dump)")
    args = ap.parse_args()

    from repository import DatasetRepository  # noqa: E402

    repo = DatasetRepository()
    if not repo.keys_match_config:
        print("[fatal] private key does not match the public key in config.js — "
              "decryption would fail. Run scripts/verify_dataset_keys.py.", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    subs = repo.list_submissions()
    print(f"[start] {len(subs)} submissions -> {out_root.resolve()}")

    n_photos = n_skipped = n_failed = 0
    for i, sub in enumerate(subs, 1):
        sid = sub["id"]
        folder = out_root / sid
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "meta.json").write_text(json.dumps(sub, indent=2, default=str))

        for which in ("front", "side"):
            dest = folder / f"{which}.jpg"
            if dest.is_file() and dest.stat().st_size > 0:
                n_skipped += 1
                continue
            try:
                dest.write_bytes(repo.get_photo_bytes(sid, which))
                n_photos += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[fail] {sid}/{which}: {exc}", file=sys.stderr)
                n_failed += 1

        if i % 20 == 0:
            print(f"  … {i}/{len(subs)} submissions "
                  f"(saved={n_photos} skipped={n_skipped} failed={n_failed})")

    print(f"[done] submissions={len(subs)} photos_saved={n_photos} "
          f"skipped={n_skipped} failed={n_failed}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
