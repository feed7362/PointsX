"""Download the entire ElasticLake / S3 bucket to a local folder.

One-shot operation, not a server task. Reads credentials from `.env`
in the PointsX repo root (same prefixes the backend accepts: ELK_*,
S3_*, R2_*, B2_*, AWS_* — first complete group wins).

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe scripts/download_bucket.py
    # downloads to ./bucket-dump/<prefix>/...   (default OUT_DIR)

    # custom destination + filter:
    .venv/Scripts/python.exe scripts/download_bucket.py \
        --out D:/pointx-archive \
        --prefix photos/measurements/

Skips files that already exist locally with the same size (resumable).
Prints a per-100-objects progress line so big buckets don't look stuck.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


_ENV_PREFIXES = ("S3_", "ELK_", "R2_", "B2_", "AWS_")


def _read_prefixed(prefix: str) -> dict[str, str | None]:
    return {
        "endpoint": os.environ.get(f"{prefix}ENDPOINT"),
        "access_key_id": os.environ.get(f"{prefix}ACCESS_KEY_ID"),
        "secret_access_key": os.environ.get(f"{prefix}SECRET_ACCESS_KEY"),
        "bucket": os.environ.get(f"{prefix}BUCKET"),
        "region": os.environ.get(f"{prefix}REGION"),
        "prefix": os.environ.get(f"{prefix}PREFIX"),
        "force_path_style": os.environ.get(f"{prefix}FORCE_PATH_STYLE"),
    }


def _resolve_config() -> dict[str, str | None] | None:
    for prefix in _ENV_PREFIXES:
        v = _read_prefixed(prefix)
        if all(v[k] for k in ("endpoint", "access_key_id", "secret_access_key", "bucket")):
            print(f"[config] using {prefix}* env-var group")
            return v
    return None


def _build_client(cfg: dict[str, str | None]):
    import boto3
    from botocore.config import Config as BotoConfig

    endpoint = cfg["endpoint"].rstrip("/")  # type: ignore[union-attr]
    raw = (cfg.get("force_path_style") or "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        force_path = True
    elif raw in ("0", "false", "no", "off"):
        force_path = False
    else:
        force_path = "amazonaws.com" not in endpoint.lower()

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=cfg["access_key_id"],
        aws_secret_access_key=cfg["secret_access_key"],
        region_name=cfg.get("region") or "us-east-1",
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "path" if force_path else "virtual"},
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "bucket-dump"),
                    help="Local destination folder (default: ./bucket-dump)")
    ap.add_argument("--prefix", default=None,
                    help="Only download keys with this prefix (e.g. photos/measurements/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List + size objects without downloading")
    args = ap.parse_args()

    # Load .env from the repo root (gitignored — holds the real S3 creds).
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        print("[warn] python-dotenv not installed; relying on shell env only")

    cfg = _resolve_config()
    if cfg is None:
        print("[fatal] no complete <PREFIX>* env-var group found. "
              "Need ENDPOINT, ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET.",
              file=sys.stderr)
        return 2

    client = _build_client(cfg)
    bucket: str = cfg["bucket"]  # type: ignore[assignment]
    bucket_prefix = (cfg.get("prefix") or "").rstrip("/")

    # Combine the <PREFIX>PREFIX (bucket-side namespace) with the optional
    # --prefix filter (per-run filter).
    list_prefix = bucket_prefix
    if args.prefix:
        list_prefix = (
            f"{bucket_prefix}/{args.prefix.lstrip('/')}"
            if bucket_prefix
            else args.prefix.lstrip("/")
        )

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[start] bucket={bucket} endpoint={cfg['endpoint']} "
          f"prefix={list_prefix or '(none)'} -> {out_root.resolve()}")

    paginator = client.get_paginator("list_objects_v2")
    page_kwargs: dict[str, object] = {"Bucket": bucket}
    if list_prefix:
        page_kwargs["Prefix"] = list_prefix

    n_total = 0
    n_downloaded = 0
    n_skipped = 0
    n_failed = 0
    bytes_downloaded = 0

    for page in paginator.paginate(**page_kwargs):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            size = int(obj["Size"])
            n_total += 1

            # Map the bucket key to a local path. Strip the bucket-side
            # prefix so the local tree mirrors the user's mental model
            # (no extra "photos/" wrapper when ELK_PREFIX=photos).
            rel = key
            if bucket_prefix and rel.startswith(bucket_prefix + "/"):
                rel = rel[len(bucket_prefix) + 1:]
            local_path = out_root / rel

            if args.dry_run:
                print(f"  {key}  ({size:,} B)")
                continue

            if local_path.is_file() and local_path.stat().st_size == size:
                n_skipped += 1
            else:
                local_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = local_path.with_suffix(local_path.suffix + ".part")
                try:
                    client.download_file(bucket, key, str(tmp))
                    tmp.replace(local_path)
                    n_downloaded += 1
                    bytes_downloaded += size
                except Exception as exc:  # noqa: BLE001
                    print(f"[fail] {key}: {exc}", file=sys.stderr)
                    if tmp.is_file():
                        try:
                            tmp.unlink()
                        except OSError:
                            pass
                    n_failed += 1

            if n_total % 100 == 0:
                print(f"  … {n_total:>6} objects scanned "
                      f"(downloaded={n_downloaded} skipped={n_skipped} "
                      f"failed={n_failed} bytes={bytes_downloaded:,})")

    print(
        f"[done] scanned={n_total} downloaded={n_downloaded} "
        f"skipped={n_skipped} failed={n_failed} bytes={bytes_downloaded:,}"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
