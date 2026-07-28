"""Verify object-storage credentials before pointing the Space at them.

Runs the exact operations the archival path uses — PutObject with an explicit
ContentLength, GetObject, then DeleteObject — through the same client config as
`webui.storage`, so a pass here means archiving will work in production.

Usage:
    cd Q:/Projects/KHNU/PointsX
    # reads S3_* / AWS_* / ELK_* from .env, first complete group wins
    .venv/Scripts/python.exe scripts/verify_s3.py

    # or point at a candidate AWS setup without touching .env:
    .venv/Scripts/python.exe scripts/verify_s3.py \
        --endpoint https://s3.eu-central-1.amazonaws.com \
        --bucket pointx-photos --region eu-central-1 \
        --key-id AKIA... --secret ...

Exit code 0 = usable for archival.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_PREFIXES = ("S3_", "AWS_", "ELK_", "R2_", "B2_")


def _from_env() -> dict | None:
    for p in _PREFIXES:
        v = {
            "endpoint": os.environ.get(f"{p}ENDPOINT"),
            "key_id": os.environ.get(f"{p}ACCESS_KEY_ID"),
            "secret": os.environ.get(f"{p}SECRET_ACCESS_KEY"),
            "bucket": os.environ.get(f"{p}BUCKET"),
            "region": os.environ.get(f"{p}REGION") or "us-east-1",
            "prefix": os.environ.get(f"{p}PREFIX") or "",
        }
        if all(v[k] for k in ("endpoint", "key_id", "secret", "bucket")):
            print(f"[config] using {p}* env group")
            return v
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    for f in ("endpoint", "bucket", "region", "key-id", "secret", "prefix"):
        ap.add_argument(f"--{f}", default=None)
    ap.add_argument("--repeat", type=int, default=3,
                    help="PUT this many times — transient providers fail intermittently, "
                         "so a single success can be misleading (default 3)")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    if args.endpoint and args.bucket and args.key_id and args.secret:
        cfg = {"endpoint": args.endpoint, "bucket": args.bucket,
               "region": args.region or "us-east-1", "key_id": args.key_id,
               "secret": args.secret, "prefix": args.prefix or ""}
        print("[config] using command-line arguments")
    else:
        cfg = _from_env()
        if cfg is None:
            print("[fatal] no complete credential group found (need ENDPOINT, "
                  "ACCESS_KEY_ID, SECRET_ACCESS_KEY, BUCKET)", file=sys.stderr)
            return 2

    import boto3
    from botocore.config import Config as BotoConfig
    from botocore.exceptions import ClientError

    endpoint = cfg["endpoint"].rstrip("/")
    virtual = "amazonaws.com" in endpoint.lower()
    print(f"[config] endpoint={endpoint} bucket={cfg['bucket']} region={cfg['region']} "
          f"addressing={'virtual' if virtual else 'path'}")

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=cfg["key_id"],
        aws_secret_access_key=cfg["secret"],
        region_name=cfg["region"],
        # Mirrors webui.storage exactly, so this is a true rehearsal.
        config=BotoConfig(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "adaptive"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "virtual" if virtual else "path"},
        ),
    )

    body = b"pointsx-verify-" + uuid.uuid4().hex.encode()
    key = f"{cfg['prefix'].rstrip('/') + '/' if cfg['prefix'] else ''}_verify/{uuid.uuid4().hex}.txt"
    ok = fail = 0

    for i in range(1, args.repeat + 1):
        t0 = time.time()
        try:
            client.put_object(Bucket=cfg["bucket"], Key=key, Body=body,
                              ContentLength=len(body), ContentType="text/plain")
            got = client.get_object(Bucket=cfg["bucket"], Key=key)["Body"].read()
            if got != body:
                print(f"  [{i}] FAIL round-trip mismatch", file=sys.stderr)
                fail += 1
                continue
            client.delete_object(Bucket=cfg["bucket"], Key=key)
            print(f"  [{i}] ok  put+get+delete in {time.time()-t0:.2f}s")
            ok += 1
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "?")
            print(f"  [{i}] FAIL {code}: {exc}", file=sys.stderr)
            fail += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
            fail += 1

    print(f"\n[result] {ok}/{args.repeat} succeeded")
    if fail == 0:
        print("[result] PASS — archival will work with these credentials")
        return 0
    if ok:
        print("[result] FLAKY — intermittent failures. The archive path retries "
              "transient codes, but an unreliable provider will still drop uploads.")
        return 1
    print("[result] FAIL — archival would not work", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
