"""Post-deploy smoke test for the HF Space.

usage: python scripts/ci/space_smoke.py https://secret0123-pointx-backend.hf.space [expected_sha]

Checks what local tests cannot: the container actually booted with the weights
it needs (pipeline_ready + the default `coco` pose backend) and answers the
mock endpoint. Retries while the Space is still starting up.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

base = sys.argv[1].rstrip("/")
deadline = time.time() + 15 * 60
last_err = "no attempt"

while time.time() < deadline:
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=30) as r:
            health = json.load(r)
        if health.get("pipeline_ready"):
            break
        last_err = f"pipeline not ready: {health}"
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        last_err = repr(exc)
    print(f"waiting… {last_err}", flush=True)
    time.sleep(30)
else:
    print(f"FAIL health never became ready: {last_err}", file=sys.stderr)
    sys.exit(1)

print("health:", health)
failures = []
if "coco" not in health.get("pose_backends", []):
    failures.append(f"default pose backend 'coco' not loaded: {health.get('pose_backends')}")
if health.get("proxy_mode"):
    failures.append("Space is running in proxy mode — POINTSX_VERCEL leaked into the container env")

req = urllib.request.Request(
    f"{base}/api/measure/mock",
    data=b"height_cm=175&sex=female",
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        n = len(json.load(r)["measurements"])
    if n != 18:
        failures.append(f"mock returned {n} measurements, expected 18")
    else:
        print("mock: 18 measurements")
except Exception as exc:  # noqa: BLE001
    failures.append(f"mock request failed: {exc!r}")

if failures:
    print("\n".join(f"FAIL {f}" for f in failures), file=sys.stderr)
    sys.exit(1)
print("space smoke: all checks passed")
