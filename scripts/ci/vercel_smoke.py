"""Vercel proxy-mode smoke test, run in a venv built ONLY from requirements.txt.

Catches the failure that no other check sees: a heavy import (torch, cv2,
ultralytics) reaching module level on the path api/index.py imports, which
kills /, /api/tts and /api/measure/mock on Vercel with no visible error.

usage: python scripts/ci/vercel_smoke.py   (cwd = repo root; no network needed)
"""
from __future__ import annotations

import os
import runpy
import sys

os.environ["POINTSX_VERCEL"] = "1"
os.environ["POINTSX_INFERENCE_ENDPOINT"] = "http://127.0.0.1:9"  # unreachable on purpose

app = runpy.run_path("api/index.py")["app"]

from fastapi.testclient import TestClient  # noqa: E402

failures: list[str] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


heavy = sorted(m for m in ("torch", "cv2", "ultralytics") if m in sys.modules)
check("no heavy modules after import", not heavy, heavy)

with TestClient(app) as client:
    health = client.get("/api/health").json()
    check("health proxy_mode", health.get("proxy_mode") is True, health)

    mock = client.post("/api/measure/mock", data={"height_cm": "175", "sex": "female"})
    check("mock 200 with 18 measurements", mock.status_code == 200 and len(mock.json()["measurements"]) == 18,
          mock.status_code)

    root = client.get("/")
    check("/ serves index.html", root.status_code == 200 and "text/html" in root.headers.get("content-type", ""),
          root.status_code)
    check("camera Permissions-Policy", root.headers.get("permissions-policy") == "camera=(self)")
    check("/dataset.html 200", client.get("/dataset.html").status_code == 200)

    jpeg = ("x.jpg", b"\xff\xd8" + b"0" * 32, "image/jpeg")
    proxied = client.post("/api/measure", data={"height_cm": "175", "sex": "female"},
                          files={"front": jpeg, "side": jpeg})
    check("measure proxied (502 on unreachable upstream)", proxied.status_code == 502, proxied.status_code)

    keep = client.get("/api/keepalive")
    check("keepalive pings upstream (502 when unreachable)", keep.status_code == 502, keep.status_code)
    os.environ["CRON_SECRET"] = "ci-test-secret"
    try:
        denied = client.get("/api/keepalive")
        allowed = client.get("/api/keepalive", headers={"Authorization": "Bearer ci-test-secret"})
    finally:
        del os.environ["CRON_SECRET"]
    check("keepalive rejects callers without CRON_SECRET", denied.status_code == 401, denied.status_code)
    check("keepalive accepts the cron bearer", allowed.status_code == 502, allowed.status_code)

heavy = sorted(m for m in ("torch", "cv2", "ultralytics") if m in sys.modules)
check("no heavy modules after requests", not heavy, heavy)

if failures:
    print(f"\n{len(failures)} check(s) failed", file=sys.stderr)
    sys.exit(1)
print("\nvercel smoke: all checks passed")
