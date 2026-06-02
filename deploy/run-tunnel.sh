#!/usr/bin/env bash
# Start pointsx-web + Cloudflare tunnel for fitmeasureai.pp.ua
# Prereqs: ~/.cloudflared/config.yml configured (see deploy/DEPLOY-CLOUDFLARE.md)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TUNNEL_NAME="${POINTSX_TUNNEL_NAME:-fitmeasure-ai}"

cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate

pointsx-web --host 127.0.0.1 --port 8000 &
WEB_PID=$!

cleanup() {
  kill "$WEB_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "pointsx-web pid=$WEB_PID → http://127.0.0.1:8000"
echo "Starting Cloudflare tunnel '$TUNNEL_NAME' → https://fitmeasureai.pp.ua"
echo "Press Ctrl+C to stop both."

sleep 2
if ! kill -0 "$WEB_PID" 2>/dev/null; then
  echo "pointsx-web failed to start." >&2
  exit 1
fi

exec cloudflared tunnel run "$TUNNEL_NAME"
