#!/usr/bin/env bash
# One-time Cloudflare Tunnel setup for https://fitmeasureai.pp.ua
# Prereqs: tunnel "fitmeasure-ai" exists in Cloudflare; zone DNS on Cloudflare.

set -euo pipefail

TUNNEL_NAME="${POINTSX_TUNNEL_NAME:-fitmeasure-ai}"
HOSTNAME="${POINTSX_TUNNEL_HOST:-fitmeasureai.pp.ua}"
ORIGIN="${POINTSX_TUNNEL_ORIGIN:-http://127.0.0.1:8000}"
CF_DIR="${HOME}/.cloudflared"

echo "==> Checking cloudflared..."
command -v cloudflared >/dev/null || { echo "Install: brew install cloudflared"; exit 1; }

if [[ ! -f "${CF_DIR}/cert.pem" ]]; then
  echo "==> Log in to Cloudflare (browser will open)..."
  cloudflared tunnel login
fi

echo "==> Resolving tunnel..."
TUNNEL_ID=$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$0 ~ n {print $1; exit}')
if [[ -z "${TUNNEL_ID}" ]]; then
  echo "Tunnel '${TUNNEL_NAME}' not found. Create it in Zero Trust → Networks → Tunnels, or:"
  echo "  cloudflared tunnel create ${TUNNEL_NAME}"
  exit 1
fi
echo "    Tunnel: ${TUNNEL_NAME} (${TUNNEL_ID})"

echo "==> Writing credentials..."
export TOK CF_DIR
TOK=$(cloudflared tunnel token "${TUNNEL_NAME}")
export TOK
python3 - <<'PY'
import base64, json, os
from pathlib import Path
tok = os.environ["TOK"]
d = json.loads(base64.b64decode(tok))
cred = {"AccountTag": d["a"], "TunnelID": d["t"], "TunnelSecret": d["s"]}
p = Path(os.environ["CF_DIR"]) / f"{d['t']}.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(cred))
print(f"    {p}")
PY

echo "==> Writing ${CF_DIR}/config.yml ..."
cat > "${CF_DIR}/config.yml" <<EOF
tunnel: ${TUNNEL_ID}
credentials-file: ${CF_DIR}/${TUNNEL_ID}.json

ingress:
  - hostname: ${HOSTNAME}
    service: ${ORIGIN}
  - service: http_status:404
EOF

cloudflared tunnel ingress validate

echo "==> DNS (skip if already set in dashboard)..."
if cloudflared tunnel route dns "${TUNNEL_NAME}" "${HOSTNAME}" 2>/dev/null; then
  echo "    Routed ${HOSTNAME} → tunnel"
else
  echo "    Could not auto-route DNS (zone may be external). Ensure ${HOSTNAME} CNAME points to this tunnel."
fi

echo ""
echo "Done. Start the stack:"
echo "  ./deploy/run-tunnel.sh"
echo "Open: https://${HOSTNAME}"
