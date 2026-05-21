# FitMeasure web UI — start guide

## Quick start (public site)

**URL:** https://fitmeasureai.pp.ua

```bash
cd /Users/max/Repos/PointsX
./deploy/run-tunnel.sh
```

Leave that terminal open. Press **Ctrl+C** to stop the app and tunnel.

Check:

```bash
curl -sS -o /dev/null -w "local=%{http_code} public=%{http_code}\n" \
  http://127.0.0.1:8000/ https://fitmeasureai.pp.ua/
```

Both should print `200`.

---

## First time only

1. Install dependencies:

   ```bash
   cd /Users/max/Repos/PointsX
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e .
   # optional TTS: pip install edge-tts
   ```

2. Ensure `models/*.pt` are present (~130 MB).

3. Install cloudflared and configure the tunnel:

   ```bash
   brew install cloudflared
   ./deploy/setup-tunnel.sh
   ```

   `setup-tunnel.sh` opens the browser once for `cloudflared tunnel login`, then writes `~/.cloudflared/config.yml` (ingress: `fitmeasureai.pp.ua` → `http://127.0.0.1:8000`).

4. In **Cloudflare Zero Trust → Networks → Tunnels → fitmeasure-ai**, confirm a **Public Hostname**:

   - Host: `fitmeasureai.pp.ua`
   - Service: **HTTP** → `127.0.0.1:8000`

Details: [`deploy/DEPLOY-CLOUDFLARE.md`](deploy/DEPLOY-CLOUDFLARE.md)

---

## Manual start (two terminals)

**Terminal 1 — app:**

```bash
cd /Users/max/Repos/PointsX
source .venv/bin/activate
pointsx-web --host 127.0.0.1 --port 8000
```

Wait for `Application startup complete`.

**Terminal 2 — tunnel:**

```bash
cloudflared tunnel run fitmeasure-ai
```

Open https://fitmeasureai.pp.ua on your phone (any network). Camera needs HTTPS; the tunnel provides it.

---

## Other ways to run

### Local Wi‑Fi only (no Cloudflare)

```bash
cd /Users/max/Repos/PointsX
mkdir -p .dev-certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout .dev-certs/key.pem -out .dev-certs/cert.pem -days 365 \
  -subj "/CN=fitmeasure-local"

ipconfig getifaddr en0   # use this IP on the phone

source .venv/bin/activate
pointsx-web --host 0.0.0.0 --port 8000 \
  --ssl-keyfile .dev-certs/key.pem --ssl-certfile .dev-certs/cert.pem
```

Log must say `https://`. On the phone: `https://<LAN-IP>:8000` (accept self-signed cert).

Run SSL flags on **one line** (a line break without `\` skips them).

### Temporary URL (ngrok)

```bash
source .venv/bin/activate
pointsx-web --host 127.0.0.1 --port 8000
```

Second terminal: `ngrok http 8000` → open the printed `https://….ngrok-free.app` URL.

### VPS (always-on)

See [`deploy/DEPLOY-CLOUDFLARE.md`](deploy/DEPLOY-CLOUDFLARE.md) and [`deploy/nginx-fitmeasureai.pp.ua.conf`](deploy/nginx-fitmeasureai.pp.ua.conf).

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| **502** on domain | Run `./deploy/run-tunnel.sh` (app + tunnel stopped) |
| **503** on domain | Re-run `./deploy/setup-tunnel.sh`; use `cloudflared tunnel run fitmeasure-ai`, not `--token` alone |
| Camera blocked | Use `https://fitmeasureai.pp.ua`, not `http://` |
| SSL command ignored | Put all `pointsx-web` flags on one line |

---

## Security

The UI has no login. Restrict access with [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) before sharing widely.
