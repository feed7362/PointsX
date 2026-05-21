# Cloudflare Tunnel → https://fitmeasureai.pp.ua

Expose `pointsx-web` on your Mac without opening router ports. Cloudflare provides HTTPS; run the app locally on **HTTP** only (`127.0.0.1:8000`).

**Tunnel URL:** https://fitmeasureai.pp.ua

---

## One-time setup

```bash
brew install cloudflared
cd /Users/max/Repos/PointsX
./deploy/setup-tunnel.sh
```

`setup-tunnel.sh` will:

1. Run `cloudflared tunnel login` if needed (browser once)
2. Use tunnel **`fitmeasure-ai`** (ID `106ce0e3-…`)
3. Write `~/.cloudflared/config.yml` with ingress → `http://127.0.0.1:8000`
4. Write tunnel credentials (from `cloudflared tunnel token`)

In **Zero Trust → Networks → Tunnels → fitmeasure-ai**, you can also set the same Public Hostname:

- Host: `fitmeasureai.pp.ua`
- Service: `http://127.0.0.1:8000`

---

## Every time you run

```bash
cd /Users/max/Repos/PointsX
./deploy/run-tunnel.sh
```

Or two terminals:

```bash
pointsx-web --host 127.0.0.1 --port 8000
cloudflared tunnel run fitmeasure-ai
```

---

## Always-on connector (optional)

```bash
sudo cloudflared service install <TOKEN_FROM_DASHBOARD>
```

Keep `pointsx-web` running separately (launchd or manual).

---

## Verify

```bash
curl -sS -o /dev/null -w "local=%{http_code}\n" http://127.0.0.1:8000/
curl -sS -o /dev/null -w "public=%{http_code}\n" https://fitmeasureai.pp.ua/
```

Both should be **200**.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| **503** | Re-run `./deploy/setup-tunnel.sh`; restart `cloudflared tunnel run fitmeasure-ai` (not `--token` alone) |
| **530** | Tunnel not running — start `cloudflared tunnel run fitmeasure-ai` |
| **502** | `pointsx-web` not on port 8000 |
| Parking page | DNS not on tunnel — fix CNAME in Cloudflare |

## Security

- Do not commit `~/.cloudflared/*.json` (credentials).
- Rotate tunnel token if exposed.
- Optional: [Cloudflare Access](https://developers.cloudflare.com/cloudflare-one/policies/access/) on `fitmeasureai.pp.ua`.
