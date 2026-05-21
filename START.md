# Start FitMeasure

## Every day

```bash
cd /Users/max/Repos/PointsX
./deploy/run-tunnel.sh
```

Open **https://fitmeasureai.pp.ua** · Stop with **Ctrl+C**

---

## First time

```bash
cd /Users/max/Repos/PointsX
python3 -m venv .venv && source .venv/bin/activate && pip install -e .
brew install cloudflared
./deploy/setup-tunnel.sh
./deploy/run-tunnel.sh
```

More detail: [RUN-WEBUI.md](RUN-WEBUI.md)
