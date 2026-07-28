# Run PointsX Locally (No Tunnel)

This guide shows how to run the entire PointsX application locally on your machine, with datasets saved to the `dataset_manual` folder.

## Quick Start

```bash
cd /Users/max/Repos/PointsX
./run-local.sh
```

The web UI will be available at **http://127.0.0.1:8000**

Press **Ctrl+C** to stop the server.

## What This Does

- Runs the FastAPI backend with all ML models loaded locally
- Serves the web UI on localhost port 8000
- Saves all captured image pairs to `dataset_manual/` folder
- No tunnel or external hosting (purely local)

## Dataset Storage

All captured images are automatically saved to:

```
/Users/max/Repos/PointsX/dataset_manual/
```

Files are named with timestamps:
- `a<timestamp>.jpg` - front view (анфас)
- `p<timestamp>.jpg` - side view (профіль)

## Requirements

Make sure you have:

1. Python virtual environment activated
2. All dependencies installed (`pip install -e .`)
3. Model files in `models/` directory (~130 MB)

If you haven't set up the project yet, run:

```bash
cd /Users/max/Repos/PointsX
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Accessing from Other Devices

### Option 1: Local Network Only (HTTP)

For testing without HTTPS, you can access from devices on the same Wi-Fi:

```bash
# Get your local IP
ipconfig getifaddr en0

# Run the server (bound to all interfaces)
source .venv/bin/activate
export POINTSX_DATASET_DIR="$(pwd)/dataset_manual"
pointsx-web --host 0.0.0.0 --port 8000
```

Then open `http://<YOUR-LOCAL-IP>:8000` on other devices.

**Note:** Camera access may not work on some browsers without HTTPS.

### Option 2: Local Network with HTTPS

Generate a self-signed certificate for camera access:

```bash
mkdir -p .dev-certs
openssl req -x509 -newkey rsa:2048 -nodes \
  -keyout .dev-certs/key.pem -out .dev-certs/cert.pem -days 365 \
  -subj "/CN=pointsx-local"

ipconfig getifaddr en0  # note this IP

source .venv/bin/activate
export POINTSX_DATASET_DIR="$(pwd)/dataset_manual"
pointsx-web --host 0.0.0.0 --port 8000 \
  --ssl-keyfile .dev-certs/key.pem --ssl-certfile .dev-certs/cert.pem
```

Open `https://<YOUR-LOCAL-IP>:8000` and accept the self-signed certificate warning.

### Option 3: Public URL with Tunnel

For access from anywhere (not just local network), use the tunnel:

```bash
./deploy/run-tunnel.sh
```

This uses Cloudflare to expose your local server at https://fitmeasureai.pp.ua

## Customization

### Change Dataset Location

Edit `run-local.sh` and modify:

```bash
export POINTSX_DATASET_DIR="/path/to/your/dataset/folder"
```

Or run manually:

```bash
source .venv/bin/activate
export POINTSX_DATASET_DIR="/custom/path"
pointsx-web --host 127.0.0.1 --port 8000
```

### Change Port

```bash
source .venv/bin/activate
export POINTSX_DATASET_DIR="$(pwd)/dataset_manual"
pointsx-web --host 127.0.0.1 --port 9000
```

### Use Different Models

Set environment variables before running:

```bash
export POINTSX_POSE_MODEL_CUSTOM="models/my-pose.pt"
export POINTSX_SEG_MODEL="models/my-seg.pt"
export POINTSX_REGRESSION_MODEL="models/my-reg.pt"
./run-local.sh
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Port already in use | Change port: `pointsx-web --port 9000` |
| Models not found | Check that `models/*.pt` files exist |
| Virtual env not found | Run `python3 -m venv .venv` first |
| Permission denied | Run `chmod +x run-local.sh` |
| Dataset folder not created | The folder is created automatically on first capture |

## Development Mode

For auto-reload during development:

```bash
source .venv/bin/activate
export POINTSX_DATASET_DIR="$(pwd)/dataset_manual"
pointsx-web --host 127.0.0.1 --port 8000 --reload
```

The server will restart automatically when you modify Python files.

## Comparison: Local vs Tunnel

| Feature | Local (`run-local.sh`) | Tunnel (`run-tunnel.sh`) |
|---------|------------------------|--------------------------|
| Speed | Fastest (no network hop) | Slightly slower |
| Access | Same machine or LAN | Anywhere with internet |
| HTTPS | Manual setup needed | Automatic |
| Dataset | Saved to `dataset_manual` | Saved to `dataset` by default |
| Setup | Simple | Requires Cloudflare config |
