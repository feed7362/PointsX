#!/usr/bin/env bash
# Run PointsX web UI locally (no tunnel)
# Datasets will be saved to dataset_manual folder

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate

# Set dataset directory to dataset_manual
export POINTSX_DATASET_DIR="$ROOT/dataset_manual"

echo "Starting PointsX Web UI locally..."
echo "Dataset directory: $POINTSX_DATASET_DIR"
echo "Server will be available at: http://127.0.0.1:8000"
echo "Press Ctrl+C to stop."
echo ""

# Run the web server
exec pointsx-web --host 127.0.0.1 --port 8000
