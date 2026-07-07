#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

pick_python() {
  if [[ -x /usr/local/bin/python3.12 ]] && /usr/local/bin/python3.12 -c "import tkinter" 2>/dev/null; then
    echo /usr/local/bin/python3.12
  elif [[ -x /opt/homebrew/bin/python3.12 ]] && /opt/homebrew/bin/python3.12 -c "import tkinter" 2>/dev/null; then
    echo /opt/homebrew/bin/python3.12
  elif /usr/bin/python3 -c "import tkinter" 2>/dev/null; then
    echo /usr/bin/python3
  else
    echo "No Python with Tk found. Install: brew install python@3.12 python-tk@3.12" >&2
    exit 1
  fi
}

if [[ ! -d .venv ]]; then
  PY="$(pick_python)"
  echo "Creating venv with ${PY}…"
  "${PY}" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install -r requirements.txt
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt
exec python app.py
