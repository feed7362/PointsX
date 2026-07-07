# Dataset Viewer (local desktop app)

Native desktop app (CustomTkinter) to browse `dataset_submissions` from Supabase: record count, list, detail view with decrypted photos, formatted metadata, and permanent delete (two-step confirmation).

**Runs locally only** — not deployed to Vercel.

## Prerequisites

- Python 3.12+ with Tkinter (included with most Python installs on macOS)
- Supabase **service_role** key (Dashboard → Settings → API)
- Offline **dataset private key** file (`dataset_private_key.txt` at repo root, or custom path)

## Setup

```bash
cd dataset-viewer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set SUPABASE_SERVICE_ROLE_KEY and DATASET_PRIVATE_KEY_FILE
```

## Run

```bash
./run.sh
```

Or:

```bash
source .venv/bin/activate
python app.py
```

## Security

- Desktop app only — no HTTP server, nothing exposed on the network.
- Delete removes the DB row **and** both encrypted photos from Storage (requires typing `DELETE` to confirm).
- Never commit `.env`, service_role key, or private key.
- Service role bypasses RLS — keep credentials offline.
