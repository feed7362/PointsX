---
title: Pointx Backend
emoji: 🏃
colorFrom: purple
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: PointsX inference API — body measurements from 2 photos
---

# PointsX — backend (Hugging Face Space)

FastAPI inference service for the PointsX body-measurement demo. Talks to:

- **Frontend** (separate Vercel repo `Pointx-frontend`) over CORS
- **S3-compatible bucket** for:
  - Model weights at `<bucket>/<MODELS_S3_KEY_PREFIX>/`
  - Per-request photo + envelope archive at `<bucket>/measurements/<uuid>/`

```
browser ⇄ Vercel (frontend) → this HF Space → S3 bucket
```

## Required env vars

Set under the Space's **Settings → Variables and secrets**. Mark anything
with key material as **Secret**.

| Name | Type | Notes |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | Variable | comma-separated origins; `*` while testing |

S3-compatible storage (loader recognises `S3_*`, `ELK_*`, `R2_*`, `B2_*`,
`AWS_*` — pick one prefix group):

| Name | Type |
|---|---|
| `ELK_ENDPOINT` (or `S3_ENDPOINT`) | Variable |
| `ELK_ACCESS_KEY_ID` | Variable |
| `ELK_SECRET_ACCESS_KEY` | **Secret** |
| `ELK_BUCKET` | Variable |
| `ELK_REGION` | Variable |
| `ELK_PREFIX` | Variable (optional) |
| `MODELS_S3_KEY_PREFIX` | Variable (default `models/`) |

## Bucket layout the container expects

```
<bucket>/
  <S3_PREFIX>/
    <MODELS_S3_KEY_PREFIX>/
      yolo26-pose.pt                    ← pulled on boot
      yolo12l-person-seg-extended.pt    ← pulled on boot
      circumference_regressor.pt        ← pulled on boot (optional)
    measurements/
      <uuid>/
        front.jpg                       ← written per /api/measure
        side.jpg
        envelope.json
```

Upload the three weight files **once** before the Space's first boot, via
the provider's web UI or `aws s3 cp`.

## How the container starts

1. `pointsx-web` loads `.env` (no-op on HF — env comes from the Space).
2. FastAPI `lifespan` resolves model paths from env vars (defaults under
   `models/`).
3. For each missing weight, `storage.download_to_path()` pulls it from
   `s3://<bucket>/<S3_PREFIX>/<MODELS_S3_KEY_PREFIX>/<file>` into the local
   path. Idempotent — warm restarts skip the download.
4. `WebuiPipeline` constructs the YOLO models.
5. `/api/measure` ready.

You'll see in the Logs tab:
```
WARNING: Object-storage: detected ELK_* env-var prefix ...
WARNING: download_to_path OK — key=models/yolo26-pose.pt → ...
WARNING: download_to_path OK — key=models/yolo12l-person-seg-extended.pt → ...
INFO:    Pipeline loaded — pose_coco=models/yolo26-pose.pt ...
WARNING: Storage probe: archival is ENABLED on startup
```

## API surface

| Route | Description |
|---|---|
| `POST /api/measure` | front + side photo + height → MeasurementEnvelope JSON |
| `POST /api/measure/mock` | same shape with synthetic data, no images, no ML |
| `POST /api/tts` | (optional) Ukrainian text-to-speech for the UI |

CORS is enabled.

## Local development

```powershell
uv sync
cp .env.example .env       # fill in S3 creds
pointsx-web --reload       # http://127.0.0.1:8000/
```

## Build the container locally

```powershell
docker build -t pointx-backend .
docker run --rm -p 7860:7860 --env-file .env pointx-backend
```

## Deploy

This repo IS the Hugging Face Space — `git push` to its remote.

```powershell
git push                  # if 'origin' is the Space
# or
git remote add space https://huggingface.co/spaces/<you>/pointx-backend
git push space main
```

First build: ~5–8 min (torch wheels). Watch progress in the **Logs** tab.

## Free CPU tier limits

- ~10–20 s per measurement.
- Container sleeps after ~48 h idle; first wake takes 30 s–2 min.
- Sub-second latency needs T4 GPU ($40/mo). Set `POINTSX_DEVICE=cuda` on
  Variables and rebuild — no code change.
