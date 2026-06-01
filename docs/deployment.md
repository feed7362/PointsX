# Deployment guide (demo branch)

Free-tier scientific demo stack: **Vercel** (static frontend) → **Hugging
Face Space** (FastAPI inference) → **S3-compatible bucket** (photo +
envelope archival). Everything below assumes the `demo` branch.

```
┌──────────────┐    ┌────────────────────────┐    ┌───────────────────────┐
│ your domain  │ →  │ Vercel (static SPA)    │ →  │ HF Space (FastAPI +   │
│ DNS @ CF     │    │ src/webui/static/      │    │ YOLO11 + envelope)    │
└──────────────┘    └────────────────────────┘    └────────────┬──────────┘
                                                                 │
                                                       any S3-compatible
                                                        object store
                                                       (R2 / B2 / S3 / …)
```

---

## Quick reference — env vars by surface

### Vercel project — exactly one variable

| Name | Value | Scope |
|---|---|---|
| `POINTSX_API_BASE` | `https://<you>-<space-name>.hf.space` | Production, Preview, Development |

No secrets needed. The value is injected into `static/config.js` at build
time and is public.

### HF Space — full list

Set in **Settings → Variables and secrets**. Anything with key material
goes under **Secret** so it's masked in the UI.

#### Required for the deployed shape

| Name | Type | Example | Notes |
|---|---|---|---|
| `CORS_ALLOW_ORIGINS` | Variable | `https://demo.yourname.dev,https://your-app.vercel.app` | Comma-separated. `*` is acceptable for an open scientific demo. |

#### Required if you want photo archival

| Name | Type | Notes |
|---|---|---|
| `S3_ENDPOINT` | Variable | Full URL — provider-specific (see table below) |
| `S3_ACCESS_KEY_ID` | Variable | |
| `S3_SECRET_ACCESS_KEY` | **Secret** | always Secret, never Variable |
| `S3_BUCKET` | Variable | bucket name only, no `s3://` prefix |
| `S3_REGION` | Variable | `auto` for R2; real region elsewhere |
| `S3_PREFIX` | Variable | optional, e.g. `prod/` to namespace |

#### Optional pipeline knobs

| Name | Type | Default | When to override |
|---|---|---|---|
| `POINTSX_DEVICE` | Variable | `cpu` (set in Dockerfile) | `cuda` if you upgrade to a GPU Space |
| `POINTSX_POSE_MODEL_COCO` | Variable | `models/yolo26-pose.pt` | only if you move weights inside the image |
| `POINTSX_POSE_MODEL_CUSTOM` | Variable | `models/pose-cus.pt` | only if you ship the LV-MHP custom model |
| `POINTSX_SEG_MODEL` | Variable | `models/yolo12l-person-seg-extended.pt` | only if you move weights |
| `POINTSX_REGRESSION_MODEL` | Variable | unset → Ramanujan ellipse fallback | set if you ship the trained regressor |
| `POINTSX_WEB_HOST` | Variable | `0.0.0.0` (set by `CMD`) | don't change on HF Spaces |
| `POINTSX_WEB_PORT` | Variable | `7860` (HF Spaces convention) | don't change |
| `POINTSX_TTS_DISABLE` | Variable | unset | `1` to skip server-side TTS calls |
| `POINTSX_TTS_VOICE` | Variable | `uk-UA-PolinaNeural` | any other [edge-tts voice id](https://github.com/rany2/edge-tts#available-voices) |

### Local dev — `.env` file in repo root

Copy `.env.example` to `.env` (gitignored) and fill in whatever you want
to test. `pointsx-web` loads it on startup via `python-dotenv` if the
file exists. Everything is optional for local dev — defaults run a
fully-functional UI without archival.

---

## `S3_ENDPOINT` URL shapes per provider

| Provider | `S3_ENDPOINT` | `S3_REGION` |
|---|---|---|
| Cloudflare R2 | `https://<account_id>.r2.cloudflarestorage.com` | `auto` |
| Backblaze B2 | `https://s3.<region>.backblazeb2.com` | the same region, e.g. `us-west-004` |
| AWS S3 | `https://s3.<region>.amazonaws.com` | `us-east-1`, `eu-west-1`, … |
| MinIO (self-hosted) | `https://minio.example.com` | whatever you set in MinIO |
| Wasabi | `https://s3.<region>.wasabisys.com` | the same region, e.g. `us-east-1` |
| iDrive e2 | `https://<region>.idrivee2-<n>.com` | per their dashboard |
| Scaleway Object Storage | `https://s3.<region>.scw.cloud` | e.g. `fr-par`, `nl-ams` |

Pick **private** bucket access — the demo never returns object URLs to the
client; only your offline analysis reads from it.

---

## Order of operations

A working chain in roughly the order you should set it up:

### 1. Object storage (provider of choice)

1. Create a bucket: `pointsx-demo` (or whatever).
2. Generate a token / API key restricted to that bucket: read + write.
3. Note the four/five values: endpoint URL, access key ID, secret access
   key, bucket name, region.
4. **Smoke-test locally:**
   ```bash
   cp .env.example .env
   # paste the S3 values into .env
   uv sync
   pointsx-web --reload
   # open http://localhost:8000, run a measurement
   # check the bucket — measurements/<uuid>/{front.jpg,side.jpg,envelope.json}
   ```

If you see `Object-storage client ready (bucket=…, endpoint=…)` on startup
and three objects appear after a measurement, the storage layer is wired.

### 2. Backend → Hugging Face Space

1. Create a new Space at https://huggingface.co/new-space:
   - SDK = **Docker**
   - Hardware = **CPU basic (free)**
   - Visibility = **Public** (free CPU requires public)
2. Track the weights via Git LFS:
   ```bash
   git lfs install
   git add models/yolo26-pose.pt \
           models/yolo12l-person-seg-extended.pt \
           models/circumference_regressor.pt
   git commit -m "lfs: ship weights for HF Space"
   ```
3. Push the `demo` branch to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/pointsx-demo
   git push space demo:main
   ```
4. First build takes ~5–7 min (torch wheels + pip install).
5. Once live, paste the env vars from the table above into the Space's
   **Settings → Variables and secrets**. The Space restarts automatically.
6. Smoke-test the Space URL directly:
   `https://<you>-pointsx-demo.hf.space/` should serve the SPA, and
   `POST /api/measure` should respond with an envelope.

### 3. Frontend → Vercel

1. **Import** the same repo into Vercel as a new project.
   - Framework preset: **Other**
   - Build command: `node scripts/build_static.mjs` (already in `vercel.json`)
   - Output directory: `public` (already in `vercel.json`)
2. **Set `POINTSX_API_BASE`** on the project's Environment Variables page.
   Use the HF Space URL exactly as it appears in the browser, no trailing
   slash.
3. Trigger a deploy. The `<vercel>.vercel.app` URL serves only the SPA;
   every API call goes to the Space.
4. Add a custom domain under **Domains** when ready.

### 4. Lock down CORS

While testing, `CORS_ALLOW_ORIGINS=*` on the Space is fine. Once you have
a stable Vercel URL and/or a custom domain, restrict to the exact origins:

```
CORS_ALLOW_ORIGINS=https://your-domain.example,https://your-app.vercel.app
```

Comma-separated, no spaces. Re-save on the Space; it restarts automatically.

---

## How to verify each layer is healthy

| Layer | Endpoint | What to look for |
|---|---|---|
| Storage | (whatever provider) | `measurements/<uuid>/` appears after a request |
| HF Space | `GET https://<space>.hf.space/api/health` (or just root) | UI loads; `/api/measure` returns 200 |
| Vercel | `GET https://<vercel>.vercel.app/` | UI loads, browser network panel shows requests going to `https://<space>.hf.space/api/measure` |
| Domain | `GET https://your-domain.example` | same as Vercel, but via custom domain |

If the browser network panel shows requests going to `/api/measure`
without the Space host, `POINTSX_API_BASE` wasn't applied — redeploy
Vercel after setting it.

If you see CORS errors in the browser console, the Space's
`CORS_ALLOW_ORIGINS` doesn't include the Vercel/domain origin — add it
and let the Space restart.

---

## Free-tier limits to be aware of

| Layer | Limit | Mitigation |
|---|---|---|
| HF Space CPU basic | Sleeps after ~48 h idle, 30 s–2 min cold start | upgrade to CPU Upgrade $9/mo |
| HF Space CPU basic | ~10–20 s per measurement | upgrade to T4 GPU $40/mo (sub-second) |
| Vercel Hobby | 100 GB bandwidth/mo, no commercial use | Pro $20/mo if monetising |
| Cloudflare R2 | 10 GB storage + 1 M class A ops / 10 M class B ops free | $0.015/GB-month after |
| Backblaze B2 | 10 GB storage + 1 GB egress/day free | $0.006/GB-month after |

GPU upgrade is the single largest UX win — sub-second pipeline. Nothing
else needs code changes when you flip the Space hardware; `POINTSX_DEVICE`
auto-detects.
