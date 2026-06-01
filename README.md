---
title: PointsX (demo)
emoji: 📏
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Body measurements from front + side photos (scientific demo)
---

# PointsX — demo branch

Scientific-demo deployment of PointsX (body measurements from two photos).
Master branch holds the algorithmic work; **this branch wires up hosting**:

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

All four layers can run on free tiers:

- **DNS**: any registrar; Cloudflare DNS is free and recommended.
- **Frontend**: Vercel Hobby — free, includes auto-SSL and CDN.
- **Backend**: Hugging Face Space, CPU basic — free, sleeps after 48 h idle.
- **Storage**: Cloudflare R2 (10 GB free), Backblaze B2 (10 GB free), or
  any other S3-compatible provider.

---

## Run locally (no Vercel, no R2)

```bash
uv sync
pointsx-web --reload
# http://127.0.0.1:8000/
```

The webui works fully — measurement, viz, results panel — without any
external service. Archival is silently disabled until S3 env vars are set.

## Run the deployment container locally

```bash
docker build -t pointsx-demo .
docker run --rm -p 7860:7860 \
    -e POINTSX_DEVICE=cpu \
    -e CORS_ALLOW_ORIGINS=* \
    pointsx-demo
# http://localhost:7860/
```

Add the storage vars (see `.env.example`) to test archival:

```bash
docker run --rm -p 7860:7860 --env-file .env pointsx-demo
```

---

## Deploy

### 1. Backend → Hugging Face Space

1. Create a Space at https://huggingface.co/new-space
   - **SDK**: Docker
   - **Hardware**: CPU basic (free)
   - **Visibility**: Public (free CPU requires public)
2. Track the model weights via Git LFS so HF Spaces clones them on build:
   ```bash
   git lfs install
   git lfs track "models/*.pt"           # already in .gitattributes
   git add models/yolo26-pose.pt models/yolo12l-person-seg-extended.pt \
           models/circumference_regressor.pt
   git commit -m "lfs: ship YOLO + regressor weights"
   ```
3. Push the demo branch to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<you>/pointsx-demo
   git push space demo:main
   ```
4. Under **Settings → Variables and secrets**, paste the storage env vars
   from `.env.example` (`S3_ENDPOINT`, `S3_ACCESS_KEY_ID`,
   `S3_SECRET_ACCESS_KEY`, `S3_BUCKET`, optional `S3_REGION`/`S3_PREFIX`).
5. First build takes ~5–7 min. Once live the Space URL serves both `/`
   (the UI for dev access) and `/api/measure` (the JSON API).

### 2. Frontend → Vercel

1. Import this repo into Vercel as a new project (Framework: **Other**).
2. Vercel reads `vercel.json` and runs `scripts/build_static.mjs` at deploy
   time. The script copies `src/webui/static/` into `public/` and rewrites
   `static/config.js` with the runtime API base.
3. Add the env var on the project → **Settings → Environment Variables**:
   ```
   POINTSX_API_BASE=https://<you>-pointsx-demo.hf.space
   ```
4. Trigger a deploy. The Vercel URL serves only the SPA; every API call
   goes to the Space.
5. Add a custom domain under **Domains** when ready.

### 3. Object storage (whichever S3-compatible provider you picked)

1. Create a bucket, e.g. `pointsx-demo`.
2. Generate a token / API key restricted to that bucket (read + write).
3. Put the endpoint URL, key id, secret, bucket name (and region if the
   provider needs one) into the HF Space's secrets panel.
4. CORS on the bucket is **not** required — only the backend writes to it.

---

## What ends up in storage

Per `/api/measure` call, when archival is configured:

```
<prefix>measurements/<uuid4>/front.jpg        (original raw upload)
<prefix>measurements/<uuid4>/side.jpg
<prefix>measurements/<uuid4>/envelope.json    (the full envelope returned to client)
```

Each object carries S3 metadata: `height_cm`, `sex`, `pose_backend`,
`created_at`. The `uuid4` is also the `request_id` in the envelope, so
client-side bug reports map 1-to-1 to stored objects.

**The bucket should be private.** The demo never returns object URLs to
the client; the only consumer is whoever runs the data analysis later.

---

## Limitations of the free stack

- Inference on a 2 vCPU container takes ~10–20 s per pair of photos.
- The Space sleeps after ~48 h of inactivity; the first request after a
  sleep takes 30 s–2 min to wake the container.
- Vercel Hobby forbids commercial use. Demo / portfolio is fine.
- HF Spaces does not allow custom domains on the free tier — Vercel is
  what your users hit; the Space URL stays behind the scenes.

Want sub-second inference? Upgrade the Space to T4 GPU (~$40/mo).
No code changes needed; set `POINTSX_DEVICE=cuda` and rebuild.
