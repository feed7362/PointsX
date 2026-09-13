# PointsX backend container for Hugging Face Spaces (Docker SDK, free CPU tier).
#
# Architecture (demo branch, two-repo split):
#   browser → custom domain → Vercel (static SPA, separate repo)
#                                       ↓
#                            this container on HF Spaces (backend repo)
#                                       ↓
#                            S3-compatible bucket
#                              ├ models/        ← weights pulled at boot
#                              └ measurements/  ← photos + envelopes per request
#
# Models are NOT baked into the image — they're downloaded from the same
# bucket used for archival, under MODELS_S3_KEY_PREFIX (default "models/").
# This keeps the image small (~1.5 GB instead of ~1.6 GB) and lets you swap
# weights without rebuilding.
#
# Local build/run:
#   docker build -t pointsx-demo .
#   docker run --rm -p 7860:7860 --env-file .env pointsx-demo
#
# HF Spaces deployment:
#   1. New Space, SDK = "Docker", hardware = "CPU basic (free)".
#   2. Upload weights to s3://<bucket>/<prefix>/models/{yolo26-pose.pt,...}
#      (once per bucket; reuse across many Spaces).
#   3. Set ELK_* / S3_* + CORS_ALLOW_ORIGINS under Settings → Variables and secrets.
#   4. Push the demo branch — Spaces auto-builds from this Dockerfile.

FROM python:3.12-slim

# OpenCV runtime libs + fonts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        # Cyrillic glyphs for the measurement-overlay labels. python:3.12-slim
        # ships with NO fonts, so PIL fell back to its bitmap default and every
        # Ukrainian label rendered as tofu squares. Installs to
        # /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf — already listed in
        # visualize.py's _FONT_CANDIDATES.
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}" \
    HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    POINTSX_DEVICE=cpu \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics

WORKDIR /home/user/app

# Deps first (build-cache stays warm across source edits).
COPY --chown=user pyproject.toml uv.lock README.md /home/user/app/
COPY --chown=user src /home/user/app/src

# CPU-only torch wheels (saves ~1.5 GB vs the default cuda wheels).
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        torch torchvision \
    && pip install --no-cache-dir --user .

# Pre-create the models/ directory with the right ownership so the boot
# script can write the downloaded weights into it without permission errors.
RUN mkdir -p /home/user/app/models

EXPOSE 7860

CMD ["pointsx-web", "--host", "0.0.0.0", "--port", "7860"]
