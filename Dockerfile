# PointsX backend container for Hugging Face Spaces (Docker SDK, free CPU tier).
#
# Architecture (demo branch):
#   user → custom domain → Vercel (static SPA) → this container on HF Spaces
#                                                       ↓
#                                              Cloudflare R2 (photos + envelopes)
#
# Local build/run:
#   docker build -t pointsx-demo .
#   docker run --rm -p 7860:7860 -e POINTSX_DEVICE=cpu pointsx-demo
#
# HF Spaces deployment:
#   1. New Space, SDK = "Docker", hardware = "CPU basic (free)".
#   2. Set the R2_* env vars under Settings → Variables and secrets.
#   3. Push this branch — Spaces auto-builds from this Dockerfile.

FROM python:3.12-slim

# OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
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

# Model weights — LFS-tracked via .gitattributes so they ride along with the
# repo automatically when HF Spaces clones it. Three files total (~66 MB).
COPY --chown=user models /home/user/app/models

EXPOSE 7860

CMD ["pointsx-web", "--host", "0.0.0.0", "--port", "7860"]
