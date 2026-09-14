"""ASGI entrypoint used by both deployments: ``webui.app:app``.

One app, two deployments:
    HF Space (Docker)  loads the models and runs /api/measure locally.
    Vercel             POINTSX_VERCEL=1 + POINTSX_INFERENCE_ENDPOINT → proxy mode:
                       no models, /api/measure is forwarded to the Space; /api/tts,
                       /api/measure/mock and the static UI are served locally. The
                       Vercel venv has no torch/opencv, so heavy imports stay lazy.

Layout (see docs/app-decomposition-plan.md):
    bootstrap/        create_app, lifespan, routers, middleware, exception handlers
    api/              thin routers: pages, health, measure, tts
    services/         measurement flow, uploads, dataset capture, proxy, mock
    schemas/          MeasurementEnvelope, TtsRequest
    infrastructure/   weight pre-fetch (storage, inference, TTS follow in later steps)
    config.py         Settings and the full list of environment variables
    errors.py         AppError + Ukrainian error texts

Speech hints use ``POST /api/tts`` (edge-tts, needs internet). If ``uv sync`` fails
(for example Torch wheels on some platforms), install TTS separately:
``.venv/bin/python -m pip install edge-tts`` then restart ``pointsx-web``.
"""
from __future__ import annotations

from webui.bootstrap import create_app

app = create_app()
