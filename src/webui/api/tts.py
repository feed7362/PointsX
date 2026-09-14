"""POST /api/tts — short spoken hints via edge-tts (browser speech is the fallback)."""
from __future__ import annotations

import logging

from fastapi import APIRouter
from fastapi.responses import Response

from webui.errors import AppError
from webui.schemas import TtsRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/tts")
async def tts_synthesize(body: TtsRequest) -> Response:
    """Synthesize speech (MP3) using a lightweight neural Edge voice."""
    from webui import tts as tts_mod

    if tts_mod.tts_disabled():
        raise AppError(503, "Синтез мовлення вимкнено на сервері.")

    try:
        mp3 = await tts_mod.synthesize_uk_speech_mp3(body.text)
    except ValueError as exc:
        raise AppError(400, str(exc)) from exc
    except ImportError as exc:
        logger.warning("TTS unavailable — install edge-tts in the server environment: %s", exc)
        raise AppError(
            503,
            "Пакет edge-tts не встановлено в середовищі сервера. "
            "Встановіть: `.venv/bin/python -m pip install edge-tts` і перезапустіть pointsx-web. "
            "Підказки спробують голос браузера.",
        ) from exc
    except Exception as exc:
        logger.warning("TTS synthesis failed: %s", exc)
        raise AppError(503, "Не вдалося синтезувати мовлення. Перевірте доступ до інтернету.") from exc

    return Response(content=mp3, media_type="audio/mpeg")
