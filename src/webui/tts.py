"""Ukrainian neural TTS via edge-tts (Microsoft Edge voices, no local GPU model).

Requires outbound HTTPS. Configure with env:
    POINTSX_TTS_VOICE   default ``uk-UA-PolinaNeural`` (alt: ``uk-UA-OstapNeural``)
    POINTSX_TTS_DISABLE  if ``1`` / ``true`` / ``yes``, ``/api/tts`` returns 503.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict

logger = logging.getLogger(__name__)

_DEFAULT_VOICE = "uk-UA-PolinaNeural"
_TTS_CACHE: OrderedDict[str, bytes] = OrderedDict()
_TTS_CACHE_MAX = 64


def tts_voice() -> str:
    return os.environ.get("POINTSX_TTS_VOICE", _DEFAULT_VOICE).strip() or _DEFAULT_VOICE


def tts_disabled() -> bool:
    raw = os.environ.get("POINTSX_TTS_DISABLE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


async def synthesize_uk_speech_mp3(text: str) -> bytes:
    """Return MP3 bytes for ``text`` using the configured Ukrainian neural voice."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty text")

    import edge_tts

    voice = tts_voice()
    key = f"{voice}\n{stripped}"
    if key in _TTS_CACHE:
        _TTS_CACHE.move_to_end(key)
        return _TTS_CACHE[key]

    communicate = edge_tts.Communicate(stripped, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("edge-tts returned no audio data")

    _TTS_CACHE[key] = data
    _TTS_CACHE.move_to_end(key)
    while len(_TTS_CACHE) > _TTS_CACHE_MAX:
        _TTS_CACHE.popitem(last=False)

    return data
