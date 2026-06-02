"""Ukrainian neural TTS for /api/tts.

Two backends, tried in order — first non-empty result wins:

  1. ``edge-tts`` (Microsoft Edge neural voices via WSS).
     Best quality, free, no API key. Blocked from many datacenter IPs
     (HF Spaces, AWS Lambda, etc.) — Microsoft started filtering
     non-residential ranges in 2024. Works fine on a laptop.

  2. ``gTTS`` (Google Translate TTS via HTTP).
     Lower quality (Translate's voice, not a true neural model), but
     does not block datacenter IPs. Reliable fallback for HF Spaces.

Configure with env vars:
    POINTSX_TTS_VOICE       edge-tts voice id (default: uk-UA-PolinaNeural)
    POINTSX_TTS_DISABLE     "1"/"true"/"yes" → /api/tts returns 503 immediately
    POINTSX_TTS_BACKEND     force one backend: "edge" or "gtts" (default: auto)
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


def _forced_backend() -> str | None:
    raw = (os.environ.get("POINTSX_TTS_BACKEND") or "").strip().lower()
    return raw if raw in ("edge", "gtts") else None


async def _synth_edge_tts(text: str) -> bytes:
    """edge-tts WSS path. Raises if Microsoft drops the connection."""
    import edge_tts

    communicate = edge_tts.Communicate(text, tts_voice())
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio" and chunk.get("data"):
            chunks.append(chunk["data"])
    data = b"".join(chunks)
    if not data:
        raise RuntimeError("edge-tts returned no audio data")
    return data


async def _synth_gtts(text: str) -> bytes:
    """gTTS HTTP path (synchronous lib; offload to a thread)."""
    import asyncio
    import io

    def _run() -> bytes:
        from gtts import gTTS

        buf = io.BytesIO()
        gTTS(text=text, lang="uk", slow=False).write_to_fp(buf)
        data = buf.getvalue()
        if not data:
            raise RuntimeError("gTTS returned no audio data")
        return data

    return await asyncio.to_thread(_run)


async def synthesize_uk_speech_mp3(text: str) -> bytes:
    """Return MP3 bytes for `text` via the first backend that responds.

    Caches successful results by `(backend, voice, text)` for the process
    lifetime — the prompt set is finite, so the cache fills quickly and
    every subsequent call is a memcpy.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("empty text")

    forced = _forced_backend()
    voice = tts_voice()
    key = f"{forced or 'auto'}\n{voice}\n{stripped}"
    if key in _TTS_CACHE:
        _TTS_CACHE.move_to_end(key)
        return _TTS_CACHE[key]

    last_err: Exception | None = None
    order = (
        [forced] if forced else
        ["edge", "gtts"]
    )

    for backend in order:
        try:
            if backend == "edge":
                data = await _synth_edge_tts(stripped)
            elif backend == "gtts":
                data = await _synth_gtts(stripped)
            else:
                continue
            logger.info("TTS synthesis OK — backend=%s text=%r bytes=%d", backend, stripped[:40], len(data))
            _TTS_CACHE[key] = data
            _TTS_CACHE.move_to_end(key)
            while len(_TTS_CACHE) > _TTS_CACHE_MAX:
                _TTS_CACHE.popitem(last=False)
            return data
        except ImportError as exc:
            logger.warning("TTS backend %s missing — `pip install` to enable: %s", backend, exc)
            last_err = exc
        except Exception as exc:  # noqa: BLE001
            logger.warning("TTS backend %s failed (%s) — falling through.", backend, exc)
            last_err = exc

    raise RuntimeError(
        f"All TTS backends failed. Last error: {last_err}. "
        "Tried: " + ", ".join(order)
    )
