/**
 * POST /api/tts — neural voice MP3 for pose hints and countdown, with a small
 * in-memory cache and a per-page kill switch (callers fall back to the browser voice).
 */

import { apiFetch } from "./client.js";

const audioBlobCache = new Map();
const AUDIO_CACHE_MAX = 48;

/** Session-level kill switch — once the server returns 503 (or fetch fails at the
 * network level) we stop calling /api/tts for the rest of the page lifetime and go
 * straight to the browser fallback. Timeouts (AbortError) do NOT trip it: the warmup
 * request can time out on a Vercel cold start while the endpoint is healthy. */
let serverTtsDisabledForSession = false;

const TTS_FETCH_TIMEOUT_MS = 8000;

function trimAudioBlobCache() {
  while (audioBlobCache.size > AUDIO_CACHE_MAX) {
    const k = audioBlobCache.keys().next().value;
    audioBlobCache.delete(k);
  }
}

/**
 * MP3 for `text` from the server (cached). The timeout covers the whole response, body included.
 * @param {string} text
 * @returns {Promise<Blob>}
 */
export async function fetchTtsMp3(text) {
  if (serverTtsDisabledForSession) throw new Error("tts:disabled-this-session");
  const cached = audioBlobCache.get(text);
  if (cached) return cached;
  const ac = new AbortController();
  const to = window.setTimeout(() => ac.abort(), TTS_FETCH_TIMEOUT_MS);
  try {
    const res = await apiFetch("/api/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "audio/mpeg" },
      body: JSON.stringify({ text }),
      signal: ac.signal,
    });
    if (!res.ok) {
      // 503 = backend has POINTSX_TTS_DISABLE=1 or edge-tts is blocked. Stop trying.
      if (res.status === 503) serverTtsDisabledForSession = true;
      throw new Error(`tts:${res.status}`);
    }
    const blob = await res.blob();
    if (blob.size < 32) throw new Error("tts:empty");
    const head = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
    const mp3 = head[0] === 0xff && (head[1] & 0xe0) === 0xe0;
    const id3 = head[0] === 0x49 && head[1] === 0x44 && head[2] === 0x33;
    if (!mp3 && !id3) throw new Error("tts:not-audio");
    audioBlobCache.set(text, blob);
    trimAudioBlobCache();
    return blob;
  } catch (err) {
    // Network-level failure (CORS, DNS, blocked host) — disable for the session.
    if (err?.name === "TypeError") serverTtsDisabledForSession = true;
    throw err;
  } finally {
    window.clearTimeout(to);
  }
}
