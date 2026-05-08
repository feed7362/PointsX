/**
 * Ukrainian TTS for pose hints and countdown:
 * Prefer server neural voice (edge-tts via ``/api/tts``, MP3 playback), fallback to SpeechSynthesis.
 */

import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";

const POSE_VOICE_MIN_INTERVAL_MS = 1800;
const POSE_VOICE_REPEAT_MS = 9000;

const COUNTDOWN_DIGIT_UK = { 3: "три", 2: "два", 1: "один" };

/** Bumped on cancel to drop stale async /api/tts completions. */
let speakGeneration = 0;

/** @type {HTMLAudioElement | null} */
let activeAudio = null;

const audioBlobCache = new Map();
const AUDIO_CACHE_MAX = 48;

/** Same-origin TTS endpoint (avoids bad resolution from import maps / subpaths). */
function ttsEndpointUrl() {
  if (typeof window === "undefined" || !window.location?.origin) return "/api/tts";
  return `${window.location.origin}/api/tts`;
}

const TTS_FETCH_TIMEOUT_MS = 8000;

function trimAudioBlobCache() {
  while (audioBlobCache.size > AUDIO_CACHE_MAX) {
    const k = audioBlobCache.keys().next().value;
    audioBlobCache.delete(k);
  }
}

function stopActiveAudio() {
  if (!activeAudio) return;
  try {
    activeAudio.pause();
    activeAudio.removeAttribute("src");
    activeAudio.load();
  } catch {
    /* ignore */
  }
  activeAudio = null;
}

async function fetchTtsMp3Blob(text) {
  const cached = audioBlobCache.get(text);
  if (cached) return cached;
  const ac = new AbortController();
  const to = window.setTimeout(() => ac.abort(), TTS_FETCH_TIMEOUT_MS);
  try {
    const res = await fetch(ttsEndpointUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "audio/mpeg" },
      body: JSON.stringify({ text }),
      signal: ac.signal,
    });
    if (!res.ok) throw new Error(`tts:${res.status}`);
    const blob = await res.blob();
    if (blob.size < 32) throw new Error("tts:empty");
    const head = new Uint8Array(await blob.slice(0, 4).arrayBuffer());
    const mp3 = head[0] === 0xff && (head[1] & 0xe0) === 0xe0;
    const id3 = head[0] === 0x49 && head[1] === 0x44 && head[2] === 0x33;
    if (!mp3 && !id3) throw new Error("tts:not-audio");
    audioBlobCache.set(text, blob);
    trimAudioBlobCache();
    return blob;
  } finally {
    window.clearTimeout(to);
  }
}

/**
 * Chrome / Safari often load speech voices only after a user gesture; call once after camera starts.
 */
export function primeVoiceAfterUserGesture() {
  if (typeof window === "undefined" || !window.speechSynthesis) return;
  ensureUkVoiceList();
  try {
    window.speechSynthesis.getVoices();
  } catch {
    /* ignore */
  }
}

/**
 * Play MP3 blob; returns when playback ends or rejects on error / play() denial.
 * @param {Blob} blob
 * @param {number} generation
 */
function playMp3Blob(blob, generation, volume = 0.95) {
  return new Promise((resolve, reject) => {
    if (generation !== speakGeneration) {
      resolve();
      return;
    }
    stopActiveAudio();
    const url = URL.createObjectURL(blob);
    const a = new Audio();
    activeAudio = a;
    const cleanup = () => {
      URL.revokeObjectURL(url);
      if (activeAudio === a) activeAudio = null;
    };
    a.addEventListener(
      "ended",
      () => {
        cleanup();
        resolve();
      },
      { once: true },
    );
    a.addEventListener(
      "error",
      () => {
        cleanup();
        reject(new Error("audio playback error"));
      },
      { once: true },
    );
    a.src = url;
    a.volume = volume;
    a.play().catch((e) => {
      cleanup();
      reject(e);
    });
  });
}

function pickPleasantUkVoice() {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  const uk = voices.filter((v) => v.lang && /^uk\b/i.test(v.lang.trim()));
  if (!uk.length) return null;
  const lowQuality = /compact|croak|bad\s+news|pipes|whisper|zira\b/i;
  const nicer = /lesya|леся|oksana|оксан|kateryna|катерин|premium|natural|enhanced|neural|google\s+uk|microsoft.*ukr/i;
  const ranked = [...uk].sort((a, b) => {
    let sa = 0;
    let sb = 0;
    if (nicer.test(a.name)) sa += 8;
    if (nicer.test(b.name)) sb += 8;
    if (lowQuality.test(a.name)) sa -= 6;
    if (lowQuality.test(b.name)) sb -= 6;
    if (a.default && !b.default) sa += 2;
    if (!a.default && b.default) sb += 2;
    if (a.localService === false) sa += 1;
    if (b.localService === false) sb += 1;
    return sb - sa;
  });
  return ranked[0] || uk[0];
}

export function ensureUkVoiceList() {
  if (typeof window === "undefined" || !("speechSynthesis" in window) || captureState.voicesChangeHooked) return;
  captureState.voicesChangeHooked = true;
  const refresh = () => {
    captureState.cachedUkVoice = pickPleasantUkVoice();
  };
  window.speechSynthesis.addEventListener("voiceschanged", refresh);
  refresh();
}

function speakWithBrowserUtterance(text, { rate = 0.9, pitch = 1.04, volume = 0.92 } = {}) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  ensureUkVoiceList();
  try {
    if (captureState.cachedUkVoice === undefined) captureState.cachedUkVoice = pickPleasantUkVoice();
    window.speechSynthesis.getVoices();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "uk-UA";
    if (captureState.cachedUkVoice) {
      u.voice = captureState.cachedUkVoice;
      if (captureState.cachedUkVoice.lang && /^uk/i.test(captureState.cachedUkVoice.lang))
        u.lang = captureState.cachedUkVoice.lang;
    }
    u.rate = rate;
    u.pitch = pitch;
    u.volume = volume;
    window.speechSynthesis.speak(u);
  } catch {
    /* ignore */
  }
}

async function speakServerPreferred(text, generation, browserOpts) {
  try {
    const blob = await fetchTtsMp3Blob(text);
    if (generation !== speakGeneration) return;
    await playMp3Blob(blob, generation, browserOpts?.volume ?? 0.95);
  } catch {
    if (generation !== speakGeneration) return;
    speakWithBrowserUtterance(text, browserOpts);
  }
}

export function cancelSpeechSynthesis() {
  speakGeneration += 1;
  stopActiveAudio();
  try {
    if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel();
  } catch {
    /* ignore */
  }
}

/** Speak Ukrainian text immediately (countdown digits; bypasses pose-hint throttling). */
export function speakImmediateUk(text) {
  const t = (text || "").trim();
  if (!t || typeof window === "undefined") return;
  const gen = speakGeneration;
  void speakServerPreferred(t, gen, { rate: 0.88, pitch: 1.05, volume: 0.95 });
}

/** Speak the Ukrainian word for countdown digit 3/2/1. */
export function speakCountdownDigit(n) {
  cancelSpeechSynthesis();
  const w = COUNTDOWN_DIGIT_UK[n];
  if (w) speakImmediateUk(w);
}

/** Speak pose hints with throttling; neural TTS when server available. */
export function speakPoseHint(msg) {
  const text = (msg || "").trim();
  if (!text || typeof window === "undefined") return;
  if (captureState.captureTimerIntervalId) return;
  const now = performance.now();
  const isSame = text === captureState.lastSpokenPoseMsg;
  if (!isSame && now - captureState.lastSpokenAtMs < POSE_VOICE_MIN_INTERVAL_MS) return;
  if (isSame && now - captureState.lastSpokenAtMs < POSE_VOICE_REPEAT_MS) return;
  captureState.lastSpokenPoseMsg = text;
  captureState.lastSpokenAtMs = now;

  cancelSpeechSynthesis();
  const gen = speakGeneration;
  void speakServerPreferred(text, gen, { rate: 0.9, pitch: 1.04, volume: 0.92 });
}

export function showCountdownOverlay(n) {
  const { countdownOverlay } = getCaptureDom();
  if (!countdownOverlay) return;
  countdownOverlay.textContent = String(n);
  countdownOverlay.hidden = false;
}

export function hideCountdownOverlay() {
  const { countdownOverlay } = getCaptureDom();
  if (!countdownOverlay) return;
  countdownOverlay.hidden = true;
  countdownOverlay.textContent = "";
}
