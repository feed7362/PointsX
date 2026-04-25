/**
 * Ukrainian TTS and countdown overlay (uses captureState + DOM).
 */

import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";

const POSE_VOICE_MIN_INTERVAL_MS = 1800;
const POSE_VOICE_REPEAT_MS = 9000;

const COUNTDOWN_DIGIT_UK = { 3: "три", 2: "два", 1: "один" };

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

/** Speak Ukrainian text immediately (countdown digits; bypasses pose-hint throttling). */
export function speakImmediateUk(text) {
  const t = (text || "").trim();
  if (!t || typeof window === "undefined" || !("speechSynthesis" in window)) return;
  ensureUkVoiceList();
  try {
    if (captureState.cachedUkVoice === undefined) captureState.cachedUkVoice = pickPleasantUkVoice();
    const u = new SpeechSynthesisUtterance(t);
    u.lang = "uk-UA";
    if (captureState.cachedUkVoice) {
      u.voice = captureState.cachedUkVoice;
      if (captureState.cachedUkVoice.lang && /^uk/i.test(captureState.cachedUkVoice.lang))
        u.lang = captureState.cachedUkVoice.lang;
    }
    u.rate = 0.88;
    u.pitch = 1.05;
    u.volume = 0.95;
    window.speechSynthesis.speak(u);
  } catch {
  }
}

/** Speak the Ukrainian word for countdown digit 3/2/1. */
export function speakCountdownDigit(n) {
  cancelSpeechSynthesis();
  const w = COUNTDOWN_DIGIT_UK[n];
  if (w) speakImmediateUk(w);
}

/** Speak pose hints with throttling; slightly slower rate for a less robotic sound. */
export function speakPoseHint(msg) {
  const text = (msg || "").trim();
  if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) return;
  if (captureState.captureTimerIntervalId) return;
  ensureUkVoiceList();
  const now = performance.now();
  const isSame = text === captureState.lastSpokenPoseMsg;
  if (!isSame && now - captureState.lastSpokenAtMs < POSE_VOICE_MIN_INTERVAL_MS) return;
  if (isSame && now - captureState.lastSpokenAtMs < POSE_VOICE_REPEAT_MS) return;
  try {
    if (captureState.cachedUkVoice === undefined) captureState.cachedUkVoice = pickPleasantUkVoice();
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "uk-UA";
    if (captureState.cachedUkVoice) {
      u.voice = captureState.cachedUkVoice;
      if (captureState.cachedUkVoice.lang && /^uk/i.test(captureState.cachedUkVoice.lang))
        u.lang = captureState.cachedUkVoice.lang;
    }
    u.rate = 0.9;
    u.pitch = 1.04;
    u.volume = 0.92;
    window.speechSynthesis.speak(u);
    captureState.lastSpokenPoseMsg = text;
    captureState.lastSpokenAtMs = now;
  } catch {
  }
}

export function cancelSpeechSynthesis() {
  try {
    if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel();
  } catch {
  }
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
