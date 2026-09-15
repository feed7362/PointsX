/**
 * Hands-free capture: stable-pose 3–2–1 countdown and the fixed-delay photo timer.
 */

import { t } from "../i18n/index.js";
import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import {
  cancelSpeechSynthesis,
  hideCountdownOverlay,
  showCountdownOverlay,
  speakCountdownDigit,
} from "./speech.js";
import { setPoseStatusVisual, setStatus } from "./ui.js";
import { captureFrameToBlob, onCaptureReady } from "./session.js";

export const CAPTURE_TIMER_SECONDS = 10;

export const STABLE_POSE_MS = 700;


/** Clear the stable-OK pose timer (starts fresh on next OK frame). */
export function resetPoseStableHold() {
  captureState.poseStableOkSinceMs = null;
}


/** Stop stable-pose countdown, hide overlay, and cancel speech. */
export function interruptAutoPoseCountdown() {
  if (!captureState.autoPoseCountdownIntervalId) return;
  clearInterval(captureState.autoPoseCountdownIntervalId);
  captureState.autoPoseCountdownIntervalId = 0;
  hideCountdownOverlay();
  cancelSpeechSynthesis();
}


/** Clear auto-capture countdown and stable-pose hold. */
export function resetAutoCaptureUi() {
  interruptAutoPoseCountdown();
  resetPoseStableHold();
}


/** Reset the timed capture button label to the default caption. */
export function resetTimerButtonLabel() {
  const { btnCaptureTimer } = getCaptureDom();
  btnCaptureTimer.textContent = t("sess-photo-timer-btn", { sec: CAPTURE_TIMER_SECONDS });
}


/** Stop the N-second manual timer if running; optionally clear status text. */
export function clearCaptureTimer(setNeutralStatus = false) {
  if (captureState.captureTimerIntervalId) {
    clearInterval(captureState.captureTimerIntervalId);
    captureState.captureTimerIntervalId = 0;
  }
  captureState.captureTimerRemaining = 0;
  resetTimerButtonLabel();
  if (setNeutralStatus) setStatus("");
}


/** Start 3–2–1 overlay and then capture when pose stayed OK for STABLE_POSE_MS. */
export function startAutoPoseCountdown() {
  if (captureState.autoPoseCountdownIntervalId || captureState.captureTimerIntervalId) return;
  if (!captureState.stream || (captureState.frontBlob && captureState.sideBlob)) return;
  cancelSpeechSynthesis();
  resetPoseStableHold();
  let n = 3;
  const stepTick = () => {
    showCountdownOverlay(n);
    setPoseStatusVisual(t("sess-taking-photo-in", { n }), "ok");
    speakCountdownDigit(n);
  };
  stepTick();
  captureState.autoPoseCountdownIntervalId = window.setInterval(() => {
    n -= 1;
    if (n <= 0) {
      clearInterval(captureState.autoPoseCountdownIntervalId);
      captureState.autoPoseCountdownIntervalId = 0;
      hideCountdownOverlay();
      captureFrameToBlob(onCaptureReady);
      return;
    }
    showCountdownOverlay(n);
    setPoseStatusVisual(t("sess-taking-photo-in", { n }), "ok");
    speakCountdownDigit(n);
  }, 1000);
}


/** Toggle or run the fixed-delay (CAPTURE_TIMER_SECONDS) capture timer. */
export function startCaptureTimer() {
  const { btnCapture, btnCaptureTimer } = getCaptureDom();
  if (!captureState.stream) {
    setStatus(t("sess-turn-on-camera-timer"), true);
    return;
  }
  if (captureState.captureTimerIntervalId) {
    clearCaptureTimer(true);
    resetAutoCaptureUi();
    setStatus(t("sess-timer-cancelled"));
    btnCapture.disabled = !captureState.lastPoseGate.ok;
    btnCaptureTimer.disabled = !captureState.lastPoseGate.ok;
    return;
  }
  resetAutoCaptureUi();
  cancelSpeechSynthesis();
  captureState.captureTimerRemaining = CAPTURE_TIMER_SECONDS;
  btnCapture.disabled = true;
  btnCaptureTimer.disabled = false;
  btnCaptureTimer.textContent = t("sess-cancel-timer-btn", { sec: captureState.captureTimerRemaining });
  setStatus(t("sess-timer-countdown-pose", { sec: captureState.captureTimerRemaining }));
  captureState.captureTimerIntervalId = window.setInterval(() => {
    if (!captureState.stream) {
      clearCaptureTimer();
      btnCapture.disabled = true;
      btnCaptureTimer.disabled = true;
      return;
    }
    captureState.captureTimerRemaining -= 1;
    if (captureState.captureTimerRemaining <= 0) {
      clearCaptureTimer();
      setStatus(t("sess-taking-photo"));
      captureFrameToBlob(onCaptureReady);
      return;
    }
    btnCaptureTimer.textContent = t("sess-cancel-timer-btn", { sec: captureState.captureTimerRemaining });
    setStatus(t("sess-timer-countdown", { sec: captureState.captureTimerRemaining }));
  }, 1000);
}
