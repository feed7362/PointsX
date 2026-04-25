/**
 * MediaPipe pose runtime, camera, capture timers, and frame-to-blob pipeline.
 */

import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { checkPoseForStep } from "./poseGate.js";
import { logPoseDebugCapture } from "./poseDebug.js";
import { syncOverlaySize } from "./overlay.js";
import {
  cancelSpeechSynthesis,
  hideCountdownOverlay,
  showCountdownOverlay,
  speakCountdownDigit,
} from "./speech.js";
import { setStatus, setPoseStatus, setPoseStatusVisual, updateUiStep } from "./ui.js";

export const MIN_VIDEO_DIMENSION = 480;
export const POSE_MIN_INTERVAL_MS = 120;
export const POSE_MP_MIN_DETECTION_CONF = 0.38;
export const POSE_MP_MIN_PRESENCE_CONF = 0.38;
export const POSE_MP_MIN_TRACKING_CONF = 0.38;
export const CAPTURE_TIMER_SECONDS = 10;
export const STABLE_POSE_MS = 700;

const MP_PKG = "https://esm.sh/@mediapipe/tasks-vision@0.10.14";
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

/** Verify camera track exists and resolution meets MIN_VIDEO_DIMENSION. */
export function checkCaptureReadiness() {
  const { video } = getCaptureDom();
  const track = captureState.stream && captureState.stream.getVideoTracks()[0];
  if (!track) {
    return { ok: false, reason: "Камера не активна. Увімкніть камеру." };
  }
  const s = track.getSettings ? track.getSettings() : {};
  const vw = s.width || video.videoWidth;
  const vh = s.height || video.videoHeight;
  if (!vw || !vh) {
    return { ok: false, reason: "Не вдалося прочитати розмір відео. Зачекайте або перезапустіть камеру." };
  }
  if (vw < MIN_VIDEO_DIMENSION || vh < MIN_VIDEO_DIMENSION) {
    return {
      ok: false,
      reason: `Занадто низька роздільна здатність (${vw}×${vh}). Потрібно щонайменше ${MIN_VIDEO_DIMENSION}px по кожній стороні.`,
    };
  }
  return { ok: true };
}

/** Lazy-load MediaPipe PoseLandmarker (GPU with CPU fallback). */
export async function loadPoseLandmarker() {
  if (captureState.poseLandmarker || captureState.poseLoadError) return;
  try {
    const { FilesetResolver, PoseLandmarker } = await import(MP_PKG);
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    try {
      captureState.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    } catch {
      captureState.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    }
  } catch (e) {
    captureState.poseLoadError = e;
    console.error(e);
  }
}

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
  btnCaptureTimer.textContent = `Фото через ${CAPTURE_TIMER_SECONDS} с`;
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

/** Revoke an object URL on a thumbnail img element. */
export function revokeThumbUrl(imgEl) {
  if (!imgEl || !imgEl.src) return;
  if (imgEl.src.startsWith("blob:")) {
    try {
      URL.revokeObjectURL(imgEl.src);
    } catch {
    }
    imgEl.removeAttribute("src");
  }
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
    setPoseStatusVisual(`Знімок через… ${n}`, "ok");
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
    setPoseStatusVisual(`Знімок через… ${n}`, "ok");
    speakCountdownDigit(n);
  }, 1000);
}

/** Throttled pose detection, gate evaluation, capture button state, and auto-countdown trigger. */
export function runPoseIfNeeded() {
  const { video, btnCapture, btnCaptureTimer } = getCaptureDom();
  const now = performance.now();
  if (now - captureState.lastPoseCheck < POSE_MIN_INTERVAL_MS) return;
  captureState.lastPoseCheck = now;

  const resChk = checkCaptureReadiness();
  if (!resChk.ok) {
    resetAutoCaptureUi();
    captureState.lastPoseGate = { ok: false, reason: resChk.reason };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (captureState.poseLoadError) {
    resetAutoCaptureUi();
    captureState.lastPoseGate = {
      ok: false,
      reason: "Не вдалося завантажити MediaPipe. Перевірте мережу та оновіть сторінку.",
    };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (!captureState.poseLandmarker || !captureState.stream || !video.videoWidth) {
    resetAutoCaptureUi();
    captureState.lastPoseGate = { ok: false, reason: "Очікування моделі пози…" };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  const result = captureState.poseLandmarker.detectForVideo(video, Math.floor(now));
  const lm = result.landmarks && result.landmarks[0];
  const worldLm = result.worldLandmarks && result.worldLandmarks[0];
  if (!lm) {
    captureState.lastRawLandmarks = null;
    resetAutoCaptureUi();
    captureState.lastPoseGate = { ok: false, reason: "Людину не видно" };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  captureState.lastRawLandmarks = lm;

  const gate = checkPoseForStep(captureState.step, lm, worldLm);
  captureState.lastPoseGate = gate;

  if (captureState.autoPoseCountdownIntervalId) {
    if (!gate.ok) {
      resetAutoCaptureUi();
      setPoseStatus(gate.reason || "Утримайте позу для зйомки", "bad");
      btnCapture.disabled = true;
      btnCaptureTimer.disabled = false;
      return;
    }
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (gate.ok) {
    setPoseStatus("Поза підходить", "ok");
    btnCapture.disabled = Boolean(captureState.captureTimerIntervalId || captureState.awaitingCaptureBlob);
    btnCaptureTimer.disabled = false;
    if (
      !captureState.captureTimerIntervalId &&
      !(captureState.frontBlob && captureState.sideBlob) &&
      !captureState.awaitingCaptureBlob
    ) {
      if (captureState.poseStableOkSinceMs == null) captureState.poseStableOkSinceMs = now;
      else if (now - captureState.poseStableOkSinceMs >= STABLE_POSE_MS) {
        startAutoPoseCountdown();
        btnCapture.disabled = true;
        btnCaptureTimer.disabled = true;
      }
    } else {
      resetPoseStableHold();
    }
  } else {
    resetAutoCaptureUi();
    setPoseStatus(gate.reason || "Виправте позу.", "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
  }
}

/** RAF loop: pose checks and overlay unless capture is suspended after completion. */
export function loop() {
  if (captureState.suspendPoseLoopAfterComplete) {
    cancelSpeechSynthesis();
    setPoseStatusVisual("", "");
    if (captureState.stream) stopCamera();
    captureState.raf = 0;
    return;
  }
  runPoseIfNeeded();
  syncOverlaySize();
  captureState.raf = requestAnimationFrame(loop);
}

/** Request user media, start pose loop, and reset capture controls. */
export async function startCamera() {
  const { video, btnStop, btnStart, btnCapture, btnCaptureTimer } = getCaptureDom();
  captureState.suspendPoseLoopAfterComplete = false;
  stopCamera();
  setStatus("");
  await loadPoseLandmarker();
  if (captureState.poseLoadError) {
    setStatus("MediaPipe не завантажився: " + (captureState.poseLoadError.message || String(captureState.poseLoadError)), true);
  }
  try {
    captureState.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = captureState.stream;
    await video.play();
    cancelAnimationFrame(captureState.raf);
    loop();
    btnStop.disabled = false;
    btnStart.textContent = "Перезапустити камеру";
    captureState.lastPoseGate = { ok: false, reason: "Аналіз пози…" };
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
  } catch (e) {
    setStatus("Не вдалося отримати доступ до камери: " + (e && e.message ? e.message : String(e)), true);
  }
}

/** Stop tracks, timers, speech, and reset guide smoothing state. */
export function stopCamera() {
  const { video, btnCapture, btnCaptureTimer, btnStop } = getCaptureDom();
  cancelSpeechSynthesis();
  clearCaptureTimer(true);
  resetAutoCaptureUi();
  cancelAnimationFrame(captureState.raf);
  if (captureState.stream) {
    captureState.stream.getTracks().forEach((t) => t.stop());
    captureState.stream = null;
  }
  video.srcObject = null;
  captureState.lastRawLandmarks = null;
  captureState.guideSmoothDelta.dx = 0;
  captureState.guideSmoothDelta.dy = 0;
  captureState.guideSmoothDelta.fitHeight = null;
  captureState.guideSmoothDelta.fitTop = null;
  captureState.guideSmoothDelta.fitLeft = null;
  btnCapture.disabled = true;
  btnCaptureTimer.disabled = true;
  btnStop.disabled = true;
  setPoseStatus("", "");
}

/**
 * Re-verify pose on the exact frame, mirror the canvas like `.preview-wrap.mirror`, then JPEG toBlob.
 */
export function captureFrameToBlob(callback) {
  const { video, captureCanvas } = getCaptureDom();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) {
    setStatus("Відео ще не готове.", true);
    return;
  }
  const chk = checkCaptureReadiness();
  if (!chk.ok) {
    setStatus(chk.reason || "Кадр не підходить.", true);
    return;
  }
  if (!captureState.lastPoseGate.ok) {
    setStatus(captureState.lastPoseGate.reason || "Поза не відповідає вимогам.", true);
    return;
  }
  let captureDebugLm = null;
  let captureDebugWorld = null;
  let captureDebugGate = null;
  if (captureState.poseLandmarker && captureState.stream) {
    const snap = captureState.poseLandmarker.detectForVideo(video, Math.floor(performance.now()));
    const snapLm = snap.landmarks && snap.landmarks[0];
    const snapWorld = snap.worldLandmarks && snap.worldLandmarks[0];
    if (!snapLm) {
      setStatus("На кадрі не видно людину — повторіть знімок.", true);
      return;
    }
    const snapGate = checkPoseForStep(captureState.step, snapLm, snapWorld);
    if (!snapGate.ok) {
      setStatus(snapGate.reason || "Поза на мить знімка не відповідає вимогам.", true);
      return;
    }
    captureDebugLm = snapLm;
    captureDebugWorld = snapWorld;
    captureDebugGate = snapGate;
  }
  setStatus("");
  captureState.awaitingCaptureBlob = true;
  const maxSide = 1024;
  let tw = vw;
  let th = vh;
  if (Math.max(tw, th) > maxSide) {
    const scale = maxSide / Math.max(tw, th);
    tw = Math.round(tw * scale);
    th = Math.round(th * scale);
  }
  captureCanvas.width = tw;
  captureCanvas.height = th;
  const cctx = captureCanvas.getContext("2d");
  if (!cctx) {
    captureState.awaitingCaptureBlob = false;
    setStatus("Не вдалося отримати контекст.", true);
    return;
  }
  cctx.translate(tw, 0);
  cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, tw, th);
  captureCanvas.toBlob(
    (blob) => {
      captureState.awaitingCaptureBlob = false;
      if (!blob) {
        setStatus("Не вдалося створити знімок.", true);
        return;
      }
      if (captureDebugLm && captureDebugGate) {
        logPoseDebugCapture(captureState.step, captureDebugLm, captureDebugWorld, captureDebugGate);
      }
      callback(blob);
    },
    "image/jpeg",
    0.92
  );
}

/** After a blob is ready: update thumbs, advance step, and stop or restart the camera as needed. */
export function onCaptureReady(blob) {
  const { thumbFront, thumbSide } = getCaptureDom();
  resetAutoCaptureUi();
  const url = URL.createObjectURL(blob);
  if (captureState.step === 1) {
    captureState.suspendPoseLoopAfterComplete = false;
    revokeThumbUrl(thumbFront);
    captureState.frontBlob = blob;
    thumbFront.src = url;
    thumbFront.hidden = false;
    if (captureState.sideBlob) {
      captureState.step = 2;
      captureState.suspendPoseLoopAfterComplete = true;
      stopCamera();
      updateUiStep();
      setStatus("Анфас оновлено. Можна «Розрахувати мірки» або перезняти кадр.");
    } else {
      captureState.step = 2;
      updateUiStep();
      void startCamera();
      setStatus("Увімкніть камеру знову для знімка в профіль.");
    }
  } else {
    revokeThumbUrl(thumbSide);
    captureState.sideBlob = blob;
    thumbSide.src = url;
    thumbSide.hidden = false;
    captureState.suspendPoseLoopAfterComplete = true;
    stopCamera();
    updateUiStep();
    setStatus("Обидва знімки готові. Натисніть «Розрахувати мірки».");
  }
}

/** Toggle or run the fixed-delay (CAPTURE_TIMER_SECONDS) capture timer. */
export function startCaptureTimer() {
  const { btnCapture, btnCaptureTimer } = getCaptureDom();
  if (!captureState.stream) {
    setStatus("Увімкніть камеру перед запуском таймера.", true);
    return;
  }
  if (captureState.captureTimerIntervalId) {
    clearCaptureTimer(true);
    resetAutoCaptureUi();
    setStatus("Таймер скасовано.");
    btnCapture.disabled = !captureState.lastPoseGate.ok;
    btnCaptureTimer.disabled = !captureState.lastPoseGate.ok;
    return;
  }
  resetAutoCaptureUi();
  cancelSpeechSynthesis();
  captureState.captureTimerRemaining = CAPTURE_TIMER_SECONDS;
  btnCapture.disabled = true;
  btnCaptureTimer.disabled = false;
  btnCaptureTimer.textContent = `Скасувати (${captureState.captureTimerRemaining} с)`;
  setStatus(`Автозйомка через ${captureState.captureTimerRemaining} с… Станьте в правильну позу.`);
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
      setStatus("Знімаю фото…");
      captureFrameToBlob(onCaptureReady);
      return;
    }
    btnCaptureTimer.textContent = `Скасувати (${captureState.captureTimerRemaining} с)`;
    setStatus(`Автозйомка через ${captureState.captureTimerRemaining} с…`);
  }, 1000);
}

/** Clear front thumbnail and restart capture from step 1. */
export function retakeFrontPhoto() {
  const { thumbFront } = getCaptureDom();
  if (!captureState.frontBlob) return;
  captureState.suspendPoseLoopAfterComplete = false;
  revokeThumbUrl(thumbFront);
  thumbFront.hidden = true;
  captureState.frontBlob = null;
  captureState.step = 1;
  resetAutoCaptureUi();
  updateUiStep();
  setStatus("Перезйомка анфасу: увімкніть камеру та встаньте в позу.");
  void startCamera();
}

/** Clear side thumbnail and restart from step 2 (or 1 if front missing). */
export function retakeSidePhoto() {
  const { thumbSide } = getCaptureDom();
  if (!captureState.sideBlob) return;
  captureState.suspendPoseLoopAfterComplete = false;
  revokeThumbUrl(thumbSide);
  thumbSide.hidden = true;
  captureState.sideBlob = null;
  captureState.step = captureState.frontBlob ? 2 : 1;
  resetAutoCaptureUi();
  updateUiStep();
  setStatus("Перезйомка профілю: увімкніть камеру та встаньте в позу.");
  void startCamera();
}
