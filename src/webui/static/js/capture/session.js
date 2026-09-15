/**
 * MediaPipe pose runtime, camera, capture timers, and frame-to-blob pipeline.
 *
 * Facade: implementation is split into the modules re-exported below; existing
 * `import * as ...` users keep the same names.
 */

export {
  CAPTURE_TIMER_SECONDS,
  STABLE_POSE_MS,
  clearCaptureTimer,
  interruptAutoPoseCountdown,
  resetAutoCaptureUi,
  resetPoseStableHold,
  resetTimerButtonLabel,
  startAutoPoseCountdown,
  startCaptureTimer,
} from "./autoCapture.js";

export {
  POSE_MP_MIN_DETECTION_CONF,
  POSE_MP_MIN_PRESENCE_CONF,
  POSE_MP_MIN_TRACKING_CONF,
  loadPoseLandmarker,
  loadPoseLandmarkerImage,
} from "./poseModel.js";

export {
  applyPrefillCapture,
  applyUploadedImage,
  revokeThumbUrl,
} from "./uploads.js";

import { t } from "../i18n/index.js";
import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { checkPoseForStep } from "./poseGate.js";
import { logPoseDebugCapture, logPoseGateCheck } from "./poseDebug.js";
import { syncOverlaySize } from "./overlay.js";
import { cancelSpeechSynthesis, primeVoiceAfterUserGesture } from "./speech.js";
import {
  setPoseStatus,
  setPoseStatusVisual,
  setStatus,
  updateUiStep,
} from "./ui.js";
import { syncThumbSlotToImage } from "./thumbLayout.js";
import { cameraErrorMessage, cameraUnavailableReason, requestCameraStream } from "./camera.js";
import { loadPoseLandmarker, nextPoseVideoTimestampMs } from "./poseModel.js";
import { imageBitmapToSquare } from "./imageOps.js";
import { revokeThumbUrl } from "./uploads.js";
import {
  STABLE_POSE_MS,
  clearCaptureTimer,
  resetAutoCaptureUi,
  resetPoseStableHold,
  startAutoPoseCountdown,
} from "./autoCapture.js";

/** Camera off: idle art inside preview; camera on: show live feed. */
function syncPreviewIdleState(cameraLive) {
  const { previewWrap, previewIdle } = getCaptureDom();
  previewWrap?.classList.toggle("preview-wrap--idle", !cameraLive);
  previewIdle?.setAttribute("aria-hidden", cameraLive ? "true" : "false");
}


export const MIN_VIDEO_DIMENSION = 480;

export const POSE_MIN_INTERVAL_MS = 120;


/** Verify camera track exists and resolution meets MIN_VIDEO_DIMENSION. */
export function checkCaptureReadiness() {
  const { video } = getCaptureDom();
  const track = captureState.stream && captureState.stream.getVideoTracks()[0];
  if (!track) {
    return { ok: false, reason: t("sess-camera-inactive") };
  }
  const s = track.getSettings ? track.getSettings() : {};
  const vw = s.width || video.videoWidth;
  const vh = s.height || video.videoHeight;
  if (!vw || !vh) {
    return { ok: false, reason: t("sess-video-size-failed") };
  }
  if (vw < MIN_VIDEO_DIMENSION || vh < MIN_VIDEO_DIMENSION) {
    return {
      ok: false,
      reason: t("sess-low-resolution", { vw, vh, min: MIN_VIDEO_DIMENSION }),
    };
  }
  return { ok: true };
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
      reason: t("sess-mediapipe-failed"),
    };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (!captureState.poseLandmarker || !captureState.stream || !video.videoWidth) {
    resetAutoCaptureUi();
    captureState.lastPoseGate = { ok: false, reason: t("sess-waiting-pose-model") };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  const result = captureState.poseLandmarker.detectForVideo(video, nextPoseVideoTimestampMs());
  const lm = result.landmarks && result.landmarks[0];
  const worldLm = result.worldLandmarks && result.worldLandmarks[0];
  if (!lm) {
    captureState.lastRawLandmarks = null;
    resetAutoCaptureUi();
    captureState.lastPoseGate = { ok: false, reason: t("sess-person-not-seen") };
    setPoseStatus(captureState.lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  captureState.lastRawLandmarks = lm;

  const gate = checkPoseForStep(captureState.step, lm, worldLm);
  captureState.lastPoseGate = gate;
  logPoseGateCheck("video", captureState.step, lm, worldLm, gate);

  if (captureState.autoPoseCountdownIntervalId) {
    if (!gate.ok) {
      resetAutoCaptureUi();
      setPoseStatus(gate.reason || t("sess-hold-pose"), "bad");
      btnCapture.disabled = true;
      btnCaptureTimer.disabled = false;
      return;
    }
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (gate.ok) {
    setPoseStatus(t("sess-pose-ok"), "ok");
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
    setPoseStatus(gate.reason || t("sess-adjust-pose"), "bad");
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
  const cameraBlocked = cameraUnavailableReason();
  if (cameraBlocked) {
    setStatus(cameraBlocked, true);
    syncPreviewIdleState(false);
    return;
  }

  await loadPoseLandmarker();
  if (captureState.poseLoadError) {
    setStatus(t("sess-mediapipe-load-failed-msg", { msg: captureState.poseLoadError.message || String(captureState.poseLoadError) }), true);
  }
  try {
    captureState.stream = await requestCameraStream();
    let mirrored = true;
    try {
      const track = captureState.stream.getVideoTracks()[0];
      if (track) {
        const settings = track.getSettings ? track.getSettings() : {};
        const facingMode = settings.facingMode || "";
        const label = (track.label || "").toLowerCase();
        if (facingMode === "environment" || label.includes("back") || label.includes("rear") || label.includes("environment")) {
          mirrored = false;
        }
      }
    } catch (e) {
      console.warn("Failed to inspect video track settings for mirroring:", e);
    }
    captureState.isMirrored = mirrored;
    const previewWrap = document.getElementById("preview-wrap");
    if (previewWrap) {
      previewWrap.classList.toggle("mirror", mirrored);
    }
    video.setAttribute("playsinline", "");
    video.setAttribute("webkit-playsinline", "true");
    video.muted = true;
    video.srcObject = captureState.stream;
    await video.play();
    void primeVoiceAfterUserGesture();
    cancelAnimationFrame(captureState.raf);
    loop();
    btnStop.disabled = false;
    btnStart.textContent = t("sess-restart-camera-btn");
    captureState.lastPoseGate = { ok: false, reason: t("sess-analyzing-pose") };
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    syncPreviewIdleState(true);
  } catch (e) {
    setStatus(t("sess-camera-access-failed", { msg: cameraErrorMessage(e) }), true);
    syncPreviewIdleState(false);
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
  syncPreviewIdleState(false);
}


/**
 * Re-verify pose on the exact frame, mirror the canvas like `.preview-wrap.mirror`,
 * scale down if needed, crop/pad to a square, then JPEG toBlob.
 */
export function captureFrameToBlob(callback) {
  const { video, captureCanvas } = getCaptureDom();
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (!vw || !vh) {
    setStatus(t("sess-video-not-ready"), true);
    return;
  }
  const chk = checkCaptureReadiness();
  if (!chk.ok) {
    setStatus(chk.reason || t("sess-frame-invalid"), true);
    return;
  }
  if (!captureState.lastPoseGate.ok) {
    setStatus(captureState.lastPoseGate.reason || t("sess-pose-invalid"), true);
    return;
  }
  let captureDebugLm = null;
  let captureDebugWorld = null;
  let captureDebugGate = null;
  if (captureState.poseLandmarker && captureState.stream) {
    const snap = captureState.poseLandmarker.detectForVideo(video, nextPoseVideoTimestampMs());
    const snapLm = snap.landmarks && snap.landmarks[0];
    const snapWorld = snap.worldLandmarks && snap.worldLandmarks[0];
    if (!snapLm) {
      setStatus(t("sess-pose-invalid-snap"), true);
      return;
    }
    const snapGate = checkPoseForStep(captureState.step, snapLm, snapWorld);
    if (!snapGate.ok) {
      setStatus(snapGate.reason || t("sess-pose-invalid-snap-requirements"), true);
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
    setStatus(t("sess-canvas-context-failed"), true);
    return;
  }
  if (captureState.isMirrored) {
    cctx.translate(tw, 0);
    cctx.scale(-1, 1);
  }
  cctx.drawImage(video, 0, 0, tw, th);

  void (async () => {
    let bmp = null;
    let sq = null;
    try {
      bmp = await createImageBitmap(captureCanvas);
      sq = await imageBitmapToSquare(bmp);
      bmp.close();
      bmp = null;
      const side = sq.width;
      captureCanvas.width = side;
      captureCanvas.height = side;
      const cctxSq = captureCanvas.getContext("2d");
      if (!cctxSq) {
        captureState.awaitingCaptureBlob = false;
        setStatus(t("sess-canvas-context-failed"), true);
        return;
      }
      cctxSq.drawImage(sq, 0, 0);
      sq.close();
      sq = null;
      captureCanvas.toBlob(
        (blob) => {
          captureState.awaitingCaptureBlob = false;
          if (!blob) {
            setStatus(t("sess-take-photo-failed"), true);
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
    } catch (e) {
      captureState.awaitingCaptureBlob = false;
      console.error(e);
      setStatus(t("sess-crop-square-failed"), true);
    } finally {
      if (bmp) bmp.close();
      if (sq) sq.close();
    }
  })();
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
    syncThumbSlotToImage(thumbFront);
    if (captureState.sideBlob) {
      captureState.step = 2;
      captureState.suspendPoseLoopAfterComplete = true;
      stopCamera();
      updateUiStep();
      setStatus(t("sess-front-updated-calculate"));
    } else {
      captureState.step = 2;
      updateUiStep();
      void startCamera().then(() => {
        syncOverlaySize();
      });
      setStatus(t("sess-turn-on-camera-side"));
    }
  } else {
    revokeThumbUrl(thumbSide);
    captureState.sideBlob = blob;
    thumbSide.src = url;
    thumbSide.hidden = false;
    syncThumbSlotToImage(thumbSide);
    captureState.suspendPoseLoopAfterComplete = true;
    stopCamera();
    updateUiStep();
    setStatus(t("sess-both-ready-calculate"));
  }
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
  setStatus(t("sess-retake-front-status"));
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
  setStatus(t("sess-retake-side-status"));
  void startCamera();
}
