/**
 * Status lines, step label, capture review layout.
 */

import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { speakPoseHint } from "./speech.js";
import { resyncVisibleCaptureThumbs } from "./thumbLayout.js";

import { t } from "./i18n.js";

export function setStatus(msg, isError) {
  const { statusEl } = getCaptureDom();
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("err", Boolean(isError));
}

export function setPoseStatus(msg, kind) {
  const { poseStatusEl } = getCaptureDom();
  poseStatusEl.textContent = msg || "";
  poseStatusEl.classList.remove("ok", "bad");
  if (kind === "ok") poseStatusEl.classList.add("ok");
  if (kind === "bad") poseStatusEl.classList.add("bad");
  speakPoseHint(msg);
}

export function setPoseStatusVisual(msg, kind) {
  const { poseStatusEl } = getCaptureDom();
  poseStatusEl.textContent = msg || "";
  poseStatusEl.classList.remove("ok", "bad");
  if (kind === "ok") poseStatusEl.classList.add("ok");
  if (kind === "bad") poseStatusEl.classList.add("bad");
}

/**
 * Enable review layout only when both blobs exist and capture is finished.
 * Avoids hiding the live preview while retaking front with side already stored.
 */
export function updateCaptureReviewUi() {
  const { sectionCapture } = getCaptureDom();
  const review = Boolean(
    captureState.frontBlob && captureState.sideBlob && captureState.suspendPoseLoopAfterComplete
  );
  if (sectionCapture) sectionCapture.classList.toggle("capture--review", review);
  const retake = document.getElementById("retake-actions");
  if (retake) retake.hidden = !review;
}

/** Update step label, measure button, and review layout from step and blobs. */
export function updateUiStep() {
  const { stepLabel, btnMeasure } = getCaptureDom();
  if (captureState.frontBlob && captureState.sideBlob && captureState.suspendPoseLoopAfterComplete) {
    stepLabel.textContent = t("both-photos-preview");
  } else if (captureState.step === 1) {
    stepLabel.textContent = t("step-1-label");
  } else {
    stepLabel.textContent = t("step-2-label");
  }
  btnMeasure.disabled = !(captureState.frontBlob && captureState.sideBlob);
  updateCaptureReviewUi();
  resyncVisibleCaptureThumbs();
}
