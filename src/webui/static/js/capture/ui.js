/**
 * Status lines, step label, capture review layout.
 */

import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { speakPoseHint } from "./speech.js";

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
    stepLabel.textContent =
      "Обидва знімки в превʼю — «Розрахувати мірки» або перезняти анфас / профіль за потреби.";
  } else if (captureState.step === 1) {
    stepLabel.textContent = "Крок 1 з 2: анфас — A-поза (пахви відкриті, ноги на ширині плечей)";
  } else {
    stepLabel.textContent =
      "Крок 2 з 2: профіль — права рука вперед ~45° (не вздовж тіла; перевіряється кут правої руки)";
  }
  btnMeasure.disabled = !(captureState.frontBlob && captureState.sideBlob);
  updateCaptureReviewUi();
}
