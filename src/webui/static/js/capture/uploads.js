/**
 * Photo uploads and prefilled captures: validation, pose gate (with auto-mirror for profile), thumbnails, wizard step.
 */

import { t } from "../i18n/index.js";
import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { logPoseGateCheck } from "./poseDebug.js";
import { syncOverlaySize } from "./overlay.js";
import { cancelSpeechSynthesis } from "./speech.js";
import {
  setPoseStatus,
  setPoseStatusVisual,
  setStatus,
  updateUiStep,
} from "./ui.js";
import { clearThumbSlotPhoto, syncThumbSlotToImage } from "./thumbLayout.js";
import { disposePoseLandmarkerImage, gatePoseOnBitmap, loadPoseLandmarkerImage } from "./poseModel.js";
import { imageBitmapToBlob, imageBitmapToSquare, readImageBitmapHorizontallyFlipped } from "./imageOps.js";
import { resetAutoCaptureUi } from "./autoCapture.js";
import { stopCamera } from "./session.js";


/** Match server `MAX_UPLOAD_BYTES` in webui/app.py */
const UPLOAD_MAX_BYTES = 5 * 1024 * 1024;

const UPLOAD_MIME = new Set(["image/jpeg", "image/png", "image/webp"]);


/**
 * Store uploaded blob(s), update thumbs and wizard step (shared by pose-validated and debug-skip paths).
 * @param {"front"|"side"} which
 * @param {File} file
 * @param {Blob} storageBlob  processed image (square) for API + thumbnail
 * @param {string} poseLine     copy for `#pose-status`
 * @param {{ mirrored?: boolean }} [opts]
 */
function commitUploadedImage(which, file, storageBlob, poseLine, opts = {}) {
  const mirrored = Boolean(opts.mirrored);
  const { thumbFront, thumbSide } = getCaptureDom();
  resetAutoCaptureUi();
  cancelSpeechSynthesis();
  // Upload flow should be silent: keep visual pose status without TTS.
  setPoseStatusVisual(poseLine, "ok");
  const url = URL.createObjectURL(storageBlob);
  if (which === "front") {
    captureState.suspendPoseLoopAfterComplete = false;
    revokeThumbUrl(thumbFront);
    captureState.frontBlob = storageBlob;
    thumbFront.src = url;
    thumbFront.hidden = false;
    syncThumbSlotToImage(thumbFront);
    if (captureState.sideBlob) {
      captureState.step = 2;
      captureState.suspendPoseLoopAfterComplete = true;
      stopCamera();
      setStatus(t("sess-front-ready-both"));
    } else {
      captureState.step = 2;
      captureState.guideSmoothDelta.dx = 0;
      captureState.guideSmoothDelta.dy = 0;
      captureState.guideSmoothDelta.fitHeight = null;
      captureState.guideSmoothDelta.fitTop = null;
      captureState.guideSmoothDelta.fitLeft = null;
      setStatus(t("sess-front-ready-need-side"));
    }
  } else {
    revokeThumbUrl(thumbSide);
    captureState.sideBlob = storageBlob;
    thumbSide.src = url;
    thumbSide.hidden = false;
    syncThumbSlotToImage(thumbSide);
    const flipHint = mirrored ? t("sess-mirrored-auto") : "";
    if (captureState.frontBlob) {
      captureState.step = 2;
      captureState.suspendPoseLoopAfterComplete = true;
      stopCamera();
      setStatus(t("sess-side-ready-both", { flipHint }));
    } else {
      captureState.step = 1;
      captureState.suspendPoseLoopAfterComplete = false;
      setStatus(t("sess-side-ready-need-front", { flipHint }));
    }
  }
  updateUiStep();
  syncOverlaySize();
}


/**
 * Use a user-picked file as front or side capture; rejects if pose gate fails (same rules as camera).
 * `File` is a `Blob` — compatible with `/api/measure` FormData.
 */
export async function applyUploadedImage(which, file) {
  if (!file) return;
  if (!UPLOAD_MIME.has(file.type)) {
    setStatus(t("sess-image-format-invalid"), true);
    return;
  }
  if (file.size > UPLOAD_MAX_BYTES) {
    setStatus(t("sess-file-too-large"), true);
    return;
  }

  if (captureState.debugSkipUploadPoseGate) {
    let bmp;
    try {
      bmp = await createImageBitmap(file);
    } catch {
      setStatus(t("sess-read-image-failed"), true);
      return;
    }
    try {
      const squared = await imageBitmapToSquare(bmp);
      bmp.close();
      bmp = null;
      const storageBlob = await imageBitmapToBlob(squared, file.type);
      squared.close();
      if (!storageBlob) {
        setStatus(t("sess-crop-square-failed"), true);
        return;
      }
      commitUploadedImage(which, file, storageBlob, t("sess-file-accepted-no-check"));
    } finally {
      if (bmp) bmp.close();
    }
    return;
  }

  const viewStep = which === "front" ? 1 : 2;
  try {
    await loadPoseLandmarkerImage();
    if (captureState.poseImageLoadError) {
      setStatus(t("sess-mediapipe-failed"), true);
      setPoseStatus(t("sess-mediapipe-failed"), "bad");
      return;
    }

    let bitmap;
    try {
      bitmap = await createImageBitmap(file);
    } catch {
      setStatus(t("sess-read-image-failed"), true);
      return;
    }

    try {
      let poseChk = gatePoseOnBitmap(viewStep, bitmap);
      logPoseGateCheck(
        "upload",
        viewStep,
        poseChk.landmarks ?? null,
        poseChk.world ?? null,
        { ok: poseChk.ok, reason: poseChk.reason }
      );

      let mirroredSide = false;
      if (!poseChk.ok && viewStep === 2) {
        let flipped = null;
        try {
          flipped = await readImageBitmapHorizontallyFlipped(bitmap);
          const poseChkF = gatePoseOnBitmap(viewStep, flipped);
          logPoseGateCheck(
            "upload",
            viewStep,
            poseChkF.landmarks ?? null,
            poseChkF.world ?? null,
            { ok: poseChkF.ok, reason: poseChkF.reason }
          );
          if (poseChkF.ok) {
            bitmap.close();
            bitmap = flipped;
            flipped = null;
            poseChk = poseChkF;
            mirroredSide = true;
          }
        } finally {
          if (flipped) flipped.close();
        }
      }

      if (!poseChk.ok) {
        setStatus(poseChk.reason || t("sess-pose-invalid"), true);
        setPoseStatus(poseChk.reason || t("sess-pose-invalid"), "bad");
        return;
      }

      let squared;
      try {
        squared = await imageBitmapToSquare(bitmap);
      } finally {
        bitmap.close();
        bitmap = null;
      }

      const storageBlob = await imageBitmapToBlob(squared, file.type);
      squared.close();
      if (!storageBlob) {
        setStatus(t("sess-crop-square-failed"), true);
        setPoseStatus("", "bad");
        return;
      }

      commitUploadedImage(which, file, storageBlob, t("sess-pose-valid-photo"), {
        mirrored: which === "side" && mirroredSide,
      });
    } finally {
      if (bitmap) bitmap.close();
    }
  } finally {
    disposePoseLandmarkerImage();
  }
}


/** Revoke an object URL on a thumbnail img element. */
export function revokeThumbUrl(imgEl) {
  if (!imgEl || !imgEl.src) return;
  clearThumbSlotPhoto(imgEl);
  if (imgEl.src.startsWith("blob:")) {
    try {
      URL.revokeObjectURL(imgEl.src);
    } catch {
    }
    imgEl.removeAttribute("src");
  }
}


/**
 * Apply pre-filled capture blobs (e.g. transferred from main results page).
 * Skips pose gate — photos were already validated on the capture flow.
 */
export function applyPrefillCapture(frontBlob, sideBlob) {
  const { thumbFront, thumbSide } = getCaptureDom();
  if (frontBlob) {
    revokeThumbUrl(thumbFront);
    captureState.frontBlob = frontBlob;
    thumbFront.src = URL.createObjectURL(frontBlob);
    thumbFront.hidden = false;
    syncThumbSlotToImage(thumbFront);
  }
  if (sideBlob) {
    revokeThumbUrl(thumbSide);
    captureState.sideBlob = sideBlob;
    thumbSide.src = URL.createObjectURL(sideBlob);
    thumbSide.hidden = false;
    syncThumbSlotToImage(thumbSide);
  }
  if (captureState.frontBlob && captureState.sideBlob) {
    captureState.step = 2;
    captureState.suspendPoseLoopAfterComplete = true;
  } else if (captureState.frontBlob) {
    captureState.step = 2;
  } else {
    captureState.step = 1;
  }
}
