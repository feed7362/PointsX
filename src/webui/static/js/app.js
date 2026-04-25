/**
 * Capture UI orchestrator: reference poses, guide outline, MediaPipe pose gate.
 * Video preview is mirrored (selfie); landmark X is flipped in gate rules to match what the user sees.
 */
import {
  guideGeom,
  loadGuideGeometry,
  loadGuideGeometryFromOptionalStaticFile,
  guidePtsToRefSvgPoints,
} from "./guideGeometry.js";
import { getCaptureDom } from "./capture/dom.js";
import { captureState } from "./capture/state.js";
import * as session from "./capture/session.js";
import * as overlay from "./capture/overlay.js";
import * as ui from "./capture/ui.js";
import * as tailoring from "./capture/tailoring.js";

/** Sync reference SVG polygons at the top of the page with the live guide geometry. */
function syncReferenceGuideSvgs() {
  const pf = document.getElementById("ref-poly-front");
  const pp = document.getElementById("ref-poly-profile");
  const pa = document.getElementById("ref-poly-arm");
  if (pf) pf.setAttribute("points", guidePtsToRefSvgPoints(guideGeom.frontPts));
  if (pp) pp.setAttribute("points", guidePtsToRefSvgPoints(guideGeom.profilePts));
  if (pa) pa.setAttribute("points", guidePtsToRefSvgPoints(guideGeom.profileArmPts));
}

loadGuideGeometry();
syncReferenceGuideSvgs();
void loadGuideGeometryFromOptionalStaticFile().then(() => {
  syncReferenceGuideSvgs();
});

function initCaptureApp() {
  getCaptureDom();

  const dom = getCaptureDom();

  dom.btnToCapture.addEventListener("click", () => {
    const target = dom.sectionParams || dom.heightInput?.closest("section");
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  dom.btnStart.addEventListener("click", () => void session.startCamera());
  dom.btnStop.addEventListener("click", () => session.stopCamera());
  if (dom.btnRetakeFront) dom.btnRetakeFront.addEventListener("click", () => session.retakeFrontPhoto());
  if (dom.btnRetakeSide) dom.btnRetakeSide.addEventListener("click", () => session.retakeSidePhoto());

  dom.heightInput.addEventListener("input", () => {
    if (dom.overlay.width) overlay.drawOverlay();
  });

  dom.btnCapture.addEventListener("click", () => {
    session.clearCaptureTimer();
    session.resetAutoCaptureUi();
    session.captureFrameToBlob(session.onCaptureReady);
  });
  dom.btnCaptureTimer.addEventListener("click", () => session.startCaptureTimer());

  tailoring.ensureSizeTabsWired();
  tailoring.attachMeasureHandler();

  void session.loadPoseLandmarker().then(() => {
    if (captureState.poseLandmarker) ui.setPoseStatus("Модель пози готова. Увімкніть камеру.", "ok");
    else if (captureState.poseLoadError) ui.setPoseStatus("Модель пози не завантажена (офлайн?).", "bad");
  });

  const previewWrap = dom.video.parentElement;
  if (previewWrap && typeof ResizeObserver !== "undefined") {
    new ResizeObserver(() => overlay.drawOverlay()).observe(previewWrap);
  }
  window.addEventListener("resize", () => overlay.drawOverlay());

  ui.updateUiStep();
  session.resetTimerButtonLabel();
}

initCaptureApp();
