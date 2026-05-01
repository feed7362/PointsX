/**
 * Capture UI orchestrator: reference poses, guide outline, MediaPipe pose gate.
 * Video preview is mirrored (selfie); landmark X is flipped in gate rules to match what the user sees.
 */
import {
  guideGeom,
  applyReloadGuideQueryParam,
  loadGuideGeometry,
  loadGuideGeometryFromOptionalStaticFile,
  drawGuideSilhouetteOnCanvas,
} from "./guideGeometry.js";
import { getCaptureDom } from "./capture/dom.js";
import { captureState } from "./capture/state.js";
import * as session from "./capture/session.js";
import * as overlay from "./capture/overlay.js";
import * as ui from "./capture/ui.js";
import * as tailoring from "./capture/tailoring.js";

/** Render top reference canvases using the same drawing code as live overlay. */
function renderReferenceGuides() {
  const canvases = [
    { id: "ref-canvas-front", step: 1 },
    { id: "ref-canvas-profile", step: 2 },
  ];
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  for (const item of canvases) {
    const canvas = /** @type {HTMLCanvasElement | null} */ (document.getElementById(item.id));
    if (!canvas) continue;
    const cssW = Math.max(1, canvas.clientWidth || 260);
    const cssH = Math.max(1, canvas.clientHeight || Math.round(cssW * 1.6));
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = cssW * 0.07;
    const box = {
      left: pad,
      top: cssH * 0.02,
      width: cssW - pad * 2,
      height: cssH * 0.94,
    };
    drawGuideSilhouetteOnCanvas(ctx, box, item.step, guideGeom);
  }
}

applyReloadGuideQueryParam();
loadGuideGeometry();
renderReferenceGuides();
void loadGuideGeometryFromOptionalStaticFile().then(() => {
  renderReferenceGuides();
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

  const uploadFrontInput = document.getElementById("upload-front");
  const uploadSideInput = document.getElementById("upload-side");
  uploadFrontInput?.addEventListener("change", async (e) => {
    const input = /** @type {HTMLInputElement} */ (e.target);
    const f = input.files?.[0];
    if (f) await session.applyUploadedImage("front", f);
    input.value = "";
  });
  uploadSideInput?.addEventListener("change", async (e) => {
    const input = /** @type {HTMLInputElement} */ (e.target);
    const f = input.files?.[0];
    if (f) await session.applyUploadedImage("side", f);
    input.value = "";
  });

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
  window.addEventListener("resize", () => {
    overlay.drawOverlay();
    renderReferenceGuides();
  });

  ui.updateUiStep();
  session.resetTimerButtonLabel();
}

initCaptureApp();
