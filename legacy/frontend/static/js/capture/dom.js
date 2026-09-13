/**
 * Cached DOM references for the capture page (script runs at end of body).
 */

/** @type {ReturnType<typeof buildDom> | null} */
let cached = null;

function buildDom() {
  return {
    video: /** @type {HTMLVideoElement} */ (document.getElementById("video")),
    overlay: /** @type {HTMLCanvasElement} */ (document.getElementById("overlay")),
    captureCanvas: /** @type {HTMLCanvasElement} */ (document.getElementById("capture-canvas")),
    heightInput: /** @type {HTMLInputElement} */ (document.getElementById("height")),
    sexSelect: /** @type {HTMLSelectElement} */ (document.getElementById("sex")),
    poseBackendSelect: /** @type {HTMLSelectElement | null} */ (document.getElementById("pose-backend")),
    stepLabel: /** @type {HTMLElement} */ (document.getElementById("step-label")),
    statusEl: /** @type {HTMLElement} */ (document.getElementById("status")),
    poseStatusEl: /** @type {HTMLElement} */ (document.getElementById("pose-status")),
    btnStart: /** @type {HTMLButtonElement} */ (document.getElementById("btn-start")),
    btnCapture: /** @type {HTMLButtonElement} */ (document.getElementById("btn-capture")),
    btnCaptureTimer: /** @type {HTMLButtonElement} */ (document.getElementById("btn-capture-timer")),
    btnStop: /** @type {HTMLButtonElement} */ (document.getElementById("btn-stop")),
    btnMeasure: /** @type {HTMLButtonElement} */ (document.getElementById("btn-measure")),
    btnMeasureTest: /** @type {HTMLButtonElement | null} */ (document.getElementById("btn-measure-test")),
    btnToCapture: /** @type {HTMLButtonElement} */ (document.getElementById("btn-to-capture")),
    sectionCapture: document.getElementById("section-capture"),
    sectionParams: document.getElementById("section-params"),
    countdownOverlay: document.getElementById("countdown-overlay"),
    previewWrap: document.getElementById("preview-wrap"),
    previewIdle: document.getElementById("preview-idle"),
    thumbFront: /** @type {HTMLImageElement} */ (document.getElementById("thumb-front")),
    thumbSide: /** @type {HTMLImageElement} */ (document.getElementById("thumb-side")),
    btnRetakeFront: /** @type {HTMLButtonElement | null} */ (document.getElementById("btn-retake-front")),
    btnRetakeSide: /** @type {HTMLButtonElement | null} */ (document.getElementById("btn-retake-side")),
    debugSkipUploadPose: /** @type {HTMLInputElement | null} */ (
      document.getElementById("debug-upload-skip-pose")
    ),
    resultsSection: /** @type {HTMLElement | null} */ (document.getElementById("results-section")),
    resultsBody: /** @type {HTMLElement | null} */ (document.getElementById("results-body")),
    tailoringIntro: /** @type {HTMLElement | null} */ (document.getElementById("tailoring-intro")),
    garmentStripWrap: /** @type {HTMLElement | null} */ (document.getElementById("garment-strip-wrap")),
    garmentStrip: /** @type {HTMLElement | null} */ (document.getElementById("garment-strip")),
    tailoringPanels: /** @type {HTMLElement | null} */ (document.getElementById("tailoring-panels")),
    tailoringMeasuresBody: /** @type {HTMLElement | null} */ (document.getElementById("tailoring-measures-body")),
    tailoringDisclaimer: /** @type {HTMLElement | null} */ (document.getElementById("tailoring-disclaimer")),
    allMeasuresDetails: document.getElementById("all-measures-details"),
    tabUa: /** @type {HTMLButtonElement | null} */ (document.getElementById("tab-ua")),
    tabEu: /** @type {HTMLButtonElement | null} */ (document.getElementById("tab-eu")),
    tabUs: /** @type {HTMLButtonElement | null} */ (document.getElementById("tab-us")),
    panelUa: /** @type {HTMLElement | null} */ (document.getElementById("panel-ua")),
    panelEu: /** @type {HTMLElement | null} */ (document.getElementById("panel-eu")),
    panelUs: /** @type {HTMLElement | null} */ (document.getElementById("panel-us")),
    patternDetails: /** @type {HTMLDetailsElement | null} */ (document.getElementById("pattern-details")),
    patternRawBody: /** @type {HTMLElement | null} */ (document.getElementById("pattern-raw-body")),
    patternSeamBody: /** @type {HTMLElement | null} */ (document.getElementById("pattern-seam-body")),
  };
}

export function getCaptureDom() {
  if (!cached) cached = buildDom();
  return cached;
}
