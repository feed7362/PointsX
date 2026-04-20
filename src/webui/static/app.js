/**
 * Capture UI: reference poses, guide outline matched to examples, MediaPipe Pose gate (browser).
 * Video preview is mirrored (selfie); landmark X is flipped before rules so checks match what the user sees.
 */
import {
  resolveGarment,
  formatSizeTabs,
  measurementsToMap,
  garmentsForSex,
} from "./sizeEngine.js";
import {
  guideGeom,
  loadGuideGeometry,
  loadGuideGeometryFromOptionalStaticFile,
  guidePtsToRefSvgPoints,
  computeGuideBox,
  computeGuideBoxTracked,
  drawPoly,
  drawOpenStroke,
} from "./guideGeometry.js";

/** Приклади на початку — ті самі полігони, що й рамка в прев’ю (`guideGeom`). */
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

const MIN_VIDEO_DIMENSION = 480;
const POSE_MIN_INTERVAL_MS = 120;
/** Нижче за дефолтні 0.5 — інакше selfie/світло часто довго дають порожній landmarks[] («Людину не видно»). */
const POSE_MP_MIN_DETECTION_CONF = 0.38;
const POSE_MP_MIN_PRESENCE_CONF = 0.38;
const POSE_MP_MIN_TRACKING_CONF = 0.38;
const VIS_MIN = 0.32;
const VIS_ANKLE_MIN = 0.22;
const VIS_FULL_BODY_MIN = 0.45;
const VIS_FULL_BODY_LIMB_MIN = 0.5;
const VIS_FULL_BODY_FEET_MIN = 0.4;

/** Діагностика гейту: `?poseDebug=1` у URL або `localStorage.poseDebug = "1"`. */
function isPoseDebug() {
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem("poseDebug") === "1") return true;
    if (typeof location !== "undefined" && new URLSearchParams(location.search).has("poseDebug")) return true;
  } catch {
    /* ignore */
  }
  return false;
}

/**
 * Кут ∠ABC у радіанах (0…π), вершина в B.
 * @param {{ x: number, y: number }} a
 * @param {{ x: number, y: number }} b
 * @param {{ x: number, y: number }} c
 */
function angleAtVertexRad(a, b, c) {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const l1 = Math.hypot(v1x, v1y);
  const l2 = Math.hypot(v2x, v2y);
  if (l1 < 1e-5 || l2 < 1e-5) return Math.PI;
  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (l1 * l2)));
  return Math.acos(cos);
}

/** Кут ∠ABC у 3D (0…π), вершина в B — для колін у worldLandmarks (2D у фронті дає хибні кути). */
function angleAtVertexRad3(a, b, c) {
  const v1x = a.x - b.x;
  const v1y = a.y - b.y;
  const v1z = (a.z ?? 0) - (b.z ?? 0);
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const v2z = (c.z ?? 0) - (b.z ?? 0);
  const l1 = Math.hypot(v1x, v1y, v1z);
  const l2 = Math.hypot(v2x, v2y, v2z);
  if (l1 < 1e-6 || l2 < 1e-6) return Math.PI;
  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y + v1z * v2z) / (l1 * l2)));
  return Math.acos(cos);
}

/**
 * Один запис у консоль після знімка — ті самі landmarks і гейт, що й у JPEG.
 * @param {number} step
 * @param {any[]} rawLm сирі лендмарки (до flip)
 * @param {any[]|null|undefined} worldLm worldLandmarks[0] з detectForVideo
 * @param {{ ok: boolean, reason?: string }} gate
 */
function logPoseDebugCapture(step, rawLm, worldLm, gate) {
  if (!isPoseDebug() || !rawLm?.length) return;
  const lm = flipLandmarks(rawLm);
  const ls = lm[11];
  const rs = lm[12];
  const nose = lm[0];
  const shoulderW = Math.abs(ls.x - rs.x);
  const hipWForFacing = Math.abs((lm[23]?.x ?? ls.x) - (lm[24]?.x ?? rs.x));
  const frontalWidth = Math.max(shoulderW, hipWForFacing);
  const shoulderMidX = (ls.x + rs.x) / 2;
  const lh = lm[23];
  const rh = lm[24];
  const lk = lm[25];
  const rk = lm[26];
  const hipY = (lh.y + rh.y) / 2;
  const kneeY = (lk.y + rk.y) / 2;
  const kneeKneeFlex2d =
    (lk.visibility ?? 0) > 0.2 && (rk.visibility ?? 0) > 0.2
      ? {
          leftDeg: (angleAtVertexRad(lh, lk, lm[27]) * 180) / Math.PI,
          rightDeg: (angleAtVertexRad(rh, rk, lm[28]) * 180) / Math.PI,
        }
      : null;
  let kneeFlexDeg3d = null;
  if (step === 1 && worldLm && worldLm.length > 28) {
    const wPt = (wm, i) => {
      const p = wm[i];
      if (!p) return null;
      return { x: p.x ?? 0, y: p.y ?? 0, z: p.z ?? 0 };
    };
    const kf = [];
    if ((lk.visibility ?? 0) > 0.18 && (lm[27]?.visibility ?? 0) > 0.18) {
      const a = wPt(worldLm, 23);
      const b = wPt(worldLm, 25);
      const c = wPt(worldLm, 27);
      if (a && b && c) kf.push((angleAtVertexRad3(a, b, c) * 180) / Math.PI);
    }
    if ((rk.visibility ?? 0) > 0.18 && (lm[28]?.visibility ?? 0) > 0.18) {
      const a = wPt(worldLm, 24);
      const b = wPt(worldLm, 26);
      const c = wPt(worldLm, 28);
      if (a && b && c) kf.push((angleAtVertexRad3(a, b, c) * 180) / Math.PI);
    }
    if (kf.length) kneeFlexDeg3d = Number(Math.min(...kf).toFixed(1));
  }
  const earL = lm[7]?.visibility ?? 0;
  const earR = lm[8]?.visibility ?? 0;
  const eyeL = lm[2]?.visibility ?? 0;
  const eyeR = lm[5]?.visibility ?? 0;
  const eyeSepX =
    step === 2 && eyeL > 0.45 && eyeR > 0.45 ? Number(Math.abs(lm[2].x - lm[5].x).toFixed(4)) : null;
  console.debug("[poseDebug:capture]", {
    step,
    ok: gate.ok,
    reason: gate.reason,
    shoulderW: Number(shoulderW.toFixed(4)),
    frontalWidth: Number(frontalWidth.toFixed(4)),
    noseShoulderDx: Number(Math.abs(nose.x - shoulderMidX).toFixed(4)),
    hipKneeDy: Number((kneeY - hipY).toFixed(4)),
    kneeFlexDeg2d: kneeKneeFlex2d,
    kneeFlexMinDeg3d: kneeFlexDeg3d,
    eyeSepX,
    earVis: { L: Number(earL.toFixed(2)), R: Number(earR.toFixed(2)) },
    eyeVis: { L: Number(eyeL.toFixed(2)), R: Number(eyeR.toFixed(2)) },
  });
}

const MP_PKG = "https://esm.sh/@mediapipe/tasks-vision@0.10.14";
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const captureCanvas = document.getElementById("capture-canvas");
const heightInput = document.getElementById("height");
const sexSelect = document.getElementById("sex");
const stepLabel = document.getElementById("step-label");
const statusEl = document.getElementById("status");
const poseStatusEl = document.getElementById("pose-status");
const btnStart = document.getElementById("btn-start");
const btnCapture = document.getElementById("btn-capture");
const btnCaptureTimer = document.getElementById("btn-capture-timer");
const btnStop = document.getElementById("btn-stop");
const btnMeasure = document.getElementById("btn-measure");
const btnToCapture = document.getElementById("btn-to-capture");
const sectionCapture = document.getElementById("section-capture");
const sectionParams = document.getElementById("section-params");
const countdownOverlay = document.getElementById("countdown-overlay");
const thumbFront = document.getElementById("thumb-front");
const thumbSide = document.getElementById("thumb-side");
const btnRetakeFront = document.getElementById("btn-retake-front");
const btnRetakeSide = document.getElementById("btn-retake-side");
const resultsSection = document.getElementById("results-section");
const resultsBody = document.getElementById("results-body");
const tailoringIntro = document.getElementById("tailoring-intro");
const garmentStripWrap = document.getElementById("garment-strip-wrap");
const garmentStrip = document.getElementById("garment-strip");
const tailoringPanels = document.getElementById("tailoring-panels");
const tailoringMeasuresBody = document.getElementById("tailoring-measures-body");
const tailoringDisclaimer = document.getElementById("tailoring-disclaimer");
const allMeasuresDetails = document.getElementById("all-measures-details");
const tabUa = document.getElementById("tab-ua");
const tabEu = document.getElementById("tab-eu");
const tabUs = document.getElementById("tab-us");
const panelUa = document.getElementById("panel-ua");
const panelEu = document.getElementById("panel-eu");
const panelUs = document.getElementById("panel-us");

let stream = null;
let step = 1;
/** @type {Blob | null} */
let frontBlob = null;
/** @type {Blob | null} */
let sideBlob = null;
/** Після успішної зйомки обох кадрів — не крутити гейт пози й не озвучувати, поки не ретейк/камера. */
let suspendPoseLoopAfterComplete = false;
let raf = 0;
/** @type {any} */
let poseLandmarker = null;
let poseLoadError = null;
let lastPoseCheck = 0;
/** @type {{ ok: boolean, reason?: string }} */
let lastPoseGate = { ok: false, reason: "Завантаження…" };
/** Останні сирі лендмарки MediaPipe (без flip) — для вирівнювання рамки під голівку. */
let lastRawLandmarks = null;
/** Згладжене зміщення рамки відносно базового guideBox (px). */
let guideSmoothDelta = { dx: 0, dy: 0, fitHeight: null, fitTop: null, fitLeft: null };

const CAPTURE_TIMER_SECONDS = 10;
let captureTimerIntervalId = 0;
let captureTimerRemaining = 0;

/** Після цієї тривалості безперервної «ok» пози стартує відлік 3–2–1. */
const STABLE_POSE_MS = 700;

/** Час початку поточної серії кадрів з gate.ok (null = серія обірвана). */
let poseStableOkSinceMs = null;
/** Інтервал зворотного відліку до автозйомки (0 = не активний). */
let autoPoseCountdownIntervalId = 0;
/** Щоб не стартував новий авто-відлік, поки `toBlob` ще не завершився. */
let awaitingCaptureBlob = false;

/** @type {any | null} */
let tailoringCatalog = null;
/** @type {any | null} */
let lastMockResponse = null;
let selectedGarmentId = "shirt";
let lastSpokenPoseMsg = "";
let lastSpokenAtMs = 0;
const POSE_VOICE_MIN_INTERVAL_MS = 1800;
const POSE_VOICE_REPEAT_MS = 9000;

/** @type {SpeechSynthesisVoice | null | undefined} undefined = ще не шукали */
let cachedUkVoice = undefined;
let voicesChangeHooked = false;

/**
 * Обираємо природніший український голос (не «Compact»), з fallback на будь-який uk.
 * Голоси часто з’являються після voiceschanged — кеш оновлюється.
 */
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

function ensureUkVoiceList() {
  if (typeof window === "undefined" || !("speechSynthesis" in window) || voicesChangeHooked) return;
  voicesChangeHooked = true;
  const refresh = () => {
    cachedUkVoice = pickPleasantUkVoice();
  };
  window.speechSynthesis.addEventListener("voiceschanged", refresh);
  refresh();
}

function setStatus(msg, isError) {
  statusEl.textContent = msg || "";
  statusEl.classList.toggle("err", Boolean(isError));
}

function setPoseStatus(msg, kind) {
  poseStatusEl.textContent = msg || "";
  poseStatusEl.classList.remove("ok", "bad");
  if (kind === "ok") poseStatusEl.classList.add("ok");
  if (kind === "bad") poseStatusEl.classList.add("bad");
  speakPoseHint(msg);
}

/** Озвучка без throttle підказок пози (цифри відліку тощо). */
function speakImmediateUk(text) {
  const t = (text || "").trim();
  if (!t || typeof window === "undefined" || !("speechSynthesis" in window)) return;
  ensureUkVoiceList();
  try {
    if (cachedUkVoice === undefined) cachedUkVoice = pickPleasantUkVoice();
    const u = new SpeechSynthesisUtterance(t);
    u.lang = "uk-UA";
    if (cachedUkVoice) {
      u.voice = cachedUkVoice;
      if (cachedUkVoice.lang && /^uk/i.test(cachedUkVoice.lang)) u.lang = cachedUkVoice.lang;
    }
    u.rate = 0.88;
    u.pitch = 1.05;
    u.volume = 0.95;
    window.speechSynthesis.speak(u);
  } catch {
    // ignore
  }
}

const COUNTDOWN_DIGIT_UK = { 3: "три", 2: "два", 1: "один" };

function speakCountdownDigit(n) {
  cancelSpeechSynthesis();
  const w = COUNTDOWN_DIGIT_UK[/** @type {1|2|3} */ (n)];
  if (w) speakImmediateUk(w);
}

function setPoseStatusVisual(msg, kind) {
  poseStatusEl.textContent = msg || "";
  poseStatusEl.classList.remove("ok", "bad");
  if (kind === "ok") poseStatusEl.classList.add("ok");
  if (kind === "bad") poseStatusEl.classList.add("bad");
}

function showCountdownOverlay(n) {
  if (!countdownOverlay) return;
  countdownOverlay.textContent = String(n);
  countdownOverlay.hidden = false;
}

function hideCountdownOverlay() {
  if (!countdownOverlay) return;
  countdownOverlay.hidden = true;
  countdownOverlay.textContent = "";
}

function resetPoseStableHold() {
  poseStableOkSinceMs = null;
}

function cancelSpeechSynthesis() {
  try {
    if (typeof window !== "undefined" && window.speechSynthesis) window.speechSynthesis.cancel();
  } catch {
    // ignore
  }
}

function interruptAutoPoseCountdown() {
  if (!autoPoseCountdownIntervalId) return;
  clearInterval(autoPoseCountdownIntervalId);
  autoPoseCountdownIntervalId = 0;
  hideCountdownOverlay();
  cancelSpeechSynthesis();
}

function resetAutoCaptureUi() {
  interruptAutoPoseCountdown();
  resetPoseStableHold();
}

function startAutoPoseCountdown() {
  if (autoPoseCountdownIntervalId || captureTimerIntervalId) return;
  if (!stream || (frontBlob && sideBlob)) return;
  cancelSpeechSynthesis();
  resetPoseStableHold();
  let n = 3;
  const stepTick = () => {
    showCountdownOverlay(n);
    setPoseStatusVisual(`Знімок через… ${n}`, "ok");
    speakCountdownDigit(n);
  };
  stepTick();
  autoPoseCountdownIntervalId = window.setInterval(() => {
    n -= 1;
    if (n <= 0) {
      clearInterval(autoPoseCountdownIntervalId);
      autoPoseCountdownIntervalId = 0;
      hideCountdownOverlay();
      captureFrameToBlob(onCaptureReady);
      return;
    }
    showCountdownOverlay(n);
    setPoseStatusVisual(`Знімок через… ${n}`, "ok");
    speakCountdownDigit(n);
  }, 1000);
}

function revokeThumbUrl(imgEl) {
  if (!imgEl || !imgEl.src) return;
  if (imgEl.src.startsWith("blob:")) {
    try {
      URL.revokeObjectURL(imgEl.src);
    } catch {
      // ignore
    }
    imgEl.removeAttribute("src");
  }
}

function updateCaptureReviewUi() {
  /* Тільки після обох знімків і зупинки циклу — інакше при перезйомці анфасу знову є обидва blob'и, але камера активна для профілю, і .capture--review ховав прев'ю (display:none), залишаючи потік/TTS. */
  const review = Boolean(frontBlob && sideBlob && suspendPoseLoopAfterComplete);
  if (sectionCapture) sectionCapture.classList.toggle("capture--review", review);
  const retake = document.getElementById("retake-actions");
  if (retake) retake.hidden = !review;
}

function speakPoseHint(msg) {
  const text = (msg || "").trim();
  if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) return;
  if (captureTimerIntervalId) return;
  ensureUkVoiceList();
  const now = performance.now();
  const isSame = text === lastSpokenPoseMsg;
  if (!isSame && now - lastSpokenAtMs < POSE_VOICE_MIN_INTERVAL_MS) return;
  if (isSame && now - lastSpokenAtMs < POSE_VOICE_REPEAT_MS) return;
  try {
    if (cachedUkVoice === undefined) cachedUkVoice = pickPleasantUkVoice();
    window.speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.lang = "uk-UA";
    if (cachedUkVoice) {
      u.voice = cachedUkVoice;
      if (cachedUkVoice.lang && /^uk/i.test(cachedUkVoice.lang)) u.lang = cachedUkVoice.lang;
    }
    /* Трохи повільніше й м’якший тон — менше «робот» у системних движках. */
    u.rate = 0.9;
    u.pitch = 1.04;
    u.volume = 0.92;
    window.speechSynthesis.speak(u);
    lastSpokenPoseMsg = text;
    lastSpokenAtMs = now;
  } catch {
    // ignore: voice is optional
  }
}

function resetTimerButtonLabel() {
  btnCaptureTimer.textContent = `Фото через ${CAPTURE_TIMER_SECONDS} с`;
}

function clearCaptureTimer(setNeutralStatus = false) {
  if (captureTimerIntervalId) {
    clearInterval(captureTimerIntervalId);
    captureTimerIntervalId = 0;
  }
  captureTimerRemaining = 0;
  resetTimerButtonLabel();
  if (setNeutralStatus) setStatus("");
}

function guideBox(cssW, cssH) {
  return computeGuideBox(cssW, cssH, heightInput.value);
}

/** Базова рамка + зсув під ніс; згладжування (стан у guideSmoothDelta). */
function guideBoxTracked(cssW, cssH, vw, vh) {
  return computeGuideBoxTracked(
    cssW,
    cssH,
    vw,
    vh,
    step,
    lastRawLandmarks,
    guideSmoothDelta,
    heightInput.value
  );
}

function drawOverlay() {
  const ctx = overlay.getContext("2d");
  if (!ctx) return;
  const cssW = Math.max(1, overlay.clientWidth || overlay.width);
  const cssH = Math.max(1, overlay.clientHeight || overlay.height);
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  const bw = Math.round(cssW * dpr);
  const bh = Math.round(cssH * dpr);
  if (overlay.width !== bw || overlay.height !== bh) {
    overlay.width = bw;
    overlay.height = bh;
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const vw = video.videoWidth || 0;
  const vh = video.videoHeight || 0;
  const box = guideBoxTracked(cssW, cssH, vw, vh);
  ctx.strokeStyle = "rgba(91, 140, 255, 0.92)";
  ctx.lineWidth = Math.max(2, cssW * 0.014);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash([8, 6]);
  if (step === 1) {
    drawPoly(ctx, guideGeom.frontPts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.07)";
    ctx.fill();
  } else {
    drawPoly(ctx, guideGeom.profilePts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.07)";
    ctx.fill();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(120, 170, 255, 0.95)";
    drawOpenStroke(ctx, guideGeom.profileArmPts, box);
    ctx.setLineDash([]);
  }
}

function syncOverlaySize() {
  drawOverlay();
}

function flipLandmarks(lm) {
  return lm.map((p) => ({ ...p, x: 1 - p.x }));
}

const DEG = (d) => (d * Math.PI) / 180;

/** Кут відведення плеча–лікоть від вертикалі вниз (0 = рука строго вниз). */
function upperArmAbductionRad(shoulder, elbow) {
  const dy = Math.max(1e-4, elbow.y - shoulder.y);
  const dx = Math.abs(elbow.x - shoulder.x);
  return Math.atan2(dx, dy);
}

/**
 * Профіль: кут правої верхньої кінцівки (плече 12 — лікоть 14) від вертикалі вниз; орієнтир «рука вперед ~45°».
 */
function checkProfileRightArmAngle(lm) {
  const rs = lm[12];
  const re = lm[14];
  const reVis = re.visibility ?? 0;
  const rightAbd = upperArmAbductionRad(rs, re);

  if (reVis < 0.24) {
    return {
      ok: false,
      reason: "Профіль: покажіть правий лікоть — перевіряється кут правої руки (плече–лікоть).",
    };
  }
  if (rightAbd < DEG(18)) {
    return {
      ok: false,
      reason: "Профіль: кут правої руки від тіла занадто малий; відведіть праву руку вперед (~45°) або збоку.",
    };
  }
  if (rightAbd > DEG(80)) {
    return {
      ok: false,
      reason: "Профіль: кут правої руки занадто великий; опустіть праву руку трохи нижче.",
    };
  }
  return { ok: true };
}

/**
 * Анфас A-поза: MediaPipe 0 nose, 11–12 shoulders, 13–14 elbows, 15–16 wrists, 23–24 hips, 25–26 knees, 27–28 ankles.
 * @param {any[]|null|undefined} rawWorldLm — worldLandmarks[0] з detectForVideo для 3D-кутів колін.
 */
function checkFrontPose(lm, rawWorldLm) {
  const nose = lm[0];
  const ls = lm[11];
  const rs = lm[12];
  const lh = lm[23];
  const rh = lm[24];
  const le = lm[13];
  const re = lm[14];
  const la = lm[27];
  const ra = lm[28];
  if (
    (nose.visibility ?? 1) < VIS_MIN ||
    (ls.visibility ?? 1) < VIS_MIN ||
    (rs.visibility ?? 1) < VIS_MIN
  ) {
    return { ok: false, reason: "Підійдіть ближче: не видно обличчя або плечей." };
  }
  const tilt = Math.abs(ls.y - rs.y);
  if (tilt > 0.12) {
    return { ok: false, reason: "Вирівняйте плечі (не нахиляйте корпус)." };
  }
  const shoulderW = Math.abs(ls.x - rs.x);
  const hipWForFacing = Math.abs((lh?.x ?? ls.x) - (rh?.x ?? rs.x));
  const frontalWidth = Math.max(shoulderW, hipWForFacing);
  /** Поріг анфасу по ширині плечей/тазу в нормалізованих координатах (див. debug: ~0.095 ще «далеко», 0.118+ ок). */
  const FRONTAL_WIDTH_MIN = 0.094;
  const shoulderMidX = (ls.x + rs.x) / 2;
  if (frontalWidth < FRONTAL_WIDTH_MIN) {
    const corridor = Math.max(0.055, shoulderW * 0.55 + 0.04);
    const noseDx = Math.abs(nose.x - shoulderMidX);
    if (tilt <= 0.12 && noseDx <= corridor) {
      return {
        ok: false,
        reason:
          "Підійдіть ближче до камери або збільшіть фігуру в кадрі: для анфасу плечі мають бути чітко розрізні в кадрі.",
      };
    }
    return { ok: false, reason: "Станьте анфасом до камери (обличчям)." };
  }
  const minSx = Math.min(ls.x, rs.x) - 0.09;
  const maxSx = Math.max(ls.x, rs.x) + 0.09;
  if (nose.x < minSx || nose.x > maxSx) {
    return { ok: false, reason: "Поверніться обличчям до камери (нос по центру)." };
  }

  if ((le.visibility ?? 0) < 0.22 || (re.visibility ?? 0) < 0.22) {
    return { ok: false, reason: "Лікті мають бути в кадрі — пахви відкриті для виміру грудей." };
  }
  const leftAbd = upperArmAbductionRad(ls, le);
  const rightAbd = upperArmAbductionRad(rs, re);
  if (leftAbd < DEG(10) || rightAbd < DEG(10)) {
    return { ok: false, reason: "A-поза: відведіть руки на ~15–20° від тіла, пахви не закривайте." };
  }
  if (leftAbd > DEG(50) || rightAbd > DEG(50)) {
    return { ok: false, reason: "Не розводьте руки занадто широко (достатньо ~15–20°)." };
  }
  const latL = Math.abs(le.x - ls.x);
  const latR = Math.abs(re.x - rs.x);
  if (latL < shoulderW * 0.11 || latR < shoulderW * 0.11) {
    return { ok: false, reason: "Лікті трохи вбік від корпусу — пахви мають залишатися відкритими." };
  }

  const ankleWx = Math.abs(la.x - ra.x);
  const kneeWx = Math.abs(lm[25].x - lm[26].x);
  const hipW = Math.abs(lm[23].x - lm[24].x);
  /* Вузький kneeWx при широкому тазі часто шум 2D, а не стійка «ноги разом». */
  if (
    hipW < shoulderW * 0.11 &&
    ankleWx < shoulderW * 0.32 &&
    kneeWx < shoulderW * 0.27
  ) {
    return { ok: false, reason: "Ноги на ширині плечей; пах між ногами не перекривайте." };
  }
  if ((lm[23].visibility ?? 0) < 0.24 || (lm[24].visibility ?? 0) < 0.24) {
    return { ok: false, reason: "Має бути видно зону стегон і паху (для внутрішнього шва)." };
  }
  if (hipW < shoulderW * 0.11) {
    return { ok: false, reason: "Не зводьте стегна — пах не має зливатися в кадрі." };
  }

  const ankleVis = ((la.visibility ?? 0) + (ra.visibility ?? 0)) / 2;
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст (стопи в кадрі)." };
  }

  const lk = lm[25];
  const rk = lm[26];
  const lv = lk.visibility ?? 0;
  const rv = rk.visibility ?? 0;
  const lav = la.visibility ?? 0;
  const rav = ra.visibility ?? 0;
  const wPt = (wm, i) => {
    const p = wm[i];
    if (!p) return null;
    return { x: p.x ?? 0, y: p.y ?? 0, z: p.z ?? 0 };
  };
  let minKneeFlexDeg3d = null;
  /** Для гейту «стоїть»: при двох ногах беремо середній кут (min дає хибні присіди від шуму одного коліна). */
  let kneeFlexForStanding = null;
  if (rawWorldLm && rawWorldLm.length > 28) {
    const flex3 = [];
    if (lv > 0.18 && lav > 0.18) {
      const a = wPt(rawWorldLm, 23);
      const b = wPt(rawWorldLm, 25);
      const c = wPt(rawWorldLm, 27);
      if (a && b && c) flex3.push((angleAtVertexRad3(a, b, c) * 180) / Math.PI);
    }
    if (rv > 0.18 && rav > 0.18) {
      const a = wPt(rawWorldLm, 24);
      const b = wPt(rawWorldLm, 26);
      const c = wPt(rawWorldLm, 28);
      if (a && b && c) flex3.push((angleAtVertexRad3(a, b, c) * 180) / Math.PI);
    }
    if (flex3.length) {
      minKneeFlexDeg3d = Math.min(...flex3);
      kneeFlexForStanding =
        flex3.length === 2 ? (flex3[0] + flex3[1]) / 2 : flex3[0];
    }
  }
  /* У фронтальній 2D-проєкції кут стегно–коліно–гомілка не відповідає реальному згину — лише world 3D. */
  if (kneeFlexForStanding != null && kneeFlexForStanding < 158) {
    return {
      ok: false,
      reason: "Станьте повним зростом на прямих ногах (не присідайте) — для зйомки потрібна стійка стоячи.",
    };
  }
  if (lv > 0.18 && rv > 0.18) {
    const hipY = (lh.y + rh.y) / 2;
    const kneeY = (lk.y + rk.y) / 2;
    const kneeGap = kneeY - hipY;
    /* Без world — лише вертикаль: присід підтягує коліна ближче до тазу в кадрі. */
    const minGap = minKneeFlexDeg3d != null ? 0.078 : 0.088;
    if (kneeGap < minGap) {
      return {
        ok: false,
        reason: "Станьте повним зростом на прямих ногах (не присідайте) — для зйомки потрібна стійка стоячи.",
      };
    }
  }

  return { ok: true };
}

function wristHangingAlongBody(w, spineX, shoulderY, hipY) {
  const vis = w.visibility ?? 0;
  if (vis < 0.24) return false;
  if (w.y < shoulderY + 0.05) return false;
  if (w.y > hipY + 0.12) return false;
  return Math.abs(w.x - spineX) < 0.11;
}

const FULL_BODY_PARTS = [
  { id: 0, name: "голову" },
  { id: 11, name: "ліве плече" },
  { id: 12, name: "праве плече" },
  { id: 13, name: "лівий лікоть" },
  { id: 14, name: "правий лікоть" },
  { id: 15, name: "ліву кисть" },
  { id: 16, name: "праву кисть" },
  { id: 23, name: "таз справа" },
  { id: 24, name: "таз зліва" },
  { id: 25, name: "ліве коліно" },
  { id: 26, name: "праве коліно" },
  { id: 27, name: "ліву щиколотку" },
  { id: 28, name: "праву щиколотку" },
];

function requiredVisibilityForPart(id) {
  if (id === 15 || id === 16 || id === 13 || id === 14) return VIS_FULL_BODY_LIMB_MIN;
  if (id === 27 || id === 28 || id === 25 || id === 26) return VIS_FULL_BODY_FEET_MIN;
  return VIS_FULL_BODY_MIN;
}

function inFrame(p) {
  return p && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
}

/**
 * Профіль за інструкцією — правий бік до камери (перевіряється права рука). Ліва половина тіла
 * не входить у вимоги «повного зросту» і не повинна підганятися в кадр.
 */
function profileSkipFullBodyIds(_lm) {
  return new Set([11, 13, 15, 23, 25, 27]);
}

function checkFullBodyVisible(lm, viewStep) {
  const invisible = [];
  const outOfFrame = [];
  const skipProfile = viewStep === 2 ? profileSkipFullBodyIds(lm) : null;
  for (const part of FULL_BODY_PARTS) {
    if (skipProfile && skipProfile.has(part.id)) continue;
    const p = lm[part.id];
    const minVis = requiredVisibilityForPart(part.id);
    if (!p || (p.visibility ?? 0) < minVis) invisible.push(part.name);
    if (!inFrame(p)) outOfFrame.push(part.name);
  }
  if (invisible.length) {
    return {
      ok: false,
      reason: `Видиме не все тіло: не видно ${invisible.slice(0, 3).join(", ")}${invisible.length > 3 ? "…" : ""}`,
    };
  }
  if (outOfFrame.length) {
    return {
      ok: false,
      reason: `Видиме не все тіло: поза кадром ${outOfFrame.slice(0, 3).join(", ")}${outOfFrame.length > 3 ? "…" : ""}.`,
    };
  }
  return { ok: true };
}

function checkProfilePose(lm) {
  const ls = lm[11];
  const rs = lm[12];
  const nose = lm[0];
  const lsV = ls.visibility ?? 0;
  const rsV = rs.visibility ?? 0;
  if (Math.max(lsV, rsV) < VIS_MIN || (nose.visibility ?? 1) < VIS_MIN) {
    return { ok: false, reason: "Підійдіть ближче: не видно силуету в профіль." };
  }
  const shoulderW = Math.abs(ls.x - rs.x);
  /**
   * Орієнтація «правий бік до камери»: у вузькому профілі не покладаємось лише на |x11−x12| і visibility — вони шумні.
   * Комбінуємо: (1) занадто анфас — широкі плечі; (2) ліва сторона явно до камери — ліва ключиця суттєво «сильніша» за праву
   * лише коли плечі ще достатньо розведені в кадрі (інакше видимість неконсистентна).
   */
  const tooFrontal = shoulderW > 0.195;
  if (tooFrontal) {
    return { ok: false, reason: "Поверніться боком на ~90° (профіль)." };
  }
  /* Ліва сторона до камери: сильний розрив видимості ловимо завжди; слабший — лише коли плечі ще не «вузький профіль» (інакше шум 11/12). */
  if (lsV > rsV + 0.112) {
    return {
      ok: false,
      reason: "У профілі стійте правим боком до камери — ліва половина тіла не має потрапляти в кадр.",
    };
  }
  if (lsV > rsV + 0.064 && shoulderW > 0.084) {
    return {
      ok: false,
      reason: "У профілі стійте правим боком до камери — ліва половина тіла не має потрапляти в кадр.",
    };
  }
  const leftEye = lm[2];
  const rightEye = lm[5];
  const eyeLv = leftEye?.visibility ?? 0;
  const eyeRv = rightEye?.visibility ?? 0;
  /* У вузькому профілі очі майже збігаються; поріг eyeSep занадто жорсткий давав хибні «розверніть голову». */
  if (eyeLv > 0.45 && eyeRv > 0.45) {
    const eyeSepX = Math.abs(leftEye.x - rightEye.x);
    const eyeSepMax =
      shoulderW < 0.11
        ? Math.max(0.02, shoulderW * 0.38)
        : Math.min(0.022, Math.max(0.017, shoulderW * 0.13));
    if (eyeSepX > eyeSepMax) {
      return {
        ok: false,
        reason: "У профілі не розвертайте голову до камери — дивіться в той самий бік, куди звернене тіло.",
      };
    }
  }
  const shoulderMidX = (ls.x + rs.x) / 2;
  const noseOffShoulderMid = Math.abs(nose.x - shoulderMidX);
  /*
   * Ніс майже на лінії mid плечей часто дає хибний «обличчям» при нормальному профілі (шум, довге обличчя).
   * Лишаємо лише явний напівфронт: широкі плечі й ніс дуже по центру. Вузький профіль — не ця перевірка.
   */
  if (shoulderW > 0.108 && noseOffShoulderMid < 0.011) {
    return {
      ok: false,
      reason: "У профілі тримайте голову вздовж тіла — дивіться в той самий бік, куди звернене тіло (не розвертайте обличчя до камери).",
    };
  }
  const tilt = Math.abs(ls.y - rs.y);
  if (tilt > 0.165) {
    return { ok: false, reason: "Не нахиляйте корпус у профіль." };
  }

  const spineX = (ls.x + rs.x + lm[23].x + lm[24].x) / 4;
  const shoulderY = (ls.y + rs.y) / 2;
  const hipY = (lm[23].y + lm[24].y) / 2;

  const hangL = wristHangingAlongBody(lm[15], spineX, shoulderY, hipY);
  const hangR = wristHangingAlongBody(lm[16], spineX, shoulderY, hipY);
  if (hangL && hangR) {
    return {
      ok: false,
      reason: "Руки не вздовж тіла: витягніть вперед ~45°, щоб не закрити талію й груди.",
    };
  }

  const wL = lm[15];
  const wR = lm[16];
  const forwardL =
    (wL.visibility ?? 0) > 0.24 &&
    wL.y > shoulderY - 0.03 &&
    wL.y < hipY + 0.1 &&
    (Math.abs(wL.x - spineX) > 0.11 || (wL.y < hipY - 0.02 && Math.abs(wL.x - spineX) > 0.07));
  const forwardR =
    (wR.visibility ?? 0) > 0.24 &&
    wR.y > shoulderY - 0.03 &&
    wR.y < hipY + 0.1 &&
    (Math.abs(wR.x - spineX) > 0.11 || (wR.y < hipY - 0.02 && Math.abs(wR.x - spineX) > 0.07));
  const leV = lm[13].visibility ?? 0;
  const reV = lm[14].visibility ?? 0;
  const leNearSpine = leV < 0.2 || Math.abs(lm[13].x - spineX) < 0.12;
  const reNearSpine = reV < 0.2 || Math.abs(lm[14].x - spineX) < 0.12;
  const elbowsTucked =
    leNearSpine &&
    reNearSpine &&
    (reV > 0.22 || leV > 0.22);

  const armsOk = forwardL || forwardR || (elbowsTucked && !hangL && !hangR);
  if (!armsOk) {
    return {
      ok: false,
      reason: "Підніміть руки вперед ~45° — інакше профіль талії/грудей закритий.",
    };
  }

  const rightArm = checkProfileRightArmAngle(lm);
  if (!rightArm.ok) return rightArm;

  const ankleVis = Math.max(lm[27].visibility ?? 0, lm[28].visibility ?? 0);
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст (стопа ближньої ноги в кадрі)." };
  }
  return { ok: true };
}

function checkPoseForStep(viewStep, landmarks, worldLandmarks) {
  const lm = flipLandmarks(landmarks);
  const fullBody = checkFullBodyVisible(lm, viewStep);
  if (!fullBody.ok) return fullBody;
  const rawWorld = worldLandmarks && worldLandmarks.length ? worldLandmarks : null;
  return viewStep === 1 ? checkFrontPose(lm, rawWorld) : checkProfilePose(lm);
}

function checkCaptureReadiness() {
  const track = stream && stream.getVideoTracks()[0];
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

async function loadPoseLandmarker() {
  if (poseLandmarker || poseLoadError) return;
  try {
    const { FilesetResolver, PoseLandmarker } = await import(MP_PKG);
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    try {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    } catch {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    }
  } catch (e) {
    poseLoadError = e;
    console.error(e);
  }
}

function runPoseIfNeeded() {
  const now = performance.now();
  if (now - lastPoseCheck < POSE_MIN_INTERVAL_MS) return;
  lastPoseCheck = now;

  const resChk = checkCaptureReadiness();
  if (!resChk.ok) {
    resetAutoCaptureUi();
    lastPoseGate = { ok: false, reason: resChk.reason };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (poseLoadError) {
    resetAutoCaptureUi();
    lastPoseGate = {
      ok: false,
      reason: "Не вдалося завантажити MediaPipe. Перевірте мережу та оновіть сторінку.",
    };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (!poseLandmarker || !stream || !video.videoWidth) {
    resetAutoCaptureUi();
    lastPoseGate = { ok: false, reason: "Очікування моделі пози…" };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  const result = poseLandmarker.detectForVideo(video, Math.floor(now));
  const lm = result.landmarks && result.landmarks[0];
  const worldLm = result.worldLandmarks && result.worldLandmarks[0];
  if (!lm) {
    lastRawLandmarks = null;
    resetAutoCaptureUi();
    lastPoseGate = { ok: false, reason: "Людину не видно. Встаньте у рамку повним зростом." };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  lastRawLandmarks = lm;

  const gate = checkPoseForStep(step, lm, worldLm);
  lastPoseGate = gate;

  if (autoPoseCountdownIntervalId) {
    if (!gate.ok) {
      resetAutoCaptureUi();
      setPoseStatus(gate.reason || "Утримайте позу для зйомки.", "bad");
      btnCapture.disabled = true;
      btnCaptureTimer.disabled = false;
      return;
    }
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (gate.ok) {
    setPoseStatus("Поза підходить — можна робити фото.", "ok");
    btnCapture.disabled = Boolean(captureTimerIntervalId || awaitingCaptureBlob);
    btnCaptureTimer.disabled = false;
    if (!captureTimerIntervalId && !(frontBlob && sideBlob) && !awaitingCaptureBlob) {
      if (poseStableOkSinceMs == null) poseStableOkSinceMs = now;
      else if (now - poseStableOkSinceMs >= STABLE_POSE_MS) {
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

function loop() {
  if (suspendPoseLoopAfterComplete) {
    cancelSpeechSynthesis();
    setPoseStatusVisual("", "");
    if (stream) stopCamera();
    raf = 0;
    return;
  }
  runPoseIfNeeded();
  syncOverlaySize();
  raf = requestAnimationFrame(loop);
}

function updateUiStep() {
  if (frontBlob && sideBlob && suspendPoseLoopAfterComplete) {
    stepLabel.textContent =
      "Обидва знімки в превʼю — «Розрахувати мірки» або перезняти анфас / профіль за потреби.";
  } else if (step === 1) {
    stepLabel.textContent = "Крок 1 з 2: анфас — A-поза (пахви відкриті, ноги на ширині плечей)";
  } else {
    stepLabel.textContent =
      "Крок 2 з 2: профіль — права рука вперед ~45° (не вздовж тіла; перевіряється кут правої руки)";
  }
  btnMeasure.disabled = !(frontBlob && sideBlob);
  updateCaptureReviewUi();
}

async function startCamera() {
  suspendPoseLoopAfterComplete = false;
  stopCamera();
  setStatus("");
  await loadPoseLandmarker();
  if (poseLoadError) {
    setStatus("MediaPipe не завантажився: " + (poseLoadError.message || String(poseLoadError)), true);
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cancelAnimationFrame(raf);
    loop();
    btnStop.disabled = false;
    btnStart.textContent = "Перезапустити камеру";
    lastPoseGate = { ok: false, reason: "Аналіз пози…" };
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
  } catch (e) {
    setStatus("Не вдалося отримати доступ до камери: " + (e && e.message ? e.message : String(e)), true);
  }
}

function stopCamera() {
  cancelSpeechSynthesis();
  clearCaptureTimer(true);
  resetAutoCaptureUi();
  cancelAnimationFrame(raf);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  lastRawLandmarks = null;
  guideSmoothDelta.dx = 0;
  guideSmoothDelta.dy = 0;
  guideSmoothDelta.fitHeight = null;
  guideSmoothDelta.fitTop = null;
  guideSmoothDelta.fitLeft = null;
  btnCapture.disabled = true;
  btnCaptureTimer.disabled = true;
  btnStop.disabled = true;
  setPoseStatus("", "");
}

function captureFrameToBlob(callback) {
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
  if (!lastPoseGate.ok) {
    setStatus(lastPoseGate.reason || "Поза не відповідає вимогам.", true);
    return;
  }
  /** Дані для poseDebug — тільки з кадру знімка (не з live loop). */
  let captureDebugLm = null;
  let captureDebugWorld = null;
  let captureDebugGate = null;
  /* Той самий кадр, що й у JPEG: інакше можна зняти присід/злам після короткого «ок» у попередньому кроці loop. */
  if (poseLandmarker && stream) {
    const snap = poseLandmarker.detectForVideo(video, Math.floor(performance.now()));
    const snapLm = snap.landmarks && snap.landmarks[0];
    const snapWorld = snap.worldLandmarks && snap.worldLandmarks[0];
    if (!snapLm) {
      setStatus("На кадрі не видно людину — повторіть знімок.", true);
      return;
    }
    const snapGate = checkPoseForStep(step, snapLm, snapWorld);
    if (!snapGate.ok) {
      setStatus(snapGate.reason || "Поза на мить знімка не відповідає вимогам.", true);
      return;
    }
    captureDebugLm = snapLm;
    captureDebugWorld = snapWorld;
    captureDebugGate = snapGate;
  }
  setStatus("");
  awaitingCaptureBlob = true;
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
    awaitingCaptureBlob = false;
    setStatus("Не вдалося отримати контекст canvas.", true);
    return;
  }
  // Match mirrored preview (.preview-wrap.mirror) so saved frames match what the user saw.
  cctx.translate(tw, 0);
  cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, tw, th);
  captureCanvas.toBlob(
    (blob) => {
      awaitingCaptureBlob = false;
      if (!blob) {
        setStatus("Не вдалося створити знімок.", true);
        return;
      }
      if (captureDebugLm && captureDebugGate) {
        logPoseDebugCapture(step, captureDebugLm, captureDebugWorld, captureDebugGate);
      }
      callback(blob);
    },
    "image/jpeg",
    0.92
  );
}

function onCaptureReady(blob) {
  resetAutoCaptureUi();
  const url = URL.createObjectURL(blob);
  if (step === 1) {
    suspendPoseLoopAfterComplete = false;
    revokeThumbUrl(thumbFront);
    frontBlob = blob;
    thumbFront.src = url;
    thumbFront.hidden = false;
    if (sideBlob) {
      /* Перезйомка лише анфасу: профіль уже є — не переводимо на крок 2 з камерою. */
      step = 2;
      suspendPoseLoopAfterComplete = true;
      stopCamera();
      updateUiStep();
      setStatus("Анфас оновлено. Можна «Розрахувати мірки» або перезняти кадр.");
    } else {
      step = 2;
      updateUiStep();
      void startCamera();
      /* Після stopCamera()/setStatus("") всередині startCamera — одразу відновлюємо підказку (синхронно до першого await). */
      setStatus("Увімкніть камеру знову для знімка в профіль.");
    }
  } else {
    revokeThumbUrl(thumbSide);
    sideBlob = blob;
    thumbSide.src = url;
    thumbSide.hidden = false;
    suspendPoseLoopAfterComplete = true;
    stopCamera();
    updateUiStep();
    setStatus("Обидва знімки готові. Натисніть «Розрахувати мірки».");
  }
}

function startCaptureTimer() {
  if (!stream) {
    setStatus("Увімкніть камеру перед запуском таймера.", true);
    return;
  }
  if (captureTimerIntervalId) {
    clearCaptureTimer(true);
    resetAutoCaptureUi();
    setStatus("Таймер скасовано.");
    btnCapture.disabled = !lastPoseGate.ok;
    btnCaptureTimer.disabled = !lastPoseGate.ok;
    return;
  }
  resetAutoCaptureUi();
  cancelSpeechSynthesis();
  captureTimerRemaining = CAPTURE_TIMER_SECONDS;
  btnCapture.disabled = true;
  btnCaptureTimer.disabled = false;
  btnCaptureTimer.textContent = `Скасувати (${captureTimerRemaining} с)`;
  setStatus(`Автозйомка через ${captureTimerRemaining} с… Станьте в правильну позу.`);
  captureTimerIntervalId = window.setInterval(() => {
    if (!stream) {
      clearCaptureTimer();
      btnCapture.disabled = true;
      btnCaptureTimer.disabled = true;
      return;
    }
    captureTimerRemaining -= 1;
    if (captureTimerRemaining <= 0) {
      clearCaptureTimer();
      setStatus("Знімаю фото…");
      captureFrameToBlob(onCaptureReady);
      return;
    }
    btnCaptureTimer.textContent = `Скасувати (${captureTimerRemaining} с)`;
    setStatus(`Автозйомка через ${captureTimerRemaining} с…`);
  }, 1000);
}

async function loadTailoringCatalog() {
  if (tailoringCatalog) return tailoringCatalog;
  const res = await fetch("/static/data/tailoring_config.json?v=2");
  if (!res.ok) throw new Error("Не вдалося завантажити tailoring_config.json");
  tailoringCatalog = await res.json();
  return tailoringCatalog;
}

function sexLabelUk(sex) {
  if (sex === "male") return "чоловік";
  if (sex === "female") return "жінка";
  return "інше";
}

function renderSizePanelContent(container, blocks) {
  if (!container) return;
  container.innerHTML = "";
  for (const b of blocks) {
    const sec = document.createElement("section");
    sec.className = "size-block";
    const h = document.createElement("h4");
    h.textContent = b.label;
    sec.appendChild(h);
    const ul = document.createElement("ul");
    for (const line of b.lines) {
      const li = document.createElement("li");
      li.textContent = line;
      ul.appendChild(li);
    }
    sec.appendChild(ul);
    if (b.warnings && b.warnings.length) {
      const w = document.createElement("p");
      w.className = "size-warn";
      w.textContent = b.warnings.join(" ");
      sec.appendChild(w);
    }
    container.appendChild(sec);
  }
}

function selectSizeTab(which) {
  if (!tabUa || !tabEu || !tabUs || !panelUa || !panelEu || !panelUs) return;
  const tabs = [
    { id: "ua", tab: tabUa, panel: panelUa },
    { id: "eu", tab: tabEu, panel: panelEu },
    { id: "us", tab: tabUs, panel: panelUs },
  ];
  for (const { id, tab, panel } of tabs) {
    const sel = id === which;
    tab.setAttribute("aria-selected", sel ? "true" : "false");
    tab.tabIndex = sel ? 0 : -1;
    panel.hidden = !sel;
  }
}

function refreshTailoringView() {
  if (!lastMockResponse || !tailoringCatalog || !tailoringMeasuresBody) return;
  const g = resolveGarment(tailoringCatalog, selectedGarmentId);
  if (!g) return;
  const map = measurementsToMap(lastMockResponse.measurements);
  tailoringMeasuresBody.innerHTML = "";
  for (const mid of g.measurement_ids || []) {
    const row = lastMockResponse.measurements.find((m) => m.id === mid);
    if (!row) continue;
    const tr = document.createElement("tr");
    tr.innerHTML =
      "<td>" +
      escapeHtml(row.label_uk) +
      "</td><td>" +
      escapeHtml(String(row.value_cm)) +
      "</td><td>" +
      escapeHtml(String(row.confidence)) +
      "</td>";
    tailoringMeasuresBody.appendChild(tr);
  }
  const { uaBlocks, euBlocks, usBlocks } = formatSizeTabs(
    tailoringCatalog,
    g,
    map,
    lastMockResponse.height_cm,
    lastMockResponse.sex
  );
  renderSizePanelContent(panelUa, uaBlocks);
  renderSizePanelContent(panelEu, euBlocks);
  renderSizePanelContent(panelUs, usBlocks);
}

function renderGarmentStrip() {
  if (!tailoringCatalog || !garmentStrip) return;
  garmentStrip.innerHTML = "";
  const sex = lastMockResponse?.sex || "other";
  const list = garmentsForSex(tailoringCatalog, sex);
  const allowed = new Set(list.map((x) => x.id));
  if (!allowed.has(selectedGarmentId) && list.length) {
    selectedGarmentId = list[0].id;
  }
  for (const g of list) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "garment-btn";
    btn.dataset.garmentId = g.id;
    btn.setAttribute("role", "radio");
    const on = g.id === selectedGarmentId;
    btn.setAttribute("aria-checked", on ? "true" : "false");
    btn.setAttribute("aria-label", g.label_uk);
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    const tmp = document.createElement("div");
    tmp.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg">${g.svg || ""}</svg>`;
    const inner = tmp.querySelector("svg");
    if (inner) {
      while (inner.firstChild) svg.appendChild(inner.firstChild);
    }
    btn.appendChild(svg);
    const cap = document.createElement("span");
    cap.textContent = g.label_uk;
    btn.appendChild(cap);
    btn.addEventListener("click", () => {
      selectedGarmentId = g.id;
      garmentStrip.querySelectorAll(".garment-btn").forEach((b) => {
        const sel = b.dataset.garmentId === selectedGarmentId;
        b.setAttribute("aria-checked", sel ? "true" : "false");
      });
      refreshTailoringView();
    });
    garmentStrip.appendChild(btn);
  }
}

let sizeTabsWired = false;
function ensureSizeTabsWired() {
  if (sizeTabsWired || !tabUa || !tabEu || !tabUs) return;
  sizeTabsWired = true;
  const order = ["ua", "eu", "us"];
  const tabById = { ua: tabUa, eu: tabEu, us: tabUs };
  for (const id of order) {
    tabById[id].addEventListener("click", () => selectSizeTab(id));
  }
  for (let i = 0; i < order.length; i++) {
    const id = order[i];
    const tab = tabById[id];
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const next = order[(i + 1) % order.length];
        selectSizeTab(next);
        tabById[next].focus();
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = order[(i - 1 + order.length) % order.length];
        selectSizeTab(prev);
        tabById[prev].focus();
      }
    });
  }
}

btnToCapture.addEventListener("click", () => {
  const target = sectionParams || heightInput?.closest("section");
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
});

function retakeFrontPhoto() {
  if (!frontBlob) return;
  suspendPoseLoopAfterComplete = false;
  revokeThumbUrl(thumbFront);
  thumbFront.hidden = true;
  frontBlob = null;
  step = 1;
  resetAutoCaptureUi();
  updateUiStep();
  setStatus("Перезйомка анфасу: увімкніть камеру та встаньте в позу.");
  void startCamera();
}

function retakeSidePhoto() {
  if (!sideBlob) return;
  suspendPoseLoopAfterComplete = false;
  revokeThumbUrl(thumbSide);
  thumbSide.hidden = true;
  sideBlob = null;
  step = frontBlob ? 2 : 1;
  resetAutoCaptureUi();
  updateUiStep();
  setStatus("Перезйомка профілю: увімкніть камеру та встаньте в позу.");
  void startCamera();
}

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);
if (btnRetakeFront) btnRetakeFront.addEventListener("click", retakeFrontPhoto);
if (btnRetakeSide) btnRetakeSide.addEventListener("click", retakeSidePhoto);

heightInput.addEventListener("input", () => {
  if (overlay.width) drawOverlay();
});

btnCapture.addEventListener("click", () => {
  clearCaptureTimer();
  resetAutoCaptureUi();
  captureFrameToBlob(onCaptureReady);
});
btnCaptureTimer.addEventListener("click", startCaptureTimer);

btnMeasure.addEventListener("click", async () => {
  if (!frontBlob || !sideBlob) return;
  if (!resultsSection || !resultsBody) {
    setStatus("Помилка: немає контейнера результатів у розмітці.", true);
    return;
  }
  setStatus("Обчислення…");
  resultsSection.hidden = true;
  const fd = new FormData();
  fd.append("height_cm", String(heightInput.value));
  fd.append("sex", sexSelect.value);
  fd.append("front", frontBlob, "front.jpg");
  fd.append("side", sideBlob, "side.jpg");
  try {
    const res = await fetch("/api/measure/mock", { method: "POST", body: fd });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || res.statusText);
    }
    const data = await res.json();
    lastMockResponse = data;
    resultsBody.innerHTML = "";
    for (const row of data.measurements) {
      const tr = document.createElement("tr");
      tr.innerHTML =
        "<td>" +
        escapeHtml(row.label_uk) +
        "</td><td>" +
        escapeHtml(String(row.value_cm)) +
        "</td><td>" +
        escapeHtml(String(row.confidence)) +
        "</td>";
      resultsBody.appendChild(tr);
    }

    ensureSizeTabsWired();
    selectSizeTab("ua");

    try {
      await loadTailoringCatalog();
      selectedGarmentId = tailoringCatalog.garments[0]?.id || "shirt";
      if (tailoringIntro) {
        tailoringIntro.textContent = `Зріст: ${data.height_cm} см. Стать: ${sexLabelUk(data.sex)}. Список типів одягу відфільтровано за статтю. Оберіть тип — мірки та орієнтовні розміри (Україна / Європа / США).`;
        tailoringIntro.hidden = false;
      }
      if (garmentStripWrap) garmentStripWrap.hidden = false;
      if (tailoringPanels) tailoringPanels.hidden = false;
      if (tailoringDisclaimer) tailoringDisclaimer.hidden = false;
      renderGarmentStrip();
      refreshTailoringView();
    } catch (cfgErr) {
      if (tailoringIntro) {
        tailoringIntro.textContent =
          "Пошив і сітки: " + (cfgErr && cfgErr.message ? cfgErr.message : String(cfgErr));
        tailoringIntro.hidden = false;
      }
      if (garmentStripWrap) garmentStripWrap.hidden = true;
      if (tailoringPanels) tailoringPanels.hidden = true;
      if (tailoringDisclaimer) tailoringDisclaimer.hidden = true;
    }

    resultsSection.hidden = false;
    setStatus("Готово (mock-дані з сервера).");
  } catch (e) {
    setStatus("Помилка запиту: " + (e && e.message ? e.message : String(e)), true);
  }
});

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

ensureSizeTabsWired();

void loadPoseLandmarker().then(() => {
  if (poseLandmarker) setPoseStatus("Модель пози готова. Увімкніть камеру.", "ok");
  else if (poseLoadError) setPoseStatus("Модель пози не завантажена (офлайн?).", "bad");
});

const previewWrap = video.parentElement;
if (previewWrap && typeof ResizeObserver !== "undefined") {
  new ResizeObserver(() => drawOverlay()).observe(previewWrap);
}
window.addEventListener("resize", () => drawOverlay());

updateUiStep();
resetTimerButtonLabel();
