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
const VIS_MIN = 0.32;
const VIS_ANKLE_MIN = 0.22;
const VIS_FULL_BODY_MIN = 0.45;
const VIS_FULL_BODY_LIMB_MIN = 0.5;
const VIS_FULL_BODY_FEET_MIN = 0.4;

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
const thumbFront = document.getElementById("thumb-front");
const thumbSide = document.getElementById("thumb-side");
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

function speakPoseHint(msg) {
  const text = (msg || "").trim();
  if (!text || typeof window === "undefined" || !("speechSynthesis" in window)) return;
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
 * Профіль: кут правої верхньої кінцівки (плече 12 — лікоть 14) від вертикалі вниз.
 * Для пози «руки за спиною» поріг інший; інакше орієнтир як «рука вперед ~45°».
 */
function checkProfileRightArmAngle(lm, elbowsTucked) {
  const rs = lm[12];
  const re = lm[14];
  const reVis = re.visibility ?? 0;
  const rightAbd = upperArmAbductionRad(rs, re);

  if (elbowsTucked) {
    if (reVis > 0.22 && rightAbd > DEG(58)) {
      return {
        ok: false,
        reason: "Профіль (руки за спиною): притисніть праву руку ближче до спини — кут від тіла занадто великий.",
      };
    }
    return { ok: true };
  }

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
 */
function checkFrontPose(lm) {
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
  if (frontalWidth < 0.11) {
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
  if (ankleWx < shoulderW * 0.32 && kneeWx < shoulderW * 0.27) {
    return { ok: false, reason: "Ноги на ширині плечей; пах між ногами не перекривайте." };
  }
  if ((lm[23].visibility ?? 0) < 0.24 || (lm[24].visibility ?? 0) < 0.24) {
    return { ok: false, reason: "Має бути видно зону стегон і паху (для внутрішнього шва)." };
  }
  const hipW = Math.abs(lm[23].x - lm[24].x);
  if (hipW < shoulderW * 0.11) {
    return { ok: false, reason: "Не зводьте стегна — пах не має зливатися в кадрі." };
  }

  const ankleVis = ((la.visibility ?? 0) + (ra.visibility ?? 0)) / 2;
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст (стопи в кадрі)." };
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

function visSum(lm, ids) {
  let s = 0;
  for (const id of ids) s += lm[id]?.visibility ?? 0;
  return s;
}

/**
 * У профілі одна половина тіла звернена від камери — не вимагаємо її в «повному зрості».
 * Обираємо сторону з меншою сумарною видимістю (рука + нога).
 */
function profileSkipFullBodyIds(lm) {
  const left = [11, 13, 15, 23, 25, 27];
  const right = [12, 14, 16, 24, 26, 28];
  const sl = visSum(lm, left);
  const sr = visSum(lm, right);
  if (sl < sr) return new Set(left);
  if (sr < sl) return new Set(right);
  return new Set([13]);
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
  if (shoulderW > 0.17) {
    return { ok: false, reason: "Поверніться боком на ~90° (профіль)." };
  }
  const shoulderMidX = (ls.x + rs.x) / 2;
  if (Math.abs(nose.x - shoulderMidX) < 0.02) {
    return { ok: false, reason: "Голова має виступати вперед відносно плечей (чіткий профіль)." };
  }
  const tilt = Math.abs(ls.y - rs.y);
  if (tilt > 0.14) {
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
      reason: "Руки не вздовж тіла: витягніть вперед ~45° або закладіть за спину, щоб не закрити талію й груди.",
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
  const wLv = wL.visibility ?? 0;
  const wRv = wR.visibility ?? 0;
  const wristsBothBehind =
    !hangL &&
    !hangR &&
    wLv > 0.22 &&
    wRv > 0.22 &&
    Math.abs(wL.x - spineX) < 0.1 &&
    Math.abs(wR.x - spineX) < 0.1 &&
    wL.y > shoulderY + 0.03 &&
    wR.y > shoulderY + 0.03 &&
    wL.y < hipY + 0.04 &&
    wR.y < hipY + 0.04;
  const wristsOneBehind =
    !hangL &&
    !hangR &&
    ((wLv > 0.22 &&
      wRv < 0.16 &&
      Math.abs(wL.x - spineX) < 0.12 &&
      wL.y > shoulderY + 0.03 &&
      wL.y < hipY + 0.06) ||
      (wRv > 0.22 &&
        wLv < 0.16 &&
        Math.abs(wR.x - spineX) < 0.12 &&
        wR.y > shoulderY + 0.03 &&
        wR.y < hipY + 0.06));
  const wristsNearSpineMid = wristsBothBehind || wristsOneBehind;
  const leV = lm[13].visibility ?? 0;
  const reV = lm[14].visibility ?? 0;
  const leNearSpine = leV < 0.2 || Math.abs(lm[13].x - spineX) < 0.12;
  const reNearSpine = reV < 0.2 || Math.abs(lm[14].x - spineX) < 0.12;
  const elbowsTucked =
    leNearSpine &&
    reNearSpine &&
    (reV > 0.22 || leV > 0.22);

  const armsOk = forwardL || forwardR || wristsNearSpineMid || (elbowsTucked && !hangL && !hangR);
  if (!armsOk) {
    return {
      ok: false,
      reason: "Підніміть руки вперед ~45° або закладіть за спину — інакше профіль талії/грудей закритий.",
    };
  }

  const rightArm = checkProfileRightArmAngle(lm, elbowsTucked);
  if (!rightArm.ok) return rightArm;

  const ankleVis = Math.max(lm[27].visibility ?? 0, lm[28].visibility ?? 0);
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст (стопа ближньої ноги в кадрі)." };
  }
  return { ok: true };
}

function checkPoseForStep(viewStep, landmarks) {
  const lm = flipLandmarks(landmarks);
  const fullBody = checkFullBodyVisible(lm, viewStep);
  if (!fullBody.ok) return fullBody;
  return viewStep === 1 ? checkFrontPose(lm) : checkProfilePose(lm);
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
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch {
      poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
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
    lastPoseGate = { ok: false, reason: resChk.reason };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = true;
    return;
  }

  if (poseLoadError) {
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
    lastPoseGate = { ok: false, reason: "Очікування моделі пози…" };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  const result = poseLandmarker.detectForVideo(video, Math.floor(now));
  const lm = result.landmarks && result.landmarks[0];
  if (!lm) {
    lastRawLandmarks = null;
    lastPoseGate = { ok: false, reason: "Людину не видно. Встаньте у рамку повним зростом." };
    setPoseStatus(lastPoseGate.reason, "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
    return;
  }

  lastRawLandmarks = lm;

  const gate = checkPoseForStep(step, lm);
  lastPoseGate = gate;
  if (gate.ok) {
    setPoseStatus("Поза підходить — можна робити фото.", "ok");
    btnCapture.disabled = Boolean(captureTimerIntervalId);
    btnCaptureTimer.disabled = false;
  } else {
    setPoseStatus(gate.reason || "Виправте позу.", "bad");
    btnCapture.disabled = true;
    btnCaptureTimer.disabled = false;
  }
}

function loop() {
  runPoseIfNeeded();
  syncOverlaySize();
  raf = requestAnimationFrame(loop);
}

function updateUiStep() {
  if (step === 1) {
    stepLabel.textContent = "Крок 1 з 2: анфас — A-поза (пахви відкриті, ноги на ширині плечей)";
  } else {
    stepLabel.textContent =
      "Крок 2 з 2: профіль — права рука вперед ~45° або за спину (не вздовж тіла; перевіряється кут правої руки)";
  }
  btnMeasure.disabled = !(frontBlob && sideBlob);
}

async function startCamera() {
  clearCaptureTimer();
  setStatus("");
  await loadPoseLandmarker();
  if (poseLoadError) {
    setStatus("MediaPipe не завантажився: " + (poseLoadError.message || String(poseLoadError)), true);
  }
  try {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
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
  clearCaptureTimer(true);
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
  setStatus("");
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
  // Match mirrored preview (.preview-wrap.mirror) so saved frames match what the user saw.
  cctx.translate(tw, 0);
  cctx.scale(-1, 1);
  cctx.drawImage(video, 0, 0, tw, th);
  captureCanvas.toBlob(
    (blob) => {
      if (!blob) {
        setStatus("Не вдалося створити знімок.", true);
        return;
      }
      callback(blob);
    },
    "image/jpeg",
    0.92
  );
}

function onCaptureReady(blob) {
  const url = URL.createObjectURL(blob);
  if (step === 1) {
    frontBlob = blob;
    thumbFront.src = url;
    thumbFront.hidden = false;
    step = 2;
    updateUiStep();
    setStatus("Увімкніть камеру знову для знімка в профіль.");
    void startCamera();
  } else {
    sideBlob = blob;
    thumbSide.src = url;
    thumbSide.hidden = false;
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
    setStatus("Таймер скасовано.");
    btnCapture.disabled = !lastPoseGate.ok;
    btnCaptureTimer.disabled = !lastPoseGate.ok;
    return;
  }
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
  sectionCapture.scrollIntoView({ behavior: "smooth", block: "start" });
});

btnStart.addEventListener("click", startCamera);
btnStop.addEventListener("click", stopCamera);

heightInput.addEventListener("input", () => {
  if (overlay.width) drawOverlay();
});

btnCapture.addEventListener("click", () => {
  clearCaptureTimer();
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
