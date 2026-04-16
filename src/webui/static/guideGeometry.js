/**
 * Налаштовувані полігони рамки-підказки, якір голови та параметри guideBox.
 * Зберігання: localStorage (ключ STORAGE_KEY). Імпорт/експорт — JSON.
 * Опційно в репозиторії: якщо localStorage порожній, підвантажується OPTIONAL_STATIC_DATA_URL.
 */

export const STORAGE_KEY = "pointsx_guide_v1";

/** Покладіть сюди експортований JSON (version: 1), щоб усі без localStorage отримали ці налаштування. */
export const OPTIONAL_STATIC_DATA_URL = "/static/data/guide-geometry.json";

/** @typedef {{ x: number, y: number }} Anchor */
/** @typedef {[number, number][]} PtList */

export const DEFAULT_GUIDE = {
  version: 1,
  headAnchorFront: { x: 0.5, y: 0.048 },
  headAnchorProfile: { x: 0.524, y: 0.071 },
  footAnchorFront: { x: 0.5, y: 0.992 },
  footAnchorProfile: { x: 0.52, y: 0.994 },
  guideFrame: {
    sidePadRatio: 0.08,
    bodyTopFrac: 0.02,
    marginXRatio: 0.02,
    marginYRatio: 0.02,
    smoothFactor: 0.82,
    smoothBlend: 0.18,
    decayNoLm: 0.88,
    decayLowVis: 0.9,
    noseVisMin: 0.22,
    snapEps: 0.4,
    fhHeightMinCm: 150,
    fhHeightMaxCm: 200,
    fhAtMin: 0.82,
    fhAtMax: 0.92,
    verticalFitMinFrac: 0.5,
    verticalFitMaxFrac: 0.93,
    ankleVisMin: 0.14,
  },
  frontPts: /** @type {PtList} */ ([
    [0.5, 0.005],
    [0.56, 0.015],
    [0.58, 0.04],
    [0.57, 0.075],
    [0.535, 0.095],
    [0.55, 0.115],
    [0.66, 0.13],
    [0.73, 0.155],
    [0.81, 0.19],
    [0.88, 0.225],
    [0.935, 0.26],
    [0.96, 0.29],
    [0.919973544973545, 0.3105329304919909],
    [0.8637566137566138, 0.28431248808161713],
    [0.8108465608465608, 0.26285939883676585],
    [0.7380952380952381, 0.22472057351258584],
    [0.671957671957672, 0.19850013110221207],
    [0.62, 0.2],
    [0.59, 0.28],
    [0.565, 0.37],
    [0.6, 0.43],
    [0.62, 0.47],
    [0.625, 0.54],
    [0.62, 0.62],
    [0.62, 0.7],
    [0.615, 0.78],
    [0.61, 0.855],
    [0.61, 0.91],
    [0.62, 0.95],
    [0.64, 0.975],
    [0.62, 0.99],
    [0.57, 0.99],
    [0.56, 0.96],
    [0.555, 0.91],
    [0.555, 0.855],
    [0.56, 0.78],
    [0.565, 0.7],
    [0.565, 0.62],
    [0.56, 0.54],
    [0.545, 0.47],
    [0.5, 0.45],
    [0.455, 0.47],
    [0.44, 0.54],
    [0.435, 0.62],
    [0.435, 0.7],
    [0.44, 0.78],
    [0.445, 0.855],
    [0.445, 0.91],
    [0.44, 0.96],
    [0.43, 0.99],
    [0.38, 0.99],
    [0.36, 0.975],
    [0.39, 0.95],
    [0.39, 0.91],
    [0.39, 0.855],
    [0.385, 0.78],
    [0.38, 0.7],
    [0.38, 0.62],
    [0.375, 0.54],
    [0.375, 0.47],
    [0.38, 0.43],
    [0.380952380952381, 0.4297167596300534],
    [0.43716931216931215, 0.37012484506102217],
    [0.41, 0.28],
    [0.38, 0.2],
    [0.3412698412698413, 0.18419807160564455],
    [0.26851851851851855, 0.21995322034706333],
    [0.208994708994709, 0.24378998617467584],
    [0.13624338624338625, 0.2747777817505721],
    [0.08664021164021164, 0.30814925390922965],
    [0.04, 0.29],
    [0.065, 0.26],
    [0.12, 0.225],
    [0.19, 0.19],
    [0.27, 0.155],
    [0.34, 0.13],
    [0.45, 0.115],
    [0.465, 0.095],
    [0.43, 0.075],
    [0.42, 0.04],
    [0.44, 0.015],
  ]),
  profilePts: /** @type {PtList} */ ([
    [0.5066137566137566, 0.01973816802536232],
    [0.4636243386243386, 0.028796139039855072],
    [0.5099206349206349, 0.01973816802536232],
    [0.5066137566137566, 0.01973816802536232],
    [0.5529100529100529, 0.012944689764492753],
    [0.5529100529100529, 0.015209182518115942],
    [0.5529100529100529, 0.015209182518115942],
    [0.5529100529100529, 0.015209182518115942],
    [0.5595238095238095, 0.015209182518115942],
    [0.5892857142857143, 0.03558961730072464],
    [0.5992063492063492, 0.05823454483695652],
    [0.6058201058201058, 0.0786149796195652],
    [0.5992063492063492, 0.09446642889492753],
    [0.5859788359788359, 0.1012599071557971],
    [0.5694444444444444, 0.11031787817028985],
    [0.5595238095238095, 0.12843382019927535],
    [0.6058201058201058, 0.14654976222826085],
    [0.6190476190476191, 0.1805171535326087],
    [0.6223544973544973, 0.2178812839673913],
    [0.6223544973544973, 0.26317113903985506],
    [0.6190476190476191, 0.2880805593297101],
    [0.6025132275132276, 0.34242838541666665],
    [0.59, 0.39],
    [0.6, 0.42],
    [0.595, 0.46],
    [0.59, 0.5],
    [0.585, 0.54],
    [0.58, 0.58],
    [0.575, 0.62],
    [0.57, 0.66],
    [0.56, 0.72],
    [0.55, 0.79],
    [0.545, 0.85],
    [0.5363756613756614, 0.8969173698512586],
    [0.5529100529100529, 0.9279051654271548],
    [0.5826719576719577, 0.9538414288949275],
    [0.6521164021164021, 0.9606349071557971],
    [0.671957671957672, 0.9810153419384058],
    [0.6554232804232805, 0.9946022984601449],
    [0.4669312169312169, 0.9900733129528986],
    [0.4669312169312169, 0.9231378122616323],
    [0.4669312169312169, 0.901684723016781],
    [0.445, 0.85],
    [0.45, 0.79],
    [0.45, 0.72],
    [0.44, 0.66],
    [0.435, 0.62],
    [0.44, 0.58],
    [0.445, 0.54],
    [0.45, 0.5],
    [0.46, 0.46],
    [0.465, 0.42],
    [0.47, 0.39],
    [0.46, 0.355],
    [0.4437830687830688, 0.3236431516971778],
    [0.4305555555555555, 0.297422709286804],
    [0.43386243386243384, 0.2640512371281465],
    [0.4305555555555555, 0.23067976496948897],
    [0.4305555555555555, 0.21161035230739894],
    [0.43386243386243384, 0.19015726306254768],
    [0.4437830687830688, 0.1663204972349352],
    [0.4470899470899471, 0.14248373140732265],
    [0.4966931216931217, 0.12164034193840578],
    [0.4636243386243386, 0.10578889266304348],
    [0.4437830687830688, 0.0808794723731884],
    [0.4503968253968254, 0.05144106657608695],
    [0.5033068783068783, 0.024267153532608696],
    [0.5099206349206349, 0.02200266077898551],
    [0.5099206349206349, 0.01973816802536232],
  ]),
  profileArmPts: /** @type {PtList} */ ([
    [0.6223544973544973, 0.18062255673150268],
    [0.876984126984127, 0.27596962004195275],
  ]),
};

function cloneGuide() {
  return structuredClone(DEFAULT_GUIDE);
}

/** Живий стан (імпортери отримують актуальне посилання). */
export let guideGeom = cloneGuide();

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

/**
 * Нормалізовані точки рамки → атрибут `points` для ref-SVG (viewBox 0 0 100×160).
 * Точка (nx, ny) у нормалізованих координатах кадру відео → локальні px у cw×ch (object-fit: cover) — див. videoNormToPreviewLocal.
 */
export function guidePtsToRefSvgPoints(pts, vbW = 100, vbH = 160) {
  return pts
    .map(([x, y]) => {
      const xr = Math.round(x * vbW * 10000) / 10000;
      const yr = Math.round(y * vbH * 10000) / 10000;
      return `${xr},${yr}`;
    })
    .join(" ");
}

export function videoNormToPreviewLocal(nx, ny, cw, ch, vw, vh) {
  const scale = Math.max(cw / vw, ch / vh);
  const dispW = vw * scale;
  const dispH = vh * scale;
  const offX = (cw - dispW) / 2;
  const offY = (ch - dispH) / 2;
  return { x: nx * vw * scale + offX, y: ny * vh * scale + offY };
}

export function frameHeightFromCm(heightCm, gf = guideGeom.guideFrame) {
  const h = Number(heightCm);
  if (!Number.isFinite(h)) return gf.fhAtMax;
  const t = Math.min(1, Math.max(0, (h - gf.fhHeightMinCm) / (gf.fhHeightMaxCm - gf.fhHeightMinCm)));
  return gf.fhAtMin + t * (gf.fhAtMax - gf.fhAtMin);
}

export function computeGuideBox(cssW, cssH, heightCmStr, geom = guideGeom) {
  const gf = geom.guideFrame;
  const fh = frameHeightFromCm(heightCmStr, gf);
  const bodyH = cssH * fh;
  const top = (cssH - bodyH) * gf.bodyTopFrac;
  const sidePad = cssW * gf.sidePadRatio;
  return { left: sidePad, top, width: cssW - 2 * sidePad, height: bodyH };
}

/** Найнижча видима щиколотка в px прев’ю (27/28). */
export function anklePreviewBottomY(lm, cssW, cssH, vw, vh, visMin = 0.14) {
  let best = -Infinity;
  for (const id of [27, 28]) {
    const p = lm[id];
    if (!p || (p.visibility ?? 0) < visMin) continue;
    const { y } = videoNormToPreviewLocal(p.x, p.y, cssW, cssH, vw, vh);
    best = Math.max(best, y);
  }
  return best > -Infinity ? best : null;
}

/**
 * Рамка: ширина як раніше; висота й положення за носом + щиколотками (якір голови / якір стопи в силуеті).
 * Мутує smoothDelta (dx, dy у режимі без щиколоток; fitHeight/fitTop/fitLeft при вертикальній підгонці).
 */
export function computeGuideBoxTracked(
  cssW,
  cssH,
  vw,
  vh,
  step,
  lastRawLandmarks,
  smoothDelta,
  heightCmStr,
  geom = guideGeom
) {
  const base = computeGuideBox(cssW, cssH, heightCmStr, geom);
  const gf = geom.guideFrame;
  const marginX = cssW * gf.marginXRatio;
  const marginY = cssH * gf.marginYRatio;
  const headA = step === 1 ? geom.headAnchorFront : geom.headAnchorProfile;
  const footA = step === 1 ? geom.footAnchorFront : geom.footAnchorProfile;
  const w = base.width;
  const vMin = gf.ankleVisMin ?? 0.14;
  const hMinF = gf.verticalFitMinFrac ?? 0.5;
  const hMaxF = gf.verticalFitMaxFrac ?? 0.93;

  function decayLegacyAndMaybeFit() {
    smoothDelta.dx *= gf.decayNoLm;
    smoothDelta.dy *= gf.decayNoLm;
    if (Math.abs(smoothDelta.dx) < gf.snapEps) smoothDelta.dx = 0;
    if (Math.abs(smoothDelta.dy) < gf.snapEps) smoothDelta.dy = 0;
    if (smoothDelta.fitHeight != null) {
      smoothDelta.fitHeight = smoothDelta.fitHeight * 0.88 + base.height * 0.12;
      smoothDelta.fitTop = smoothDelta.fitTop * 0.88 + base.top * 0.12;
      smoothDelta.fitLeft = smoothDelta.fitLeft * 0.88 + base.left * 0.12;
      if (
        Math.abs(smoothDelta.fitHeight - base.height) < 2 &&
        Math.abs(smoothDelta.fitTop - base.top) < 2 &&
        Math.abs(smoothDelta.fitLeft - base.left) < 2
      ) {
        smoothDelta.fitHeight = null;
        smoothDelta.fitTop = null;
        smoothDelta.fitLeft = null;
      }
      return {
        left: smoothDelta.fitLeft,
        top: smoothDelta.fitTop,
        width: w,
        height: smoothDelta.fitHeight,
      };
    }
    return {
      ...base,
      left: base.left + smoothDelta.dx,
      top: base.top + smoothDelta.dy,
    };
  }

  if (!lastRawLandmarks || !lastRawLandmarks[0] || !vw || !vh) {
    return decayLegacyAndMaybeFit();
  }

  const nose = lastRawLandmarks[0];
  if ((nose.visibility ?? 1) < gf.noseVisMin) {
    smoothDelta.dx *= gf.decayLowVis;
    smoothDelta.dy *= gf.decayLowVis;
    return decayLegacyAndMaybeFit();
  }

  const { x: px, y: py } = videoNormToPreviewLocal(nose.x, nose.y, cssW, cssH, vw, vh);
  const ankleY = anklePreviewBottomY(lastRawLandmarks, cssW, cssH, vw, vh, vMin);
  const denom = footA.y - headA.y;

  const canStretch = ankleY != null && denom > 0.055 && ankleY > py + cssH * 0.07;

  if (canStretch) {
    let H = (ankleY - py) / denom;
    H = clamp(H, cssH * hMinF, cssH * hMaxF);
    let top = py - headA.y * H;
    let left = px - headA.x * w;
    top = clamp(top, marginY, cssH - marginY - H);
    left = clamp(left, marginX, cssW - marginX - w);

    if (smoothDelta.fitHeight == null) {
      smoothDelta.fitHeight = H;
      smoothDelta.fitTop = top;
      smoothDelta.fitLeft = left;
    } else {
      smoothDelta.fitHeight = smoothDelta.fitHeight * gf.smoothFactor + H * gf.smoothBlend;
      smoothDelta.fitTop = smoothDelta.fitTop * gf.smoothFactor + top * gf.smoothBlend;
      smoothDelta.fitLeft = smoothDelta.fitLeft * gf.smoothFactor + left * gf.smoothBlend;
    }
    smoothDelta.dx = 0;
    smoothDelta.dy = 0;
    return {
      left: smoothDelta.fitLeft,
      top: smoothDelta.fitTop,
      width: w,
      height: smoothDelta.fitHeight,
    };
  }

  if (smoothDelta.fitHeight != null) {
    smoothDelta.fitHeight = smoothDelta.fitHeight * 0.9 + base.height * 0.1;
    smoothDelta.fitTop = smoothDelta.fitTop * 0.9 + base.top * 0.1;
    smoothDelta.fitLeft = smoothDelta.fitLeft * 0.9 + base.left * 0.1;
  } else {
    const targetX = base.left + headA.x * w;
    const targetY = base.top + headA.y * base.height;
    let rawLeft = base.left + (px - targetX);
    let rawTop = base.top + (py - targetY);
    rawLeft = clamp(rawLeft, marginX, cssW - marginX - w);
    rawTop = clamp(rawTop, marginY, cssH - marginY - base.height);
    const tgtDx = rawLeft - base.left;
    const tgtDy = rawTop - base.top;
    smoothDelta.dx = smoothDelta.dx * gf.smoothFactor + tgtDx * gf.smoothBlend;
    smoothDelta.dy = smoothDelta.dy * gf.smoothFactor + tgtDy * gf.smoothBlend;
  }

  if (smoothDelta.fitHeight != null) {
    if (Math.abs(smoothDelta.fitHeight - base.height) < 3) {
      smoothDelta.fitHeight = null;
      smoothDelta.fitTop = null;
      smoothDelta.fitLeft = null;
    } else {
      return {
        left: smoothDelta.fitLeft,
        top: smoothDelta.fitTop,
        width: w,
        height: smoothDelta.fitHeight,
      };
    }
  }

  return {
    ...base,
    left: base.left + smoothDelta.dx,
    top: base.top + smoothDelta.dy,
  };
}

/** Поточна рамка на екрані без додаткового кроку згладжування (для знімка = останній кадр прев’ю). */
export function computeGuideBoxWithDelta(cssW, cssH, heightCmStr, smoothDelta, geom = guideGeom) {
  const base = computeGuideBox(cssW, cssH, heightCmStr, geom);
  if (smoothDelta.fitHeight != null && smoothDelta.fitTop != null && smoothDelta.fitLeft != null) {
    return {
      left: smoothDelta.fitLeft,
      top: smoothDelta.fitTop,
      width: base.width,
      height: smoothDelta.fitHeight,
    };
  }
  return {
    ...base,
    left: base.left + smoothDelta.dx,
    top: base.top + smoothDelta.dy,
  };
}

export function drawPoly(ctx, pts, box) {
  if (!pts.length) return;
  ctx.beginPath();
  ctx.moveTo(box.left + pts[0][0] * box.width, box.top + pts[0][1] * box.height);
  for (let i = 1; i < pts.length; i++)
    ctx.lineTo(box.left + pts[i][0] * box.width, box.top + pts[i][1] * box.height);
  ctx.closePath();
}

export function drawOpenStroke(ctx, pts, box) {
  if (pts.length < 2) return;
  ctx.beginPath();
  ctx.moveTo(box.left + pts[0][0] * box.width, box.top + pts[0][1] * box.height);
  for (let i = 1; i < pts.length; i++)
    ctx.lineTo(box.left + pts[i][0] * box.width, box.top + pts[i][1] * box.height);
  ctx.stroke();
}

/**
 * Малює силует на canvas, де вже активний той самий mirror transform, що й для drawImage(video).
 * @param {{ left: number, top: number, width: number, height: number }} box — з computeGuideBoxTracked або computeGuideBoxWithDelta
 */
export function drawGuideSilhouetteOnCanvas(ctx, box, currentStep, geom = guideGeom) {
  const cw = ctx.canvas && ctx.canvas.width ? ctx.canvas.width : box.width;
  ctx.strokeStyle = "rgba(91, 140, 255, 0.92)";
  ctx.lineWidth = Math.max(2, cw * 0.014);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash([8, 6]);
  if (currentStep === 1) {
    drawPoly(ctx, geom.frontPts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.07)";
    ctx.fill();
  } else {
    drawPoly(ctx, geom.profilePts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.07)";
    ctx.fill();
    ctx.setLineDash([5, 5]);
    ctx.strokeStyle = "rgba(120, 170, 255, 0.95)";
    drawOpenStroke(ctx, geom.profileArmPts, box);
    ctx.setLineDash([]);
  }
}

function mergeGuideFrame(target, src) {
  if (!src || typeof src !== "object") return;
  for (const k of Object.keys(target)) {
    if (k in src && typeof src[k] === "number" && Number.isFinite(src[k])) {
      target[k] = src[k];
    }
  }
}

function isPtList(x, minPts = 3) {
  return (
    Array.isArray(x) &&
    x.length >= minPts &&
    x.every((p) => Array.isArray(p) && p.length === 2 && p.every((n) => typeof n === "number" && Number.isFinite(n)))
  );
}

export function applyGuidePayload(o) {
  if (!o || o.version !== 1) return false;
  if (o.headAnchorFront && typeof o.headAnchorFront.x === "number" && typeof o.headAnchorFront.y === "number") {
    guideGeom.headAnchorFront = { x: o.headAnchorFront.x, y: o.headAnchorFront.y };
  }
  if (o.headAnchorProfile && typeof o.headAnchorProfile.x === "number" && typeof o.headAnchorProfile.y === "number") {
    guideGeom.headAnchorProfile = { x: o.headAnchorProfile.x, y: o.headAnchorProfile.y };
  }
  if (o.footAnchorFront && typeof o.footAnchorFront.x === "number" && typeof o.footAnchorFront.y === "number") {
    guideGeom.footAnchorFront = { x: o.footAnchorFront.x, y: o.footAnchorFront.y };
  }
  if (o.footAnchorProfile && typeof o.footAnchorProfile.x === "number" && typeof o.footAnchorProfile.y === "number") {
    guideGeom.footAnchorProfile = { x: o.footAnchorProfile.x, y: o.footAnchorProfile.y };
  }
  if (o.guideFrame && typeof o.guideFrame === "object") {
    mergeGuideFrame(guideGeom.guideFrame, o.guideFrame);
  }
  if (isPtList(o.frontPts)) guideGeom.frontPts = o.frontPts.map((p) => [p[0], p[1]]);
  if (isPtList(o.profilePts)) guideGeom.profilePts = o.profilePts.map((p) => [p[0], p[1]]);
  if (isPtList(o.profileArmPts, 2)) {
    guideGeom.profileArmPts = o.profileArmPts.map((p) => [p[0], p[1]]);
  }
  return true;
}

export function loadGuideGeometry() {
  try {
    if (typeof localStorage === "undefined") return;
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return;
    const o = JSON.parse(raw);
    applyGuidePayload(o);
  } catch {
    // ignore
  }
}

/**
 * Якщо в браузері ще немає збереженої рамки — пробує завантажити JSON з репозиторію.
 * @returns {Promise<boolean>} чи застосовано файл
 */
export async function loadGuideGeometryFromOptionalStaticFile() {
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem(STORAGE_KEY)) return false;
    if (typeof fetch === "undefined") return false;
    const res = await fetch(OPTIONAL_STATIC_DATA_URL, { cache: "no-cache" });
    if (!res.ok) return false;
    const o = await res.json();
    return applyGuidePayload(o);
  } catch {
    return false;
  }
}

/** Завжди новий запит (для прикладів на початку сторінки після зміни файлу на сервері). */
export function saveGuideGeometry() {
  if (typeof localStorage === "undefined") return;
  const payload = {
    version: guideGeom.version,
    headAnchorFront: guideGeom.headAnchorFront,
    headAnchorProfile: guideGeom.headAnchorProfile,
    footAnchorFront: guideGeom.footAnchorFront,
    footAnchorProfile: guideGeom.footAnchorProfile,
    guideFrame: { ...guideGeom.guideFrame },
    frontPts: guideGeom.frontPts.map((p) => [p[0], p[1]]),
    profilePts: guideGeom.profilePts.map((p) => [p[0], p[1]]),
    profileArmPts: guideGeom.profileArmPts.map((p) => [p[0], p[1]]),
  };
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

export function resetGuideGeometry() {
  guideGeom = cloneGuide();
  if (typeof localStorage !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export function exportGuideGeometryJson() {
  return JSON.stringify(
    {
      version: guideGeom.version,
      headAnchorFront: guideGeom.headAnchorFront,
      headAnchorProfile: guideGeom.headAnchorProfile,
      footAnchorFront: guideGeom.footAnchorFront,
      footAnchorProfile: guideGeom.footAnchorProfile,
      guideFrame: { ...guideGeom.guideFrame },
      frontPts: guideGeom.frontPts,
      profilePts: guideGeom.profilePts,
      profileArmPts: guideGeom.profileArmPts,
    },
    null,
    2
  );
}

export function importGuideGeometryJson(str) {
  try {
    const o = JSON.parse(str);
    return applyGuidePayload(o);
  } catch {
    return false;
  }
}
