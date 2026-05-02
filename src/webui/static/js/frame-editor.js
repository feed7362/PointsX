/**
 * Canvas editor for guide silhouette polygons, anchors, and guideFrame parameters.
 */

import {
  guideGeom,
  applyReloadGuideQueryParam,
  loadGuideGeometry,
  loadGuideGeometryFromOptionalStaticFile,
  saveGuideGeometry,
  resetGuideGeometry,
  exportGuideGeometryJson,
  importGuideGeometryJson,
  computeGuideBox,
  drawPoly,
} from "./guideGeometry.js";

const MIN_MAIN_POLY = 3;
const EDGE_HIT_PX = 12;

const cv = document.getElementById("cv");
const ctx = cv.getContext("2d");
const msg = document.getElementById("msg");
const heightPreview = document.getElementById("height-preview");
const step1 = document.getElementById("step1");
const step2 = document.getElementById("step2");
const haFx = document.getElementById("ha-fx");
const haFy = document.getElementById("ha-fy");
const haPx = document.getElementById("ha-px");
const haPy = document.getElementById("ha-py");
const faFx = document.getElementById("fa-fx");
const faFy = document.getElementById("fa-fy");
const faPx = document.getElementById("fa-px");
const faPy = document.getElementById("fa-py");
const gfFields = document.getElementById("gf-fields");

cv.setAttribute("tabindex", "0");

function setMsg(text, isErr) {
  msg.textContent = text || "";
  msg.classList.toggle("err", Boolean(isErr));
}

applyReloadGuideQueryParam();
loadGuideGeometry();
void loadGuideGeometryFromOptionalStaticFile().then((fromFile) => {
  if (fromFile) {
    syncFormFromGeom();
    draw();
  }
});

/** @type {number} */
let editorStep = 1;
/** @type {number | null} */
let dragIdx = null;
/** @type {number | null} */
let selectedVertex = null;

const GF_KEYS = [
  ["sidePadRatio", "Бокові поля (частка ширини)"],
  ["bodyTopFrac", "Відступ зверху (частка)"],
  ["marginXRatio", "Маржа X для зсуву"],
  ["marginYRatio", "Маржа Y для зсуву"],
  ["smoothFactor", "Згладжування пози (0–1)"],
  ["smoothBlend", "Частка нової позиції"],
  ["decayNoLm", "Згасання без лендмарків"],
  ["decayLowVis", "Згасання при низькій видимості носа"],
  ["noseVisMin", "Мін. видимість носа"],
  ["snapEps", "Поріг «нуля» зсуву (px)"],
  ["fhHeightMinCm", "Зріст мін (см) для висоти рамки"],
  ["fhHeightMaxCm", "Зріст макс (см)"],
  ["fhAtMin", "Висота тіла при мін. зрості (0–1)"],
  ["fhAtMax", "Висота тіла при макс. зрості (0–1)"],
  ["verticalFitMinFrac", "Підгонка: мін. висота рамки (частка екрану)"],
  ["verticalFitMaxFrac", "Підгонка: макс. висота рамки"],
  ["ankleVisMin", "Мін. видимість щиколотки для підгонки"],
];

/** Push guide geometry into the anchor and guideFrame form fields. */
function syncFormFromGeom() {
  haFx.value = String(guideGeom.headAnchorFront.x);
  haFy.value = String(guideGeom.headAnchorFront.y);
  haPx.value = String(guideGeom.headAnchorProfile.x);
  haPy.value = String(guideGeom.headAnchorProfile.y);
  faFx.value = String(guideGeom.footAnchorFront.x);
  faFy.value = String(guideGeom.footAnchorFront.y);
  faPx.value = String(guideGeom.footAnchorProfile.x);
  faPy.value = String(guideGeom.footAnchorProfile.y);
  for (const el of gfFields.querySelectorAll("input[data-gf]")) {
    const k = el.getAttribute("data-gf");
    if (k && k in guideGeom.guideFrame) {
      el.value = String(guideGeom.guideFrame[k]);
    }
  }
}

/** Read head anchor inputs into guideGeom when all values are finite. */
function readAnchorsFromForm() {
  const fx = Number(haFx.value);
  const fy = Number(haFy.value);
  const px = Number(haPx.value);
  const py = Number(haPy.value);
  if ([fx, fy, px, py].every((n) => Number.isFinite(n))) {
    guideGeom.headAnchorFront = { x: fx, y: fy };
    guideGeom.headAnchorProfile = { x: px, y: py };
  }
  const ffx = Number(faFx.value);
  const ffy = Number(faFy.value);
  const fpx = Number(faPx.value);
  const fpy = Number(faPy.value);
  if ([ffx, ffy, fpx, fpy].every((n) => Number.isFinite(n))) {
    guideGeom.footAnchorFront = { x: ffx, y: ffy };
    guideGeom.footAnchorProfile = { x: fpx, y: fpy };
  }
}

/** Read numeric guideFrame fields from the dynamic form. */
function readGuideFrameFromForm() {
  for (const el of gfFields.querySelectorAll("input[data-gf]")) {
    const k = el.getAttribute("data-gf");
    const v = Number(el.value);
    if (k && Number.isFinite(v) && k in guideGeom.guideFrame) {
      guideGeom.guideFrame[k] = v;
    }
  }
}

/** Build guideFrame labeled inputs from GF_KEYS. */
function buildGfFields() {
  gfFields.innerHTML = "";
  for (const [key, label] of GF_KEYS) {
    const lab = document.createElement("label");
    lab.textContent = label;
    const inp = document.createElement("input");
    inp.type = "number";
    inp.step = "0.001";
    inp.setAttribute("data-gf", key);
    lab.appendChild(inp);
    gfFields.appendChild(lab);
  }
}

/** Active main polygon: front (step 1) or profile (step 2). */
function currentMainPoly() {
  return editorStep === 1 ? guideGeom.frontPts : guideGeom.profilePts;
}

/** Map pointer client coordinates to canvas pixel space. */
function clientToCanvas(clientX, clientY) {
  const r = cv.getBoundingClientRect();
  const sx = cv.width / r.width;
  const sy = cv.height / r.height;
  return { x: (clientX - r.left) * sx, y: (clientY - r.top) * sy };
}

/** Hit-test a vertex on the main polygon (front or profile). */
function pickVertex(mx, my) {
  const h = heightPreview.value;
  const box = computeGuideBox(cv.width, cv.height, h);
  const pts = currentMainPoly();
  let best = -1;
  let bestD = 196;
  for (let i = 0; i < pts.length; i++) {
    const px = box.left + pts[i][0] * box.width;
    const py = box.top + pts[i][1] * box.height;
    const d = (mx - px) ** 2 + (my - py) ** 2;
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  if (best >= 0 && bestD < 14 * 14) return best;
  return null;
}

/** Closest point on segment AB to M; returns distance squared and point. */
function closestOnSeg(mx, my, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const amx = mx - ax;
  const amy = my - ay;
  const len2 = abx * abx + aby * aby || 1e-12;
  let t = (amx * abx + amy * aby) / len2;
  t = Math.max(0, Math.min(1, t));
  const px = ax + t * abx;
  const py = ay + t * aby;
  const dx = mx - px;
  const dy = my - py;
  return { distSq: dx * dx + dy * dy, px, py };
}

/** Find edge insert position on a closed polygon near (mx, my). */
function findInsertClosed(pts, mx, my, box, threshPx) {
  const thr2 = threshPx * threshPx;
  const n = pts.length;
  let bestI = -1;
  let bestD = Infinity;
  let bestPx = 0;
  let bestPy = 0;
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    const ax = box.left + pts[i][0] * box.width;
    const ay = box.top + pts[i][1] * box.height;
    const bx = box.left + pts[j][0] * box.width;
    const by = box.top + pts[j][1] * box.height;
    const { distSq, px, py } = closestOnSeg(mx, my, ax, ay, bx, by);
    if (distSq < bestD) {
      bestD = distSq;
      bestI = i;
      bestPx = px;
      bestPy = py;
    }
  }
  if (bestI < 0 || bestD > thr2) return null;
  const nx = Math.min(1, Math.max(0, (bestPx - box.left) / box.width));
  const ny = Math.min(1, Math.max(0, (bestPy - box.top) / box.height));
  return { afterIdx: bestI, nx, ny };
}

/**
 * Shift-click: insert a point on the nearest edge of the main closed polygon.
 */
function tryInsertShiftClick(mx, my) {
  const h = heightPreview.value;
  const box = computeGuideBox(cv.width, cv.height, h);
  const pts = editorStep === 1 ? guideGeom.frontPts : guideGeom.profilePts;
  const r = findInsertClosed(pts, mx, my, box, EDGE_HIT_PX);
  if (!r) return false;
  pts.splice(r.afterIdx + 1, 0, [r.nx, r.ny]);
  return true;
}

/** Delete the selected vertex if counts stay above minimums. */
function removeSelectedVertex() {
  if (selectedVertex == null) {
    setMsg("Спочатку клацніть по точці на контуру, щоб її виділити.", true);
    return false;
  }
  const pts = currentMainPoly();
  if (pts.length <= MIN_MAIN_POLY) {
    setMsg(`Мінімум ${MIN_MAIN_POLY} точки для замкненого контуру.`, true);
    return false;
  }
  pts.splice(selectedVertex, 1);
  selectedVertex = null;
  return true;
}

/** Redraw the canvas from form state and guideGeom. */
function draw() {
  readAnchorsFromForm();
  readGuideFrameFromForm();
  const h = heightPreview.value;
  const box = computeGuideBox(cv.width, cv.height, h);
  ctx.fillStyle = "#121722";
  ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.strokeStyle = "rgba(91, 140, 255, 0.85)";
  ctx.lineWidth = 2;
  ctx.strokeRect(box.left + 0.5, box.top + 0.5, box.width - 1, box.height - 1);

  ctx.strokeStyle = "rgba(91, 140, 255, 0.92)";
  ctx.lineWidth = Math.max(2, cv.width * 0.014);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash([8, 6]);
  if (editorStep === 1) {
    drawPoly(ctx, guideGeom.frontPts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.08)";
    ctx.fill();
  } else {
    drawPoly(ctx, guideGeom.profilePts, box);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(91, 140, 255, 0.08)";
    ctx.fill();
  }

  const pts = currentMainPoly();
  for (let i = 0; i < pts.length; i++) {
    const px = box.left + pts[i][0] * box.width;
    const py = box.top + pts[i][1] * box.height;
    const sel = selectedVertex != null && selectedVertex === i;
    ctx.beginPath();
    ctx.arc(px, py, sel ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = sel ? "#7dffb3" : "#fff";
    ctx.fill();
    if (sel) {
      ctx.strokeStyle = "#0a3";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  const ax = box.left + guideGeom.headAnchorFront.x * box.width;
  const ay = box.top + guideGeom.headAnchorFront.y * box.height;
  const bx = box.left + guideGeom.headAnchorProfile.x * box.width;
  const by = box.top + guideGeom.headAnchorProfile.y * box.height;
  ctx.fillStyle = editorStep === 1 ? "rgba(255,100,120,0.9)" : "rgba(255,100,120,0.35)";
  ctx.beginPath();
  ctx.arc(ax, ay, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = editorStep === 2 ? "rgba(255,100,120,0.9)" : "rgba(255,100,120,0.35)";
  ctx.beginPath();
  ctx.arc(bx, by, 5, 0, Math.PI * 2);
  ctx.fill();

  const f1x = box.left + guideGeom.footAnchorFront.x * box.width;
  const f1y = box.top + guideGeom.footAnchorFront.y * box.height;
  const f2x = box.left + guideGeom.footAnchorProfile.x * box.width;
  const f2y = box.top + guideGeom.footAnchorProfile.y * box.height;
  ctx.fillStyle = editorStep === 1 ? "rgba(120,220,255,0.95)" : "rgba(120,220,255,0.35)";
  ctx.beginPath();
  ctx.arc(f1x, f1y, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = editorStep === 2 ? "rgba(120,220,255,0.95)" : "rgba(120,220,255,0.35)";
  ctx.beginPath();
  ctx.arc(f2x, f2y, 5, 0, Math.PI * 2);
  ctx.fill();
}

buildGfFields();
syncFormFromGeom();
draw();

for (const el of [
  haFx,
  haFy,
  haPx,
  haPy,
  faFx,
  faFy,
  faPx,
  faPy,
  heightPreview,
  ...gfFields.querySelectorAll("input"),
]) {
  el.addEventListener("input", () => draw());
}

step1.addEventListener("change", () => {
  if (step1.checked) {
    editorStep = 1;
    selectedVertex = null;
    draw();
  }
});
step2.addEventListener("change", () => {
  if (step2.checked) {
    editorStep = 2;
    selectedVertex = null;
    draw();
  }
});

cv.addEventListener("mousedown", (e) => {
  const { x, y } = clientToCanvas(e.clientX, e.clientY);
  if (e.shiftKey) {
    if (tryInsertShiftClick(x, y)) {
      setMsg("Точку додано на ребрі.");
      draw();
    } else {
      setMsg("Немає ребра поруч (спробуйте ближче до лінії контуру).", true);
    }
    return;
  }
  const hit = pickVertex(x, y);
  if (hit != null) selectedVertex = hit;
  else selectedVertex = null;
  if (hit == null) return;
  dragIdx = hit;
  draw();
});

window.addEventListener("mousemove", (e) => {
  if (dragIdx == null) return;
  const { x, y } = clientToCanvas(e.clientX, e.clientY);
  const h = heightPreview.value;
  const box = computeGuideBox(cv.width, cv.height, h);
  let nx = (x - box.left) / box.width;
  let ny = (y - box.top) / box.height;
  nx = Math.min(1, Math.max(0, nx));
  ny = Math.min(1, Math.max(0, ny));
  const pts = currentMainPoly();
  pts[dragIdx][0] = nx;
  pts[dragIdx][1] = ny;
  draw();
});

window.addEventListener("mouseup", () => {
  dragIdx = null;
});

window.addEventListener("keydown", (e) => {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if (e.key === "Delete" || e.key === "Backspace") {
    if (removeSelectedVertex()) {
      e.preventDefault();
      setMsg("Точку видалено.");
      draw();
    }
  }
});

document.getElementById("btn-del-point").addEventListener("click", () => {
  if (removeSelectedVertex()) {
    setMsg("Точку видалено.");
    draw();
  }
});

document.getElementById("btn-save").addEventListener("click", () => {
  readAnchorsFromForm();
  readGuideFrameFromForm();
  saveGuideGeometry();
  setMsg("Збережено в localStorage. Оновіть головну сторінку, щоб підтягнулось у зйомці.");
});

document.getElementById("btn-reset").addEventListener("click", () => {
  if (!confirm("Скинути всі точки та параметри до типових?")) return;
  resetGuideGeometry();
  selectedVertex = null;
  syncFormFromGeom();
  draw();
  setMsg("Скинуто до типових (localStorage очищено).");
});

document.getElementById("btn-export").addEventListener("click", () => {
  readAnchorsFromForm();
  readGuideFrameFromForm();
  const blob = new Blob([exportGuideGeometryJson()], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "guide-geometry.json";
  a.click();
  URL.revokeObjectURL(a.href);
  setMsg("Файл завантажено (guide-geometry.json).");
});

document.getElementById("file-import").addEventListener("change", async (e) => {
  const f = e.target.files && e.target.files[0];
  e.target.value = "";
  if (!f) return;
  try {
    const text = await f.text();
    const ok = importGuideGeometryJson(text);
    if (!ok) {
      setMsg("Файл не підходить (очікується JSON з version: 1).", true);
      return;
    }
    selectedVertex = null;
    syncFormFromGeom();
    draw();
    setMsg("Імпортовано з файлу (ще не збережено в браузері — натисніть «Зберегти»).");
  } catch (err) {
    setMsg("Помилка імпорту: " + (err && err.message ? err.message : String(err)), true);
  }
});
