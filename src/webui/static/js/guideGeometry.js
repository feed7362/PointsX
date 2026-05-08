/**
 * Configurable guide-frame polygons, head/foot anchors, and guideBox parameters.
 * Persistence: localStorage (STORAGE_KEY). Import/export as JSON.
 * Optional repo file: fetched only when localStorage has no saved geometry (see applyReloadGuideQueryParam).
 */

import guideDefaultPayload from "./guide-default.json" with { type: "json" };

export const STORAGE_KEY = "pointsx_guide_v2";

/**
 * If the page URL contains `reloadGuide=1`, clears saved geometry so the next load pulls from
 * OPTIONAL_STATIC_DATA_URL (browser cache may still apply; use hard refresh if needed).
 * Call once before loadGuideGeometry().
 */
export function applyReloadGuideQueryParam() {
  if (typeof localStorage === "undefined" || typeof location === "undefined") return;
  try {
    const q = new URLSearchParams(location.search);
    if (q.get("reloadGuide") === "1") {
      localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    /* ignore */
  }
}

/** Optional static JSON (version: 1) served with the app when users have no localStorage entry. */
export const OPTIONAL_STATIC_DATA_URL = "/static/data/guide-geometry.json";

/** @typedef {{ x: number, y: number }} Anchor */
/** @typedef {[number, number][]} PtList */

/**
 * Built-in defaults from `./guide-default.json` (copied from `static/data/guide-geometry.json`).
 * After editing the data file, sync: `cp src/webui/static/data/guide-geometry.json src/webui/static/js/guide-default.json`
 */
export const DEFAULT_GUIDE = structuredClone(guideDefaultPayload);

function cloneGuide() {
  return structuredClone(DEFAULT_GUIDE);
}

/** Live mutable geometry; importers should use this binding. */
export let guideGeom = cloneGuide();

function clamp(n, lo, hi) {
  return Math.min(hi, Math.max(lo, n));
}

/**
 * Convert normalized frame points to an SVG `points` attribute (ref SVG viewBox 0 0 100×160).
 * Maps video-normalized (nx, ny) through object-fit: cover; see videoNormToPreviewLocal.
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

function _mapGuidePtsToSvgSpace(pts, vbW, vbH) {
  return pts.map(([x, y]) => [x * vbW, y * vbH]);
}

/**
 * Smooth closed contour path for SVG `<path d="...">` (matches canvas smoothing).
 */
export function guidePtsToRefSvgPathClosed(pts, vbW = 100, vbH = 160) {
  if (!Array.isArray(pts) || pts.length < 3) return "";
  const mapped = _mapGuidePtsToSvgSpace(pts, vbW, vbH);
  const first = mapped[0];
  const second = mapped[1];
  const sx = (first[0] + second[0]) * 0.5;
  const sy = (first[1] + second[1]) * 0.5;
  let d = `M ${sx.toFixed(4)} ${sy.toFixed(4)}`;
  for (let i = 1; i < mapped.length; i++) {
    const cur = mapped[i];
    const next = mapped[(i + 1) % mapped.length];
    const mx = (cur[0] + next[0]) * 0.5;
    const my = (cur[1] + next[1]) * 0.5;
    d += ` Q ${cur[0].toFixed(4)} ${cur[1].toFixed(4)} ${mx.toFixed(4)} ${my.toFixed(4)}`;
  }
  return `${d} Z`;
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

/**
 * Horizontal map for profile silhouette + anchors: optional X scale about a pivot (default 1.3).
 * Pivot defaults to silhouette midline (~0.5); optional shift nudges the whole figure left/right in norm space.
 */
export function profileNormX(x, gf = guideGeom.guideFrame) {
  const rawS = Number(gf.profileDrawScaleX);
  const scale = Number.isFinite(rawS) && rawS > 0.05 ? rawS : 1.3;
  const pv = Number(gf.profilePivotX);
  const sh = Number(gf.profileDrawShiftNormX);
  const pivot = (Number.isFinite(pv) ? pv : 0.5) + (Number.isFinite(sh) ? sh : 0);
  if (Math.abs(scale - 1) < 1e-6) return x;
  return pivot + (x - pivot) * scale;
}

/** Inverse of `profileNormX`: map silhouette/box-normalized X back to stored `profilePts` X. */
export function profileDenormX(mappedX, gf = guideGeom.guideFrame) {
  const rawS = Number(gf.profileDrawScaleX);
  const scale = Number.isFinite(rawS) && rawS > 0.05 ? rawS : 1.3;
  const pv = Number(gf.profilePivotX);
  const sh = Number(gf.profileDrawShiftNormX);
  const pivot = (Number.isFinite(pv) ? pv : 0.5) + (Number.isFinite(sh) ? sh : 0);
  if (Math.abs(scale - 1) < 1e-6) return mappedX;
  return pivot + (mappedX - pivot) / scale;
}

/** Min/max mapped X of profile art (for clamping box position so silhouette stays inside the canvas). */
function profileMappedXExtent(geom, gf) {
  if (!geom?.profilePts?.length) return null;
  let minX = Infinity;
  let maxX = -Infinity;
  for (const p of geom.profilePts) {
    const mx = profileNormX(p[0], gf);
    minX = Math.min(minX, mx);
    maxX = Math.max(maxX, mx);
  }
  const paths = geom.profileInteriorPaths;
  if (Array.isArray(paths)) {
    for (const path of paths) {
      if (!path?.length) continue;
      for (const p of path) {
        const mx = profileNormX(p[0], gf);
        minX = Math.min(minX, mx);
        maxX = Math.max(maxX, mx);
      }
    }
  }
  if (!Number.isFinite(minX) || !Number.isFinite(maxX)) return null;
  return { minX, maxX };
}

/**
 * Horizontal clamp: letterbox-aware range (aligns with video crop) intersected on profile step with
 * silhouette bounds so strokes stay on-screen without using the full box width as the stop edge.
 */
function clampGuideBoxLeft(rawLeft, cssW, cssH, vw, vh, w, headAx, footAx, marginX, step, geom) {
  if (!Number.isFinite(rawLeft) || !Number.isFinite(w) || w <= 0) return rawLeft;
  if (step !== 2) {
    return clamp(rawLeft, marginX, cssW - marginX - w);
  }

  let lo;
  let hi;
  if (!vw || !vh || vw <= 0 || vh <= 0) {
    lo = marginX;
    hi = cssW - marginX - w;
  } else {
    const scale = Math.max(cssW / vw, cssH / vh);
    const dispW = vw * scale;
    const offX = (cssW - dispW) / 2;
    const axHi = Math.max(headAx, footAx);
    const axLo = Math.min(headAx, footAx);
    const leftMinAlign = offX - axHi * w;
    const leftMaxAlign = offX + dispW - axLo * w;
    const minL = Math.min(marginX, leftMinAlign);
    const maxL = Math.max(cssW - marginX - w, leftMaxAlign);
    lo = Math.min(minL, maxL);
    hi = Math.max(minL, maxL);
  }

  if (step === 2 && geom) {
    const ext = profileMappedXExtent(geom, geom.guideFrame);
    if (ext) {
      const visualLo = marginX - ext.minX * w;
      const visualHi = cssW - marginX - ext.maxX * w;
      const nLo = Math.max(lo, visualLo);
      const nHi = Math.min(hi, visualHi);
      if (nLo <= nHi) {
        lo = nLo;
        hi = nHi;
      } else {
        lo = visualLo;
        hi = visualHi;
        if (lo > hi) {
          const t = lo;
          lo = hi;
          hi = t;
        }
        lo = clamp(lo, 0, cssW - w);
        hi = clamp(hi, 0, cssW - w);
      }
    }
  }

  return clamp(rawLeft, lo, hi);
}

export function computeGuideBox(cssW, cssH, heightCmStr, geom = guideGeom) {
  const gf = geom.guideFrame;
  const fh = frameHeightFromCm(heightCmStr, gf);
  const bodyH = cssH * fh;
  const top = (cssH - bodyH) * gf.bodyTopFrac;
  const sidePad = cssW * gf.sidePadRatio;
  return { left: sidePad, top, width: cssW - 2 * sidePad, height: bodyH };
}

/** Lowest visible ankle Y in preview pixels (landmarks 27/28). */
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
 * Landmark used to pin the guide horizontally (and vertically) to the head.
 * Profile: nose is often low-confidence; fall back to ears or eyes so the frame keeps tracking.
 * @returns {{ x: number, y: number, vis: number } | null}
 */
function headLandmarkForGuide(step, lm, noseVisMin) {
  const nose = lm[0];
  const noseVis = nose?.visibility ?? 1;
  if (nose && noseVis >= noseVisMin) {
    return { x: nose.x, y: nose.y, vis: noseVis };
  }
  if (step !== 2) return null;
  const cands = [];
  for (const id of [7, 8]) {
    const p = lm[id];
    const v = p?.visibility ?? 0;
    if (p && v >= 0.14) cands.push({ x: p.x, y: p.y, vis: v });
  }
  if (cands.length) {
    cands.sort((a, b) => b.vis - a.vis);
    return cands[0];
  }
  const le = lm[2];
  const re = lm[5];
  const lv = le?.visibility ?? 0;
  const rv = re?.visibility ?? 0;
  if (le && re && lv >= 0.12 && rv >= 0.12) {
    return { x: (le.x + re.x) * 0.5, y: (le.y + re.y) * 0.5, vis: Math.min(lv, rv) };
  }
  return null;
}

/**
 * Best ankle landmark for horizontal guide alignment on profile step (prefer higher visibility).
 * @returns {{ x: number, y: number, vis: number } | null}
 */
function profileAnkleLandmarkForGuide(lm, visMin) {
  const a27 = lm[27];
  const a28 = lm[28];
  const v27 = a27?.visibility ?? 0;
  const v28 = a28?.visibility ?? 0;
  const pickRight = v28 >= v27;
  const p = pickRight ? a28 : a27;
  const v = pickRight ? v28 : v27;
  if (!p || v < visMin * 0.82) return null;
  return { x: p.x, y: p.y, vis: v };
}

/**
 * Blend weight [0–1]: how much horizontal box position trusts the ankle vs head on profile step.
 */
function profileFootHorizBlend(footLm, headVis, visMin, gf) {
  if (!footLm) return 0;
  const minW = gf.profileFootHorizBlendMin ?? 0.22;
  const maxW = gf.profileFootHorizBlendMax ?? 0.5;
  if (footLm.vis < visMin * 0.85) return 0;
  const v = clamp(footLm.vis + (headVis ?? 0) * 0.08, visMin, 1);
  const t = (v - visMin) / (0.62 - visMin || 1e-3);
  return clamp(minW + (maxW - minW) * clamp(t, 0, 1), minW, maxW);
}

/**
 * Guide box with optional vertical fit from nose and ankles; mutates smoothDelta for smoothing
 * (dx/dy when ankles are weak; fitHeight/fitTop/fitLeft when stretching to feet).
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
  const headAnchorX = step === 2 ? profileNormX(headA.x, gf) : headA.x;
  const footAnchorX = step === 2 ? profileNormX(footA.x, gf) : footA.x;
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

  const headLm = headLandmarkForGuide(step, lastRawLandmarks, gf.noseVisMin);
  if (!headLm) {
    smoothDelta.dx *= gf.decayLowVis;
    smoothDelta.dy *= gf.decayLowVis;
    return decayLegacyAndMaybeFit();
  }

  const { x: px, y: py } = videoNormToPreviewLocal(headLm.x, headLm.y, cssW, cssH, vw, vh);
  const ankleY = anklePreviewBottomY(lastRawLandmarks, cssW, cssH, vw, vh, vMin);
  const denom = footA.y - headA.y;

  const footLmGuide = step === 2 ? profileAnkleLandmarkForGuide(lastRawLandmarks, vMin) : null;
  const { x: fpx } = footLmGuide
    ? videoNormToPreviewLocal(footLmGuide.x, footLmGuide.y, cssW, cssH, vw, vh)
    : { x: px };
  const fhBlend =
    step === 2 ? profileFootHorizBlend(footLmGuide, headLm.vis, vMin, gf) : 0;

  function horizontalTrackedLeft(pxHead, pxFoot) {
    const leftH = pxHead - headAnchorX * w;
    const leftF = pxFoot - footAnchorX * w;
    return leftH * (1 - fhBlend) + leftF * fhBlend;
  }

  const canStretch = ankleY != null && denom > 0.055 && ankleY > py + cssH * 0.07;

  if (canStretch) {
    let H = (ankleY - py) / denom;
    H = clamp(H, cssH * hMinF, cssH * hMaxF);
    let top = py - headA.y * H;
    let left = horizontalTrackedLeft(px, fpx);
    top = clamp(top, marginY, cssH - marginY - H);
    left = clampGuideBoxLeft(left, cssW, cssH, vw, vh, w, headAnchorX, footAnchorX, marginX, step, geom);

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
    const targetXHead = base.left + headAnchorX * w;
    const targetXFoot = base.left + footAnchorX * w;
    const targetY = base.top + headA.y * base.height;
    let rawLeft =
      base.left + (px - targetXHead) * (1 - fhBlend) + (fpx - targetXFoot) * fhBlend;
    let rawTop = base.top + (py - targetY);
    rawLeft = clampGuideBoxLeft(
      rawLeft,
      cssW,
      cssH,
      vw,
      vh,
      w,
      headAnchorX,
      footAnchorX,
      marginX,
      step,
      geom
    );
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

/** Current box on screen using accumulated delta (matches last preview frame before capture). */
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

export function drawPoly(ctx, pts, box, opts = {}) {
  if (!pts.length) return;
  if (pts.length < 3) return;
  const linear = Boolean(opts.linear);
  const mapPt = typeof opts.mapNormPt === "function" ? opts.mapNormPt : (p) => p;
  ctx.beginPath();
  const mapped = pts.map((p) => {
    const [x, y] = mapPt(p);
    return [box.left + x * box.width, box.top + y * box.height];
  });
  if (linear) {
    ctx.moveTo(mapped[0][0], mapped[0][1]);
    for (let i = 1; i < mapped.length; i++) {
      ctx.lineTo(mapped[i][0], mapped[i][1]);
    }
    ctx.closePath();
    return;
  }
  const first = mapped[0];
  const second = mapped[1];
  const startX = (first[0] + second[0]) * 0.5;
  const startY = (first[1] + second[1]) * 0.5;
  ctx.moveTo(startX, startY);
  for (let i = 1; i < mapped.length; i++) {
    const cur = mapped[i];
    const next = mapped[(i + 1) % mapped.length];
    const midX = (cur[0] + next[0]) * 0.5;
    const midY = (cur[1] + next[1]) * 0.5;
    ctx.quadraticCurveTo(cur[0], cur[1], midX, midY);
  }
  ctx.closePath();
}

/** @param {[number, number][][] | undefined} paths Closed loops in normalized coords. */
export function drawProfileInteriorPaths(ctx, box, paths, style, mapNormPt) {
  if (!paths || !paths.length) return;
  const cw = box.width || (ctx.canvas && ctx.canvas.width ? ctx.canvas.width : 0);
  const lineWScale = style?.lineWidthScale ?? 1;
  const mapPt = typeof mapNormPt === "function" ? mapNormPt : (p) => p;
  ctx.save();
  ctx.strokeStyle = style?.profileInteriorStroke ?? "rgba(142, 196, 255, 0.78)";
  ctx.lineWidth = Math.max(1, cw * 0.0042 * lineWScale);
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowBlur = 0;
  ctx.setLineDash([]);
  for (const pts of paths) {
    if (!pts || pts.length < 3) continue;
    ctx.beginPath();
    const p0 = mapPt(pts[0]);
    ctx.moveTo(box.left + p0[0] * box.width, box.top + p0[1] * box.height);
    for (let i = 1; i < pts.length; i++) {
      const pi = mapPt(pts[i]);
      ctx.lineTo(box.left + pi[0] * box.width, box.top + pi[1] * box.height);
    }
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * Draw the silhouette on a canvas that already uses the same mirror transform as drawImage(video).
 * @param {{ left: number, top: number, width: number, height: number }} box From computeGuideBoxTracked or computeGuideBoxWithDelta.
 * @param {{ strokeStyle?: string, fillStyle?: string, shadowColor?: string, shadowBlur?: number, lineWidthScale?: number, skipProfileInteriorPaths?: boolean, profileInteriorStroke?: string } | undefined} style Optional; omit for live preview colors.
 */
export function drawGuideSilhouetteOnCanvas(ctx, box, currentStep, geom = guideGeom, style) {
  const cw = box.width || (ctx.canvas && ctx.canvas.width ? ctx.canvas.width : 0);
  const lineWScale = style?.lineWidthScale ?? 1;
  const lineFloor = style ? 1.15 : 2;
  const lineWidth = Math.max(lineFloor, cw * 0.0125 * lineWScale);
  ctx.strokeStyle = style?.strokeStyle ?? "rgba(165, 202, 255, 0.92)";
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowColor = style?.shadowColor ?? "rgba(66, 147, 255, 0.25)";
  ctx.shadowBlur =
    style?.shadowBlur !== undefined ? style.shadowBlur : Math.max(3, cw * 0.008);
  ctx.setLineDash([]);
  if (currentStep === 1) {
    drawPoly(ctx, geom.frontPts, box);
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
    ctx.fillStyle = style?.fillStyle ?? "rgba(118, 170, 255, 0.12)";
    ctx.fill();
  } else {
    const gf = geom.guideFrame;
    const mapProfile = (/** @type {[number, number]} */ p) => [profileNormX(p[0], gf), p[1]];
    drawPoly(ctx, geom.profilePts, box, { linear: true, mapNormPt: mapProfile });
    ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.setLineDash([]);
    ctx.fillStyle = style?.fillStyle ?? "rgba(118, 170, 255, 0.12)";
    ctx.fill();
    if (!style?.skipProfileInteriorPaths && geom.profileInteriorPaths?.length) {
      drawProfileInteriorPaths(ctx, box, geom.profileInteriorPaths, style, mapProfile);
    }
  }
}

function mergeGuideFrame(target, src) {
  if (!src || typeof src !== "object") return;
  for (const [k, v] of Object.entries(src)) {
    if (typeof v === "number" && Number.isFinite(v)) target[k] = v;
  }
}

function isPtList(x, minPts = 3) {
  return (
    Array.isArray(x) &&
    x.length >= minPts &&
    x.every((p) => Array.isArray(p) && p.length === 2 && p.every((n) => typeof n === "number" && Number.isFinite(n)))
  );
}

function isInteriorPathsList(x) {
  return Array.isArray(x) && x.length > 0 && x.every((path) => isPtList(path, 3));
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
  if (isPtList(o.profilePts)) {
    guideGeom.profilePts = o.profilePts.map((p) => [p[0], p[1]]);
    if (isInteriorPathsList(o.profileInteriorPaths)) {
      guideGeom.profileInteriorPaths = o.profileInteriorPaths.map((path) => path.map((p) => [p[0], p[1]]));
    } else {
      delete guideGeom.profileInteriorPaths;
    }
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
  }
}

/**
 * If the browser has no saved geometry, try loading JSON from the optional static URL.
 * @returns {Promise<boolean>} Whether a file payload was applied.
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

/** Serialize current geometry to localStorage. */
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
  };
  if (
    guideGeom.profileInteriorPaths &&
    Array.isArray(guideGeom.profileInteriorPaths) &&
    guideGeom.profileInteriorPaths.length > 0
  ) {
    payload.profileInteriorPaths = guideGeom.profileInteriorPaths.map((path) =>
      path.map((p) => [p[0], p[1]])
    );
  }
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
}

export function resetGuideGeometry() {
  guideGeom = cloneGuide();
  if (typeof localStorage !== "undefined") {
    localStorage.removeItem(STORAGE_KEY);
  }
}

export function exportGuideGeometryJson() {
  /** @type {Record<string, unknown>} */
  const payload = {
    version: guideGeom.version,
    headAnchorFront: guideGeom.headAnchorFront,
    headAnchorProfile: guideGeom.headAnchorProfile,
    footAnchorFront: guideGeom.footAnchorFront,
    footAnchorProfile: guideGeom.footAnchorProfile,
    guideFrame: { ...guideGeom.guideFrame },
    frontPts: guideGeom.frontPts,
    profilePts: guideGeom.profilePts,
  };
  if (
    guideGeom.profileInteriorPaths &&
    Array.isArray(guideGeom.profileInteriorPaths) &&
    guideGeom.profileInteriorPaths.length > 0
  ) {
    payload.profileInteriorPaths = guideGeom.profileInteriorPaths;
  }
  return JSON.stringify(payload, null, 2);
}

export function importGuideGeometryJson(str) {
  try {
    const o = JSON.parse(str);
    return applyGuidePayload(o);
  } catch {
    return false;
  }
}
