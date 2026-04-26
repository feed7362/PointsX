/**
 * Guide silhouette overlay on the video preview canvas.
 */

import {
  guideGeom,
  computeGuideBox,
  computeGuideBoxTracked,
  drawPoly,
  drawOpenStroke,
} from "../guideGeometry.js";
import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";

/** Base guide box from height only. */
export function guideBox(cssW, cssH) {
  const { heightInput } = getCaptureDom();
  return computeGuideBox(cssW, cssH, heightInput.value);
}

/** Tracked guide box: nose/ankle alignment and smoothed offset in guideSmoothDelta. */
export function guideBoxTracked(cssW, cssH, vw, vh) {
  const { heightInput } = getCaptureDom();
  return computeGuideBoxTracked(
    cssW,
    cssH,
    vw,
    vh,
    captureState.step,
    captureState.lastRawLandmarks,
    captureState.guideSmoothDelta,
    heightInput.value
  );
}

/** Draw the dashed silhouette overlay on the video preview canvas. */
export function drawOverlay() {
  const { overlay, video } = getCaptureDom();
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
  if (captureState.step === 1) {
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

/** Resize/redraw the overlay to match the preview box. */
export function syncOverlaySize() {
  drawOverlay();
}
