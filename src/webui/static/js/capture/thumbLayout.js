/**
 * Thumbnail slots match each photo’s aspect ratio (no letterboxing).
 */

import { getCaptureDom } from "./dom.js";

function thumbDisplayCaps() {
  const { sectionCapture } = getCaptureDom();
  const review = sectionCapture?.classList.contains("capture--review");
  if (review) {
    return {
      maxW: Math.min(400, typeof window !== "undefined" ? window.innerWidth * 0.92 : 400),
      maxH: Math.min(520, typeof window !== "undefined" ? window.innerHeight * 0.65 : 520),
    };
  }
  return {
    maxW: Math.min(220, typeof window !== "undefined" ? window.innerWidth * 0.44 : 220),
    maxH: Math.min(360, typeof window !== "undefined" ? window.innerHeight * 0.4 : 360),
  };
}

/** Size `.thumb-slot` so the inner content box matches image aspect ratio (no sliver gaps). */
export function syncThumbSlotToImage(imgEl) {
  const slot = imgEl.closest(".thumb-slot");
  if (!slot || imgEl.hidden) return;
  const BORDER_PX = 1;
  const apply = () => {
    const iw = imgEl.naturalWidth;
    const ih = imgEl.naturalHeight;
    if (!(iw > 0 && ih > 0)) return;
    const { maxW, maxH } = thumbDisplayCaps();
    const maxInnerW = Math.max(1, maxW - 2 * BORDER_PX);
    const maxInnerH = Math.max(1, maxH - 2 * BORDER_PX);
    const scale = Math.min(maxInnerW / iw, maxInnerH / ih);
    let cw = Math.max(1, Math.round(iw * scale));
    let ch = Math.max(1, Math.round((cw * ih) / iw));
    if (ch > maxInnerH) {
      ch = Math.max(1, Math.floor(maxInnerH));
      cw = Math.max(1, Math.round((ch * iw) / ih));
    }
    if (cw > maxInnerW) {
      cw = Math.max(1, Math.floor(maxInnerW));
      ch = Math.max(1, Math.round((cw * ih) / iw));
    }
    slot.style.width = `${cw + 2 * BORDER_PX}px`;
    slot.style.height = `${ch + 2 * BORDER_PX}px`;
    slot.classList.add("thumb-slot--photo");
  };
  if (imgEl.complete && imgEl.naturalWidth > 0) apply();
  else imgEl.addEventListener("load", apply, { once: true });
}

export function clearThumbSlotPhoto(imgEl) {
  const slot = imgEl.closest(".thumb-slot");
  if (!slot) return;
  slot.style.removeProperty("width");
  slot.style.removeProperty("height");
  slot.classList.remove("thumb-slot--photo");
}

export function resyncVisibleCaptureThumbs() {
  const { thumbFront, thumbSide } = getCaptureDom();
  if (thumbFront?.src && !thumbFront.hidden) syncThumbSlotToImage(thumbFront);
  if (thumbSide?.src && !thumbSide.hidden) syncThumbSlotToImage(thumbSide);
}
