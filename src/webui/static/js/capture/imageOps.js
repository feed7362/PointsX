/**
 * ImageBitmap helpers: horizontal flip, square crop/pad for the models, encode to Blob.
 */


export async function readImageBitmapHorizontallyFlipped(bitmap) {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D unavailable");
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(bitmap, 0, 0);
  return createImageBitmap(canvas);
}


export async function imageBitmapToBlob(bitmap, preferredMime) {
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(bitmap, 0, 0);
  const types =
    preferredMime && /^image\/(jpeg|png|webp)$/i.test(preferredMime)
      ? [preferredMime, "image/jpeg"]
      : ["image/jpeg"];
  for (const mime of types) {
    const q = mime === "image/jpeg" ? 0.92 : undefined;
    /** @type {Blob | null} */
    const blob = await new Promise((res) => canvas.toBlob((b) => res(b), mime, q));
    if (blob && blob.size > 0) return blob;
  }
  return null;
}


/**
 * Square output for pose/seg models: landscape → centered crop (H×H); portrait → pad sides to H×H.
 * Near-square images return a fresh ImageBitmap copy.
 */
export async function imageBitmapToSquare(bitmap) {
  const w = bitmap.width;
  const h = bitmap.height;
  if (Math.abs(w - h) <= 1) {
    return createImageBitmap(bitmap);
  }
  const canvas = document.createElement("canvas");
  canvas.width = h;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D unavailable");
  if (w > h) {
    const sx = (w - h) / 2;
    ctx.drawImage(bitmap, sx, 0, h, h, 0, 0, h, h);
  } else {
    ctx.fillStyle = "#0f1218";
    ctx.fillRect(0, 0, h, h);
    ctx.drawImage(bitmap, (h - w) / 2, 0);
  }
  return createImageBitmap(canvas);
}
