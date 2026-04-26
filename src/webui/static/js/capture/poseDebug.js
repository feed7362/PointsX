/**
 * Optional pose gate debug logging after capture.
 */

import { flipLandmarks, angleAtVertexRad, angleAtVertexRad3 } from "./poseGate.js";

/** Whether pose gate debug logging is enabled (URL `?poseDebug=1` or localStorage.poseDebug = "1"). */
export function isPoseDebug() {
  try {
    if (typeof localStorage !== "undefined" && localStorage.getItem("poseDebug") === "1") return true;
    if (typeof location !== "undefined" && new URLSearchParams(location.search).has("poseDebug")) return true;
  } catch {
  }
  return false;
}

/**
 * Log one debug line after capture using the same landmarks and gate as the JPEG.
 * @param {number} step
 * @param {any[]} rawLm Raw landmarks before horizontal flip.
 * @param {any[]|null|undefined} worldLm worldLandmarks[0] from detectForVideo.
 * @param {{ ok: boolean, reason?: string }} gate
 */
export function logPoseDebugCapture(step, rawLm, worldLm, gate) {
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
