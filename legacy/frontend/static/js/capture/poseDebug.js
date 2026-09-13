/**
 * Optional pose gate debug: enable with URL `?poseDebug=1` or `localStorage.poseDebug = "1"`.
 *
 * Landmarker `visibility` is model confidence (0–1), not “visible in the photo”; occluded ears
 * can still score ~1. Use earSepX / geometry together with earVisMP when interpreting profile.
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
 * Log one pose gate result and derived metrics (pass or fail).
 * @param {"upload" | "video" | "capture"} source
 * @param {number} step 1 = front, 2 = profile
 * @param {any[] | null | undefined} rawLm Raw landmarks before horizontal flip (MediaPipe output).
 * @param {any[] | null | undefined} worldLm worldLandmarks[0] when available (front knee flex in 3D).
 * @param {{ ok: boolean, reason?: string }} gate
 */
export function logPoseGateCheck(source, step, rawLm, worldLm, gate) {
  if (!isPoseDebug()) return;
  if (!rawLm?.length) {
    console.debug("[poseDebug:gate]", { source, step, ok: gate?.ok, reason: gate?.reason, landmarks: false });
    return;
  }
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
  const ankleY = ((lm[27]?.y ?? 0) + (lm[28]?.y ?? 0)) / 2;
  const torsoLen = hipY - (ls.y + rs.y) / 2;
  const legLen = ankleY - hipY;
  const kneeKneeFlex2d =
    (lk.visibility ?? 0) > 0.2 && (rk.visibility ?? 0) > 0.2
      ? {
          leftDeg: (angleAtVertexRad(lh, lk, lm[27]) * 180) / Math.PI,
          rightDeg: (angleAtVertexRad(rh, rk, lm[28]) * 180) / Math.PI,
        }
      : null;
  let kneeFlexDeg3d = null;
  let kneeFlexAvg3d = null;
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
    if (kf.length) {
      kneeFlexDeg3d = Number(Math.min(...kf).toFixed(1));
      kneeFlexAvg3d = Number((kf.reduce((s, v) => s + v, 0) / kf.length).toFixed(1));
    }
  }
  const earL = lm[7]?.visibility ?? 0;
  const earR = lm[8]?.visibility ?? 0;
  const eyeL = lm[2]?.visibility ?? 0;
  const eyeR = lm[5]?.visibility ?? 0;
  const eyeSepXNorm =
    step === 2 ? Number(Math.abs((lm[2]?.x ?? 0) - (lm[5]?.x ?? 0)).toFixed(4)) : null;
  const earSepXNorm =
    step === 2 ? Number(Math.abs((lm[7]?.x ?? 0) - (lm[8]?.x ?? 0)).toFixed(4)) : null;
  /** Profile: knee angle and shin tilt (same geometry as poseGate checkProfilePose). */
  let profileRightKneeDeg = null;
  let profileRightLegTiltDeg = null;
  if (step === 2 && (rh?.visibility ?? 0) > 0.32 && (rk?.visibility ?? 0) > 0.32 && (lm[28]?.visibility ?? 0) > 0.32) {
    const ra = lm[28];
    profileRightKneeDeg = Number(((angleAtVertexRad(rh, rk, ra) * 180) / Math.PI).toFixed(1));
    profileRightLegTiltDeg = Number(
      ((Math.atan2(Math.abs((ra?.x ?? 0) - (rh?.x ?? 0)), Math.max(1e-4, (ra?.y ?? 0) - (rh?.y ?? 0))) * 180) /
        Math.PI).toFixed(1)
    );
  }
  console.debug("[poseDebug:gate]", {
    source,
    step,
    ok: gate.ok,
    reason: gate.reason,
    shoulderW: Number(shoulderW.toFixed(4)),
    frontalWidth: Number(frontalWidth.toFixed(4)),
    noseShoulderDx: Number(Math.abs(nose.x - shoulderMidX).toFixed(4)),
    hipKneeDy: Number((kneeY - hipY).toFixed(4)),
    legTorsoRatio: torsoLen > 0.06 && legLen > 0 ? Number((legLen / torsoLen).toFixed(3)) : null,
    kneeFlexDeg2d: kneeKneeFlex2d,
    kneeFlexMinDeg3d: kneeFlexDeg3d,
    kneeFlexAvgDeg3d: kneeFlexAvg3d,
    profileRightKneeDeg,
    profileRightLegTiltDeg,
    eyeSepX: eyeSepXNorm,
    earSepX: earSepXNorm,
    earVisMP: { L: Number(earL.toFixed(3)), R: Number(earR.toFixed(3)) },
    eyeVisMP: { L: Number(eyeL.toFixed(3)), R: Number(eyeR.toFixed(3)) },
  });
}

/**
 * Log after mirrored JPEG capture (same as logPoseGateCheck with source "capture").
 * @param {number} step
 * @param {any[]} rawLm Raw landmarks before horizontal flip.
 * @param {any[]|null|undefined} worldLm worldLandmarks[0] from detectForVideo.
 * @param {{ ok: boolean, reason?: string }} gate
 */
export function logPoseDebugCapture(step, rawLm, worldLm, gate) {
  logPoseGateCheck("capture", step, rawLm, worldLm, gate);
}
