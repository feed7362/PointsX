/**
 * Pose gate rules (mirrored landmark space). No DOM or MediaPipe.
 */

export const VIS_MIN = 0.32;
export const VIS_ANKLE_MIN = 0.22;
export const VIS_FULL_BODY_MIN = 0.45;
export const VIS_FULL_BODY_LIMB_MIN = 0.5;
export const VIS_FULL_BODY_FEET_MIN = 0.4;

/**
 * Angle ∠ABC in radians (0…π) with vertex at B (2D).
 * @param {{ x: number, y: number }} a
 * @param {{ x: number, y: number }} b
 * @param {{ x: number, y: number }} c
 */
export function angleAtVertexRad(a, b, c) {
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

/** Angle ∠ABC in 3D (0…π) at B; used for knee flex from worldLandmarks (2D front view is misleading). */
export function angleAtVertexRad3(a, b, c) {
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

/** Mirror landmark X so rules match the mirrored selfie preview. */
export function flipLandmarks(lm) {
  return lm.map((p) => ({ ...p, x: 1 - p.x }));
}

const DEG = (d) => (d * Math.PI) / 180;

/** Upper-arm angle from vertical downward (0 = arm straight down). */
function upperArmAbductionRad(shoulder, elbow) {
  const dy = Math.max(1e-4, elbow.y - shoulder.y);
  const dx = Math.abs(elbow.x - shoulder.x);
  return Math.atan2(dx, dy);
}

/**
 * Profile step: right upper arm roughly along the torso (~arms at sides mirror pose).
 * `upperArmAbductionRad`: 0 = straight down behind/beside torso; larger = elbow forward in plane.
 */
function checkProfileRightArmAngle(lm) {
  const rs = lm[12];
  const re = lm[14];
  const reVis = re.visibility ?? 0;
  const rightAbd = upperArmAbductionRad(rs, re);

  if (reVis < 0.24) {
    return {
      ok: false,
      reason: "Покажіть правий лікоть",
    };
  }
  if (rightAbd > DEG(52)) {
    return {
      ok: false,
      reason: "Опустіть руку вздовж тіла",
    };
  }
  return { ok: true };
}

/**
 * Front A-pose gate: nose 0, shoulders 11–12, elbows 13–14, wrists 15–16, hips 23–24, knees 25–26, ankles 27–28.
 * @param {any[]|null|undefined} rawWorldLm worldLandmarks[0] from detectForVideo for knee flex.
 */
function checkFrontPose(lm, rawWorldLm) {
  const nose = lm[0];
  const leye = lm[2];
  const reye = lm[5];
  const lear = lm[7];
  const rear = lm[8];
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
    return { ok: false, reason: "Покажіть обличчя і плечі в кадрі." };
  }
  const tilt = Math.abs(ls.y - rs.y);
  if (tilt > 0.12) {
    return { ok: false, reason: "Вирівняйте плечі (не нахиляйте корпус)." };
  }
  const shoulderW = Math.abs(ls.x - rs.x);
  const hipWForFacing = Math.abs((lh?.x ?? ls.x) - (rh?.x ?? rs.x));
  const frontalWidth = Math.max(shoulderW, hipWForFacing);
  // Slightly less strict distance gate: avoid false "come closer" prompts.
  // const FRONTAL_WIDTH_MIN = 0.086;
  // const shoulderMidX = (ls.x + rs.x) / 2;
  // if (frontalWidth < FRONTAL_WIDTH_MIN) {
  //   const corridor = Math.max(0.055, shoulderW * 0.55 + 0.04);
  //   const noseDx = Math.abs(nose.x - shoulderMidX);
  //   if (tilt <= 0.12 && noseDx <= corridor) {
  //     return {
  //       ok: false,
  //       reason:
  //         "Покажіть себе крупніше в кадрі",
  //     };
  //   }
  //   return { ok: false, reason: "Станьте обличчям до камери" };
  // }
  const minSx = Math.min(ls.x, rs.x) - 0.09;
  const maxSx = Math.max(ls.x, rs.x) + 0.09;
  if (nose.x < minSx || nose.x > maxSx) {
    return { ok: false, reason: "Поверніться обличчям до камери" };
  }

  const eyeVisMax = Math.max(leye?.visibility ?? 0, reye?.visibility ?? 0);
  const earVisMax = Math.max(lear?.visibility ?? 0, rear?.visibility ?? 0);
  if (eyeVisMax < 0.24 || earVisMax < 0.2) {
    return { ok: false, reason: "Покажіть всю голову в кадрі" };
  }
  if (nose.y < 0.09 || ((leye?.y ?? 1) < 0.065 && (reye?.y ?? 1) < 0.065)) {
    return { ok: false, reason: "Опустіть камеру або відійдіть трохи, вся голова має бути в кадрі" };
  }

  if ((le.visibility ?? 0) < 0.22 || (re.visibility ?? 0) < 0.22) {
    return { ok: false, reason: "Лікті мають бути в кадрі" };
  }
  const leftAbd = upperArmAbductionRad(ls, le);
  const rightAbd = upperArmAbductionRad(rs, re);
  if (leftAbd < DEG(10) || rightAbd < DEG(10)) {
    return { ok: false, reason: "A-поза: відведіть руки на ~15–20° від тіла" };
  }
  if (leftAbd > DEG(50) || rightAbd > DEG(50)) {
    return { ok: false, reason: "Не розводьте руки занадто широко (достатньо ~15–20°)." };
  }
  const latL = Math.abs(le.x - ls.x);
  const latR = Math.abs(re.x - rs.x);
  if (latL < shoulderW * 0.11 || latR < shoulderW * 0.11) {
    return { ok: false, reason: "Лікті трохи вбік від корпусу, пахви мають залишатися відкритими" };
  }

  const ankleWx = Math.abs(la.x - ra.x);
  const kneeWx = Math.abs(lm[25].x - lm[26].x);
  const hipW = Math.abs(lm[23].x - lm[24].x);
  const minAnkleSpread = Math.max(0.054, shoulderW * 0.55);
  const minKneeSpread = Math.max(0.04, shoulderW * 0.43);
  const lowerBodySpreadScore = (ankleWx * 0.65 + kneeWx * 0.35) / Math.max(1e-6, shoulderW);
  if (
    (ankleWx < minAnkleSpread && kneeWx < minKneeSpread) ||
    lowerBodySpreadScore < 0.62
  ) {
    return { ok: false, reason: "Спробуйте поставити ноги трохи ширше — приблизно на ширині плечей" };
  }
  if ((lm[23].visibility ?? 0) < 0.24 || (lm[24].visibility ?? 0) < 0.24) {
    return { ok: false, reason: "Має бути видно зону стегон і паху" };
  }
  if (hipW < shoulderW * 0.11) {
    return { ok: false, reason: "Не зводьте стегна" };
  }

  const ankleVis = ((la.visibility ?? 0) + (ra.visibility ?? 0)) / 2;
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст, стопи повинні бути в кадрі" };
  }
  const lHeel = lm[29];
  const rHeel = lm[30];
  const lFootIndex = lm[31];
  const rFootIndex = lm[32];
  const footPts = [lHeel, rHeel, lFootIndex, rFootIndex].filter(Boolean);
  if (footPts.length >= 2) {
    const footVisiblePts = footPts.filter((p) => (p.visibility ?? 0) >= 0.2);
    if (footVisiblePts.length < 2 || footVisiblePts.some((p) => !inFrame(p) || p.y > 0.992)) {
      return {
        ok: false,
        reason: "Стопи мають бути повністю в кадрі",
      };
    }
  }

  if (!inFrame(la) || !inFrame(ra) || Math.max(la.y, ra.y) > 0.985) {
    return {
      ok: false,
      reason: "Стопи мають бути повністю в кадрі",
    };
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
  let kneeFlexForStanding = null;
  let kneeFlexSamples3d = 0;
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
      kneeFlexSamples3d = flex3.length;
      minKneeFlexDeg3d = Math.min(...flex3);
      kneeFlexForStanding =
        flex3.length === 2 ? (flex3[0] + flex3[1]) / 2 : flex3[0];
    }
  }
  if (lv > 0.18 && rv > 0.18) {
    const shoulderY = (ls.y + rs.y) / 2;
    const hipY = (lh.y + rh.y) / 2;
    const kneeY = (lk.y + rk.y) / 2;
    const ankleY = (la.y + ra.y) / 2;
    const torsoLen = hipY - shoulderY;
    const legLen = ankleY - hipY;
    const kneeGap = kneeY - hipY;
    const minGap = minKneeFlexDeg3d != null ? 0.066 : 0.076;
    if (
      kneeFlexForStanding != null &&
      (kneeFlexForStanding < 142 ||
        (kneeFlexSamples3d === 2 && kneeFlexForStanding < 150 && kneeGap < minGap + 0.012))
    ) {
      return {
        ok: false,
        reason: "Станьте повним зростом на прямих ногах, не присідайте",
      };
    }
    if (torsoLen > 0.06 && legLen > 0 && legLen / torsoLen < 1.16) {
      return {
        ok: false,
        reason: "Станьте повним зростом на прямих ногах, не присідайте",
      };
    }
    if (kneeGap < minGap) {
      return {
        ok: false,
        reason: "Станьте повним зростом на прямих ногах, не присідайте",
      };
    }
  }

  return { ok: true };
}

const FULL_BODY_PARTS = [
  { id: 0, name: "голову" },
  { id: 11, name: "ліве плече" },
  { id: 12, name: "праве плече" },
  { id: 13, name: "лівий лікоть" },
  { id: 14, name: "правий лікоть" },
  { id: 15, name: "ліву кисть" },
  { id: 16, name: "праву кисть" },
  { id: 23, name: "таз зліва" },
  { id: 24, name: "таз справа" },
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
      reason: `Не видно ${invisible.slice(0, 3).join(", ")}${invisible.length > 3 ? "…" : ""}`,
    };
  }
  if (outOfFrame.length) {
    return {
      ok: false,
      reason: `Поза кадром ${outOfFrame.slice(0, 3).join(", ")}${outOfFrame.length > 3 ? "…" : ""}.`,
    };
  }
  return { ok: true };
}

/**
 * Profile pose: right side to camera. Uses shoulder width, visibility asymmetry, eye separation,
 * and head alignment; rejects too-frontal or left-side-dominant poses where landmarks are noisy.
 */
function checkProfilePose(lm) {
  const ls = lm[11];
  const rs = lm[12];
  const nose = lm[0];
  const lsV = ls.visibility ?? 0;
  const rsV = rs.visibility ?? 0;
  if (Math.max(lsV, rsV) < VIS_MIN || (nose.visibility ?? 1) < VIS_MIN) {
    return { ok: false, reason: "Покажіть себе крупніше в кадрі" };
  }

  const leftEye = lm[2];
  const rightEye = lm[5];
  const leftEar = lm[7];
  const rightEar = lm[8];
  const eyeVisMaxEarly = Math.max(leftEye?.visibility ?? 0, rightEye?.visibility ?? 0);
  const earVisMaxEarly = Math.max(leftEar?.visibility ?? 0, rightEar?.visibility ?? 0);
  if (eyeVisMaxEarly < 0.2 && earVisMaxEarly < 0.16) {
    return { ok: false, reason: "Покажіть всю голову в кадрі" };
  }
  if (nose.y < 0.085 || ((leftEye?.y ?? 1) < 0.062 && (rightEye?.y ?? 1) < 0.062)) {
    return { ok: false, reason: "Опустіть камеру або відійдіть трохи, вся голова має бути в кадрі" };
  }

  const shoulderW = Math.abs(ls.x - rs.x);
  const hipW = Math.abs((lm[23]?.x ?? ls.x) - (lm[24]?.x ?? rs.x));
  const frontalWidth = Math.max(shoulderW, hipW);
  const torsoFrontality = shoulderW * 0.62 + hipW * 0.38;
  const tooFrontal = shoulderW > 0.195;
  if (tooFrontal) {
    return { ok: false, reason: "Поверніться боком на ~90°" };
  }
  if (frontalWidth > 0.086 || torsoFrontality > 0.082) {
    return { ok: false, reason: "Поверніться боком на ~90°" };
  }
  if (lsV > rsV + 0.112) {
    return {
      ok: false,
      reason: "У профілі стійте правим боком до камери",
    };
  }
  if (lsV > rsV + 0.064 && shoulderW > 0.084) {
    return {
      ok: false,
      reason: "У профілі стійте правим боком до камери",
    };
  }
  const eyeLv = leftEye?.visibility ?? 0;
  const eyeRv = rightEye?.visibility ?? 0;
  const earLv = leftEar?.visibility ?? 0;
  const earRv = rightEar?.visibility ?? 0;
  const eyeSepX = Math.abs((leftEye?.x ?? 0) - (rightEye?.x ?? 0));
  const earSepX = Math.abs((leftEar?.x ?? 0) - (rightEar?.x ?? 0));

  if (eyeLv > 0.42 && eyeRv > 0.42 && earLv > 0.42 && earRv > 0.42) {
    const eyeSepMaxProfile = Math.max(0.0105, shoulderW * 0.46);
    const earSepMaxProfile = Math.max(0.016, shoulderW * 0.95);
    if (eyeSepX > eyeSepMaxProfile && earSepX > earSepMaxProfile) {
      return {
        ok: false,
        reason: "Не розвертайте голову до камери",
      };
    }
  }
  if (shoulderW < 0.02 && eyeLv > 0.5 && eyeRv > 0.5 && earLv > 0.5 && earRv > 0.5) {
    if (eyeSepX > 0.0075 && earSepX > 0.012) {
      return {
        ok: false,
        reason: "Не повертайте голову до камери",
      };
    }
  }
  // Strong profile => very narrow shoulders; eye separation alone is unreliable (projection + model).
  if (shoulderW < 0.03 && eyeLv > 0.45 && eyeRv > 0.45) {
    const eyeSepTurnThreshold = 0.009 + shoulderW * 0.22;
    if (eyeSepX > eyeSepTurnThreshold) {
      return {
        ok: false,
        reason: "Не повертайте голову до камери",
      };
    }
  }

  if (eyeLv > 0.45 && eyeRv > 0.45) {
    const eyeSepMax =
      shoulderW < 0.11
        ? Math.max(0.02, shoulderW * 0.38)
        : Math.min(0.022, Math.max(0.017, shoulderW * 0.13));

    const eyeVisBalance = Math.min(eyeLv, eyeRv) / Math.max(eyeLv, eyeRv);
    if (shoulderW > 0.075 && eyeVisBalance > 0.92 && eyeSepX > 0.018) {
      return {
        ok: false,
        reason: "Не розвертайте голову до камери",
      };
    }
    if (shoulderW > 0.09 && eyeVisBalance > 0.9 && eyeSepX > eyeSepMax * 1.18) {
      return {
        ok: false,
        reason: "Не розвертайте голову до камери",
      };
    }
    if (eyeSepX > eyeSepMax) {
      return {
        ok: false,
        reason: "Не розвертайте голову до камери",
      };
    }
  }
  const shoulderMidX = (ls.x + rs.x) / 2;
  const noseOffShoulderMid = Math.abs(nose.x - shoulderMidX);
  if (shoulderW > 0.108 && noseOffShoulderMid < 0.011) {
    return {
      ok: false,
      reason: "Дивіться в той самий бік, куди звернене тіло .",
    };
  }
  const tilt = Math.abs(ls.y - rs.y);
  if (tilt > 0.165) {
    return { ok: false, reason: "Не нахиляйте корпус у профіль" };
  }

  const shoulderY = (ls.y + rs.y) / 2;

  const wR = lm[16];
  if ((wR.visibility ?? 0) > 0.24 && wR.y < shoulderY - 0.04) {
    return {
      ok: false,
      reason: "Опустіть руку вздовж тулуба",
    };
  }

  const rightArm = checkProfileRightArmAngle(lm);
  if (!rightArm.ok) return rightArm;

  const rk = lm[26];
  const rh = lm[24];
  const rAnkle = lm[28];
  if (
    (rh?.visibility ?? 0) > 0.32 &&
    (rk?.visibility ?? 0) > 0.32 &&
    (rAnkle?.visibility ?? 0) > 0.32
  ) {
    const rightKneeAngle = (angleAtVertexRad(rh, rk, rAnkle) * 180) / Math.PI;
    // 2D profile angle is noisy vs true knee extension; allow margin below ~180°.
    if (rightKneeAngle < 155) {
      return {
        ok: false,
        reason: "Станьте на рівних ногах",
      };
    }
    const rightLegTilt = Math.atan2(
      Math.abs((rAnkle?.x ?? 0) - (rh?.x ?? 0)),
      Math.max(1e-4, (rAnkle?.y ?? 0) - (rh?.y ?? 0))
    );
    if (rightLegTilt > DEG(14)) {
      return {
        ok: false,
        reason: "Станьте на рівних ногах",
      };
    }
  }

  const la = lm[27];
  const lav = la?.visibility ?? 0;
  const rav = rAnkle?.visibility ?? 0;
  const ankleVis = Math.max(lav, rav);
  if (ankleVis < VIS_ANKLE_MIN) {
    return { ok: false, reason: "Покажіть повний зріст" };
  }
  const bothAnklesTracked = lav >= 0.14 && rav >= 0.14;
  if (bothAnklesTracked) {
    if (!inFrame(la) || !inFrame(rAnkle) || Math.max(la.y, rAnkle.y) > 0.985) {
      return {
        ok: false,
        reason: "Стопи мають бути повністю в кадрі",
      };
    }
  } else {
    const dom = rav >= lav ? rAnkle : la;
    if (!dom || !inFrame(dom) || dom.y > 0.986) {
      return {
        ok: false,
        reason: "Стопи мають бути повністю в кадрі",
      };
    }
  }

  const lHeel = lm[29];
  const rHeel = lm[30];
  const lFootIndex = lm[31];
  const rFootIndex = lm[32];
  const footPts = [lHeel, rHeel, lFootIndex, rFootIndex].filter(Boolean);
  if (footPts.length >= 2) {
    const footVisiblePts = footPts.filter((p) => (p.visibility ?? 0) >= 0.18);
    if (footVisiblePts.length >= 2) {
      if (footVisiblePts.some((p) => !inFrame(p) || p.y > 0.992)) {
        return {
          ok: false,
          reason: "Стопи мають бути повністю в кадрі",
        };
      }
    }
  }

  return { ok: true };
}

/** Run full-body check then front or profile rules on mirrored landmarks. */
export function checkPoseForStep(viewStep, landmarks, worldLandmarks) {
  const lm = flipLandmarks(landmarks);
  const fullBody = checkFullBodyVisible(lm, viewStep);
  if (!fullBody.ok) return fullBody;
  const rawWorld = worldLandmarks && worldLandmarks.length ? worldLandmarks : null;
  return viewStep === 1 ? checkFrontPose(lm, rawWorld) : checkProfilePose(lm);
}
