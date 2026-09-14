/**
 * MediaPipe PoseLandmarker loading (VIDEO for the live camera, IMAGE for uploads) and the pose gate on stills.
 */

import { t } from "../i18n/index.js";
import { captureState } from "./state.js";
import { checkPoseForStep } from "./poseGate.js";

export const POSE_MP_MIN_DETECTION_CONF = 0.38;

export const POSE_MP_MIN_PRESENCE_CONF = 0.38;

export const POSE_MP_MIN_TRACKING_CONF = 0.38;


// MediaPipe module sources, tried in order. esm.sh alone was a single point of
// failure: an outage or an ad-blocker filtering it silently disabled the live
// pose guidance (and broke dataset upload via the same CDN). jsdelivr's /+esm
// endpoint is separate infrastructure serving self-contained ES modules.
const MP_PKGS = [
  "https://esm.sh/@mediapipe/tasks-vision@0.10.14",
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm",
];

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";


async function importMediaPipePackage() {
  const failures = [];
  for (const url of MP_PKGS) {
    try {
      return await import(/* @vite-ignore */ url);
    } catch (err) {
      failures.push(`${new URL(url).host}: ${err?.message ?? err}`);
      console.warn(`[capture] MediaPipe failed from ${url}`, err);
    }
  }
  throw new Error(`MediaPipe unavailable — ${failures.join(" | ")}`);
}


/**
 * VIDEO-mode PoseLandmarker requires strictly monotonic timestamp_ms on every detectForVideo call
 * (same instance). Mixing 0 for uploads with camera frames causes graph errors and a wedged UI.
 */
let lastPoseVideoTimestampMs = -1;


export function nextPoseVideoTimestampMs() {
  let t = Math.floor(performance.now());
  if (t <= lastPoseVideoTimestampMs) t = lastPoseVideoTimestampMs + 1;
  lastPoseVideoTimestampMs = t;
  return t;
}


/** Lazy-load MediaPipe PoseLandmarker (GPU with CPU fallback). */
export async function loadPoseLandmarker() {
  if (captureState.poseLandmarker || captureState.poseLoadError) return;
  try {
    const { FilesetResolver, PoseLandmarker } = await importMediaPipePackage();
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    try {
      captureState.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    } catch {
      captureState.poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
        minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
        minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
      });
    }
  } catch (e) {
    captureState.poseLoadError = e;
    console.error(e);
  }
}


/** Release upload landmarker so the next file gets a clean WASM graph (avoids odd state across runs). */
export function disposePoseLandmarkerImage() {
  const m = captureState.poseLandmarkerImage;
  if (m) {
    try {
      m.close();
    } catch {
      /* ignore */
    }
  }
  captureState.poseLandmarkerImage = null;
  captureState.poseImageLoadError = null;
}


/** Lazy-load a second PoseLandmarker in IMAGE mode for file uploads (no video timestamps / tracking). */
export async function loadPoseLandmarkerImage() {
  if (captureState.poseLandmarkerImage || captureState.poseImageLoadError) return;
  try {
    const { FilesetResolver, PoseLandmarker } = await importMediaPipePackage();
    const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: MODEL_URL, delegate },
      runningMode: "IMAGE",
      numPoses: 1,
      minPoseDetectionConfidence: POSE_MP_MIN_DETECTION_CONF,
      minPosePresenceConfidence: POSE_MP_MIN_PRESENCE_CONF,
      minTrackingConfidence: POSE_MP_MIN_TRACKING_CONF,
    });
    try {
      captureState.poseLandmarkerImage = await PoseLandmarker.createFromOptions(vision, opts("GPU"));
    } catch {
      captureState.poseLandmarkerImage = await PoseLandmarker.createFromOptions(vision, opts("CPU"));
    }
  } catch (e) {
    captureState.poseImageLoadError = e;
    console.error(e);
  }
}


/**
 * Run the same pose gate on a still using the IMAGE landmarker (independent of VIDEO session state).
 * @param {1|2} viewStep 1 = front, 2 = profile
 * @param {ImageBitmap} bitmap
 * @returns {{ ok: boolean, reason?: string, landmarks?: any, world?: any, gate?: { ok: boolean, reason?: string } }}
 */
export function gatePoseOnBitmap(viewStep, bitmap) {
  const marker = captureState.poseLandmarkerImage;
  if (!marker) {
    return { ok: false, reason: t("sess-pose-model-not-ready") };
  }
  const result = marker.detect(bitmap);
  const lm = result.landmarks && result.landmarks[0];
  const worldLm = result.worldLandmarks && result.worldLandmarks[0];
  if (!lm) {
    return { ok: false, reason: t("sess-no-person") };
  }
  const gate = checkPoseForStep(viewStep, lm, worldLm);
  if (!gate.ok) {
    return {
      ok: false,
      reason: gate.reason || t("sess-pose-invalid"),
      landmarks: lm,
      world: worldLm,
      gate,
    };
  }
  return { ok: true, landmarks: lm, world: worldLm, gate };
}
