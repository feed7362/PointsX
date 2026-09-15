import { t } from "../i18n/index.js";

/** @returns {string | null} Ukrainian/English message when camera API cannot be used. */
export function cameraUnavailableReason() {
  if (!navigator.mediaDevices?.getUserMedia) {
    if (!window.isSecureContext) {
      const host = window.location.hostname || "";
      const port = window.location.port ? `:${window.location.port}` : "";
      const url = `${window.location.protocol}//${host}${port}`;
      return t("camera-https-warning", { url });
    }
    return t("camera-browser-unsupported");
  }
  return null;
}

/** Map DOMException / Error to a short hint. */
export function cameraErrorMessage(err) {
  const name = err?.name || "";
  const msg = err?.message ? String(err.message) : String(err);
  if (!window.isSecureContext) {
    return cameraUnavailableReason() || msg;
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return t("camera-permission-denied");
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return t("camera-not-found");
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return t("camera-busy");
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return t("camera-params-unsupported");
  }
  return msg || t("camera-unknown-error");
}

const CONSTRAINT_ATTEMPTS = [
  {
    video: {
      facingMode: "user",
      width: { ideal: 1280 },
      height: { ideal: 720 },
    },
    audio: false,
  },
  { video: { facingMode: "user" }, audio: false },
  { video: true, audio: false },
];

/**
 * Request front-facing camera with progressively simpler constraints (mobile-safe).
 * @returns {Promise<MediaStream>}
 */
export async function requestCameraStream() {
  const blocked = cameraUnavailableReason();
  if (blocked) {
    throw new Error(blocked);
  }

  let lastErr = /** @type {unknown} */ (null);
  for (const constraints of CONSTRAINT_ATTEMPTS) {
    try {
      return await navigator.mediaDevices.getUserMedia(constraints);
    } catch (e) {
      lastErr = e;
      const n = /** @type {DOMException} */ (e).name;
      if (n === "NotAllowedError" || n === "PermissionDeniedError" || n === "SecurityError") {
        throw e;
      }
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error(String(lastErr));
}
