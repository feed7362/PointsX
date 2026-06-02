/**
 * getUserMedia helpers: secure-context checks, mobile constraint fallbacks, UA hints.
 */

/** @returns {string | null} Ukrainian message when camera API cannot be used. */
export function cameraUnavailableReason() {
  if (!navigator.mediaDevices?.getUserMedia) {
    if (!window.isSecureContext) {
      const host = window.location.hostname || "";
      const port = window.location.port ? `:${window.location.port}` : "";
      return (
        "Камера в браузері потребує захищеного з'єднання (HTTPS) або localhost. " +
        `Зараз відкрито: ${window.location.protocol}//${host}${port}. ` +
        "На телефоні в локальній мережі запустіть сервер з HTTPS (див. RUN-WEBUI.md) " +
        "або завантажте фото з галереї замість камери."
      );
    }
    return "Цей браузер не підтримує доступ до камери. Спробуйте Chrome або Safari або завантажте фото.";
  }
  return null;
}

/** Map DOMException / Error to a short Ukrainian hint. */
export function cameraErrorMessage(err) {
  const name = err?.name || "";
  const msg = err?.message ? String(err.message) : String(err);
  if (!window.isSecureContext) {
    return cameraUnavailableReason() || msg;
  }
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return (
      "Доступ до камери заборонено. Дозвольте камеру для цього сайту в налаштуваннях браузера " +
      "(іконка замка / «Дозволи сайту») і натисніть «Увімкнути камеру» знову."
    );
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "Камеру не знайдено на цьому пристрої.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Камера зайнята іншим застосунком або недоступна. Закрийте інші програми з камерою.";
  }
  if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
    return "Камера не підтримує обрані параметри. Спробуйте ще раз — застосунок спробує простіший режим.";
  }
  return msg || "Невідома помилка камери.";
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
