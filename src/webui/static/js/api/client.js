/**
 * Single HTTP entry point for the capture UI.
 *
 * The page and the API are always served from the same origin (Vercel function
 * in proxy mode, or uvicorn locally), so every call goes to an absolute
 * same-origin URL. Feature modules (measure.js, tts.js) build on this; UI code
 * does not call fetch() directly.
 */

/** Non-2xx API response; `message` is the server's human-readable `detail`. */
export class ApiError extends Error {
  /**
   * @param {string} message
   * @param {number} status
   */
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/** Absolute same-origin URL (avoids bad resolution from import maps / sub-paths). */
export function apiUrl(path) {
  if (typeof window === "undefined" || !window.location?.origin) return path;
  return `${window.location.origin}${path}`;
}

/**
 * Human-readable error text from a non-2xx body: FastAPI `{detail: string}`,
 * pydantic `{detail: [{msg}]}`, or the raw text / status text as fallback.
 * @param {string} bodyText
 * @param {string} statusText
 */
export function errorDetailFromBody(bodyText, statusText) {
  const fallback = (bodyText && bodyText.trim()) || statusText;
  try {
    const j = JSON.parse(bodyText);
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) {
      const msgs = j.detail
        .map((x) =>
          typeof x === "object" && x !== null && typeof x.msg === "string" ? x.msg : ""
        )
        .filter(Boolean);
      if (msgs.length) return msgs.join(" ");
    }
  } catch {
    return fallback;
  }
  return fallback;
}

/**
 * fetch() against the app's own origin.
 * @param {string} path  absolute path, e.g. "/api/measure"
 * @param {RequestInit} [init]
 */
export function apiFetch(path, init) {
  return fetch(apiUrl(path), init);
}

/** Throw ApiError with the parsed `detail` when the response is not 2xx. */
export async function throwIfNotOk(res) {
  if (res.ok) return res;
  const text = await res.text();
  throw new ApiError(errorDetailFromBody(text, res.statusText), res.status);
}
