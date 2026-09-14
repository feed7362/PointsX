/**
 * POST /api/measure (photos → MeasurementEnvelope v2) and /api/measure/mock (no photos).
 */

import { apiFetch, throwIfNotOk } from "./client.js";

/**
 * @param {FormData} formData  height_cm, sex, and (real run) pose_backend + front/side files
 * @param {{ mock?: boolean }} [opts]
 * @returns {Promise<any>} MeasurementEnvelope JSON
 * @throws {import("./client.js").ApiError} on a non-2xx response
 */
export async function requestMeasurement(formData, { mock = false } = {}) {
  const res = await apiFetch(mock ? "/api/measure/mock" : "/api/measure", {
    method: "POST",
    body: formData,
  });
  await throwIfNotOk(res);
  return res.json();
}
