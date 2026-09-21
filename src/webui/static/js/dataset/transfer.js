/**
 * Transfer measurement results + photos from the main app to the dataset page.
 * Uses IndexedDB so photo blobs are not limited by sessionStorage size.
 */

const DB_NAME = "pointsx-dataset-transfer";
const DB_VERSION = 1;
const STORE = "prefill";
const KEY = "latest";

/** Map MeasurementEnvelope ids → dataset manual field ids. */
export const ENVELOPE_TO_DATASET = {
  chest_circumference: "chest_circumference",
  waist_circumference: "waist_circumference",
  hip_circumference: "hip_circumference",
  neck_circumference: "neck_circumference",
  neck_base_height: "neck_base_height_from_floor",
  // shoulder_width (full width) has no envelope counterpart — left for manual entry.
  shoulder_slope_width: "shoulder_slope_width",
  back_width_scapular: "back_width",
  chest_width_front: "chest_width",
  back_length_to_waist: "back_length_to_waist",
  front_length_to_waist: "front_length_to_waist",
  arm_length_shoulder_to_wrist: "sleeve_length",
  upper_arm_circumference: "arm_circumference_bicep",
  leg_length_inner_seam: "inner_seam",
  leg_length_outer_seam: "outer_seam",
  thigh_circumference: "thigh_circumference",
};

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

/**
 * @param {object} envelope MeasurementEnvelope v2 from /api/measure
 * @param {Blob} frontBlob
 * @param {Blob} sideBlob
 */
export function envelopeToDatasetMeasurements(envelope, heightCm) {
  /** @type {Record<string, number>} */
  const values = {};
  if (Number.isFinite(heightCm)) {
    values.height = heightCm;
  }
  const rows = Array.isArray(envelope?.measurements) ? envelope.measurements : [];
  for (const row of rows) {
    const envId = String(row?.id ?? "");
    const datasetId = ENVELOPE_TO_DATASET[envId];
    const val = Number(row?.value_cm);
    if (datasetId && Number.isFinite(val)) {
      values[datasetId] = val;
    }
  }
  return values;
}

/**
 * @param {{ heightCm: number, sex: string, measurements: Record<string, number>, frontBlob: Blob, sideBlob: Blob }} payload
 */
export async function saveDatasetPrefill(payload) {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).put(
      {
        ...payload,
        savedAt: new Date().toISOString(),
      },
      KEY
    );
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}

/** @returns {Promise<{ heightCm: number, sex: string, measurements: Record<string, number>, frontBlob: Blob, sideBlob: Blob, savedAt?: string } | null>} */
export async function loadDatasetPrefill() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(KEY);
    req.onsuccess = () => {
      db.close();
      resolve(req.result ?? null);
    };
    req.onerror = () => {
      db.close();
      reject(req.error);
    };
  });
}

export async function clearDatasetPrefill() {
  const db = await openDb();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.objectStore(STORE).delete(KEY);
    tx.oncomplete = () => {
      db.close();
      resolve();
    };
    tx.onerror = () => {
      db.close();
      reject(tx.error);
    };
  });
}
