/**
 * Tailoring view: measurement table, sizing report and pattern for the selected garment; garment strip.
 */

import { garmentsForSex, resolveGarment } from "../sizeEngine.js";
import { sizeAndPatternHandler } from "../patternEngine.js";
import { captureState } from "../capture/state.js";
import { getCaptureDom } from "../capture/dom.js";
import { t, translateGarment } from "../i18n/index.js";
import { escapeHtml } from "./text.js";
import { HIDDEN_MEASUREMENT_IDS, orderMeasurementsManual } from "./measurementOrder.js";
import { garmentIconUrl, renderConsensusBlock, renderPatternBlock } from "./render.js";


// ---------------------------------------------------------------------------
// Tailoring view refresh
// ---------------------------------------------------------------------------

export function refreshTailoringView() {
  const { tailoringMeasuresBody, panelUa, panelEu, panelUs, patternDetails, patternRawBody, patternSeamBody } = getCaptureDom();
  const env = captureState.lastMockResponse;
  if (!env || !captureState.tailoringCatalog || !tailoringMeasuresBody) return;

  const catalog  = captureState.tailoringCatalog;
  const garmentId = captureState.selectedGarmentId;
  const garment  = resolveGarment(catalog, garmentId);
  if (!garment) return;

  const sex          = env.subject?.sex ?? "other";
  const measurements = orderMeasurementsManual(env.measurements ?? []);

  // Render measurements table
  const midsForGarment = (garment.measurement_ids || []).filter(
    (mid) => !HIDDEN_MEASUREMENT_IDS.has(String(mid))
  );
  tailoringMeasuresBody.innerHTML = "";
  const garmentRows = midsForGarment
    .map((mid) => measurements.find((m) => m.id === mid))
    .filter(Boolean);
  for (const row of orderMeasurementsManual(garmentRows)) {
    if (!row) continue;
    const tr = document.createElement("tr");
    const label = t(row.id) !== row.id ? t(row.id) : (row.label_uk ?? row.id);
    tr.innerHTML =
      "<td>" + escapeHtml(label) +
      "</td><td>" + escapeHtml(String(row.value_cm)) + "</td>";
    tailoringMeasuresBody.appendChild(tr);
  }

  try {
    const envelope = _normaliseEnvelope(env);
    const report   = sizeAndPatternHandler(envelope, catalog, {
      garmentId,
      fit:     captureState.fitPreference ?? "regular",
      regions: ["ua", "eu", "us"],
    });
    if (panelUa) renderConsensusBlock(panelUa, report.sizing.ua, "Україна");
    if (panelEu) renderConsensusBlock(panelEu, report.sizing.eu, "Європа");
    if (panelUs) renderConsensusBlock(panelUs, report.sizing.us, "США");

    // Render pattern block (individual sewing parameters)
    if (patternDetails && patternRawBody && patternSeamBody) {
      renderPatternBlock(report.pattern, patternRawBody, patternSeamBody);
      patternDetails.hidden = false;
    }
  } catch (err) {
    console.error("[tailoring] sizing failed:", err);
    [panelUa, panelEu, panelUs].forEach((p) => {
      if (p) p.innerHTML = `<p class="size-warn">${escapeHtml(t("sizing-failed", { msg: err?.message ?? String(err) }))}</p>`;
    });
    if (patternDetails) patternDetails.hidden = true;
  }
}


/**
 * Ensure the response from the server is always in v2 MeasurementEnvelope shape.
 * If the server still returns a v1 MockMeasureResponse (height_cm at root, no subject),
 * wraps it into the v2 envelope structure so validateEnvelope passes.
 */
function _normaliseEnvelope(data) {
  if (data.schema === "pointsx.measurement.envelope" && (data.schema_version ?? 0) >= 2) {
    return data;
  }
  // Shim for v1 responses (during transition)
  return {
    schema:         "pointsx.measurement.envelope",
    schema_version: 2,
    request_id:     data.request_id ?? crypto.randomUUID(),
    created_at:     data.created_at ?? new Date().toISOString(),
    pipeline:       { source: "mock", model_version: "mock-0.1.0", unit_system: "metric" },
    subject: {
      height_cm:     data.height_cm ?? 170,
      sex:           data.sex ?? "female",
      posture_flags: [],
    },
    capture: {
      front: { quality: 0.85, pose_ok: true, occlusions: [] },
      side:  { quality: 0.82, pose_ok: true, occlusions: [] },
    },
    measurements: (data.measurements || []).map((m) => ({
      ...m,
      uncertainty_cm: m.uncertainty_cm ?? (1 - (m.confidence ?? 0.75)) * m.value_cm * 0.05,
      source:         m.source ?? "fused",
      quality_flags:  m.quality_flags ?? [],
    })),
    derived:  {},
    warnings: [],
  };
}


// ---------------------------------------------------------------------------
// Garment strip
// ---------------------------------------------------------------------------

export function renderGarmentStrip() {
  const { garmentStrip } = getCaptureDom();
  if (!captureState.tailoringCatalog || !garmentStrip) return;
  garmentStrip.innerHTML = "";
  const env  = captureState.lastMockResponse;
  const sex  = env?.subject?.sex ?? env?.sex ?? "other";
  const list = garmentsForSex(captureState.tailoringCatalog, sex);
  const allowed = new Set(list.map((x) => x.id));
  if (!allowed.has(captureState.selectedGarmentId) && list.length) {
    captureState.selectedGarmentId = list[0].id;
  }
  for (const g of list) {
    const btn = document.createElement("button");
    btn.type  = "button";
    btn.className = "garment-btn";
    btn.dataset.garmentId = g.id;
    btn.setAttribute("role", "radio");
    btn.setAttribute("aria-checked", g.id === captureState.selectedGarmentId ? "true" : "false");
    const label = translateGarment(g.id, g.label_uk);
    btn.setAttribute("aria-label", label);

    const img = document.createElement("img");
    img.className = "garment-icon";
    img.alt = "";
    img.loading = "lazy";
    img.src = garmentIconUrl(g.id);
    btn.appendChild(img);

    const cap = document.createElement("span");
    cap.textContent = label;
    btn.appendChild(cap);

    btn.addEventListener("click", () => {
      captureState.selectedGarmentId = g.id;
      garmentStrip.querySelectorAll(".garment-btn").forEach((b) => {
        b.setAttribute("aria-checked", b.dataset.garmentId === captureState.selectedGarmentId ? "true" : "false");
      });
      refreshTailoringView();
    });
    garmentStrip.appendChild(btn);
  }
}
