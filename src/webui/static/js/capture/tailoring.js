/**
 * Tailoring UI — garment strip, regional size tabs, and measurement table.
 *
 * /api/measure → MeasurementEnvelope v2
 * → sizeAndPatternHandler → SizingReport
 * → renderConsensusBlock (UA / EU / US panels)
 */

import {
  resolveGarment,
  garmentsForSex,
} from "../sizeEngine.js";
import { sizeAndPatternHandler, validateEnvelope } from "../patternEngine.js";
import { captureState } from "./state.js";
import { getCaptureDom } from "./dom.js";
import { setStatus } from "./ui.js";

// ---------------------------------------------------------------------------
// Catalog loading
// ---------------------------------------------------------------------------

export async function loadTailoringCatalog() {
  if (captureState.tailoringCatalog) return captureState.tailoringCatalog;
  const res = await fetch("/static/data/tailoring_config.json?v=3");
  if (!res.ok) throw new Error("Не вдалося завантажити tailoring_config.json");
  captureState.tailoringCatalog = await res.json();
  return captureState.tailoringCatalog;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

export function sexLabelUk(sex) {
  if (sex === "male")   return "чоловік";
  if (sex === "female") return "жінка";
  return "інше";
}

export function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function formatMeasureHttpError(res, bodyText) {
  const fallback = (bodyText && bodyText.trim()) || res.statusText;
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

const MEASUREMENT_MANUAL_ORDER = [
  "chest_circumference",
  "waist_circumference",
  "hip_circumference",
  "thigh_circumference",
 
  "neck_base_height",
  "chest_width_front",
  "back_width_scapular",
  "shoulder_slope_width",

  "arm_length_shoulder_to_wrist",
  "leg_length_outer_seam",
  "leg_length_inner_seam",
  
  "back_length_to_waist",
  "front_length_to_waist",

  "neck_circumference",
  "upper_arm_circumference", 
  "wrist_circumference",
  "calf_circumference",
  "ankle_circumference",
];

const HIDDEN_MEASUREMENT_IDS = new Set([
  "neck_circumference",
  "upper_arm_circumference",
  "wrist_circumference",
  "calf_circumference",
  "ankle_circumference",
  "back_width_scapular",
  "front_length_to_waist",
]);

function orderMeasurementsManual(list) {
  const rows = (Array.isArray(list) ? list : []).filter(
    (r) => !HIDDEN_MEASUREMENT_IDS.has(String(r?.id ?? ""))
  );
  const byId = new Map(rows.map((r) => [r?.id, r]));
  const ordered = [];
  for (const id of MEASUREMENT_MANUAL_ORDER) {
    const row = byId.get(id);
    if (row) ordered.push(row);
  }
  for (const row of rows) {
    if (!MEASUREMENT_MANUAL_ORDER.includes(row?.id)) ordered.push(row);
  }
  return ordered;
}

const PIPELINE_VIZ_ROWS = [
  {
    view: "front",
    items: [
      ["viz_front_pose_png_b64", "Анфас — поза"],
      ["viz_front_seg_png_b64", "Анфас — силует"],
      ["viz_front_measures_png_b64", "Анфас — лінії зняття мірок"],
    ],
  },
  {
    view: "side",
    items: [
      ["viz_side_pose_png_b64", "Профіль — поза"],
      ["viz_side_seg_png_b64", "Профіль — силует"],
      ["viz_side_measures_png_b64", "Профіль — лінії зняття мірок"],
    ],
  },
];

/** Show base64 PNGs from envelope `derived` (server pipeline debug).
 *
 * Layout: анфас tiles share the first row, профіль tiles share the second.
 */
function renderPipelineModelViz(derived) {
  const wrap = document.getElementById("model-viz");
  const grid = document.getElementById("model-viz-grid");
  if (!wrap || !grid) return;
  grid.innerHTML = "";
  const d = derived && typeof derived === "object" ? derived : {};
  let any = false;
  for (const row of PIPELINE_VIZ_ROWS) {
    const rowEl = document.createElement("div");
    rowEl.className = `model-viz-row model-viz-row--${row.view}`;
    let rowAny = false;
    for (const [key, caption] of row.items) {
      const b64 = d[key];
      if (typeof b64 !== "string" || !b64.length) continue;
      rowAny = true;
      const fig = document.createElement("figure");
      fig.className = "model-viz-item";
      const cap = document.createElement("figcaption");
      cap.textContent = caption;
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64;
      img.alt = caption;
      fig.append(cap, img);
      rowEl.appendChild(fig);
    }
    if (rowAny) {
      any = true;
      grid.appendChild(rowEl);
    }
  }
  wrap.hidden = !any;
}

/**
 * Local, license-safe garment icons.
 * These are app-owned static assets under /static/images/garments.
 */
function garmentIconUrl(garmentId) {
  const byId = {
    shirt: "/static/images/garments/shirt.png",
    tshirt: "/static/images/garments/tshirt.png",
    polo: "/static/images/garments/polo.png",
    blouse: "/static/images/garments/blouse.png",
    jacket: "/static/images/garments/jacket.png",
    coat: "/static/images/garments/coat.png",
    vest: "/static/images/garments/vest.png",
    hoodie: "/static/images/garments/hoodie.png",
    cardigan: "/static/images/garments/cardigan.png",
    sport_top: "/static/images/garments/sport_top.png",
    raincoat: "/static/images/garments/raincoat.png",
    pajama_top: "/static/images/garments/pajama_top.png",
    pants: "/static/images/garments/pants.png",
    jeans: "/static/images/garments/jeans.png",
    shorts: "/static/images/garments/shorts.png",
    pajama_bottom: "/static/images/garments/pajama_bottom.png",
    skirt: "/static/images/garments/skirt.png",
    pencil_skirt: "/static/images/garments/pencil_skirt.png",
    dress: "/static/images/garments/dress.png",
    dress_fitted: "/static/images/garments/dress_fitted.png",
    overalls: "/static/images/garments/overalls.png",
  };
  return byId[garmentId] ?? "/static/images/garments/top-generic.svg";
}

// ---------------------------------------------------------------------------
// Verdict badge helpers
// ---------------------------------------------------------------------------

const VERDICT_LABEL = {
  unanimous:      "Однозначно",
  unanimous_edge: "Однозначно (крайній розмір)",
  majority:       "Більшість 2/3",
  majority_edge:  "Більшість (крайній розмір)",
  no_consensus:   "Неоднозначно",
  insufficient:   "Недостатньо даних",
};

const VERDICT_CLASS = {
  unanimous:      "verdict--unanimous",
  unanimous_edge: "verdict--majority",
  majority:       "verdict--majority",
  majority_edge:  "verdict--low",
  no_consensus:   "verdict--low",
  insufficient:   "verdict--none",
};

// ---------------------------------------------------------------------------
// v2 consensus panel renderer
// ---------------------------------------------------------------------------

/**
 * Render a SizingBlock (from evaluateGarmentSize) into a container element.
 * @param {HTMLElement} container
 * @param {any}         block      SizingBlock from evaluateGarmentSize
 * @param {string}      regionLabel e.g. "Україна", "Європа", "США"
 */
function renderConsensusBlock(container, block, regionLabel) {
  container.innerHTML = "";

  const sec = document.createElement("section");
  sec.className = "size-block size-block--v2";

  // Primary size display
  const primary = document.createElement("div");
  primary.className = "size-primary";
  if (block.verdict === "no_consensus") {
    primary.textContent = block.rangeCode ?? "—";
    const indication = document.createElement("span");
    indication.className = "size-indication";
    indication.textContent = `орієнт. ${block.code}`;
    primary.appendChild(indication);
  } else if (block.verdict === "insufficient") {
    primary.textContent = block.provisional?.code ?? "—";
  } else {
    primary.textContent = block.code ?? "—";
  }
  sec.appendChild(primary);

  // Verdict badge
  const badge = document.createElement("span");
  badge.className = `verdict-badge ${VERDICT_CLASS[block.verdict] ?? ""}`;
  badge.textContent = VERDICT_LABEL[block.verdict] ?? block.verdict;
  sec.appendChild(badge);

  // Between-sizes note
  if (block.between) {
    const bw = document.createElement("p");
    bw.className = "size-between";
    bw.textContent = `На межі розмірів: ${block.between[0]} / ${block.between[1]}`;
    sec.appendChild(bw);
  }

  // Votes breakdown
  if (Array.isArray(block.votes) && block.votes.length > 0) {
    const voteList = document.createElement("ul");
    voteList.className = "vote-list";
    for (const v of block.votes) {
      const li = document.createElement("li");
      const isOutlier  = block.outlier?.mid === v.mid;
      const isMissing  = v.missing != null;
      const isLowConf  = v.lowConf;
      li.className = [
        "vote-item",
        isOutlier  ? "vote-item--outlier" : "",
        isMissing  ? "vote-item--missing" : "",
        isLowConf  ? "vote-item--lowconf" : "",
      ].filter(Boolean).join(" ");

      const label = document.createElement("span");
      label.className = "vote-label";
      label.textContent = escapeHtml(v.label || v.mid);

      const result = document.createElement("span");
      result.className = "vote-result";
      if (isMissing) {
        result.textContent = v.missing === "grid" ? "немає таблиці" : "немає мірки";
      } else {
        result.textContent = v.code ?? "—";
        if (isLowConf) result.textContent += " ⚠";
      }

      li.appendChild(label);
      li.appendChild(result);
      voteList.appendChild(li);
    }
    sec.appendChild(voteList);
  }

  // Tailoring hint
  if (block.tailoringHint) {
    const hint = document.createElement("p");
    hint.className = "size-hint";
    hint.textContent = block.tailoringHint;
    sec.appendChild(hint);
  }

  // Warnings
  if (block.warnings?.length) {
    const wEl = document.createElement("p");
    wEl.className = "size-warn";
    wEl.textContent = block.warnings.join(" ");
    sec.appendChild(wEl);
  }

  container.appendChild(sec);
}

// ---------------------------------------------------------------------------
// Pattern block renderer
// ---------------------------------------------------------------------------

const PATTERN_LABELS_UK = {
  armscye_depth:          "Глибина пройми",
  sleeve_cap_height:      "Висота окату рукава",
  sleeve_cap_width:       "Ширина окату",
  chest_pattern:          "Обхват грудей (лекало)",
  waist_pattern:          "Обхват талії (лекало)",
  hip_pattern:            "Обхват стегон (лекало)",
  back_length_cb:         "Довжина спини",
  hip_line_depth:         "Рівень лінії стегон",
  cross_back:             "Ширина спини",
  cross_front:            "Ширина переду",
  shoulder_seam:          "Довжина плечового шва",
  neck_width:             "Ширина горловини",
  neck_depth_front:       "Глибина горловини (перед)",
  neck_depth_back:        "Глибина горловини (спинка)",
  dart_intake_total:      "Загальна виточка",
  dart_back:              "Виточка спинки",
  dart_side:              "Виточка бокова",
  dart_front:             "Виточка переду",
  sleeve_underarm_length: "Довжина рукава (від пахви)",
  bicep_pattern:          "Обхват біцепса (лекало)",
  cuff_pattern:           "Обхват манжета",
  crotch_depth:           "Глибина сидіння",
  knee_pattern:           "Обхват коліна (лекало)",
  hem_width_pants:        "Ширина низу штанини",
};

/**
 * Render pattern block dimensions (raw + with seam allowances) into the two tables.
 * @param {any} pattern  { raw: {...}, with_seam_allowances: {...}, fit: string }
 * @param {HTMLElement} rawBody
 * @param {HTMLElement} seamBody
 */
function renderPatternBlock(pattern, rawBody, seamBody) {
  rawBody.innerHTML = "";
  seamBody.innerHTML = "";

  const raw = pattern.raw ?? {};
  const seam = pattern.with_seam_allowances ?? {};

  // Order: bodice, sleeves, pants
  const order = [
    "armscye_depth", "sleeve_cap_height", "sleeve_cap_width",
    "chest_pattern", "waist_pattern", "hip_pattern",
    "back_length_cb", "hip_line_depth",
    "cross_back", "cross_front", "shoulder_seam",
    "neck_width", "neck_depth_front", "neck_depth_back",
    "dart_intake_total", "dart_back", "dart_side", "dart_front",
    "sleeve_underarm_length", "bicep_pattern", "cuff_pattern",
    "crotch_depth", "knee_pattern", "hem_width_pants",
  ];

  for (const key of order) {
    if (raw[key] == null) continue;
    const tr1 = document.createElement("tr");
    tr1.innerHTML = `<td>${escapeHtml(PATTERN_LABELS_UK[key] ?? key)}</td><td>${raw[key]}</td>`;
    rawBody.appendChild(tr1);

    if (seam[key] != null) {
      const tr2 = document.createElement("tr");
      tr2.innerHTML = `<td>${escapeHtml(PATTERN_LABELS_UK[key] ?? key)}</td><td>${seam[key]}</td>`;
      seamBody.appendChild(tr2);
    }
  }
}

// ---------------------------------------------------------------------------
// Tab wiring
// ---------------------------------------------------------------------------

export function selectSizeTab(which) {
  const { tabUa, tabEu, tabUs, panelUa, panelEu, panelUs } = getCaptureDom();
  if (!tabUa || !tabEu || !tabUs || !panelUa || !panelEu || !panelUs) return;
  const tabs = [
    { id: "ua", tab: tabUa, panel: panelUa },
    { id: "eu", tab: tabEu, panel: panelEu },
    { id: "us", tab: tabUs, panel: panelUs },
  ];
  for (const { id, tab, panel } of tabs) {
    const sel = id === which;
    tab.setAttribute("aria-selected", sel ? "true" : "false");
    tab.tabIndex = sel ? 0 : -1;
    panel.hidden = !sel;
  }
}

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
    tr.innerHTML =
      "<td>" + escapeHtml(row.label_uk ?? row.id) +
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
      if (p) p.innerHTML = `<p class="size-warn">Помилка: ${escapeHtml(err?.message ?? String(err))}</p>`;
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
    btn.setAttribute("aria-label", g.label_uk);

    const img = document.createElement("img");
    img.className = "garment-icon";
    img.alt = "";
    img.loading = "lazy";
    img.src = garmentIconUrl(g.id);
    btn.appendChild(img);

    const cap = document.createElement("span");
    cap.textContent = g.label_uk;
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

// ---------------------------------------------------------------------------
// Tab keyboard wiring
// ---------------------------------------------------------------------------

export function ensureSizeTabsWired() {
  const { tabUa, tabEu, tabUs } = getCaptureDom();
  if (captureState.sizeTabsWired || !tabUa || !tabEu || !tabUs) return;
  captureState.sizeTabsWired = true;
  const order    = ["ua", "eu", "us"];
  const tabById  = { ua: tabUa, eu: tabEu, us: tabUs };
  for (const id of order) {
    tabById[id].addEventListener("click", () => selectSizeTab(id));
  }
  for (let i = 0; i < order.length; i++) {
    const id  = order[i];
    const tab = tabById[id];
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const next = order[(i + 1) % order.length];
        selectSizeTab(next);
        tabById[next].focus();
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = order[(i - 1 + order.length) % order.length];
        selectSizeTab(prev);
        tabById[prev].focus();
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Measure button handler
// ---------------------------------------------------------------------------

export function attachMeasureHandler() {
    const {
    btnMeasure,
    btnMeasureTest,
    resultsSection,
    resultsBody,
    heightInput,
    sexSelect,
    poseBackendSelect,
    tailoringIntro,
    garmentStripWrap,
    tailoringPanels,
    tailoringDisclaimer,
  } = getCaptureDom();

  async function runMeasureRequest(mode = "capture") {
    const useTestImages = mode === "test";
    if (!useTestImages && (!captureState.frontBlob || !captureState.sideBlob)) {
      setStatus("Потрібні обидва знімки — анфас і профіль.", true);
      return;
    }
    if (!resultsSection || !resultsBody) {
      setStatus("Помилка: немає контейнера результатів у розмітці.", true);
      return;
    }
    const heightCmNum = Number(String(heightInput.value).replace(",", "."));
    if (!Number.isFinite(heightCmNum) || heightCmNum < 100 || heightCmNum > 250) {
      setStatus("Вкажіть зріст від 100 до 250 см.", true);
      return;
    }
    setStatus(useTestImages ? "Тестовий розрахунок…" : "Обчислення…");
    resultsSection.hidden = true;
    const modelVizEl = document.getElementById("model-viz");
    if (modelVizEl) modelVizEl.hidden = true;

    const fd = new FormData();
    fd.append("height_cm", String(heightCmNum));
    fd.append("sex",       sexSelect.value);
    // POINTSX_API_BASE is injected by the static host (Vercel) via
    // /static/config.js — empty string falls back to same-origin (HF Space
    // dev where frontend + backend share the host).
    const apiBase = (window.POINTSX_API_BASE || "").replace(/\/+$/, "");
    const measureUrl = apiBase + (useTestImages ? "/api/measure/mock" : "/api/measure");
    if (!useTestImages) {
      if (poseBackendSelect && poseBackendSelect.value) {
        fd.append("pose_backend", poseBackendSelect.value);
      }
      const frontName =
        captureState.frontBlob instanceof File && captureState.frontBlob.name
          ? captureState.frontBlob.name
          : "front.jpg";
      const sideName =
        captureState.sideBlob instanceof File && captureState.sideBlob.name
          ? captureState.sideBlob.name
          : "side.jpg";
      fd.append("front", captureState.frontBlob, frontName);
      fd.append("side", captureState.sideBlob, sideName);
    }

    try {
      const res = await fetch(measureUrl, { method: "POST", body: fd });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(formatMeasureHttpError(res, text));
      }
      const data = await res.json();
      console.log("[PointsX] Full model output:", data);
      const measurements = orderMeasurementsManual(data.measurements ?? []);
      if (!measurements.length) {
        captureState.lastMockResponse = null;
        const mvEmpty = document.getElementById("model-viz");
        if (mvEmpty) mvEmpty.hidden = true;
        resultsBody.innerHTML = "";
        if (tailoringIntro) tailoringIntro.hidden = true;
        if (garmentStripWrap) garmentStripWrap.hidden = true;
        if (tailoringPanels) tailoringPanels.hidden = true;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = true;
        const patternDetailsEl = document.getElementById("pattern-details");
        if (patternDetailsEl) patternDetailsEl.hidden = true;
        resultsSection.hidden = false;
        setStatus(
          "Сервер повернув порожній список мірок. Спробуйте інші знімки або перевірте позу й освітлення.",
          true
        );
        return;
      }

      captureState.lastMockResponse = data;
      renderPipelineModelViz(data.derived);

      const sex      = data.subject?.sex ?? sexSelect.value;
      const heightCm = data.subject?.height_cm ?? heightCmNum;
      resultsBody.innerHTML = "";
      for (const row of measurements) {
        const tr = document.createElement("tr");
        tr.innerHTML =
          "<td>" + escapeHtml(row.label_uk ?? row.id) +
          "</td><td>" + escapeHtml(String(row.value_cm)) + "</td>";
        resultsBody.appendChild(tr);
      }

      ensureSizeTabsWired();
      selectSizeTab("ua");

      try {
        await loadTailoringCatalog();
        captureState.selectedGarmentId = captureState.tailoringCatalog.garments[0]?.id || "shirt";

        if (tailoringIntro) {
          tailoringIntro.textContent = `Зріст: ${heightCm} см. Стать: ${sexLabelUk(sex)}. Оберіть тип одягу для орієнтовних розмірів (Україна / Європа / США).`;
          tailoringIntro.hidden = false;
        }
        if (garmentStripWrap)   garmentStripWrap.hidden   = false;
        if (tailoringPanels)    tailoringPanels.hidden    = false;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = false;

        renderGarmentStrip();
        refreshTailoringView();
      } catch (cfgErr) {
        if (tailoringIntro) {
          tailoringIntro.textContent = "Пошив і сітки: " + (cfgErr?.message ?? String(cfgErr));
          tailoringIntro.hidden = false;
        }
        if (garmentStripWrap)   garmentStripWrap.hidden   = true;
        if (tailoringPanels)    tailoringPanels.hidden    = true;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = true;
      }

      resultsSection.hidden = false;
      setStatus("Готово.");
    } catch (e) {
      const mv = document.getElementById("model-viz");
      if (mv) mv.hidden = true;
      setStatus("Помилка запиту: " + (e?.message ?? String(e)), true);
    }
  }

  btnMeasure.addEventListener("click", async () => {
    await runMeasureRequest("capture");
  });

  if (btnMeasureTest) {
    btnMeasureTest.addEventListener("click", async () => {
      await runMeasureRequest("test");
    });
  }
}
