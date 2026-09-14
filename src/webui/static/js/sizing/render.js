/**
 * Renderers: pipeline debug images, garment icons, consensus size blocks, pattern tables.
 */

import { t } from "../i18n/index.js";
import { escapeHtml } from "./text.js";


const PIPELINE_VIZ_ROWS = [
  {
    view: "front",
    items: [
      ["viz_front_pose_png_b64", "viz-front-pose"],
      ["viz_front_seg_png_b64", "viz-front-seg"],
      ["viz_front_measures_png_b64", "viz-front-measures"],
    ],
  },
  {
    view: "side",
    items: [
      ["viz_side_pose_png_b64", "viz-side-pose"],
      ["viz_side_seg_png_b64", "viz-side-seg"],
      ["viz_side_measures_png_b64", "viz-side-measures"],
    ],
  },
];


/** Show base64 PNGs from envelope `derived` (server pipeline debug).
 *
 * Layout: анфас tiles share the first row, профіль tiles share the second.
 */
export function renderPipelineModelViz(derived) {
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
      cap.textContent = t(caption);
      const img = document.createElement("img");
      img.src = "data:image/png;base64," + b64;
      img.alt = t(caption);
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
export function garmentIconUrl(garmentId) {
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

function getVerdictLabel(verdict) {
  const keys = {
    unanimous:      "verdict-unanimous",
    unanimous_edge: "verdict-unanimous-edge",
    majority:       "verdict-majority",
    majority_edge:  "verdict-majority-edge",
    no_consensus:   "verdict-no-consensus",
    insufficient:   "verdict-insufficient",
  };
  const key = keys[verdict];
  return key ? t(key) : verdict;
}


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
export function renderConsensusBlock(container, block, regionLabel) {
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
    indication.textContent = t("approx-verdict", { code: block.code });
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
  badge.textContent = getVerdictLabel(block.verdict);
  sec.appendChild(badge);

  // Between-sizes note
  if (block.between) {
    const bw = document.createElement("p");
    bw.className = "size-between";
    bw.textContent = t("borderline-size", { size1: block.between[0], size2: block.between[1] });
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
      // Try to translate measurement ID v.mid, fallback to v.label or v.mid
      const labelText = t(v.mid) !== v.mid ? t(v.mid) : (v.label || v.mid);
      label.textContent = escapeHtml(labelText);

      const result = document.createElement("span");
      result.className = "vote-result";
      if (isMissing) {
        result.textContent = v.missing === "grid" ? t("no-table") : t("no-measurement");
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
export function renderPatternBlock(pattern, rawBody, seamBody) {
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
    const label = t(key) !== key ? t(key) : (PATTERN_LABELS_UK[key] ?? key);
    tr1.innerHTML = `<td>${escapeHtml(label)}</td><td>${raw[key]}</td>`;
    rawBody.appendChild(tr1);

    if (seam[key] != null) {
      const tr2 = document.createElement("tr");
      const labelSeam = t(key) !== key ? t(key) : (PATTERN_LABELS_UK[key] ?? key);
      tr2.innerHTML = `<td>${escapeHtml(labelSeam)}</td><td>${seam[key]}</td>`;
      seamBody.appendChild(tr2);
    }
  }
}
