/**
 * Sizing engine — v2 (3-measure consensus)
 *
 * v2 path:   evaluateGarmentSize  → used when catalog.version >= 2
 * Entry:     formatSizeTabs       → dispatches on catalog.version
 */

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/** @param {any[]} measurements */
export function measurementsToMap(measurements) {
  /** @type {Record<string, number>} */
  const m = {};
  for (const row of measurements) {
    const id = row.id;
    if (id in m) continue;
    m[id] = Number(row.value_cm);
  }
  return m;
}

/** @param {any[]} measurements */
export function confidencesToMap(measurements) {
  /** @type {Record<string, number>} */
  const c = {};
  for (const row of measurements) {
    if (row.id in c) continue;
    c[row.id] = typeof row.confidence === "number" ? row.confidence : 0.75;
  }
  return c;
}

/** @param {{ garments: any[] }} catalog @param {string} id */
export function resolveGarment(catalog, id) {
  const g = catalog.garments.find((x) => x.id === id);
  if (!g) return null;
  if (g.extends) {
    const base = resolveGarment(catalog, g.extends);
    if (!base) return { ...g, audience: g.audience ?? "unisex" };
    const ids = [...new Set([...(base.measurement_ids || []), ...(g.measurement_ids || [])])];
    const audience =
      g.audience !== undefined && g.audience !== null && g.audience !== ""
        ? g.audience
        : base.audience ?? "unisex";
    const sizingMeasurements = g.sizing_measurements ?? base.sizing_measurements ?? null;
    const fitDefault = g.fit_preference_default ?? base.fit_preference_default ?? "regular";
    return {
      ...base,
      ...g,
      measurement_ids: ids,
      sizing_measurements: sizingMeasurements,
      fit_preference_default: fitDefault,
      audience,
      extends: undefined,
    };
  }
  return { ...g, audience: g.audience ?? "unisex" };
}

/** @param {any} resolvedGarment @param {"male"|"female"|"other"} sex */
export function garmentVisibleForSex(resolvedGarment, sex) {
  if (!resolvedGarment) return false;
  const aud = resolvedGarment.audience || "unisex";
  if (sex === "other") return true;
  if (aud === "unisex") return true;
  return aud === sex;
}

/** @param {{ garments: any[] }} catalog @param {"male"|"female"|"other"} sex */
export function garmentsForSex(catalog, sex) {
  return (catalog.garments || []).filter((raw) =>
    garmentVisibleForSex(resolveGarment(catalog, raw.id), sex)
  );
}

// ---------------------------------------------------------------------------
// v2 — 3-measure consensus engine
// ---------------------------------------------------------------------------

const MEAS_SLUG = {
  chest_circumference: "chest",
  waist_circumference: "waist",
  hip_circumference:   "hip",
  thigh_circumference: "thigh",
};

const MEAS_LABEL_UK = {
  chest_circumference: "груди",
  waist_circumference: "талія",
  hip_circumference:   "стегна",
  thigh_circumference: "стегно",
};

const BAND_EDGE_TOLERANCE_CM = 0.7;

/**
 * Half-open band lookup [min, max).
 * Returns { ordinal, code, edge: null|"below"|"above", betweenSizes: null|[a,b] }
 */
function pickBand(bands, v) {
  const sorted = [...bands].sort((a, b) => a.min_cm - b.min_cm);

  // Check boundary proximity for any band edge
  let betweenSizes = null;
  for (let i = 0; i < sorted.length - 1; i++) {
    const edge = sorted[i].max_cm;  // == sorted[i+1].min_cm
    if (Math.abs(v - edge) <= BAND_EDGE_TOLERANCE_CM) {
      betweenSizes = [sorted[i].code, sorted[i + 1].code];
      break;
    }
  }

  // Half-open hit
  for (let i = 0; i < sorted.length; i++) {
    const b = sorted[i];
    const isLast = i === sorted.length - 1;
    if (isLast ? (v >= b.min_cm && v <= b.max_cm) : (v >= b.min_cm && v < b.max_cm)) {
      return { ordinal: b.ordinal, code: b.code, edge: null, betweenSizes };
    }
  }

  // Out of table
  if (v < sorted[0].min_cm) {
    return { ordinal: sorted[0].ordinal, code: sorted[0].code, edge: "below", betweenSizes: null };
  }
  const last = sorted[sorted.length - 1];
  return { ordinal: last.ordinal, code: last.code, edge: "above", betweenSizes: null };
}

/**
 * Find the v2 consensus_band grid for (region, sex, measurementId, family).
 * Uses an exact ID match: {region}_{sexKey}_{slug}_{family}.
 * Falls back to null when no grid is registered.
 */
function findGrid(catalog, region, sex, measureId, family) {
  const slug   = MEAS_SLUG[measureId];
  const sexKey = sex === "male" ? "men" : "women";
  if (!slug || !family) return null;
  const targetId = `${region}_${sexKey}_${slug}_${family}`;
  return (catalog.grids || []).find(
    (g) => g.kind === "consensus_band" &&
           g.id === targetId &&
           g.measurement_key === measureId
  ) ?? null;
}

/**
 * Weighted median ordinal using weight × confidence as vote mass.
 * Returns the fractional median rounded to nearest integer.
 */
function weightedMedianOrdinal(votes) {
  const items = votes
    .filter((v) => v.ordinal != null)
    .map((v) => ({ ord: v.ordinal, mass: v.weight * v.conf }))
    .sort((a, b) => a.ord - b.ord);

  const total = items.reduce((s, x) => s + x.mass, 0);
  if (total === 0) return items[Math.floor(items.length / 2)].ord;

  let cum = 0;
  const half = total / 2;
  for (const item of items) {
    cum += item.mass;
    if (cum >= half) return item.ord;
  }
  return items[items.length - 1].ord;
}

/**
 * The most-frequent ordinal among present votes; null if all distinct.
 */
function modeOf(ords) {
  const counts = {};
  for (const o of ords) counts[o] = (counts[o] || 0) + 1;
  const maxCount = Math.max(...Object.values(counts));
  if (maxCount < 2) return null;
  return Number(Object.keys(counts).find((k) => counts[k] === maxCount));
}

function outlierVote(votes, mode) {
  return votes.find((v) => v.ordinal != null && v.ordinal !== mode) ?? null;
}

function codeForOrdinal(catalog, scaleKey, ordinal) {
  const scale = catalog.ordinal_scales?.[scaleKey];
  if (!scale) return String(ordinal);
  return scale[ordinal] ?? String(ordinal);
}

/**
 * Build a tailoring hint for no-consensus: identifies the measurement that deviates most
 * from the weighted-median ordinal and suggests a tailoring action.
 */
function buildTailoringHint(votes, medianOrd, catalog, scaleKey) {
  const present = votes.filter((v) => v.ordinal != null);
  if (!present.length) return null;

  let worstVote = present[0];
  let worstDev  = Math.abs(present[0].ordinal - medianOrd);
  for (const v of present) {
    const d = Math.abs(v.ordinal - medianOrd);
    if (d > worstDev) { worstDev = d; worstVote = v; }
  }
  if (worstDev === 0) return null;

  const label    = MEAS_LABEL_UK[worstVote.mid] || worstVote.mid;
  const medCode  = codeForOrdinal(catalog, scaleKey, medianOrd);
  const devCode  = codeForOrdinal(catalog, scaleKey, worstVote.ordinal);
  const dir      = worstVote.ordinal > medianOrd ? "більше" : "менше";
  return `${label} дає розмір ${devCode} (${dir} на ${worstDev} позиції відносно ${medCode}) — рекомендовано індивідуальне пошиття.`;
}

/**
 * Main v2 sizing function.
 * Returns a SizingBlock: { verdict, ordinal, code, votes, outlier?, between?, tailoringHint?, confidence }
 *
 * @param {any}    catalog
 * @param {any}    garment   resolvedGarment with sizing_measurements
 * @param {Record<string,number>} measures  { id → value_cm }
 * @param {Record<string,number>} confidences { id → 0..1 }
 * @param {string} region    "ua"|"eu"|"us"
 * @param {"male"|"female"|"other"} sex
 * @param {{ fit?: string, label?: string }} [opts]
 */
export function evaluateGarmentSize(catalog, garment, measures, confidences, region, sex, opts = {}) {
  const trio = garment.sizing_measurements;
  if (!trio || trio.length < 3) {
    return { verdict: "insufficient", votes: [], warnings: ["sizing_measurements не налаштовано для цього виробу"] };
  }

  const effectiveSex = sex === "other" ? "female" : sex;
  const votes = [];
  const warnings = [];
  if (sex === "other") {
    warnings.push("Стать «інше»: застосовано орієнтовні таблиці жіночої сітки.");
  }
  let scaleKey = null;

  for (const m of trio) {
    const grid = findGrid(catalog, region, effectiveSex, m.id, m.family);
    if (!grid) {
      votes.push({ mid: m.id, label: MEAS_LABEL_UK[m.id] || m.id, missing: "grid", weight: m.weight });
      continue;
    }
    if (!scaleKey) scaleKey = grid.scale_key;

    const v = measures[m.id];
    if (v == null || !Number.isFinite(v)) {
      votes.push({ mid: m.id, label: MEAS_LABEL_UK[m.id] || m.id, missing: "value", weight: m.weight });
      continue;
    }

    const rawConf = confidences[m.id] ?? 0.75;
    const effectiveWeight = rawConf < 0.5 ? m.weight * 0.5 : m.weight;

    const { ordinal, code, edge, betweenSizes } = pickBand(grid.bands, v);
    votes.push({
      mid:      m.id,
      label:    MEAS_LABEL_UK[m.id] || m.id,
      raw:      v,
      ordinal,
      code,
      conf:     rawConf,
      weight:   effectiveWeight,
      edge:     edge ?? null,
      between:  betweenSizes,
      lowConf:  rawConf < 0.5,
    });

    if (edge) warnings.push(`${MEAS_LABEL_UK[m.id] || m.id}: значення ${v} см за межами таблиці (${edge}) — рекомендовано замовне пошиття.`);
    if (rawConf < 0.5) warnings.push(`Низька достовірність мірки "${MEAS_LABEL_UK[m.id] || m.id}" — переробіть вимірювання.`);
  }

  const present = votes.filter((v) => v.ordinal != null);
  if (present.length < 3) {
    const provisional = present.length > 0 ? present.reduce((a, b) => (a.weight > b.weight ? a : b)) : null;
    return {
      verdict: "insufficient",
      provisional: provisional ? { ordinal: provisional.ordinal, code: provisional.code } : null,
      votes,
      warnings,
      confidence: "none",
    };
  }

  const ords   = present.map((v) => v.ordinal).sort((a, b) => a - b);
  const spread = ords[2] - ords[0];
  const anyEdge = present.some((v) => v.edge);
  const between = present.find((v) => v.between)?.between ?? null;

  if (spread === 0) {
    const winOrd = ords[0];
    const winCode = present[0].code;
    return {
      verdict:    anyEdge ? "unanimous_edge" : "unanimous",
      ordinal:    winOrd,
      code:       winCode,
      votes,
      between,
      warnings,
      confidence: anyEdge ? "medium" : "high",
    };
  }

  const mode = modeOf(ords);
  if (mode != null && spread <= 1) {
    const outlier    = outlierVote(votes, mode);
    const modeVote   = present.find((v) => v.ordinal === mode);
    const lowConfOut = outlier?.lowConf;
    if (lowConfOut) {
      warnings.push(`Низька достовірність мірки "${outlier.label}" — не враховується як відхилення від більшості.`);
    }
    return {
      verdict:      anyEdge ? "majority_edge" : "majority",
      ordinal:      mode,
      code:         modeVote?.code ?? String(mode),
      outlier:      outlier ? { mid: outlier.mid, label: outlier.label, code: outlier.code, ordinal: outlier.ordinal } : null,
      votes,
      between,
      warnings,
      confidence:   anyEdge ? "low" : "medium",
    };
  }

  // No consensus — compute weighted median
  const medianOrd = weightedMedianOrdinal(present);
  const medVote   = present.find((v) => v.ordinal === medianOrd) ?? present[0];
  const hint      = buildTailoringHint(votes, medianOrd, catalog, scaleKey ?? "top.women");
  const minCode   = present.find((v) => v.ordinal === ords[0])?.code ?? String(ords[0]);
  const maxCode   = present.find((v) => v.ordinal === ords[ords.length - 1])?.code ?? String(ords[ords.length - 1]);

  return {
    verdict:       "no_consensus",
    ordinal:       medianOrd,
    code:          medVote?.code ?? String(medianOrd),
    rangeCode:     `${minCode}–${maxCode}`,
    votes,
    between,
    tailoringHint: hint,
    warnings,
    confidence:    "low",
  };
}

// ---------------------------------------------------------------------------
// v2 — formatSizeTabs dispatcher (v2 path)
// ---------------------------------------------------------------------------

/**
 * Build UA/EU/US sizing blocks using the v2 consensus engine.
 * Returns { uaBlocks, euBlocks, usBlocks } each as SizingBlock[] for renderSizePanelContent.
 */
export function formatSizeTabsV2(catalog, garmentResolved, measures, confidences, sex) {
  const regions = ["ua", "eu", "us"];
  const blocks  = {};
  for (const r of regions) {
    blocks[r] = evaluateGarmentSize(catalog, garmentResolved, measures, confidences, r, sex);
  }
  return { uaBlocks: [blocks.ua], euBlocks: [blocks.eu], usBlocks: [blocks.us] };
}

/** Convenience alias — always uses the v2 consensus engine. */
export function formatSizeTabs(catalog, garmentResolved, measures, _height_cm, sex, confidences = {}) {
  return formatSizeTabsV2(catalog, garmentResolved, measures, confidences, sex);
}
