/**
 * Довідкові розміри з tailoring_config.json (не медична точність).
 * @param {any[]} measurements
 */
export function measurementsToMap(measurements) {
  /** @type {Record<string, number>} */
  const m = {};
  for (const row of measurements) {
    m[row.id] = Number(row.value_cm);
  }
  return m;
}

/**
 * @param {{ garments: any[] }} catalog
 * @param {string} id
 */
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
    return {
      ...base,
      ...g,
      measurement_ids: ids,
      audience,
      extends: undefined,
    };
  }
  return { ...g, audience: g.audience ?? "unisex" };
}

/**
 * @param {any} resolvedGarment
 * @param {"male"|"female"|"other"} sex
 */
export function garmentVisibleForSex(resolvedGarment, sex) {
  if (!resolvedGarment) return false;
  const aud = resolvedGarment.audience || "unisex";
  if (sex === "other") return true;
  if (aud === "unisex") return true;
  return aud === sex;
}

/**
 * @param {{ garments: any[] }} catalog
 * @param {"male"|"female"|"other"} sex
 */
export function garmentsForSex(catalog, sex) {
  return (catalog.garments || []).filter((raw) =>
    garmentVisibleForSex(resolveGarment(catalog, raw.id), sex)
  );
}

/** @param {"male"|"female"|"other"} sex */
function sizingSex(sex) {
  if (sex === "male") return "male";
  if (sex === "female") return "female";
  return "female";
}

/**
 * @param {string} category
 * @param {"male"|"female"|"other"} sex
 */
export function tabGridPlan(category, sex) {
  const s = sizingSex(sex);
  const cat = category || "upper";
  if (cat === "upper") {
    return {
      ua: s === "male" ? "ua_men_top" : "ua_women_top",
      eu: s === "male" ? "eu_men_top" : "eu_women_top",
      us: s === "male" ? "us_men_top" : "us_women_top",
    };
  }
  if (cat === "lower") {
    return {
      ua: s === "male" ? "ua_men_bottom" : "ua_women_bottom",
      eu: s === "male" ? "eu_men_bottom" : "eu_women_bottom",
      us: s === "male" ? "us_men_bottom" : "us_women_bottom",
    };
  }
  if (cat === "dress") {
    return {
      ua: s === "male" ? "ua_men_dress" : "ua_women_dress",
      eu: s === "male" ? "eu_men_dress" : "eu_women_dress",
      us: s === "male" ? "us_men_top" : "us_women_dress",
    };
  }
  if (cat === "full") {
    return {
      ua: s === "male" ? ["ua_men_dress", "ua_men_bottom"] : ["ua_women_dress", "ua_women_bottom"],
      eu: s === "male" ? ["eu_men_dress", "eu_men_bottom"] : ["eu_women_dress", "eu_women_bottom"],
      us: s === "male" ? ["us_men_full", "us_men_bottom"] : ["us_women_dress", "us_women_bottom"],
    };
  }
  return { ua: "ua_women_top", eu: "eu_women_top", us: "us_women_top" };
}

/**
 * @param {any} grid
 * @param {Record<string, number>} measures
 * @param {number} _height_cm
 * @param {"male"|"female"|"other"} sex
 */
export function evaluateGrid(grid, measures, _height_cm, sex) {
  const warnings = [];
  const s = sizingSex(sex);
  if (grid.sex && grid.sex !== s && sex !== "other") {
    /* allow mixed tables — our grids are sex-tagged */
  }
  if (sex === "other") {
    warnings.push("Стать «інше»: сітка орієнтована на жіночі діапазони як приклад.");
  }

  if (grid.kind === "circumference_band") {
    const v = measures[grid.measurement_key];
    if (v == null || Number.isNaN(v)) {
      return { lines: ["Немає мірки для цієї сітки."], warnings };
    }
    const { code, warn } = pickCircumferenceBand(grid.bands, v);
    if (warn) warnings.push(warn);
    return { lines: [`Розмір (умовно): ${code}`, `Обхват: ${v} см`], warnings };
  }

  if (grid.kind === "circumference_band_dual") {
    const v = measures[grid.measurement_key];
    if (v == null || Number.isNaN(v)) {
      return { lines: ["Немає мірки для цієї сітки."], warnings };
    }
    const { band, warn } = pickCircumferenceBandDual(grid.bands, v);
    if (warn) warnings.push(warn);
    if (!band) return { lines: [`Обхват: ${v} см`], warnings };
    return {
      lines: [`Літери (US): ${band.letter}`, `Числовий діапазон (misses тощо): ${band.numeric}`, `Обхват грудей: ${v} см`],
      warnings,
    };
  }

  if (grid.kind === "two_key_band") {
    const w = measures[grid.primary_key];
    const ins = measures[grid.secondary_key];
    if (w == null || ins == null) {
      return { lines: ["Потрібні обхват талії та внутрішній шов."], warnings };
    }
    const { code, warn } = pickTwoKeyRow(grid.rows, w, ins);
    if (warn) warnings.push(warn);
    return { lines: [code, `Талія: ${w} см, внутр. шов: ${ins} см`], warnings };
  }

  if (grid.kind === "waist_inseam_inches") {
    const w = measures[grid.waist_key];
    const ins = measures[grid.inseam_key];
    if (w == null || ins == null) {
      return { lines: ["Потрібні обхват талії та внутрішній шов."], warnings };
    }
    const wIn = w / 2.54;
    const iIn = ins / 2.54;
    const W = snapToList(wIn, grid.waist_even_inches);
    const L = snapToList(iIn, grid.inseam_even_inches);
    const warn =
      wIn < Math.min(...grid.waist_even_inches) - 1 || wIn > Math.max(...grid.waist_even_inches) + 1
        ? "Талія за межами типової сітки — значення округлено до найближчого."
        : "";
    if (warn) warnings.push(warn);
    return {
      lines: [
        `Умовний W×L: ${W}×${L} (дюйми)`,
        `Талія ≈ ${wIn.toFixed(1)}″, внутр. шов ≈ ${iIn.toFixed(1)}″`,
        `У сантиметрах: талія ${w} см, шов ${ins} см`,
      ],
      warnings,
    };
  }

  return { lines: ["Невідомий тип сітки."], warnings };
}

function pickCircumferenceBand(bands, v) {
  const sorted = [...bands].sort((a, b) => a.min_cm - b.min_cm);
  for (let i = 0; i < sorted.length; i++) {
    const b = sorted[i];
    const last = i === sorted.length - 1;
    if (v >= b.min_cm && (last ? v <= b.max_cm : v < b.max_cm)) {
      return { code: b.code, warn: "" };
    }
  }
  if (v < sorted[0].min_cm) {
    return { code: sorted[0].code, warn: "Значення менше таблиці — показано найменший розмір." };
  }
  const lastB = sorted[sorted.length - 1];
  return { code: lastB.code, warn: "Значення більше таблиці — показано найбільший розмір." };
}

function pickCircumferenceBandDual(bands, v) {
  const sorted = [...bands].sort((a, b) => a.min_cm - b.min_cm);
  for (let i = 0; i < sorted.length; i++) {
    const b = sorted[i];
    const last = i === sorted.length - 1;
    if (v >= b.min_cm && (last ? v <= b.max_cm : v < b.max_cm)) {
      return { band: b, warn: "" };
    }
  }
  if (v < sorted[0].min_cm) {
    return { band: sorted[0], warn: "Значення менше таблиці — показано найменший розмір." };
  }
  const lastB = sorted[sorted.length - 1];
  return { band: lastB, warn: "Значення більше таблиці — показано найбільший розмір." };
}

function pickTwoKeyRow(rows, w, ins) {
  const hit = rows.find((r) => w >= r.w_min && w < r.w_max && ins >= r.i_min && ins <= r.i_max);
  if (hit) return { code: hit.code, warn: "" };
  let best = rows[0];
  let bestScore = Infinity;
  for (const r of rows) {
    const dw = intervalDistance(w, r.w_min, r.w_max);
    const di = intervalDistance(ins, r.i_min, r.i_max);
    const sc = dw + di;
    if (sc < bestScore) {
      bestScore = sc;
      best = r;
    }
  }
  return { code: best.code, warn: "Комбінація талії/шва між рядками таблиці — показано найближчий рядок." };
}

function intervalDistance(v, lo, hi) {
  if (v < lo) return lo - v;
  if (v > hi) return v - hi;
  return 0;
}

function snapToList(value, list) {
  let best = list[0];
  let d = Math.abs(value - best);
  for (const x of list) {
    const dd = Math.abs(value - x);
    if (dd < d) {
      d = dd;
      best = x;
    }
  }
  return best;
}

/**
 * @param {any} catalog
 * @param {any} garmentResolved
 * @param {Record<string, number>} measures
 * @param {number} height_cm
 * @param {"male"|"female"|"other"} sex
 */
export function formatSizeTabs(catalog, garmentResolved, measures, height_cm, sex) {
  const plan = tabGridPlan(garmentResolved.category, sex);
  const grids = Object.fromEntries((catalog.grids || []).map((g) => [g.id, g]));

  /** @type {{ id: string, label: string, lines: string[], warnings: string[] }[]} */
  const uaBlocks = [];
  /** @type {{ id: string, label: string, lines: string[], warnings: string[] }[]} */
  const euBlocks = [];
  /** @type {{ id: string, label: string, lines: string[], warnings: string[] }[]} */
  const usBlocks = [];

  const pushEval = (gridId, bucket) => {
    const grid = grids[gridId];
    if (!grid) return;
    const { lines, warnings } = evaluateGrid(grid, measures, height_cm, sex);
    bucket.push({ id: gridId, label: grid.label_uk, lines, warnings });
  };

  if (Array.isArray(plan.ua)) {
    for (const gid of plan.ua) pushEval(gid, uaBlocks);
  } else {
    pushEval(plan.ua, uaBlocks);
  }

  if (Array.isArray(plan.eu)) {
    for (const gid of plan.eu) pushEval(gid, euBlocks);
  } else if (plan.eu) {
    pushEval(plan.eu, euBlocks);
  }

  if (Array.isArray(plan.us)) {
    for (const gid of plan.us) pushEval(gid, usBlocks);
  } else {
    pushEval(plan.us, usBlocks);
  }

  return { uaBlocks, euBlocks, usBlocks };
}
