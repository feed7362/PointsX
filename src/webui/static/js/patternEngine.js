/**
 * Pattern engine — bespoke tailoring math + MeasurementEnvelope v2 handler.
 *
 * Exports:
 *   sizeAndPatternHandler(envelope, catalog, options) → SizingReport
 *
 * Formula sources: Müller & Sohn, Winifred Aldrich (Metric Pattern Cutting, 7th ed.),
 * See docs/sizing-baseline.md §Pattern Formula Reference.
 */

import { evaluateGarmentSize, resolveGarment, measurementsToMap, confidencesToMap } from "./sizeEngine.js";

const SCHEMA_ID          = "pointsx.measurement.envelope";
const MIN_SCHEMA_VERSION = 2;

// ---------------------------------------------------------------------------
// Envelope validation
// ---------------------------------------------------------------------------

/**
 * Throw a descriptive Error if the envelope is not v2-compatible.
 * @param {any} envelope
 */
export function validateEnvelope(envelope) {
  if (!envelope || envelope.schema !== SCHEMA_ID) {
    throw new Error(`Expected schema="${SCHEMA_ID}", got "${envelope?.schema}"`);
  }
  if ((envelope.schema_version ?? 0) < MIN_SCHEMA_VERSION) {
    throw new Error(`schema_version must be >= ${MIN_SCHEMA_VERSION}`);
  }
  if (!envelope.subject || typeof envelope.subject.height_cm !== "number") {
    throw new Error("subject.height_cm required and must be a number");
  }
  if (!["male", "female", "other"].includes(envelope.subject.sex)) {
    throw new Error('subject.sex must be "male", "female", or "other"');
  }
  if (!Array.isArray(envelope.measurements) || envelope.measurements.length === 0) {
    throw new Error("measurements[] must be a non-empty array");
  }
}

// ---------------------------------------------------------------------------
// Measurement indexing (hardened against duplicates)
// ---------------------------------------------------------------------------

/**
 * Build { id → { value_cm, confidence, uncertainty_cm, source } }.
 * When duplicate IDs appear, the entry with the higher confidence is kept.
 * Synthesises uncertainty_cm if absent (legacy v1 responses).
 * @param {any} envelope
 * @returns {Record<string, { value_cm: number, confidence: number, uncertainty_cm: number, source: string }>}
 */
export function indexMeasurements(envelope) {
  const idx = Object.create(null);
  for (const m of envelope.measurements) {
    const prev = idx[m.id];
    if (prev && (m.confidence ?? 0) <= (prev.confidence ?? 0)) continue;
    const conf = m.confidence ?? 0.75;
    idx[m.id] = {
      value_cm:       m.value_cm,
      confidence:     conf,
      uncertainty_cm: m.uncertainty_cm ?? (1 - conf) * m.value_cm * 0.05,
      source:         m.source ?? "fused",
    };
  }
  return idx;
}

// ---------------------------------------------------------------------------
// Ease helper
// ---------------------------------------------------------------------------

/**
 * Return the default ease allowance (cm) for a (measurementId, fit) pair from the catalog.
 * `table[measureId]` is `[min, max, default]`; index 2 is used.
 */
function easeFor(catalog, garment, measureId, fit) {
  // Caller-supplied fit takes priority over garment default (allows slim/relaxed overrides)
  const profile = fit ?? garment.fit_preference_default ?? "regular";
  const table   = catalog.ease_profiles?.[profile];
  if (!table) return 0;
  const entry = table[measureId];
  if (!entry) return 0;
  return entry[2] ?? 0;
}

// ---------------------------------------------------------------------------
// Bespoke pattern block
// ---------------------------------------------------------------------------

/**
 * Compute the bespoke pattern block from body measurements.
 * All inputs and outputs are in cm.
 * Outputs are BEFORE seam allowances are applied.
 *
 * Formulas per docs/sizing-baseline.md §Pattern Formula Reference:
 *   Armscye depth:       C/10 + 10.5 (F) / 11.0 (M)
 *   Cross-back:          BW + 0.8
 *   Cross-front:         FW + 0.4 (F) / 0.8 (M)
 *   Shoulder seam:       SS / 2
 *   Neck width:          N/5 + 0.5
 *   Neck depth front:    neck_width + 0.5
 *   Neck depth back:     2.0 (constant)
 *   Dart intake:         (H − W) / 4, split 55/30/15 %
 *   Hip line depth:      BL/2 + 2
 *   Sleeve cap height:   AD × 0.75 (F) / 0.80 (M)
 *   Sleeve cap width:    C × 0.18 + 2
 *   Sleeve underarm:     AL − sleeve_cap_height
 *   Bicep pattern:       UA + E_bicep (slim=2, reg=4, relaxed=6)
 *   Cuff pattern:        wrist + 3 (button) / +1 (knit)
 *   Crotch depth:        H/4 + 1 (F) / +1.5 (M)
 *   Knee pattern:        calf + 4
 *   Hem width (pants):   ankle × 2 + 4
 *
 * @param {Record<string, { value_cm: number }>} body  indexed measurements
 * @param {"male"|"female"|"other"} sex
 * @param {any}    garment   resolved garment object
 * @param {any}    catalog
 * @param {string} [fit]     "slim"|"regular"|"relaxed" (overrides garment default)
 * @returns {{ pattern: Record<string,number>, warnings: string[] }}
 */
export function computePatternBlock(body, sex, garment, catalog, fit = "regular") {
  const get = (id) => body[id]?.value_cm ?? null;

  const C  = get("chest_circumference");
  const W  = get("waist_circumference");
  const H  = get("hip_circumference");
  const N  = get("neck_circumference");
  const BL = get("back_length_to_waist");
  const BW = get("back_width_scapular");
  const FW = get("chest_width_front");
  const SS = get("shoulder_slope_width");
  const AL = get("arm_length_shoulder_to_wrist");
  const UA = get("upper_arm_circumference");
  const WR = get("wrist_circumference");
  const CA = get("calf_circumference");
  const AN = get("ankle_circumference");

  const isMale  = sex === "male";
  const pat     = {};
  const warnings = [];

  // Bodice — chest group
  if (C != null) {
    pat.armscye_depth     = r(C / 10 + (isMale ? 11.0 : 10.5));
    pat.sleeve_cap_height = r((isMale ? 0.80 : 0.75) * pat.armscye_depth);
    pat.sleeve_cap_width  = r(C * 0.18 + 2);
    pat.chest_pattern     = r(C + easeFor(catalog, garment, "chest_circumference", fit));
  }

  if (W != null) pat.waist_pattern = r(W + easeFor(catalog, garment, "waist_circumference", fit));
  if (H != null) pat.hip_pattern   = r(H + easeFor(catalog, garment, "hip_circumference",   fit));

  // Back / length
  if (BL != null) {
    pat.back_length_cb = r(BL);
    pat.hip_line_depth = r(BL / 2 + 2);
  }

  // Cross-back and cross-front
  if (BW != null) {
    pat.cross_back = r(BW + 0.8);
    if (C != null && pat.cross_back > C / 2 - 2) {
      warnings.push("cross_back_too_wide: ширина спини перевищує половину грудей мінус 2 см");
    }
  }
  if (FW != null) pat.cross_front = r(FW + (isMale ? 0.8 : 0.4));

  // Shoulder
  if (SS != null) {
    pat.shoulder_seam = r(SS / 2);
  }

  // Neck
  if (N != null) {
    pat.neck_width       = r(N / 5 + 0.5);
    pat.neck_depth_front = r(pat.neck_width + 0.5);
    pat.neck_depth_back  = 2.0;
  }

  // Dart intake (Müller split: back 55%, side 30%, front 15%)
  if (W != null && H != null) {
    if (H < W) {
      warnings.push("waist_gt_hip: обхват талії більший за обхват стегон — потрібна перевірка");
    }
    const DI            = Math.max(0, (H - W) / 4);
    pat.dart_intake_total = r(DI);
    pat.dart_back         = r(DI * 0.55);
    pat.dart_side         = r(DI * 0.30);
    pat.dart_front        = r(DI * 0.15);
  }

  // Sleeve
  if (AL != null && pat.sleeve_cap_height != null) {
    pat.sleeve_underarm_length = r(AL - pat.sleeve_cap_height);
    if (pat.sleeve_underarm_length <= 0) {
      warnings.push("sleeve_underarm_non_positive: довжина рукава коротша за висоту окату");
    }
  }

  if (UA != null) {
    const Ebi         = fit === "slim" ? 2 : fit === "relaxed" ? 6 : 4;
    pat.bicep_pattern = r(UA + Ebi);
  }

  if (WR != null) pat.cuff_pattern = r(WR + 3);

  // Armscye sanity — only meaningful when depth is overridden from BL (BL-based path not yet implemented)
  // The formula C/10 + 10.5 always satisfies C/10 + 8, so this fires only if depth is set externally.
  if (C != null && pat.armscye_depth != null && pat.armscye_depth < C / 10 + 8) {
    warnings.push("armscye_too_shallow: глибина пройми занадто мала — рукав обмежуватиме рух");
  }

  // Pants / skirt block
  if (H != null) {
    pat.crotch_depth = r(H / 4 + (isMale ? 1.5 : 1.0));
  }
  if (CA != null) pat.knee_pattern   = r(CA + 4);
  if (AN != null) pat.hem_width_pants = r(AN * 2 + 4);

  return { pattern: pat, warnings };
}

function r(v) {
  return +v.toFixed(2);
}

// ---------------------------------------------------------------------------
// Seam allowances
// ---------------------------------------------------------------------------

/**
 * Apply seam allowances from `catalog.seam_allowances` to a pattern block.
 * Non-destructive — returns a new object.
 *
 * Allowances per docs/sizing-baseline.md §6.3:
 *   side seams: +1.0  shoulder: +1.0  armhole: +1.0  sleeve cap: +0.7
 *   neckline: +0.7 (facing)   hem blouse/shirt: +2.5  dress: +3.0  pants: +4.0
 */
export function applySeamAllowances(pattern, catalog, garment) {
  const sa  = catalog.seam_allowances?.[garment.category] ??
               catalog.seam_allowances?.default ?? {};
  const out = { ...pattern };

  const add = (key, allowance) => {
    if (out[key] != null && allowance != null) out[key] = r(out[key] + allowance);
  };

  // Circumferential patterns add seam per BOTH side seams (×2 total seam)
  if (sa.side_seam != null) {
    add("chest_pattern", sa.side_seam * 2);
    add("waist_pattern", sa.side_seam * 2);
    add("hip_pattern",   sa.side_seam * 2);
  }

  add("shoulder_seam",          sa.shoulder_seam);
  add("cross_back",             sa.shoulder_seam);   // structural — same allowance as shoulder
  add("sleeve_underarm_length", sa.hem_sleeve);
  add("cuff_pattern",           sa.side_seam);
  add("back_length_cb",         sa.hem_body);        // body hem added to length

  return out;
}

// ---------------------------------------------------------------------------
// Main handler — SizingReport
// ---------------------------------------------------------------------------

/**
 * Ingest a MeasurementEnvelope v2, run 3-measure consensus for UA/EU/US,
 * compute the bespoke pattern block, apply seam allowances, and return a SizingReport.
 *
 * @param {any}    envelope   MeasurementEnvelope v2 (from /api/measure/mock)
 * @param {any}    catalog    tailoring_config.json v2
 * @param {{ garmentId?: string, fit?: string, regions?: string[] }} [options]
 * @returns {SizingReport}
 *
 * @typedef {{
 *   schema: string, schema_version: number, request_id: string,
 *   garment_id: string, subject: any,
 *   sizing: { ua: any, eu: any, us: any },
 *   pattern: { raw: any, with_seam_allowances: any, fit: string },
 *   warnings: string[]
 * }} SizingReport
 */
export function sizeAndPatternHandler(envelope, catalog, options = {}) {
  validateEnvelope(envelope);

  const {
    garmentId = catalog.garments?.[0]?.id ?? "shirt",
    fit       = "regular",
    regions   = ["ua", "eu", "us"],
  } = options;

  const garment = resolveGarment(catalog, garmentId);
  if (!garment) throw new Error(`Unknown garment: "${garmentId}"`);

  // Build flat measure + confidence maps from the indexed body
  const body        = indexMeasurements(envelope);
  const measures    = Object.fromEntries(Object.entries(body).map(([k, v]) => [k, v.value_cm]));
  const confidences = Object.fromEntries(Object.entries(body).map(([k, v]) => [k, v.confidence]));

  // 3-measure consensus per region
  const sizing = {};
  for (const region of regions) {
    sizing[region] = evaluateGarmentSize(
      catalog, garment, measures, confidences, region, envelope.subject.sex, { fit }
    );
  }

  // Bespoke pattern math
  const { pattern, warnings: patWarnings } = computePatternBlock(
    body, envelope.subject.sex, garment, catalog, fit
  );
  const patternWithSa = applySeamAllowances(pattern, catalog, garment);

  // Merge envelope warnings + pattern sanity warnings (deduplicate)
  const allWarnings = [...new Set([...(envelope.warnings ?? []), ...patWarnings])];

  return {
    schema:         "pointsx.sizing.report",
    schema_version: 1,
    request_id:     envelope.request_id,
    garment_id:     garment.id,
    subject:        envelope.subject,
    sizing,
    pattern: {
      raw:                   pattern,
      with_seam_allowances:  patternWithSa,
      fit,
    },
    warnings: allWarnings,
  };
}
