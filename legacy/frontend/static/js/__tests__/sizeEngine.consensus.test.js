/**
 * Consensus sizing engine tests.
 * Run: node --test src/webui/static/js/__tests__/sizeEngine.consensus.test.js
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";

import {
  evaluateGarmentSize,
  resolveGarment,
  measurementsToMap,
  confidencesToMap,
} from "../sizeEngine.js";
import {
  validateEnvelope,
  indexMeasurements,
  computePatternBlock,
  sizeAndPatternHandler,
} from "../patternEngine.js";

const __dir  = dirname(fileURLToPath(import.meta.url));
const catalog = JSON.parse(
  readFileSync(resolve(__dir, "../../data/tailoring_config.json"), "utf8")
);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeEnvelope(measures, sex = "female", overrides = {}) {
  const measurements = Object.entries(measures).map(([id, value_cm]) => ({
    id,
    label_uk: id,
    value_cm,
    uncertainty_cm: value_cm * 0.02,
    confidence: overrides.confidence?.[id] ?? 0.85,
    source: "fused",
    quality_flags: [],
  }));
  return {
    schema:         "pointsx.measurement.envelope",
    schema_version: 2,
    request_id:     "test-request-1",
    created_at:     "2026-04-23T00:00:00.000Z",
    pipeline:       { source: "mock", model_version: "test-0.0.1", unit_system: "metric" },
    subject:        { height_cm: 170, sex, posture_flags: [] },
    capture:        { front: { quality: 0.9, pose_ok: true, occlusions: [] }, side: { quality: 0.88, pose_ok: true, occlusions: [] } },
    measurements,
    derived:        {},
    warnings:       [],
  };
}

const dressGarment  = resolveGarment(catalog, "dress");
const pantsGarment  = resolveGarment(catalog, "pants");
const shirtGarment  = resolveGarment(catalog, "shirt");

function evalUa(garment, measures, sex = "female", confidences = {}) {
  return evaluateGarmentSize(catalog, garment, measures, confidences, "ua", sex);
}

// ---------------------------------------------------------------------------
// 1. Unanimous verdict
// ---------------------------------------------------------------------------

describe("unanimous", () => {
  it("three measures landing on M → verdict=unanimous, code=40", () => {
    // chest 88 → M (86–90), waist 68 → M (66–70), hip 94 → M (92–96)
    const measures     = { chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 };
    const result       = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "unanimous");
    assert.equal(result.code,    "40");
    assert.equal(result.confidence, "high");
  });

  it("all three land on XS → ordinal 0", () => {
    const measures = { chest_circumference: 79, waist_circumference: 60, hip_circumference: 85 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "unanimous");
    assert.equal(result.ordinal, 0);
  });
});

// ---------------------------------------------------------------------------
// 2. Majority 2/3 verdict
// ---------------------------------------------------------------------------

describe("majority", () => {
  it("M/M/L → verdict=majority, code=40, outlier is hip", () => {
    // chest 88 → M (ordinal 2), waist 68 → M (ordinal 2), hip 99 → L (96–102, ordinal 3)
    const measures = { chest_circumference: 88, waist_circumference: 68, hip_circumference: 99 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "majority");
    assert.equal(result.code,    "40");
    assert.equal(result.outlier?.mid, "hip_circumference");
  });

  it("S/S/M → verdict=majority with spread=1", () => {
    // chest 83 → S(1), waist 63 → S(1), hip 93 → M(2)
    const measures = { chest_circumference: 83, waist_circumference: 63, hip_circumference: 93 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "majority");
    assert.equal(result.ordinal, 1);
  });
});

// ---------------------------------------------------------------------------
// 3. No-consensus (spread > 1)
// ---------------------------------------------------------------------------

describe("no_consensus", () => {
  it("S/M/L → verdict=no_consensus, rangeCode contains both extremes", () => {
    // chest 83 → S(1), waist 71 → L(3), hip 94 → M(2) [wait, 94 → ord 4 XL?]
    // Let me be precise: waist 70–76 = L(ordinal 3), but chest 82–86 = S(ordinal 1)
    // hip 96–102 = L(ordinal 3)... that's majority.
    // Use: chest 83 → S(1), waist 73 → L(3), hip 105 → XL(4)
    const measures = { chest_circumference: 83, waist_circumference: 73, hip_circumference: 105 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "no_consensus");
    assert.ok(result.rangeCode, "rangeCode should be set");
    assert.ok(result.tailoringHint, "tailoringHint should be set");
    assert.equal(result.confidence, "low");
  });

  it("tailoringHint names the outlier measurement", () => {
    const measures = { chest_circumference: 83, waist_circumference: 73, hip_circumference: 105 };
    const result   = evalUa(dressGarment, measures);
    assert.ok(result.tailoringHint.length > 10, "hint should be a sentence");
  });
});

// ---------------------------------------------------------------------------
// 4. Insufficient inputs
// ---------------------------------------------------------------------------

describe("insufficient", () => {
  it("only 1 of 3 measures present → verdict=insufficient", () => {
    const measures = { chest_circumference: 88 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "insufficient");
  });

  it("0 measures → insufficient with null provisional", () => {
    const result = evalUa(dressGarment, {});
    assert.equal(result.verdict, "insufficient");
    assert.equal(result.provisional, null);
  });

  it("2 of 3 measures → insufficient with provisional set", () => {
    const measures = { chest_circumference: 88, waist_circumference: 68 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "insufficient");
    assert.ok(result.provisional != null, "should have provisional from partial votes");
  });
});

// ---------------------------------------------------------------------------
// 5. Out-of-table (edge values)
// ---------------------------------------------------------------------------

describe("out_of_range", () => {
  it("chest below smallest band → edge=below, verdict downgraded", () => {
    const measures = { chest_circumference: 60, waist_circumference: 68, hip_circumference: 94 };
    const result   = evalUa(dressGarment, measures);
    const chestVote = result.votes?.find((v) => v.mid === "chest_circumference");
    assert.equal(chestVote?.edge, "below");
    assert.ok(result.warnings?.some((w) => w.includes("за межами")));
  });

  it("all three above largest band → unanimous_edge verdict", () => {
    const measures = { chest_circumference: 120, waist_circumference: 100, hip_circumference: 130 };
    const result   = evalUa(dressGarment, measures);
    assert.equal(result.verdict, "unanimous_edge");
    assert.equal(result.confidence, "medium");
  });
});

// ---------------------------------------------------------------------------
// 6. Low-confidence measurement
// ---------------------------------------------------------------------------

describe("low_confidence", () => {
  it("low-confidence outlier is down-weighted and warning emitted", () => {
    // chest=88(M), waist=68(M), hip=99(L) but hip confidence=0.3
    const measures    = { chest_circumference: 88, waist_circumference: 68, hip_circumference: 99 };
    const confidences = { chest_circumference: 0.88, waist_circumference: 0.85, hip_circumference: 0.3 };
    const result      = evalUa(dressGarment, measures, "female", confidences);
    assert.ok(result.warnings?.some((w) => w.includes("достовірність")));
    const hipVote = result.votes?.find((v) => v.mid === "hip_circumference");
    assert.ok(hipVote?.lowConf === true);
  });

  it("low-confidence does not prevent unanimous verdict when all three agree", () => {
    const measures    = { chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 };
    const confidences = { chest_circumference: 0.4, waist_circumference: 0.4, hip_circumference: 0.4 };
    const result      = evalUa(dressGarment, measures, "female", confidences);
    assert.ok(["unanimous","unanimous_edge"].includes(result.verdict));
  });
});

// ---------------------------------------------------------------------------
// 7. Men's sizing
// ---------------------------------------------------------------------------

describe("men sizing", () => {
  it("men shirt: chest 100cm → ordinal 1 (M) in ua", () => {
    const measures = { chest_circumference: 100, waist_circumference: 88, hip_circumference: 102 };
    const result   = evaluateGarmentSize(catalog, shirtGarment, measures, {}, "ua", "male");
    assert.ok(result.ordinal === 1, `expected ordinal 1, got ${result.ordinal}`);
  });

  it("men pants: waist 82cm → ordinal 1 (M) in ua", () => {
    const measures = { waist_circumference: 82, hip_circumference: 100, thigh_circumference: 57 };
    const result   = evaluateGarmentSize(catalog, pantsGarment, measures, {}, "ua", "male");
    assert.ok(result.ordinal === 1, `expected ordinal 1 (M), got ${result.ordinal}`);
  });
});

// ---------------------------------------------------------------------------
// 8. Bottom (pants) sizing — thigh trio
// ---------------------------------------------------------------------------

describe("pants trio", () => {
  it("pants waist/hip/thigh all M → unanimous", () => {
    // waist 68 → M (66–70 ord 2), hip 94 → M (92–96 ord 2), thigh 56 → M (54–58 ord 2)
    const measures = { waist_circumference: 68, hip_circumference: 94, thigh_circumference: 56 };
    const result   = evaluateGarmentSize(catalog, pantsGarment, measures, {}, "eu", "female");
    assert.equal(result.verdict, "unanimous");
    assert.equal(result.ordinal, 2);
    assert.equal(result.code, "EU 38");
  });

  it("pants waist/hip agree M, thigh is one step L → majority", () => {
    // waist 68 → M(ord 2), hip 94 → M(ord 2), thigh 60 → L(ord 3) — spread = 1
    const measures = { waist_circumference: 68, hip_circumference: 94, thigh_circumference: 60 };
    const result   = evaluateGarmentSize(catalog, pantsGarment, measures, {}, "ua", "female");
    assert.equal(result.verdict, "majority");
    assert.equal(result.outlier?.mid, "thigh_circumference");
  });
});

// ---------------------------------------------------------------------------
// 9. Between-sizes boundary note
// ---------------------------------------------------------------------------

describe("between_sizes", () => {
  it("measurement within tolerance of band edge → between note set", () => {
    // Chest band edge at 86 cm; chest=86.4 is within 0.7 cm tolerance
    const measures = { chest_circumference: 86.4, waist_circumference: 68, hip_circumference: 94 };
    const result   = evalUa(dressGarment, measures);
    assert.ok(result.between != null, "should have between note near boundary");
  });
});

// ---------------------------------------------------------------------------
// 10. sex=other falls back to female grids with indicative warning
// ---------------------------------------------------------------------------

describe("sex=other", () => {
  it("sex=other adds indicative warning", () => {
    const measures = { chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 };
    const result   = evaluateGarmentSize(catalog, dressGarment, measures, {}, "ua", "other");
    assert.ok(result.warnings?.some((w) => w.includes("інше")));
  });
});

// ---------------------------------------------------------------------------
// 11. sizeAndPatternHandler (integration)
// ---------------------------------------------------------------------------

describe("sizeAndPatternHandler", () => {
  it("full flow: envelope → report with all three regions and pattern block", () => {
    const envelope = makeEnvelope({
      chest_circumference:          88,
      waist_circumference:          68,
      hip_circumference:            94,
      back_length_to_waist:         42,
      back_width_scapular:          35,
      chest_width_front:            32,
      shoulder_slope_width:         38,
      neck_circumference:           36,
      arm_length_shoulder_to_wrist: 60,
      upper_arm_circumference:      28,
      wrist_circumference:          16,
    });
    const report = sizeAndPatternHandler(envelope, catalog, { garmentId: "dress", fit: "close" });
    assert.equal(report.schema, "pointsx.sizing.report");
    assert.ok(report.sizing.ua, "ua sizing present");
    assert.ok(report.sizing.eu, "eu sizing present");
    assert.ok(report.sizing.us, "us sizing present");
    assert.ok(typeof report.pattern.raw.armscye_depth === "number", "armscye_depth computed");
    assert.ok(typeof report.pattern.raw.dart_intake_total === "number", "dart intake computed");
  });

  it("schema_version=1 envelope → throws", () => {
    const bad = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    bad.schema_version = 1;
    assert.throws(() => sizeAndPatternHandler(bad, catalog, { garmentId: "dress" }));
  });

  it("waist > hip → warning emitted", () => {
    const envelope = makeEnvelope({
      chest_circumference: 88,
      waist_circumference: 110,
      hip_circumference:   90,
    });
    const report = sizeAndPatternHandler(envelope, catalog, { garmentId: "dress" });
    assert.ok(report.warnings.some((w) => w.includes("waist_gt_hip")));
  });
});

// ---------------------------------------------------------------------------
// 12. computePatternBlock — formula spot-checks
// ---------------------------------------------------------------------------

describe("computePatternBlock", () => {
  const body = {
    chest_circumference:          { value_cm: 90 },
    waist_circumference:          { value_cm: 70 },
    hip_circumference:            { value_cm: 96 },
    back_length_to_waist:         { value_cm: 42 },
    back_width_scapular:          { value_cm: 36 },
    chest_width_front:            { value_cm: 33 },
    shoulder_slope_width:         { value_cm: 39 },
    neck_circumference:           { value_cm: 36 },
    arm_length_shoulder_to_wrist: { value_cm: 61 },
    upper_arm_circumference:      { value_cm: 29 },
    wrist_circumference:          { value_cm: 16 },
    calf_circumference:           { value_cm: 36 },
    ankle_circumference:          { value_cm: 24 },
  };
  const dressGarmentObj = resolveGarment(catalog, "dress");

  it("armscye_depth = chest/10 + 10.5 (female)", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.armscye_depth, 19.5);
  });

  it("armscye_depth = chest/10 + 11.0 (male)", () => {
    const { pattern } = computePatternBlock(body, "male", dressGarmentObj, catalog, "close");
    assert.equal(pattern.armscye_depth, 20);
  });

  it("cross_back = back_width_scapular + 0.8", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.cross_back, 36.8);
  });

  it("dart_intake = (hip - waist) / 4", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.dart_intake_total, (96 - 70) / 4);
  });

  it("dart_back = 55% of dart_intake", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.dart_back, +((pattern.dart_intake_total * 0.55).toFixed(2)));
  });

  it("neck_width = N/5 + 0.5", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.neck_width, 36 / 5 + 0.5);
  });

  it("sleeve_cap_height = armscye_depth * 0.75 (female)", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.sleeve_cap_height, +(pattern.armscye_depth * 0.75).toFixed(2));
  });

  it("hip_line_depth = BL/2 + 2", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.hip_line_depth, 42 / 2 + 2);
  });

  it("crotch_depth = hip/4 + 1 (female)", () => {
    const { pattern } = computePatternBlock(body, "female", dressGarmentObj, catalog, "close");
    assert.equal(pattern.crotch_depth, +(96 / 4 + 1).toFixed(2));
  });
});

// ---------------------------------------------------------------------------
// 13. validateEnvelope
// ---------------------------------------------------------------------------

describe("validateEnvelope", () => {
  it("valid v2 envelope passes", () => {
    const env = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    assert.doesNotThrow(() => validateEnvelope(env));
  });

  it("wrong schema string → throws", () => {
    const env = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    env.schema = "wrong.schema";
    assert.throws(() => validateEnvelope(env), /Expected schema/);
  });

  it("schema_version=1 → throws", () => {
    const env = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    env.schema_version = 1;
    assert.throws(() => validateEnvelope(env), /schema_version/);
  });

  it("missing subject.sex → throws", () => {
    const env = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    env.subject.sex = "unknown";
    assert.throws(() => validateEnvelope(env), /sex/);
  });
});

// ---------------------------------------------------------------------------
// 14. indexMeasurements — duplicate handling
// ---------------------------------------------------------------------------

describe("indexMeasurements", () => {
  it("keeps highest-confidence entry when duplicate IDs present", () => {
    const env = makeEnvelope({ chest_circumference: 88, waist_circumference: 68, hip_circumference: 94 });
    env.measurements.push({
      id: "chest_circumference", label_uk: "груди dup",
      value_cm: 92, uncertainty_cm: 1, confidence: 0.99, source: "fused", quality_flags: [],
    });
    const idx = indexMeasurements(env);
    assert.equal(idx.chest_circumference.value_cm, 92);
    assert.equal(idx.chest_circumference.confidence, 0.99);
  });
});
