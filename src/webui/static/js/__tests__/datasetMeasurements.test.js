/**
 * Dataset form measurement definitions: ids, i18n keys and the envelope → form prefill map.
 * Run: node --test src/webui/static/js/__tests__/datasetMeasurements.test.js
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { MEASUREMENTS, validateMeasurements, areAllMeasurementsFilled } from "../dataset/measurements.js";
import { ENVELOPE_TO_DATASET, envelopeToDatasetMeasurements } from "../dataset/transfer.js";
import uk from "../i18n/uk.js";
import en from "../i18n/en.js";

const ids = MEASUREMENTS.map((m) => m.id);

describe("dataset MEASUREMENTS", () => {
  it("has 17 unique ids incl. shoulder_slope_width", () => {
    assert.equal(ids.length, 17);
    assert.equal(new Set(ids).size, ids.length);
    assert.ok(ids.includes("shoulder_width"));
    assert.ok(ids.includes("shoulder_slope_width"));
  });

  it("every labelKey exists in uk and en", () => {
    for (const m of MEASUREMENTS) {
      assert.equal(typeof uk[m.labelKey], "string", `uk missing ${m.labelKey}`);
      assert.equal(typeof en[m.labelKey], "string", `en missing ${m.labelKey}`);
    }
  });

  it("the 'fill all N' error text matches the field count", () => {
    for (const dict of [uk, en]) {
      assert.ok(dict["dataset-err-need-measurements"].includes(String(ids.length)));
    }
  });

  it("shoulder slope band matches the envelope (8–25 cm) and flags full-width values", () => {
    const slope = MEASUREMENTS.find((m) => m.id === "shoulder_slope_width");
    assert.deepEqual([slope.plausibleMin, slope.plausibleMax], [8, 25]);
    assert.deepEqual(validateMeasurements({ shoulder_slope_width: 14.5 }), []);
    assert.deepEqual(validateMeasurements({ shoulder_slope_width: 45 }), ["shoulder_slope_width"]);
  });

  it("areAllMeasurementsFilled requires every field", () => {
    const all = Object.fromEntries(ids.map((id) => [id, 50]));
    assert.equal(areAllMeasurementsFilled(all), true);
    delete all.shoulder_slope_width;
    assert.equal(areAllMeasurementsFilled(all), false);
  });
});

describe("ENVELOPE_TO_DATASET", () => {
  it("targets only existing form fields", () => {
    for (const [envId, formId] of Object.entries(ENVELOPE_TO_DATASET)) {
      assert.ok(ids.includes(formId), `${envId} -> ${formId} is not a form field`);
    }
  });

  it("prefills the slope field, never the full shoulder width", () => {
    assert.equal(ENVELOPE_TO_DATASET.shoulder_slope_width, "shoulder_slope_width");
    assert.ok(!Object.values(ENVELOPE_TO_DATASET).includes("shoulder_width"));
    const values = envelopeToDatasetMeasurements(
      { measurements: [{ id: "shoulder_slope_width", value_cm: 15.2 }] },
      176
    );
    assert.deepEqual(values, { height: 176, shoulder_slope_width: 15.2 });
  });
});
