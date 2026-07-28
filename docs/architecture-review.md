# Pipeline Architecture Review — SSOT / DRY / KISS / SOLID

> Written 2026-07-20 after the thigh + waist investigations. Every finding below
> is backed by evidence from this codebase, and most were *demonstrated* by bugs
> this session actually hit — not hypothetical code-smell.

---

## The core thesis

**The pipeline mixes four things that should be separate layers:**

| layer | what it is | today |
|---|---|---|
| **Geometry engine** | generic "measure a width at a row" code | ✅ exists (`silhouette.py` helpers) |
| **Anatomical definitions** | *which* row is the waist, *which* band excludes arms | ❌ hardcoded inline as 64 float literals |
| **Fitted coefficients** | per-sex bias scales, learned girth maps | ❌ hand-edited Python dicts in source |
| **Eval** | fits + validates the two layers above | ⚠️ exists but duplicated, and can't feed them back |

The user's instinct is right: **the constants are learned parameters wearing the
costume of source code.** The thigh fix proved it — `0.10` was *discovered by
scanning against ground truth*, not derived. Anything discovered from data is
data, and belongs in a versioned artifact the eval can refit and A/B — not in a
`.py` file that must be hand-edited in two repos.

---

## 1. 🔴 CRITICAL — Two repos, one codebase (SSOT)

**Evidence:** `PointsX/src` and `Pointx-backend/src` hold **29 byte-identical
files and 11 diverged ones.**

**This actively caused bugs this session:**
- I fixed the synthetic GT in the wrong repo, then had to port it.
- The thigh fix + refitted constants had to be hand-copied, and **had to move
  together** — shipping constants calibrated on the fixed extraction to a repo
  with the old extraction would have made the deployed service *worse*.
- `blender_render.py`, `pipeline.py`, `save_body_obj` have silently diverged.

**Fix (highest leverage of anything in this document):** one source of truth.
- `Pointx-backend` becomes a **thin deployment shell** (Dockerfile, HF Space
  config, `app.py`) that depends on the `pointsx` package rather than vendoring it.
- Or: extract `pointsx` into a package both consume via a pinned dependency.
- Interim, if neither is possible today: a `make sync` + CI check that fails when
  shared paths diverge. **Anything is better than manual copying.**

---

## 2. 🔴 Domain knowledge duplicated with *conflicting* values (SSOT)

The plausible waist range is defined **three times, with two different answers**:

| file | range |
|---|---|
| `pointsx/postprocess.py` (`ABSOLUTE_CHECKS`) | **55–150** |
| `scripts/build_eval_csv.py` (`GT_RANGES`) | **45–170** |
| `synthetic/measurements_gt.py` (`_BOUNDS`) | 45–170 |

A body accepted by one component is rejected by another. Same story for the
measurement-id vocabulary, which exists in `envelope.py::CANONICAL_MEASUREMENTS`,
`build_eval_csv.py::GT_MAP`, the dataset form, and the synthetic GT dataclass —
four hand-maintained lists that must agree and have no mechanism forcing them to.

**Fix — one measurement registry.** A single module that is the *only* place a
measurement is defined:

```python
@dataclass(frozen=True)
class MeasurementDef:
    id: str                      # "waist_circumference"
    label_uk: str
    source: Literal["front", "side", "fused"]
    plausible_cm: tuple[float, float]
    gt_aliases: tuple[str, ...]  # dataset-form ids that map here
```

Everything else — validation, envelope, eval CSV, synthetic GT, the dataset form
— *derives* from it. Adding a measurement becomes one entry, not five edits.

---

## 3. 🟠 Anatomical constants hardcoded inline (the main ask)

**64 bare float literals** across the measurement path
(`silhouette.py` 34, `measurements.py` 17, `postprocess.py` 10, `calibration.py` 3),
and the same concept is repeated per view:

```python
y_mid = y_pelvis + 0.4 * (upper_neck_y - y_pelvis)   # front waist
y_mid = y_pelvis + 0.4 * (upper_neck_y - y_pelvis)   # side waist  (same rule, retyped)
y_start = y_pelvis + 0.05 * (y_knee - y_pelvis)      # front hip
y_start = y_pelvis + 0.05 * (y_knee - y_pelvis)      # side hip    (again)
```

**Documentation has already drifted from code:** `silhouette.py:452` says
*"pelvis + 0.5*(upper_neck - pelvis)"* while line 455 computes **0.4**. Nobody
noticed because nothing tests it.

**Fix — declarative definitions + one generic executor.** Replace the 200-line
`extract_all_widths` branch-fest with a table:

```python
WAIST = SliceDef(
    anchor_a=KP.PELVIS, anchor_b=KP.UPPER_NECK,
    span=(0.0, 0.40),        # search window along anchor_a -> anchor_b
    strategy="min",          # narrowest continuous run in the window
    band=TorsoBand.SHOULDER_HIP,
)
THIGH = SliceDef(
    anchor_a=KP.HIP_MID, anchor_b=KP.KNEE_MID,
    span=(0.10, 0.10),       # fixed level, empirically fitted
    strategy="at", halve_if_straddles_midline=True,
)
```

Then extraction is **one function** driving the table for every measurement and
both views. This is the "dynamic calculation, constants only as coefficients"
the user asked for, and it collapses three violations at once:
- **DRY** — the front/side duplication disappears (same def, two masks).
- **Open/Closed** — a new measurement is a new row, not a new `if` branch.
- **Single Responsibility** — anchoring, banding, and strategy stop being tangled.

It also makes the *fractions themselves fittable*: the level scan I ran manually
for the thigh (0.10 → +0.72 correlation) becomes a first-class eval capability
rather than a throwaway script.

---

## 4. 🟠 Fitted coefficients live in source code

`_SEX_CIRCUMFERENCE_SCALES_PCT` is a hand-edited dict in `envelope.py`. Consequences
seen this session: it must be refit by running an eval, **pasted by hand**, synced
across two repos, and its provenance (n per cell, fit date, which extraction it was
calibrated against) survives only as a comment I wrote.

**It went stale twice without any signal** — first fitted on n=3, then invalidated
by the thigh fix, because nothing links a coefficient to the code version it was
fitted against.

**Fix — ship coefficients as a versioned data artifact:**

```json
{
  "schema": "girth-scales-v1",
  "fitted_on": "2026-07-20",
  "pipeline_fingerprint": "silhouette@<git-sha>",
  "corpus": {"n_female": 8, "n_male": 3},
  "loocv_mae_cm": 6.50,
  "scales_pct": {"female": {"waist_circumference": -14.0, ...}}
}
```

Loaded at startup; **warn loudly when `pipeline_fingerprint` ≠ current code**.
Then a refit is a data change (reviewable, revertable, A/B-able) instead of a
source edit, and the "constants are stale" failure becomes impossible to miss.

---

## 5. 🟡 Silent fallbacks that fabricate data

A recurring pattern, and the most dangerous class of bug found this session:

| location | fallback | consequence |
|---|---|---|
| `synthetic/measurements_gt.py` | `chest=90, waist=75, …` on slice failure | **every** body got identical fake GT (fixed) |
| `body_generator.py` | `raw_height_m = 1.7` on degenerate mesh | fabricated height (fixed) |
| `silhouette.py` thigh | crotch scan fails → measure at pelvis | returned *hip width* labelled thigh (fixed) |
| `measure_width_in_band_at_y` | no pixels in band → unclipped width | silently measures arms too (**still live**) |

**Principle to adopt repo-wide: never fabricate, always abstain.** Return `None`,
let the caller decide, and surface it. A missing measurement is honest; a
confidently wrong one reaches a user and cannot be detected downstream.

---

## 6. 🟡 Eval duplication (mine — same sin, freshly committed)

`eval/pipelines.py` (BodyM) and `eval/app_pipelines.py` (app corpus) each define
pipelines A/B/C/D with *slightly different* implementations of the same maths.
They already disagree: the app version gained a `D_MIN_SAMPLES` guard the BodyM
one lacks.

**Fix:** one pipeline module; two thin **data adapters** (BodyM masks, app photos)
producing a common `{sex, widths, gt}` record. The pipelines then can't drift, and
a result difference means a *data* difference — which is the entire point of
running both.

---

## 7. 🟡 No tests, so every check costs minutes

`CLAUDE.md` notes there is no test suite. Consequence: validating the one-line
thigh change required a full pipeline run over 11 subjects (~minutes each
iteration), and the doc/code drift at `silhouette.py:452` went unnoticed
indefinitely.

The geometry helpers are **pure functions over numpy arrays** — the cheapest
possible things to test. A synthetic mask (two rectangles = torso + legs) with a
known answer would have caught the thigh bug instantly:

```python
def test_thigh_halves_merged_legs():
    mask = two_legs_touching(width_px=40)     # one 40px run
    assert thigh_width(mask, ...) == 20       # halved at the midline
```

---

## Prioritised plan

| # | Fix | Effort | Why now |
|---|---|---|---|
| 1 | **Kill the two-repo duplication** | M | Already caused wrong-repo edits + manual porting this session |
| 2 | **Measurement registry (SSOT)** | M | Three conflicting waist ranges are live today |
| 3 | **Coefficients → versioned artifact + fingerprint** | S | They silently went stale twice |
| 4 | **Declarative `SliceDef` + generic executor** | L | Removes 64 magic numbers; makes fractions fittable |
| 5 | **Ban fabricating fallbacks** | S | One still live (`measure_width_in_band_at_y`) |
| 6 | **Unify eval pipelines** | S | They've already drifted |
| 7 | **Unit tests on pure geometry** | S | Seconds instead of minutes per iteration |

**Do 1–3 first.** They're the ones that already bit us, and 3 is nearly free.
4 is the big architectural win but should follow the registry, since `SliceDef`
entries should reference registry ids.

---

## What NOT to do

- **Don't tune the magic numbers in place.** Moving `0.4` to `0.45` without the
  declarative layer just relocates the problem — and the waist investigation showed
  the binding constraint there is *clothing*, not the constant.
- **Don't unify the two repos by copying harder.** A `cp`-based sync script is the
  same violation with extra steps.
- **Don't add coefficients for chest.** Its bias is not statistically significant
  (CI spans zero on both datasets) — see the note in `envelope.py`.
