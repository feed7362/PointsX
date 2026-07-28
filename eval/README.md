# PointsX Eval — BodyM golden benchmark

> Design-doc-before-code. Follows the BuildLab evaluation-polygon canon
> (`shared/decisions/2026-06-28-evaluation-polygon-platform.md` +
> `shared/docs/adopt-from-llm-under-hood-2026-07-11.md`), adapted from string
> classification to **numeric measurement regression**.

## Feasibility verdict: build it — but know exactly what it measures

**Feasible: yes.** BodyM (Amazon, `s3://amazon-bodym`, public, CC BY-NC 4.0) gives
2,018 train + 2,844 test subjects with **real scan-derived ground truth** for our
exact hard metrics: `chest, waist, hip, thigh, arm-length, leg-length,
shoulder-breadth` + `height` for calibration. This is the labeled corpus we do
not otherwise have (18 Supabase + 150 S3 samples, none with tape-measure GT).

**Good approach: yes, with one honest caveat that changes how you read the number.**

BodyM ships **binary silhouettes only — no RGB, no keypoints.** Consequences:

1. **It cannot run our production pipeline.** Production is `RGB → pose(keypoints)
   + seg(mask) → measure`. With no RGB there is no pose stage and no seg stage to
   exercise. The eval bypasses the **two noisiest stages** of the real app.
2. **So the BodyM number is a CEILING, not the end-to-end number.** It answers:
   *"given a near-perfect silhouette + the true height, how good is our
   width→circumference math?"* Production will be **worse** by exactly the
   segmentation + pose/calibration error this eval removes. Reaching ±3 cm on
   BodyM does **not** guarantee ±3 cm in the app — it's the best case you could
   reach *if* segmentation were perfect.
3. **Proxy risk (the polygon draft-review's named anti-pattern).** BodyM forces a
   **silhouette-only landmark finder** (find waist/hip/chest rows from the mask
   width profile) that is *not* our production keypoint-anchored path. If we tune
   that finder to BodyM we optimize a proxy. **Mitigation, load-bearing:** the
   eval imports the **real** `pointsx.circumference` /
   `pointsx.silhouette.measure_width_at_y` modules. Be exact about the slice:
   **calibration, landmark row-finding, and arm-clipping are all eval-side new
   code** — the *only* production code BodyM exercises is
   `ramanujan_ellipse_circumference` + `measure_width_at_y`. So the true, narrow
   claim is: *"is two-widths→ellipse accurate against real scan GT, given correct
   widths?"* — not "tests the circumference core." Stated, not hidden.

**Therefore: two evals, two purposes.**
- **This one (BodyM):** isolated measurement-core ceiling. Big, real GT, honest
  upper bound. Also the training corpus for the circumference regressor
  (demo/research track only — NC license bars commercial use).
- **A separate small real-photo end-to-end eval** (the growing Supabase set, once
  it has any GT): the actual production number, seg + pose error included. Not
  built here; noted so the BodyM number is never mistaken for it.

## Why Python here (BuildLab defaults to TS)

The polygon ADR mandates TS *"until ML is the core."* PointsX **is** ML-core
(torch, ultralytics, numpy). The ADR's own carve-out — Python "earns its place
once ML is the core" — selects Python. The eval imports the app's real numpy
measurement modules; a TS runner would need a second copy. One language, no seam.

## Probe result (empirical, 10 testA subjects — 2026-07-13)

A 10-subject comparability probe (real `ramanujan_ellipse_circumference` +
`measure_width_at_y`, rough adapter) answered the "are the definitions even
comparable?" question with data:

- **Side depth is anatomically perfect** (23–28 cm for chest/hip) → lateral view +
  per-mask calibration work; the ellipse math is sound.
- **Front widths are contaminated by arms** — BodyM's frontal pose is A-pose, so
  the naive silhouette width at the hip reads 59–94 cm (real hip breadth ≈ 33–38).
  Production clips this with `measure_width_in_band_at_y` using **shoulder/hip
  keypoints** — which BodyM doesn't have.

**Conclusion: the entire adapter difficulty reduces to one silhouette-only
problem — find the torso x-band from the mask alone** (torso = central contiguous
column; arms = side lobes separable in the row width-profile). Definitions are
compatible; clip the arms and chest/hip land on GT. This is the one thing to build
and exactly what BodyM stresses.

## Architecture — flat script first (per the polygon ADR's own build order)

The ADR mandates instance-1 = a **flat `scripts/eval/`**, and its draft-review
*punishes* jumping to the Stage/evaluator/monorepo shell at n=1. So:

**Now (instance 1) — one flat script:**
```
eval/
  README.md          # this file
  bodym.py           # fetch sample → adapter(arm-clip + calibrate + rows) →
                     #   real ellipse math → per-measurement MAE/bias/%-within-tol
                     #   over a split, vs per-sex-mean baseline. Prints a table.
  golden/            # cached CSVs + a jsonl index (built on first run)
```
The one piece of genuine new code is the **silhouette torso-band finder** inside
`bodym.py`. Everything downstream imports `pointsx.*` live — no vendored copy.

**Orchestrator + separated pipelines (2026-07-13).** `bodym.py` is now a generic
orchestrator: the expensive silhouette work (calibrate → arm-clip → rows → widths
→ ellipse) runs **once per subject**, then every pipeline in `pipelines.py`
replays over that `raw` output as a cheap pure transform and is scored against the
frozen baseline A. Add a pipeline to the `PIPELINES` registry → it's scored
automatically; no orchestrator change. (Same shape as the backend `pointsx-eval`:
run the heavy stage once, replay combos.)

`eval/pipelines.py`:
- **A — `raw_ellipse` (frozen baseline).** Identity. Never edited; the reference.
- **B — `corrected`.** A × per-`(sex, measurement)` scale fitted as `median(gt/pred)`
  — the L1-optimal closed form the backend's `--fit-offsets` uses ("pulp but
  lighter", no LP). **Fit on `train`, applied to test** (real generalization, not
  fit-on-test; `--fit-n` caps the fit set — the median stabilizes by ~300).
- **C — `gated_corrected(tau=0.05)`.** B, but **skip cells the fit deems already-good**
  — if the fitted scale is within `tau` of 1.0 (≈no bias to remove), leave raw
  untouched (correcting an unbiased cell only adds error — see testA chest under B).
  Gates on fitted-scale magnitude, so it can't separate an already-good cell from a
  biased one when their scales coincide — a blunt but honest guard, same spirit as
  the backend dropping sub-0.25% scales.
- **D — `learned_girth`.** **Replaces the ellipse formula** with a learned nonlinear
  width→girth map: `circ = c0 + c1·front + c2·side + c3·(front·side)`, a bilinear
  least-squares fit per `(sex, measurement)` on `train`. The cross term makes it
  nonlinear; it subsumes B's constant scale *and* captures how the real
  cross-section drifts from an ellipse with size. **Cheaper than a neural net,
  richer than the fixed Ramanujan ellipse** — the "improve the geometric model"
  middle ground (superellipse/Lamé is the same idea; the learned map is more
  flexible). Unlike A/B/C it consumes the raw front/side *widths*, not the ellipse
  output — which is why the adapter now surfaces `(circ, front_w, side_w)` per row.

The orchestrator prints, per measurement, each experimental pipeline's MAE + Δ-vs-A
with a ✓/✗ — **are we going the right direction?**

B/C scales are for the *ceiling benchmark only* — they scale UP (silhouette
under-reads), the opposite sign to the app's `_SEX_CIRCUMFERENCE_SCALES_PCT` (which
scale DOWN); they do not transfer to the app. D's learned coefs are likewise
fit to the *silhouette* front-end, not the app's keypoint widths — a separate fit
would be needed for production. All BodyM numbers stay a perfect-mask ceiling.

Next pipeline slot: the `CircumferenceRegressor` (full MLP on the whole contour,
train on BodyM `train`) drops in as a fifth registry entry, no orchestrator change.

## Built + results (2026-07-13, `bodym.py`, full splits)

Runs end-to-end: fetch (unsigned S3, mask cache) → adapter → real
`ramanujan_ellipse_circumference` → table + buckets + `reports/run_NNN/metrics.json`.
Coverage = all measured subjects, one photo each: **testA n=87** (controlled),
**testB n=400** (in-the-wild).

Params fit on 300 train subjects (scales converge — n=60 ≈ n=300). MAE in cm.

**testA (controlled) — overall A 6.4 · B 5.8 · C 5.8 · D 3.7:**

| measure | A raw | B scale | C gated | **D learned girth** |
|---|---|---|---|---|
| chest | 7.8 | 8.9 ✗ | 8.7 ✗ | **4.6 ✓** |
| waist | 5.9 | 3.6 ✓ | 3.6 ✓ | **4.0 ✓** |
| hip | 5.4 | 3.6 ✓ | 4.0 ✓ | **2.8 ✓** |
| thigh | 6.7 | 7.0 ✗ | 7.0 ✗ | **3.5 ✓** |

**testB (in-the-wild) — overall A 9.4 · B 6.3 · C 6.8 · D 4.6:**

| measure | A raw | B scale | C gated | **D learned girth** |
|---|---|---|---|---|
| chest | 10.2 | 8.4 ✓ | 9.4 ✓ | **5.8 ✓** |
| waist | 9.7 | 4.9 ✓ | 4.9 ✓ | **4.6 ✓** |
| hip | 7.9 | 5.8 ✓ | 7.0 ✓ | **4.3 ✓** |
| thigh | 10.1 | 6.0 ✓ | 6.0 ✓ | **3.7 ✓** |

**Verdict — D (learned girth) wins decisively; the ellipse was the floor, not the ceiling.**
The learned nonlinear width→girth map cuts error **40–50% vs the ellipse (A)** and
beats *every* measurement on *both* splits — including chest and thigh, which the
scale corrections (B/C) made *worse*. It subsumes B's scaling and additionally
corrects the cross-section shape, at 4 coefficients per cell (no neural net). At
~4–5 cm it approaches the ~2–3 cm the literature reports for a full-contour MLP
regressor — the only remaining step up, at much higher cost.

Notes on B/C (kept as instructive baselines): B (uniform per-sex scale) helps
proportional to the raw bias but overshoots already-unbiased cells (testA
chest/thigh ✗). C (gated) is a no-op-or-worse *here* because BodyM's underestimate
is uniform (no already-good cells to protect) — it's the mechanism that would pay
off for the *app*'s heterogeneous biases, not for this ceiling.

**BMI>30 is the worst bucket everywhere (13–14 cm raw)** — the ellipse cross-section
breaks down on obese bodies; D's shape term absorbs part of this, but it's the
hardest regime.

**Two findings the eval surfaced:**
1. **Systematic underestimate — on testB the bias IS the error.** Every
   circumference reads low (testA −2 to −5 cm; testB −6 to −9 cm). On testB
   `bias ≈ MAE` (waist −9.0 / 9.7) → almost all error is a *fixed offset*, not
   scatter. **A per-measurement bias correction (production already does a per-sex
   one, not applied here) could roughly halve testB error** — the single
   highest-leverage lever. The eval doing its job: quantifying a known correction.
2. **Chest is not cleanly recoverable from an arms-down frontal silhouette.** The
   arm fuses to the torso above the armpit and occludes the bust line (breadth
   plateaus, then cliffs where the arm separates). The adapter detects that cliff
   and measures the widest clean torso row below it — an underbust-ish proxy.
   Higher variance is inherent to the pose, **not** a bug. Production mitigates
   with arms-out capture + keypoint occlusion handling, neither of which BodyM has
   — another reason BodyM is a *lower bound on input quality*, harder than the app.

### Adapter internals (the new code, all in `bodym.py`)
- **Calibration:** per-mask `height_cm / body_pixel_length` (front & side independent).
- **Arm clip:** `torso_width_at` splits each row into segments (`_find_segments`,
  the same helper production uses) and keeps the one straddling the body axis —
  detached A-pose arms drop out.
- **Rows:** waist = narrowest torso row; hip = widest below; chest = widest clean
  row below the armpit cliff; thigh = a single leg below the crotch split.

### Golden set (`golden/*.jsonl`)
One row per subject (measurements are per-subject; masks map via
`subject_to_photo_map.csv` — pick one front+side pair per subject):
```json
{"subject_id":"...", "split":"testA", "sex":"female", "height_cm":164.4,
 "mask_front":"testA/mask/<id>.png", "mask_side":"testA/mask_left/<id>.png",
 "expected":{"chest":99.3,"waist":87.3,"hip":92.1,"thigh":52.4,
             "arm":46.8,"leg":75.1,"shoulder":33.8}}
```
Fixed external dataset ⇒ **no golden-drift risk** (the ADR's #1 failure surface) —
an advantage over hand-labeled corpora.

### Baseline-first (adoption doc #5)
Row 0 = predict the per-sex training mean for every measurement. Every pipeline
reports **lift over this floor**. If ellipse geometry can't beat "always guess the
average woman," that's the headline finding.

### Metrics (numeric — NOT precision/recall)
The polygon ADR's open caveat ("pet-growth may be numeric prediction, schema may
not transfer") **is exactly our case.** Per measurement, per split:
- **MAE**, **RMSE**, **mean bias** (signed — reveals systematic over/under-estimate)
- **% within ±1 / ±3 / ±5 cm** (the tolerance ladder)
- **Safety metric = wrong-size-rate**: % of predictions off by more than one
  garment-size step (≈2 cm chest). The numeric analog of the ADR's "wrong-fold
  rate" — high accuracy + high wrong-size = silently sizing people wrong.
- **cost / latency** columns (adoption doc #5): ellipse ≈ free, regressor cheap —
  record anyway so future model swaps are a measured decision.

### Error buckets
Tally MAE by: **split** (testA controlled vs testB in-the-wild — robustness),
**sex**, **BMI band** (from hwg weight+height — obesity is where ellipse breaks),
**measurement**. The histogram shows where the ROI is (per the 63.6%→1% bucket
story in the adoption doc).

### Diff report
Each run vs baseline (and vs previous run): per-measurement **wins** and
**regressions**. The most useful artifact — decisions ride numbers, not feelings.

## Sequencing (rule-based first, per ADR)
1. **Baseline** (per-sex mean) — the floor.
2. **Ellipse** (A) — the current production math, silhouette-fed. This is the real
   ceiling question: how much of our error is the ellipse model vs segmentation?
3. **Regressor** (B) — train `CircumferenceRegressor` on BodyM `train`, eval on
   `testA`/`testB`. Only pursue hard if A plateaus below target (the ADR's
   "don't build ML until the rule-based ceiling is hit").

## Relationship to the existing backend eval (`pointsx-eval`)

`Pointx-backend/src/pointsx/eval.py` already exists and is the **end-to-end app
eval**: it runs the real RGB pipeline (pose + seg + circumference + envelope) on a
GT-labeled subjects CSV, has a pose×regressor×offset **grid**, and **`--fit-offsets`**
which fits per-`(sex, measurement)` multiplicative corrections as the L1-optimal
`median(gt/pred)` and prints a paste-ready `_SEX_CIRCUMFERENCE_SCALES_PCT` for
`envelope.py`. This BodyM eval does **not** replace it — they measure different
things and must not be conflated:

| | `pointsx-eval` (backend) | `eval/bodym.py` (this) |
|---|---|---|
| input | RGB front+side + GT CSV | silhouette masks + GT (BodyM) |
| exercises | full pipeline (pose+seg+circ) | geometry only (widths→ellipse) |
| answers | the **app's live accuracy** | the **ceiling** given perfect masks |

**Critical: the biases point opposite ways.** The app's `_SEX_CIRCUMFERENCE_SCALES_PCT`
are all negative (−5…−24%: the app *over*-predicts). BodyM's silhouette path
*under*-predicts (needs positive correction). So **BodyM-fit constants do NOT
transfer to the app** — the app's corrections absorb its keypoint+regressor
front-end bias, which BodyM doesn't share. The app's constants can only be refit on
real GT-labeled RGB subjects shot through the app (today: n=3 — the actual gap).

## Scope boundary
`mask snapshot → eval → report`. Read-only. Nothing writes back into `src/`. The
adapter and pipelines import `src/pointsx` live; they never copy it.
