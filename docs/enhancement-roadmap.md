# PointsX → 10/10: Full Project Review & Enhancement Roadmap

> Evidence-based. Every claim below traces to a measured number (eval runs
> `eval/reports/run_001…014`), a production log incident, or cited literature —
> not vibes. Written 2026-07-18, after the BodyM eval sprint.

---

## 1. What "10/10" means (measurable, or it doesn't count)

| Dimension | Today (measured/estimated) | 10/10 target |
|---|---|---|
| Circumference MAE, end-to-end real photos | **unknown** (n=3 eval); geometry ceiling 3.7–4.6 cm | **≤ 2.5–3 cm** |
| Length/width MAE | unknown; lengths are easier | ≤ 1.5–2 cm |
| Repeatability (same person, re-capture σ) | never measured | **≤ 1.5 cm** |
| Wrong-size rate (error > 5 cm, silent) | 50–87 % at ceiling (raw ellipse) | **< 5 %**, rest abstains |
| Latency (2 vCPU free tier) | 11.5–13 s | ≤ 5 s CPU / ≤ 1 s GPU |
| Trust surface | warnings list | per-measurement confidence interval |
| Privacy | E2E dataset flow; but prod archives photos unencrypted | process-and-discard or E2E, consistently |
| Eval discipline | BodyM orchestrator (A–D) + backend `pointsx-eval` | every change gated by both tracks |

**Repeatability deserves equal billing with accuracy.** A tailor re-measures and
gets the same number; if two captures of the same person differ by 4 cm, no B2B
buyer trusts the API regardless of average MAE. It has never been measured and
costs one afternoon (§7c).

---

## 2. Where the error actually comes from (error budget)

The session's evals let us decompose the pipeline. Approximate per-stage
contribution to circumference error, evidence-tagged:

| Stage | Contribution | Evidence |
|---|---|---|
| **Capture** (pose variance, tilt, distance, clothing) | 2–4 cm | Prod warnings on real requests: `leg/height 0.65` vs range [0.43–0.53], `Neck 78.2 cm`, `shoulder/height 0.20`; testA (controlled) vs testB (wild) gap = 6.4 → 9.4 MAE on identical code |
| **Calibration** (px→cm) | 1–3 cm | head-top-on-hair offset; front/side scale mismatch warn threshold ±15 % ≈ ±3 cm on a 100 cm girth; the `leg/height 0.65` incident is a calibration/keypoint failure reaching the user |
| **Keypoint row placement** (which y-row is "waist") | 1–3 cm | chest arm-occlusion finding (BodyM probe: breadth plateaus, cliffs at armpit); prod ratio warnings |
| **Segmentation edges** | 0.5–1 cm | ±2 px at ~0.2 cm/px, two edges, ×π for girth |
| **Widths→girth model** | ellipse 5–9 cm → **learned bilinear 3.7–4.6** → contour CNN ~2–3 (lit.) → SMPL ~1–2 (lit.) | runs 013/014; ICPR-2020 silhouette benchmarks; arXiv 2205.14347 |
| **Bias constants** | unbounded risk | live `_SEX_CIRCUMFERENCE_SCALES_PCT` fit on **n=3** |

Two structural conclusions:

1. **The ellipse was the floor, not the ceiling.** Proven: a 4-coefficient
   bilinear width→girth fit (pipeline D) cuts geometry error 40–50 % with no
   neural net. The measurement core is no longer the biggest unknown.
2. **The biggest *unmanaged* error is upstream: capture + calibration + rows.**
   The testA→testB degradation (identical code, wilder photos) is the largest
   single jump in the whole table. Model work cannot buy this back.

---

## 3. The 10/10 architecture

```
CLIENT (browser)                          SERVER
┌─────────────────────────┐   ┌─────────────────────────────────┐
│ Burst capture (5 frames)│   │ Seg (INT8) ─┐                   │
│ MediaPipe hard gates:   │   │ Pose (INT8) ─┤→ fused calibration│
│  arm angle, tilt (gyro),│──▶│              │  (kp + mask extent│
│  distance, profile check│   │              │   + outlier reject)│
│ best-frame selection    │   │ rows: kp band → silhouette      │
│ reject BEFORE upload    │   │        extremum refinement       │
└─────────────────────────┘   │ widths → LEARNED GIRTH MAP (D)  │
                              │   → later: contour regressor     │
        data flywheel         │   → later: 2-view SMPL-X        │
┌─────────────────────────┐   │ conformal CI per measurement     │
│ Dataset page (E2E, GT)  │◀──│ abstain-with-reason if CI wide  │
│ Synthetic SMPL-X engine │   └─────────────────────────────────┘
└─────────────────────────┘
```

Staged. Each stage is shippable and eval-gated; no stage bets on the next.

### Stage 0 — this week, no model work (cuts the worst risks)
- **Refit bias constants on the Supabase GT corpus.** The dataset page collects
  *user-entered tape measurements* (required fields) + E2E photos — **n=18 real
  GT subjects exist right now**, vs the n=3 the live constants were fit on. Run
  the existing `pointsx-eval --fit-offsets` over them. Zero new code.
- Client hard gates before upload (§4.1a–b). Kills the `leg/height 0.65` class.
- Fix the privacy contradiction (§4.11a): marketing says process-and-discard;
  prod archives raw photos to S3. Decide one way, today it's both.
- CORS pin + rate limit + body-size cap (§4.11c).

### Stage 1 — 2–4 weeks → end-to-end ≤ 4–5 cm, ≤ 5 s
- Port the eval's silhouette row-refinement into production (proven on BodyM).
- Calibration fusion: keypoint + mask-extent + outlier rejection.
- ONNX/OpenVINO INT8 → seg 4–6 s → ~1.5–2 s.
- Ship the D-form bilinear girth map fit on app GT once n ≥ 30–50.
- Conformal intervals per measurement; abstain-with-reason.

### Stage 2 — 1–2 months → ≤ 3 cm
- Scale the synthetic SMPL-X engine (scaffold already in repo) to 10–50 k
  bodies with domain randomization → license-clean training data.
- Retargeted contour regressor (existing `CircumferenceRegressor`, +chest,
  +sex, full-profile features) trained on synthetic + app GT.
- Seg/pose fine-tune on the same synthetic renders. Burst capture.

### Stage 3 — 3–6 months → ≤ 2 cm, scanner-adjacent
- Two-view learned SMPL-X regression (arXiv 2205.14347 shape — literally our
  input setup), trained on synthetic, fine-tuned on app GT; measurements read
  off the fitted mesh. GPU tier when a B2B pilot pays for it.

---

## 4. Aspect-by-aspect enhancement report

Format: **current score → what limits it → approaches ranked cheap→heavy**
(effort: S ≤ 1 day, M ≤ 1 week, L ≤ 1 month, XL beyond).

### 4.1 Capture UX & photo QA — 6/10
MediaPipe gate + voice guidance exist and are good. But: single frame, two
sequential poses, no tilt/distance control — and prod warnings prove bad
captures reach the pipeline.
- **(a, S)** Hard client gates: arm-torso angle 20–45°, both ankles + head-top
  in frame with margin, frontal symmetry check (shoulder x-symmetry), side-view
  profile check (shoulder overlap). Reject *before* upload with the specific
  fix spoken aloud (TTS already wired).
- **(b, S)** Gyro tilt gate (`DeviceOrientation`): phone vertical ±3°, warn on
  height ≠ ~chest level. Perspective foreshortening is a *systematic* bias —
  cheaper to prevent than correct.
- **(c, M)** Burst capture: 5 frames, on-device pose scores each, upload best
  (or two best → server averages masks). Kills blur + pose jitter.
- **(d, M–L)** Tilt rectification server-side: homography from gyro angle (sent
  as metadata) before measurement. Only after (b) shows residual tilt matters.
- **(e, S–M)** Loose-clothing detector: silhouette convexity/roughness
  heuristic → "tight clothing" warning. Clothing is unmodeled error today.

### 4.2 Calibration (px→cm) — 5/10
Keypoint head-top→ankle-midpoint with fixed-ratio fallbacks (8 % head, 22 %
ankle-knee). Failure modes: HEAD_TOP lands on hair (+2–4 cm height → −2 %
scale → −2 cm on a 100 cm girth), keypoint drift (the 0.65 incident), tilt.
- **(a, S)** Mask-extent cross-check: mask top/bottom vs keypoint height; if
  they disagree > 3 %, prefer the mask + warn. The BodyM adapter proved mask
  extent is clean (side-depth 23–28 cm anatomically exact).
- **(b, M)** Fusion: median of {kp height, mask extent, proportion-implied
  heights from shoulder-hip and hip-ankle} with outlier rejection. One robust
  scale instead of one fragile one.
- **(c, M)** Row-dependent scale under known tilt (from 4.1b metadata):
  px/cm(y) linear in y. Removes the systematic hip/thigh bias of a tilted phone.
- **(d, S–M)** Optional reference-object mode for B2B (A4 sheet on the floor):
  sub-1 % scale for users who opt in.

### 4.3 Segmentation — 7/10
yolo12l-person-seg is stable (the nano experiment was reverted for good
reason). Edges and loose clothing are the residual.
- **(a, S)** Post-process guarantee: largest component, hole fill, 1-px
  morphological smooth — cheap insurance if not already exhaustive.
- **(b, S + eval)** Quantify before optimizing: perturb masks ±1–2 px in the
  eval → Δcm. If Δ < 0.5 cm, *stop investing here* (suspected outcome).
- **(c, M)** Torso-band matting refinement (MODNet/BiRefNet-lite ONNX on a
  crop, ~0.3 s) — only if (b) says edges matter.
- **(d, M–L)** Fine-tune on synthetic renders + hard real cases (low contrast,
  loose clothes). Shares the Stage-2 synthetic corpus.

### 4.4 Pose / landmarks / measurement rows — 5/10
The rows (which y is "waist") anchor everything. Prod ratio warnings + the
BodyM chest-occlusion finding show this is a live error source.
- **(a, S–M, proven)** Hybrid rows in production: keypoints give the band,
  silhouette width-profile extrema refine within it (waist = narrowest, hip =
  widest below, chest = widest clean row below the armpit cliff). This is the
  eval adapter's logic — already validated on 487 real subjects.
- **(b, M–L)** Fine-tune pose on the synthetic corpus: the synthetic pipeline
  already emits 25 landmarks *including bust/waist/hip lines* — train the
  detector on the exact anatomical definitions the tape measure uses.
- **(c, L)** Dense landmark head (bust/waist/hip lines as direct regression
  targets). Stage-3 material.

### 4.5 Widths→girth core — was 4/10, path proven
Measured on BodyM (perfect masks): raw ellipse 6.4/9.4 cm (testA/testB) →
**bilinear learned girth 3.7/4.6 cm**, winning every measurement on both splits
including the ones scale-corrections made worse.
- **(a, S once data)** Ship pipeline D's form in production: per-sex
  `c₀+c₁·w_f+c₂·w_s+c₃·w_f·w_s`, **fit on app widths + app GT** (the BodyM
  coefficients do NOT transfer — the app's front-end bias has the opposite
  sign, proven in this sprint). Needs n ≥ 30–50; until then keep current scales.
- **(b, S–M)** BMI-aware term (worst bucket everywhere, 13–14 cm): add a
  height-normalized width or width² feature, or a per-BMI-band exponent
  (superellipse). Directly targets the failure regime.
- **(c, M)** Retarget `CircumferenceRegressor`: +chest output, +sex input,
  features from the full width *profile* rather than 7 rows. Train: synthetic +
  app GT (commercial-clean) / +BodyM (research track only — CC BY-NC).
- **(d, L)** Two-view contour encoder → measurements (~1.6–3 cm in lit.).
- **(e, XL)** Two-view SMPL-X regression, measure on mesh (~1–2 cm). Endgame.

### 4.6 Bias constants / correction — 3/10 (the scariest live number)
`_SEX_CIRCUMFERENCE_SCALES_PCT` (−5…−24 %) was fit on **three people** and
multiplies every circumference the product returns.
- **(a, S, do first)** Refit on the 18 Supabase GT subjects with the existing
  `pointsx-eval --fit-offsets` (it prints the paste-ready dict). Verify the
  measurements field is tape-GT during import (the form requires user entry).
- **(b, process)** Grow to 50–100 subjects → stable per-sex fits; 300+ →
  per-BMI-band. The university showcase is a collection event: the page is
  built, E2E works, the viewer works.
- **(c, S)** Replace constant scales with the D-form bilinear at n ≥ 50 (§4.5a
  subsumes this aspect entirely).

### 4.7 Data engine (real GT) — 4/10 infra, 2/10 volume — **the moat**
Everything model-side is bounded by this. Infrastructure exists (dataset page,
E2E NaCl, Supabase, tkinter viewer, decrypt scripts); volume is n=18.
- **(a, S)** GT protocol one-pager on the dataset page: how to hold the tape,
  measure twice, garment state, capture immediately after measuring.
- **(b, process)** Showcase campaign target: 50 subjects. Every subject is
  simultaneously a bias-fit point AND a regressor training row AND an eval row.
- **(c, S)** Test-retest subset: 10 volunteers × 3 captures → the repeatability
  number (§1). Doubles as the abstention-threshold calibration set.
- **(d, S–M)** Mine the 150 archived S3 sessions (photos + predictions, no GT)
  for a QA-failure taxonomy: how often do validation warnings fire, on what
  photo conditions → feeds the capture-gate rules (§4.1a).

### 4.8 Data engine (synthetic) — scaffold 6/10, usage 1/10
`src/pointsx/synthetic/` already does SMPL-X sampling, 25 landmarks, GT
measurements from mesh, Blender headless rendering. It has never been run at
scale — yet it solves three problems at once: training data for seg + pose +
girth heads, all license-clean (BodyM is CC BY-NC and cannot ship in the paid
product; synthetic can).
- **(a, M–L)** Scale to 10–50 k bodies with domain randomization: body shape
  from measured population priors, A-pose jitter ±10°, camera pitch/height/
  distance jitter, lens FOV, HDRI backgrounds, cloth displacement noise.
- **(b, S per run)** Domain-gap gate: a synthetic-trained pipeline must be
  scored on BodyM testB + app GT before any real-data conclusions.
- **(c, —)** The same renders back every fine-tune in §4.3d and §4.4b — one
  corpus, three consumers.

### 4.9 Uncertainty & abstention — 4/10
Warnings exist and measurements are omitted on hard failures (good instinct,
already better than most competitors). Missing: calibrated per-measurement
uncertainty.
- **(a, S–M)** Conformal prediction intervals from eval residuals, per
  measurement × BMI band: ship `[lo, hi]` in the envelope, render "waist 87 ±3"
  in the UI. Distribution-free, no model change, honest by construction.
- **(b, S)** Abstain-with-reason: if capture-QA score low or CI wider than the
  garment-size step → "retake: turn 90°, arms slightly out" instead of a
  number. The wrong-size rate is the metric this drives down.
- **(c, S)** With burst capture (§4.1c): report empirical inter-frame spread as
  a free second uncertainty signal.

### 4.10 Serving & performance — 6/10
Async FastAPI, model warmup, per-phase timing — solid. 11.5–13 s total on
2 vCPU (seg 3.5–6 s per view is the bottleneck).
- **(a, M)** ONNX Runtime / OpenVINO INT8 for both models: seg → ~1.5–2 s,
  total → ~4–5 s. The single biggest UX win per engineering-hour.
- **(b, S + eval gate)** Try 960 px seg input (from 1280): accept only if the
  eval shows Δcm ≈ 0.
- **(c, S–M)** Batch front+side as batch=2 in one forward pass per model.
- **(d, —)** GPU only when a pilot pays: HF A10G or a ~$20/mo VM → ~1 s.

### 4.11 Privacy & security — 7/10 design, with one contradiction
E2E dataset flow (NaCl, keys never on server) is genuinely ahead of the
competitors. But:
- **(a, S, policy — do in Stage 0)** The contradiction: marketing-report.md
  claims "no photo storage, process and discard", while `/api/measure` archives
  raw photos + envelope to S3 on every request. Either stop archiving, or make
  it an explicit consent checkbox routed through the E2E path. For EU (stated
  target): this is not optional.
- **(b, S–M)** Any retained data → the existing NaCl encrypt-at-rest flow.
- **(c, S)** CORS pinned to the Vercel origin (currently `*`), per-IP rate
  limit, request body cap, key rotation off the flaky provider (AWS migration
  already in progress).
- **(d, docs)** EU readiness: retention schedule, DPA template, on-prem story
  for B2B (the container is already self-contained — this is a docs task).

### 4.12 Product & API (B2B) — 5/10
Envelope JSON + debug overlay + `/api/health` are right. Missing the B2B shell:
- **(a, S–M)** API keys + per-key usage metering (header + counter — no more).
- **(b, M)** Embeddable widget build: iframe + `postMessage` result contract —
  the marketing report's stated Stage-1 deliverable.
- **(c, S–M)** Envelope schema versioning + webhook push.
- **(d, S)** SLA endpoint: p50/p95 latency, abstention rate, uptime — B2B
  buyers ask for exactly these three numbers.

---

## 5. Eval discipline (the meta-aspect that makes the rest safe)

Already built this sprint: the BodyM orchestrator (`eval/bodym.py` +
`eval/pipelines.py`, A–D registry, train-fit/test-apply, per-bucket reports,
persisted runs) and the backend `pointsx-eval` (end-to-end RGB, grid,
`--fit-offsets`). Keep two tracks, never conflate them:

| Track | Question it answers | Data |
|---|---|---|
| BodyM orchestrator | did the *geometry* improve? | 487 real subjects, perfect masks — a ceiling |
| `pointsx-eval` on app GT | did the *product* improve? | Supabase corpus (n=18 → 50 → …) |

Rules (BuildLab polygon canon, now proven useful here twice):
1. No pipeline change ships without both tracks run; regressions block.
2. New pipeline = new registry entry, scored against frozen A automatically.
3. Every fitted constant states its n in a comment next to the values.
4. BodyM-derived weights/constants never ship in the commercial product (NC
   license + proven opposite-sign bias).

### 5.1 Benchmark hierarchy — train / dev / final (and two finals, not one)

Standard train/dev/final ML hygiene, adapted to this task. **All non-owned sets
are evaluation-only** (research licenses permit benchmarking, never training,
never shipping derived weights — same rule as BodyM).

| Dataset | Role | Measures | Usable now? |
|---|---|---|---|
| Own synthetic | **train** | — | after the clothing rebuild (`synthetic-data-review.md`) |
| **BodyM** | **final — algorithm** | circumferences, cm | ✅ built (`eval/bodym.py`) |
| **App GT (Supabase)** | **final — product** | circumferences, cm | ✅ (n=18 → grow) via `pointsx-eval` |
| **SHAPY / HBW** | **dev — measurement** | measurements, cm | ✅ next: closest published task |
| AGORA | dev — front-end / mesh | PVE/MPJPE, mm | ⚠️ only with the SMPL-X (Level C) route |
| BEDLAM | dev — synthetic transfer | PVE/MPJPE, mm | ⚠️ only with Level C |

Three things the naive "just add benchmarks" plan gets wrong:
- **Two finals, not one.** BodyM = *algorithm* final (accuracy given perfect masks
  + true height — a ceiling that bypasses seg/pose/calibration/clothing). App GT =
  *product* final (real end-to-end). Neither substitutes for the other.
- **SHAPY is a *measurement* benchmark, not a shape proxy.** Its HBW split has real
  cm measurement GT and it's the nearest published task (attributes + image →
  measurements) — so it's the dev number to cite against prior work, not just a
  shape-quality monitor.
- **AGORA/BEDLAM score meshes (PVE/MPJPE), not circumferences.** With the current
  geometric pipeline (A–D, no mesh) they cannot score measurements at all — they
  only validate the seg/pose front-end. They become measurement-relevant *only*
  after the SMPL-X regression route (Level C) produces a mesh to measure. Do not
  build these harnesses before Level C + a paper goal; until then they're academic
  overhead with near-zero product ROI.

**Build order:** BodyM (done) → SHAPY/HBW (next, reuses measurement plumbing) →
AGORA/BEDLAM (gated on Level C). For a paper the four-benchmark story is strong
(measurement=BodyM+SHAPY, real-world=AGORA, synthetic-transfer=BEDLAM); for the
product only the two cm-measurement finals + SHAPY matter.

---

## 6. What NOT to do

- **Don't fit production constants on BodyM.** Proven this sprint: its bias has
  the opposite sign to the app's (silhouette under-reads, app over-reads).
- **Don't buy accuracy with model size before capture QA ships.** The
  testA→testB gap says photos, not weights, are the binding constraint.
- **Don't ship BodyM-trained weights commercially** (CC BY-NC). Synthetic
  corpus is the clean path.
- **Don't chase segmentation SOTA before §4.3b quantifies edge sensitivity.**
- **Don't build the full SMPL-X track before Stage 1 lands.** It's the best
  endgame and the worst first move: heavy, slow to validate, and its gains are
  masked by capture noise until the upstream is fixed.
- **Don't keep the archival contradiction.** It's a one-line policy decision
  with EU-market consequences.

---

## 7. Sequencing at a glance

| When | Do | Expected outcome |
|---|---|---|
| **Week 0** | Refit constants on Supabase n=18 · client hard gates · privacy contradiction fix · CORS/rate-limit | worst live risks closed, zero model work |
| **Weeks 1–4** | prod row-refinement · calibration fusion · INT8 · D-form girth (n≥30) · conformal CIs | **≤4–5 cm end-to-end, ≤5 s, honest intervals** |
| **Months 1–2** | synthetic engine at scale · retargeted contour regressor · seg/pose fine-tune · burst capture | **≤3 cm** |
| **Months 3–6** | two-view SMPL-X regression · mesh measurements · GPU tier with pilot revenue | **≤2 cm, scanner-adjacent** |
| **Continuous** | GT collection (50→300) · test-retest metric · eval gates on every change | the moat compounds |
