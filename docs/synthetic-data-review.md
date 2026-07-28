# Synthetic Data Generator — Review + Deep Research

> Review of `src/pointsx/synthetic/` against what production-grade synthetic-human
> pipelines actually do (BEDLAM, SURREAL, CLOTH3D, SynBody, …), with concrete
> GitHub/paper references for our exact case: **clothed people → body measurements**.
> Legal frame: this is the license-clean training source that replaces
> [[bodym-eval-only-license|BodyM (eval-only)]]. Date: 2026-07-18.

---

## 1. Verdict: real potential, sound bones — but not measurement-grade yet

**Keep the architecture. Rebuild the clothing and the body-shape realism.**

The 4-phase orchestration is genuinely good and matches how SURREAL/BEDLAM are
built: `sample bodies → SMPL-X forward → GT measurements from the mesh → Blender
render + YOLO labels`, manifest-driven, parallel Blender jobs, both pose and seg
datasets from one SMPL-X pass, UV/SMPLitex skin textures, GT measurements read off
the mesh with the *same* definitions used at inference. That last point is the
single most valuable property — train and inference measure identically by
construction. This is not a throwaway scaffold; the skeleton is right.

The problem is **fidelity where it matters for our task**. The whole downstream
pipeline consumes a *silhouette*, and the generator's silhouette is wrong in ways
that would teach the model the opposite of the real-world failure mode.

---

## 1b. Current state — it does not run, and the GT is unreliable (analysed 2026-07-18)

**Runnability: BLOCKED.** Nothing can be generated locally right now:

| Requirement | State | Note |
|---|---|---|
| `smplx`, `torch`, `scipy`, `tqdm` (py pkgs) | ✅ present | in the PointsX venv |
| `trimesh` | ❌ missing | needed by parts of the mesh path |
| **`smpl_anthropometry`** | ❌ **missing** | the correct GT measurement library — see below |
| **SMPL-X model files** (`models/smplx/SMPLX_*.npz`) | ❌ absent | registration-gated download (smpl-x.is.tue.mpg.de) |
| **Blender executable** | ❌ not installed | required for the render phase |
| **`assets/`** (clothing OBJs, HDRIs, skin/SMPLitex textures) | ❌ absent | required for the render phase |

**Two-phase consequence:** the SMPL-X phase (body sample → mesh → GT measurement)
needs only `smplx` + the model files + a measurement lib — **not** Blender/assets.
So the GT can be de-risked cheaply and first, before any rendering.

### ✅ Fragility #0 — GROUND TRUTH: FIXED (2026-07-18)
Root cause confirmed and repaired. The code called `MeasurementComputer(...).compute(...)`
— a class/method that **does not exist** in SMPL-Anthropometry (real API:
`MeasureBody("smplx").from_verts().measure()`), the `ImportError`/`AttributeError`
was swallowed, so **every** body silently got the crude geometric + hard-coded
constant GT. Fix shipped:
- Vendored SMPL-Anthropometry (MIT) → `synthetic/anthropometry/` + a headless
  `runner.measure_smplx(verts, joints, faces)` wrapper (plane-slice + body-part
  face-segmentation + convex-hull perimeter; arms can't contaminate waist/hip).
- Rewrote `measurements_gt.py`: correct method, **all hard-coded constant
  fallbacks deleted**, unmeasurable fields → `None`, plausibility bounds discard
  (never clamp) bad slices, `has_core_measurements()` gate.
- `pipeline.py` skips any body missing the four core circumferences.
- Validated: measurements match BodyM real ranges, monotonic with BMI; a
  degenerate mesh returns empty (no constants, no crash). 6/6 real bodies usable.

Original description of the defect (kept for the record):

### 🔴 Fragility #0 — the GROUND TRUTH was the deepest problem (`measurements_gt.py`)
For a *measurement* dataset the labels matter more than the pixels, and the labels
are currently unreliable:
- `compute_measurements` tries `smpl_anthropometry` (the correct, vetted library)
  first — but it is **not installed**, and the `ImportError` is swallowed
  silently → it **always falls back** to `_geometric_measurements`.
- `_geometric_measurements` is crude: it convex-hulls the vertices in a ±1.5 cm
  Y-band with a fixed `max_x_radius` to "exclude arms" — but in A-pose the arms
  hang beside the torso at waist/hip height, so a 0.22–0.30 m radius **includes the
  arms** → inflated waist/hip GT (the very arm-contamination we fought in the eval,
  now baked into the labels).
- Anatomical Y-levels are hand-guessed joint offsets (`chest = shoulder_y − 0.12`,
  `waist = spine1 joint`, `hip = pelvis joint`) — not the tape-measure definitions
  (max bust / min waist / max hip), so even a clean slice is biased.
- **Worst: hard-coded constant fallbacks.** When a slice returns below threshold it
  silently substitutes population constants (`chest=90, waist=75, hips=95,
  thigh=50, neck=35, wrist=16`). Any body whose slice fails gets **identical fake
  GT** — training on that teaches the model to regress toward constants.

**This must be fixed before anything else** — realistic clothing on top of wrong
labels still yields a worthless measurement dataset. Fix = install/repair
`smpl_anthropometry` (verify its real API matches the calls), or replace the GT
with a vetted mesh-slicer (the blueprint's `mesh_measure.py` convex-hull-per-plane
is the right shape), and **delete the constant fallbacks** — a failed measurement
must be dropped, never faked.

## 2. Fragilities, ranked by impact on measurement accuracy

### 🔴 1. Mask mode strips the clothing (`blender_render.py:671`) — the critical one
Seg/mask renders the **nude body** (`mode=="mask"` → no clothing). But:
- Production segmentation sees **clothed** people, and the mask it produces is the
  **clothed outline**.
- The entire measurement chain (widths → girth) runs on that mask.

So the model would learn body-silhouette→measurement, then be fed clothed
silhouettes at inference — a guaranteed domain gap, and it never learns the #1
real error (loose clothing inflates the silhouette). **The mask must be the
clothed silhouette.** Ideally emit *both* (clothed mask as input, body mask as a
learning target) so a model can learn to see the body *under* clothing.

### 🔴 2. Clothing is shrink-wrapped, not draped (`import_clothing:375`)
Garments are static OBJs with a `SHRINKWRAP` modifier → they hug the body surface.
Real clothing **drapes**: it hangs off the body, folds, and makes the silhouette
*larger* than the body (loose tops/trousers add 2–10 cm of apparent girth). A model
trained on shrink-wrapped clothes learns "clothed silhouette ≈ body silhouette" —
precisely the assumption that breaks on real photos. Needs **physics cloth
simulation with tightness variation**, not shrinkwrap.

### 🟠 3. Height by uniform mesh scaling — REMOVED, exposed a deeper issue (`run_smplx_forward`)
The mesh was uniformly scaled (`vertices *= target/raw`) to hit target height →
**every circumference scaled linearly with height** (wrong allometry), and it also
silently faked height on a degenerate pass (`raw_height_m = 1.7`). **Both deleted**
(2026-07-18): height is now emergent, mesh floor-aligned only, degenerate pass
raises (caller skips). GT measured from the mesh is self-consistent with its height.
- **Newly surfaced (was hidden by the rescale): the β→height mapping is broken.**
  `MEAN_HEIGHT_M`/`HEIGHT_STD_M` barely control height once random `β[2:]·0.8`
  proportions are added — a "target 198 cm" body can emerge at 153 cm. Per-body GT
  stays correct (real mesh, real girths, emergent height recorded), but the height
  *distribution* is uncontrolled. **Next fix (fragility #4): calibrate β sampling to
  an anthropometric prior** so heights land where intended without any rescale.
  `target_height_cm` is now vestigial (a β[0] seed); `actual_height_cm` (emergent)
  is the truth.

### ✅ 4. Shape / height sampling — FIXED (2026-07-18)
Now: stature sampled as a truncated normal per sex (≈NHANES/ANSUR M 176±7, F
163±6); **β[0] SOLVED** to hit target height via a *measured* dheight/dβ0 slope
(no hand-guessed constant, no mesh rescale → correct allometry); β[2:] from the
model's own N(0,1) prior (clipped ±2.5); β[1] still BMI-stratified for tail
coverage. Validated: **height solve MAE 0.02 cm**, realistic 149–177 cm spread,
two 175 cm males now carry different girths (chest 117 vs 95) — decoupling proven.
Lower-priority refinement remains: a measurement-conditioned/ANSUR-fit β prior
(SHAPY-style) would match the joint girth distribution exactly; revisit if the
domain-gap eval shows mismatch.

Original defect (kept for record):
Only `β0` (height) and `β1` (BMI class) are meaningful; `β2:10 = randn·0.8`, and
the "BMI class → β1 range" mapping is a hand-guess (SMPL-X betas aren't cleanly
"height"/"weight" on PC0/PC1). Result: a narrow shape manifold that may not match
real populations. Fix: fit β to a **real anthropometric measurement distribution**
(ANSUR II / CAESAR), or measurement-condition the sampling so the synthetic
population's girths match reality (this is exactly SHAPY's contribution).

### 🟡 5. Pose diversity is a-pose-only (`POSE_WEIGHTS=[1,0,0]`)
Fine for clean width extraction, but gives the model zero robustness to the pose
jitter real users produce. The renderer *does* randomize HDRI lighting, background,
grain, and camera (good) — but not **capture-realistic** nuisances: phone tilt,
camera height ≠ chest level, distance/foreshortening. Those are the exact things
production logs show breaking calibration. Add them as domain randomization.

### 🟡 6. Hand-rolled Blender script vs a framework
`blender_render.py` reinvents scene setup / domain randomization / multi-modal
output that **BlenderProc** (DLR-RM) provides as a maintained library. Not urgent,
but for prod-scale it's less brittle to adopt BlenderProc than to grow a bespoke
600-line render script.

---

## 3. Deep research — how the real pipelines do it (and what to borrow)

### Reference pipelines / datasets

| Project | What it proves / provides | Relevance to us | Link |
|---|---|---|---|
| **BEDLAM** (CVPR 2023) | Synthetic-ONLY training reaches SOTA on 3D human shape; 111 garments **physics-simulated** (CLO3D) on SMPL-X | The proof our whole synthetic bet is valid — *if* clothing is simulated, not shrink-wrapped | [site](https://bedlam.is.tue.mpg.de/) · [arXiv](https://arxiv.org/abs/2306.16940) |
| **SURREAL** (CVPR 2017) | The canonical SMPL+Blender synthetic-human pipeline; our orchestration mirrors it | Structural precedent; validates the 4-phase design | [github](https://github.com/gulvarol/surreal) |
| **CLOTH3D** | 7K+ sequences, garments with variability in type/topology/shape/size/**tightness**/fabric, cloth-simulated on SMPL | The reference for *tightness variation* — the exact axis we're missing | [github](https://github.com/hbertiche/CLOTH3D) · [arXiv](https://arxiv.org/abs/1912.02792) |
| **CLOTH4D** (CVPR 2023) | Physically-plausible draped dynamic meshes + textures, multi-view | Higher-fidelity drape reference | [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zou_CLOTH4D_A_Dataset_for_Clothed_Human_Reconstruction_CVPR_2023_paper.pdf) |
| **SynBody** | 1.2M images, **layered** human models (body + clothing layers), 10k bodies | Layered model = emit body-mask AND clothed-mask (our fix #1) | [site](https://synbody.github.io/) |
| **SHAPY** (CVPR 2022) | Body shape from **metric + semantic attributes** (height/weight/measurements) | Directly fixes #3/#4: attribute-conditioned β instead of scaling/random | [arXiv](https://arxiv.org/abs/2206.07036) |
| **ShapeBoost** | Clothing-**preserving** augmentation for shape estimation | A cheaper middle path than full cloth sim | [arXiv](https://arxiv.org/abs/2403.01345) |
| **AGORA-CLOTH** | Cloth types + accurate 3D body annotations, from AGORA | Another clothed-body annotation reference | (via SynBody/AGORA) |

### Tooling for the clothing rebuild

| Tool | Approach | Cost / license | Fit |
|---|---|---|---|
| **Blender cloth modifier** | Physics sim of garment meshes on the body; tightness via garment size + fabric params | Free, your own output = **license-clean** | ✅ default choice for a commercial product |
| **Garment-Pattern-Generator** (Korosteleva) | Sewing patterns → 3D garments, dataset-scale | research code | garment *variability* source | [github](https://github.com/maria-korosteleva/Garment-Pattern-Generator) |
| **GarmentCode** | Sewing patterns → draped mesh on SMPL via **Nvidia-Warp** cloth sim | open | programmatic garment generation + drape |
| **DrapeNet** | Learned self-supervised draping — fast, no per-garment sim | research | scale path once sim is too slow | [arXiv](https://arxiv.org/abs/2211.11277) |
| **CLO3D / Marvelous Designer** | Commercial garment sim (what BEDLAM used) | paid | best realism, manual-ish |
| **BlenderProc** (DLR-RM) | Procedural synthetic-data framework on Blender | free (BSD) | replaces the bespoke render script | [github](https://github.com/DLR-RM/BlenderProc) |

### ⚠️ License trap (same lesson as BodyM)
Most clothed-human datasets — **CLOTH3D, CLOTH4D, SynBody, AGORA, BEDLAM** — are
**research / non-commercial**. Reusing their meshes or renders in a *commercial*
product repeats the BodyM problem. For the shippable path, the safe choice is
**generate our own** (Blender cloth sim + commercially-licensed or self-made
garment assets), using the research datasets as *reference/validation*, not as
training data that ends up in the product.

---

## 4. Rebuild plan → production-ready (priority order)

Each step is independently shippable and gated by the domain-gap check (§5).

1. **Clothed masks (fix #1).** Make `mode=="mask"` render the **draped-clothing**
   silhouette; optionally emit a second body-only mask as a learning target.
   *Highest leverage — it's what the whole pipeline consumes.*
2. **Physics cloth sim with tightness variation (fix #2).** Replace shrinkwrap with
   the Blender cloth modifier (or GarmentCode/Warp): per-render sample garment
   size (tight→loose) and fabric stiffness. Two tightness regimes minimum, matching
   the real "облягаючий одяг vs вільний одяг" split.
3. **Correct height (fix #3).** Stop uniform-scaling the mesh. Either accept the
   emergent SMPL-X height and record it, or solve β to hit target height while
   preserving shape (SHAPY-style). Verify limb/girth ratios stay allometric.
4. **Realistic shape prior (fix #4).** Sample β from an ANSUR/CAESAR-fit
   distribution (or measurement-condition it) so synthetic girths match real
   populations, especially the BMI>30 tail (our worst eval bucket).
5. **Capture-realistic domain randomization (fix #5).** Add phone-tilt, camera
   height ≠ chest, distance/FOV jitter to the renderer's existing HDRI/lighting
   randomization — reproduce the exact nuisances production logs show.
6. **Consider BlenderProc (fix #6).** Optional infra hardening for scale.

---

## 5. The non-negotiable gate: domain-gap validation

Synthetic quality is only real if it transfers. **Every synthetic iteration must be
scored by training on synthetic and evaluating on held-out real data** — precisely
the two-track eval already built:
- Train a measurement head on synthetic → run `eval/bodym.py` against BodyM testB
  (in-the-wild, held out, never trained on — legal).
- Also score on the app's own GT corpus (`pointsx-eval`, the Supabase set).

If synthetic-trained accuracy on BodyM held-out doesn't approach the geometric
ceiling (D: ~4–5 cm) and then beat it, the synthetic isn't ready — iterate on the
fidelity gaps above before scaling to 10–50k bodies. **This is how BEDLAM proved
synthetic-only works, and it's how we avoid shipping a confidently-wrong model
trained on pretty-but-unreal renders.**

---

## 6. One-line answer

The generator has strong bones and is worth keeping — but as a *measurement*
training source it's currently fragile because **it never learns clothing**: masks
are nude and garments are shrink-wrapped, so it teaches the opposite of the real
loose-clothing error. Fix clothing first (drape + clothed masks), then body-shape
realism (drop the height-scaling hack, use an anthropometric β prior), validate
against BodyM held-out at every step, and generate our own assets to stay
license-clean for the commercial product.
