# Synthetic dataset: audit + plan (2026-09-20)

> Goal: a **license-clean training dataset** built by us (Blender + GPU), measured against
> **licensed datasets used as eval-only baselines** (never trained on, never shipped).
> Companion docs: `synthetic-data-review.md` (2026-07-18 code review), `accuracy-approaches-survey.md`
> (§1a: what was measured 09-15/16), `ml-blueprint.md` §8-9.

## 1. Audit: what exists today

| Item | Where | Size | State | Verdict |
|---|---|---|---|---|
| Generator scaffold (4 phases: sample bodies → SMPL-X forward → GT from mesh → Blender render + YOLO labels) | `src/pointsx/synthetic/` (2 635 lines) | — | Architecture sound (review §1). Body = **SMPL-X (non-commercial)**. Clothing shrink-wrapped, mask mode strips clothing, A-pose only. Cannot run: no SMPL-X files, Blender not on PATH, `assets/clothing/` empty. | **Keep scaffold, swap body model, rebuild clothing** |
| Vendored SMPL-Anthropometry (MIT) | `synthetic/anthropometry/` | — | Correct GT method since 07-18 (plane slice + face segmentation). Landmarks are SMPL-X vertex ids. | **Keep**; remap landmarks to the new topology |
| Synthetic pose dataset v1 | `data/synthetic-pose/` + `.zip` | 5 000 bodies, 35k files, **6.4 GB + 2.5 GB** | Generated **before** the GT fix: `body_00001` = "thin" male 159 cm with waist 110 / hips 133 cm. GT is wrong for every body. SMPL-X-derived. | **Delete** (both) |
| Synthetic seg dataset v1 | `data/synthetic-seg/` | 8k images, 172 MB | Same generation run, same taint; masks strip clothing (review fix #1). | **Delete** |
| SMPLitex skin textures | `assets/textures/smplitex/` | 106 MB | Derived from SMPL; license to be checked before any commercial use. UV layout is SMPL-X's — useless on another topology anyway. | **Delete when the body model changes** |
| HDRIs (5) | `assets/hdri/` | 31 MB | Poly Haven, CC0. | Keep |
| Clothing assets | `assets/clothing/` | 0 | Empty. | Rebuild (see §4) |
| LV-MHP-v2 (+ pose conversion) | `data/LV-MHP-v2*/` | 9.5 GB | Research dataset; used only for the optional `pose-cus.pt` fine-tune. Not in the production path (`coco` backend). | Keep offline; not part of the commercial track |
| BodyM cache | `eval/.cache/` | 14 MB | 87 + 400 + 300 masks with scan GT. **CC BY-NC, eval-only** (guard in code). | Keep — baseline #1 |
| App GT corpus | `supabase-dump/` | 27 submissions → 16 scored | Own data, self-measured tape (σ 3-4 cm), 12 F / 4 M. | Keep — product final; grow |
| `data/eval/photos` | 14 photos | — | Old ad-hoc eval photos, no GT file next to them. | Delete or fold into supabase-dump |
| Hardware | RTX 5070 Ti 12 GB; Blender installed (not on PATH) | | Enough for cloth sim + rendering ~1-2k bodies/day. | — |

Net: **~9.2 GB of tainted, wrong-GT synthetic data to delete**; a good scaffold and a correct
GT library to keep; a body model and a clothing pipeline to replace.

## 2. Eval baselines (licensed, eval-only) — what our dataset is aimed at

Rule (same as BodyM): these are **held-out tests**. No model, constant or asset trained/fit on
them ships. Guard banner in each runner.

| Baseline | What it gives | License | Sees which stages | Status |
|---|---|---|---|---|
| **BodyM** testA/testB (Amazon) | 487 subjects, front+side **silhouettes**, scan GT (chest/waist/hip/thigh…), height, weight | CC BY-NC | width→girth math only (perfect masks) | built (`eval/bodym.py`), in the ledger |
| **SHAPY / HBW** (MPI) | 35 subjects, **RGB photos in the wild** in normal clothes, 3D-scan GT: height, chest, waist, hips | NC (registration) | **whole pipeline**: pose, seg, calibration, rows, ellipse | **to build** — the missing end-to-end baseline with mm-grade GT |
| **App GT** (ours) | 16 subjects, real phone photos, tape GT | ours | whole pipeline, product conditions | built (`pointsx-eval`) |
| ANSUR II | tables only (no photos) | public domain | priors, plausibility, shape sampling for synthetic | in use (`scripts/ansur/`) |
| CAESAR | 4.4k scans + measurements | paid | — | not needed now |

Why HBW is the priority: it is the only licensed set with **photos + accurate measurements**, so
it can score a body-under-clothing segmenter or a new width extractor end to end — BodyM cannot
(no RGB), app GT cannot below ~3 cm (tape noise). n=35 is small but the GT σ is millimetres.

Baseline numbers to beat, current production (`006c909`):
- BodyM testB A_raw 9.45 / B 6.26 / D 4.61 cm (ceiling benchmark)
- App GT displayed 3.40 cm (in-sample), 3.69 held-out on corrected targets
- HBW: to be measured first (step 0 below) — that number becomes the target.

## 3. What the synthetic dataset is for (and not)

It trains the components that need pixel-level supervision we cannot get from real data:
1. **Body-under-clothing segmentation** (survey C2): clothed render → nude-body mask. The only
   way to attack loose clothing without 3D at runtime; this is where the width over-read comes from.
2. Optionally later: a width→girth head trained on synthetic widths (only after 1, and only with a
   domain shift fitted on real GT).

It does **not** replace: ANSUR priors/constants (real people beat renders), the measured cohort
(the product number still needs real tape/scan GT), or HBW/BodyM (validation must be real).

## 4. Dataset design (license-clean)

| Layer | Choice | License | Note |
|---|---|---|---|
| Body model | **MPFB2 / MakeHuman** inside Blender (primary) or **Anny** (Naver, Apache 2.0) | MPFB2 code GPL, **outputs CC0**; Anny Apache 2.0 | MPFB2 wins on assets: the MakeHuman library has **CC0 clothing** and skins; phenotype sliders (height, weight, proportions). Anny if we need smplx-compatible topology for the vendored anthropometry. Decide in step 2 by a 20-body smoke test of both. |
| Shape sampling | ANSUR II per-sex distributions (height, weight, chest/waist/hip circ.) → solve model params to hit them | public domain | replaces random β; covers the BMI>30 tail |
| Clothing | CC0 garments (MakeHuman assets, Poly Haven, self-made) + **Blender cloth modifier**, two tightness regimes (tight / loose) + fabric stiffness jitter | CC0 / ours | replaces shrinkwrap; tightness is the axis the product cares about |
| Skin/hair | MakeHuman CC0 skins; hair as mesh or none | CC0 | SMPLitex dropped |
| Camera | phone-like: 26-28 mm eq., height 0.9-1.4 m, distance 1.8-3 m, tilt ±8°, distortion | — | matches capture guidance (`A4`) |
| Environment | 5 HDRIs now → 20+ Poly Haven CC0, indoor bias | CC0 | |
| Outputs per body | front + side RGB; **clothed mask**; **nude-body mask**; 16 keypoints (COCO-mapped); GT measurements from the nude mesh with our definitions; height, weight, sex, tightness label | | both masks is the key change vs v1 |
| GT definitions | vendored SMPL-Anthropometry definitions, remapped to the new topology; validated against ANSUR ranges (monotonic with BMI, per-sex means within 2 cm of ANSUR) | MIT | same definitions as inference |
| Scale | smoke 20 → pilot 500 → 5k; scale further only after the gate in §5 passes | | RTX 5070 Ti: ~1 min/body with cloth sim → 500/night |
| Provenance | `manifest.json` per body: generator commit, asset ids + licenses, seeds; a LICENSES.md listing every asset source | | required for the commercial claim |

## 5. Gates (every iteration, in this order)

1. **GT sanity** (no render needed): per-sex means/σ of chest/waist/hip vs ANSUR within 2 cm / 20 %;
   `gt_sanity.py` ratio bounds pass for ≥ 99 % of bodies.
2. **Silhouette realism**: run the *production* pipeline on the renders' clothed masks; the raw
   over-read (silhouette girth ÷ GT girth per site) must match what we see on real photos
   (app corpus: waist ≈ 1.20-1.25, hip ≈ 1.07, thigh ≈ 1.12, chest ≈ 1.03). If synthetic over-read
   is ≈ 1.0, clothing is still shrink-wrapped and the data teaches the wrong thing.
3. **Transfer**: train the segmenter on synthetic only → measure end to end on **HBW** (held out,
   NC) and app GT via `eval_track.py`. Accept only if displayed MAE improves on both, or HBW
   improves and app GT does not regress (app GT is too noisy to see < 0.3 cm).
4. Only then scale from pilot to 5k+.

## 6. Plan

| Step | What | Output | Effort |
|---|---|---|---|
| 0 | **HBW baseline**: register, download (eval-only), `eval/hbw.py` running the real pipeline, add to `eval_track.py` as a third benchmark | HBW MAE of production; `ledger` gets a `hbw` column | 2-3 days |
| 1 | **Cleanup**: delete v1 synthetic data (9.2 GB), `data/eval/photos`, SMPLitex; keep scaffold + anthropometry; write `assets/LICENSES.md` | disk, clean legal state | 1 h |
| 2 | ~~Body model smoke test~~ **DONE 2026-09-22: Anny** (see §8) | `scripts/synthetic/` | — |
| 3 | **Clothing + masks**: cloth sim, tight/loose, clothed + nude masks; phone camera model; gate 2 on 50 bodies | pipeline v2 | 1-2 weeks |
| 4 | **Pilot 500** + train the body-under-clothing segmenter (U-Net/light seg, CPU-fast at runtime); gate 3 | first transfer number on HBW + app GT | 1-2 weeks |
| 5 | Scale 5k, iterate on the gate-2/3 gaps; ship the segmenter only if gate 3 passes | | ongoing |
| ∥ | **Measured cohort** (20-30 people, ISO 8559-1, tight clothing, weight) — still the product final | | organising |

Not before step 0: without HBW there is no end-to-end baseline with GT good enough to see whether
a synthetic-trained model helps.

## 7. Risks

- **Domain gap** stays the main risk (review §5): renders that look fine can still teach the wrong
  silhouette. Gate 2 is designed to catch that before training.
- **Licenses**: verify every asset (MakeHuman CC0 packs are per-asset; some community assets are
  CC BY). SMPL-X, SMPLitex, CLOTH3D/4D, SynBody, BEDLAM, AGORA, BodyM, HBW: reference/eval only.
- **Small HBW** (35): report per-subject and bootstrap CI, not just MAE.
- **Blender time**: cloth sim per body dominates; DrapeNet/GarmentCode only if sim is too slow at 5k.

## 8. Step 2 result (2026-09-22): the body model is Anny

`pip install anny` (Apache 2.0, MakeHuman-derived assets CC0, differentiable PyTorch, GPU via Warp).
MPFB2 was not tested: Anny gives the same MakeHuman parameter space through a scriptable API instead
of a Blender add-on, so there was nothing left for the comparison to decide.

**What it can do** (measured, `scripts/synthetic/smoke_anny.py`, female at ANSUR-mean stature):

| control | range |
|---|---|
| `height` phenotype | stature 118.7 -> 220.7 cm, linear |
| `weight` phenotype (with `extrapolate_phenotypes=True`) | waist 65.8 -> 90.2 cm at fixed stature |
| `measure-waist-circ-incr` | waist 64.2 -> 84.2 cm |
| `measure-hips-circ-incr` | hip 78.2 -> 105.3 cm |
| `measure-bust-circ-incr` | chest 72.2 -> 104.9 cm |
| `measure-thigh-circ-incr` | thigh 41.2 -> 61.0 cm |
| `measure-neck-circ-incr` | neck 33.9 -> 43.5 cm |

`local_changes="all"` exposes **256 modifiers, 20 of them `measure-*`** — bust, underbust, waist,
hips, thigh, calf, knee, ankle, wrist and neck circumference plus arm/leg lengths and shoulder
distance. They map almost one-to-one onto our canonical ids, so a body can be **built to hit a
measurement vector sampled from ANSUR II** instead of sampled at random and hoped for. That is what
the old generator got wrong (random betas -> waist mean 111.6 cm).

**Measuring the mesh** — `scripts/synthetic/measure_mesh.py`, topology-agnostic (the vendored
SMPL-Anthropometry is tied to SMPL-X vertex ids): slice horizontally at the ANSUR landmark height for
the site, keep the cross-section that straddles the body axis (torso) or lies to one side (limb),
take the convex-hull perimeter — what a tape does. Bugs found and fixed while validating:
- `age` below 0.5 is a child in MakeHuman semantics: the first run produced a 127 cm "male".
- the neck ring lost to ear/hair fragments near the axis (necks of 2-7 cm) -> minimum perimeter of
  15 cm per component, and the neck is the narrowest slice in a band, as in `pointsx.silhouette`.
After both fixes 9/10 bodies pass `pointsx.gt_sanity` (the tenth is a neck ratio of 0.180 against a
bound of 0.18). Cost: 13 ms to build a body, 0.6 s to measure it -> ~1 h for 5 000 bodies.

**Still open before rendering (step 3):** solve phenotype + `measure-*` values per body to hit an
ANSUR target vector (bisection; the mappings above are monotone), then gate 1a is satisfied by
construction rather than by luck.

## 9. Step 3 clothing (2026-09-22): garments from the body, draped, and gate 2 passes

No garment assets were bought or downloaded — each one is the body's own surface pushed outward by
an ease allowance (`scripts/synthetic/make_garment.py`), so the licence question never arises and
the garment fits the body it was made for. Ease is the axis the product cares about: how much room
the garment has over the body.

The shell alone is the shrink-wrap the 2026-07 review rejected, so `blender_render.py` now drapes it
with Blender's cloth solver against the body as a collider, and PINS the top edge (4 % of the
garment's height). Without the pin gravity simply pulled the clothes off: the first run rendered a
chest over-read of 1.000x (no garment there at all) and 3.30x at the hip (trousers heaped at the
ankles). Fabric stiffness jitters 5-25 per render.

**Gate 2 — does synthetic clothing over-read like real clothing?** Front-view silhouette width,
clothed / body, against the app corpus's suit-vs-own-clothes pairs:

| site | synthetic ease 2 cm | synthetic ease 6 cm | real, own clothes |
|---|---|---|---|
| chest | 1.071x | 1.213x | 1.03x |
| waist | **1.204x** | 1.401x | **1.20-1.25x** |
| hip | 1.119x | 1.393x | 1.07x |

Ease ~2 cm reproduces the real profile, with the waist landing inside the measured band; 6 cm is a
genuinely loose regime for the other end of the distribution. This is the gate the deleted dataset
never passed — its masks stripped the clothing, teaching that the silhouette IS the body.

Remaining before the pilot: sleeves (the top is sleeveless, so the arm/torso overlap that T6 proved
geometry cannot resolve is not yet represented), per-site ease rather than one uniform value (real
clothing is looser at the waist than at the hip), and a garment-vs-body collision check on a wider
range of body shapes.
