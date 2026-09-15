# PointsX

Body measurement extraction from 2D photos using YOLO pose estimation + segmentation (FitMeasure AI).

**Input:** Front + side view photos + known height (cm)
**Output:** Body measurements (widths, lengths, circumferences) in cm

## Tech Stack

- Python 3.12+, PyTorch, Ultralytics (YOLO), OpenCV, NumPy, SciPy; FastAPI web app
- Frontend: vanilla JS ES modules, no bundler (served statically by Vercel / uvicorn)
- Package manager: **uv** (`uv sync` to install)
- Build system: hatchling (PEP 517)
- Linter: Ruff (line-length: 120, target: py312)

## Project Structure

```
src/pointsx/
  cli.py              # CLI entry point (`pointsx` command)
  pipeline.py          # Top-level orchestrator (MeasurementPipeline), downscale_for_inference
  models.py            # YOLO model wrappers (pose + seg)
  schemas.py           # Dataclasses: Keypoints, SilhouetteMask, BodyMeasurements
  keypoints.py         # 16-point LV-MHP-v2 skeleton enum + geometric helpers
  calibration.py       # Pixel-to-cm scale from known height
  silhouette.py        # Width extraction from segmentation masks
  measurements.py      # Core measurement computation
  circumference.py     # Ellipse circumference (Ramanujan formula) or regression
  postprocess.py       # Validation, ratio checks, warnings
  eval.py              # `pointsx-eval`: accuracy on labelled subjects, --fit-offsets
  train_pose.py        # YOLO pose finetuning script
  convert_pose.py      # LV-MHP-v2 .mat → YOLO format converter
  regression/          # MLP circumference regressor (28 features → 6 outputs)
  synthetic/           # Synthetic data generation (SMPL-X + Blender)

src/webui/            # web app — layered like Video_streaming's backend (docs/app-decomposition-plan.md)
  app.py               # `app = create_app()` — the entrypoint both deploys use (webui.app:app)
  config.py            # Settings + the full list of environment variables
  errors.py            # AppError + Ukrainian error texts (part of the API contract)
  bootstrap/           # factory, lifespan (model loading), routers, middleware, exception handlers
  api/                 # thin routers: pages, health (+keepalive), measure (+mock, proxy), tts, dependencies
  services/            # measurement flow, uploads, dataset_capture, proxy, mock — no FastAPI imports
  schemas/             # MeasurementEnvelope v2, TtsRequest
  envelope/            # catalog, corrections (fitted tables + provenance), derive, build
  visualize/           # primitives, mask_spans, measure_lines (one function per overlay)
  infrastructure/      # storage/ {s3, supabase, local, crypto, common}, inference, weights, tts_edge
  static/js/
    api/               # client.js (only place that calls fetch), measure.js, tts.js
    i18n/              # uk.js, en.js (same keys), index.js (t, setLang, translatePage, …)
    capture/           # session.js facade + poseModel, imageOps, uploads, autoCapture; ui, speech, camera, …
    sizing/            # catalog, text, measurementOrder, render, sizeTabs, view, measureFlow
    dataset/           # dataset collection page (owned by MaksShu; not yet split)
    sizeEngine.js, patternEngine.js, guideGeometry.js

models/               # YOLO weights (see Models)
notebooks/            # Jupyter: exploration, training, synthetic generation
runs/                 # Training outputs (auto-generated)
```

## Monorepo layout & deploy (since 2026-09-13)

One repo (`feed7362/PointsX`, default branch `main`) replaces PointsX + Pointx-backend + Pointx-frontend
(the latter two imported with history via `git subtree`, then archived). Plan/log: `docs/monorepo-cicd-plan.md`,
merge decisions: `docs/migration-phase0-report.md`.

```
src/pointsx/          measurement core (shared by CLI, webui, eval)
src/webui/            FastAPI app + envelope + static UI — one app, two deploy modes
api/index.py          Vercel entry (POINTSX_VERCEL=1 → proxy mode: /api/measure forwarded to the Space)
apps/backend/         HF Space packaging: Dockerfile, Space README card, .dockerignore, .gitattributes
scripts/ci/           stage_hf_space.sh, stage_vercel.sh, vercel_smoke.py, space_smoke.py, check_vercelignore.py
tests/                unit + HTTP contract + geometry snapshot gate (procedural bodies, no real photos)
.github/workflows/    ci.yml (lint, test, vercel-smoke, docker, web → deploy), deploy.yml, keepalive.yml
```

- Push to `main` → `ci` → on green `deploy`: web tree committed on top of the `vercel` branch (Vercel builds it),
  Space tree committed on top of the HF Space git (`HF_TOKEN` secret) + smoke test. Nothing is force-pushed.
- `vercel` is a **generated deploy branch**, never a dev branch. Manual `git push origin <commit>:vercel` still deploys
  (hotfix path) but must also land on `main`, or the next deploy overlays it.
- Vercel function has no torch/cv2: keep heavy imports lazy on the path `api/index.py` imports (`vercel-smoke` enforces).
- `.vercelignore` patterns must be root-anchored (`/dataset`), CI fails if a deployable file would be dropped.
- Accuracy constants in `webui/envelope/corrections.py` are fitted against the ellipse; regressor only with
  `POINTSX_USE_REGRESSOR=1`. Refit with `pointsx-eval --fit-offsets` and keep the provenance comments with the numbers.
- Input images are capped at 1280 px (`pointsx.pipeline.downscale_for_inference`) at every entry point.

## Web app layering rules

### Backend (`src/webui`)

1. `app.py` only calls `create_app()`. Assembly lives in `bootstrap/`; tests use `create_app(use_lifespan=False)`.
2. `api/*` routers are thin: parse the request, take `Depends(...)`, call one service, return a schema. No logic.
3. `services/*` never import FastAPI. They raise `AppError(status, detail_uk)`; `bootstrap/exceptions.py` turns it into
   `{"detail": ...}`. Routes read uploads and pass bytes (`PhotoUpload`) and a background scheduler in.
4. The "models not loaded" 503 is checked **inside** `services/measurement.py`, never in a raising dependency —
   form validation (422) must win. `tests/test_api_contract.py` pins status codes, shapes and the Ukrainian texts.
5. External systems live in `infrastructure/` (storage backends, YOLO pipeline, weight pre-fetch, edge-tts).
   `cv2`, `torch`, `ultralytics`, `boto3`, `nacl` are imported inside functions only.
6. Environment variables are read in `config.py` (`Settings.from_env`, cached `get_settings()`). Per-request reads are
   the documented exceptions (`CRON_SECRET`, TTS, storage `*_S3_*`, `HF_*`).
7. Domain code stays in its package: `envelope/` (catalog → corrections → derive → build), `visualize/`
   (`measure_lines._SECTIONS` order decides label placement — keep it stable).
8. Each package's `__init__.py` re-exports its public API; import from the package, not from sibling internals.

### Frontend (`src/webui/static/js`)

1. Network calls go through `api/client.js` (`apiFetch`, `throwIfNotOk`, `ApiError`) and the feature wrappers
   `api/measure.js`, `api/tts.js`. Exceptions: `guideGeometry.js` (optional static data) and `dataset/*`.
2. UI strings live in `i18n/uk.js` + `i18n/en.js` with identical keys (uk is the fallback); import `t` from
   `i18n/index.js`. `capture/i18n.js` is a re-export kept only for the dataset page.
3. `capture/session.js` and `capture/tailoring.js` are facades: `app.js` and `dataset/app.js` use them as
   `import * as session / tailoring`, so their export sets must not change. New code imports the focused modules.
4. Import cycles between modules are allowed only for function-level use, never for values read at module top level.
5. When the module graph changes, bump the `?v=` cache-buster on the entry `<script>` in `index.html` / `dataset.html`.
6. JS tests: `node --test 'src/webui/static/js/__tests__/*.test.js'` (CI `web` job).

### Accuracy changes: re-measure every time

Any change that can move a measurement (GT corpus or gate, pose/seg models or versions, calibration, rows,
widths, ellipse, envelope corrections/derivations) is followed by
`.venv/Scripts/python scripts/eval_track.py run --label "<what changed>"` before commit.

- It runs both benchmarks (app GT on real photos; BodyM perfect silhouettes testA + testB), appends to
  `runs/eval/ledger.jsonl` and compares with the previous entry. Exit 1 = regression.
- Read the detection map it prints: BodyM sees only the ellipse math for chest/waist/hip/thigh. "BodyM
  unchanged" after a pose, row or envelope change means blind, not safe.
- If the scored set changed (GT gate, new subjects), full-set MAE is not comparable; use the intersection
  deltas it prints.
- Correction constants: report leave-one-out, never in-sample MAE (see `envelope/corrections.py`).
- Ledger and reports stay local (per-subject errors on real people). Only aggregates go into docs/commits.

### Refactor checklist (behaviour must not change)

- `pytest` (contract, services, storage, config, bootstrap, geometry snapshot) green.
- `scripts/ci/vercel_smoke.py` in a venv built only from `requirements.txt`.
- `pointsx-eval --subjects supabase-dump/subjects.csv --pose-backend coco` report byte-identical to before (local only).
- UI changes: compare old vs new build side by side in a browser (same scenario, hashes of rendered DOM/text).
- Real photos and anything derived from them stay local: never commit, upload or paste them.

## Commands

```bash
# Install
uv sync

# Run measurement pipeline
pointsx --front front.jpg --side side.jpg --height 175.0 --output table

# Web app locally (static UI + API)
pointsx-web

# Tests
.venv/Scripts/python -m pytest -q
node --test 'src/webui/static/js/__tests__/*.test.js'

# Convert dataset annotations
python -m pointsx.convert_pose

# Train YOLO pose model
python -m pointsx.train_pose

# Train circumference regressor
python -m pointsx.regression.train --data features.npz --output models/regressor.pt

# Generate synthetic data
python -m pointsx.synthetic.pipeline --n-bodies 500 --blender-exe /path/to/blender --out-dir data/synthetic-pose
```

## Pipeline Flow

1. **Pose estimation** — YOLO pose (COCO-17 mapped to 16 points, or custom 16-point) → keypoints (largest person)
2. **Segmentation** — YOLO person segmentation → binary silhouette mask
3. **Calibration** — px_per_cm from head-to-ankle distance vs known height
4. **Width extraction** — horizontal mask extent at anatomical y-coordinates
5. **Circumference** — Ramanujan ellipse approximation (front + side widths) or regression
6. **Validation** — anthropometric ratio bounds, symmetry checks, warnings
7. **Envelope** (web) — 18 canonical measurements, bias corrections, plausibility flags, optional debug overlays

## Code Conventions

- Snake_case functions/variables, PascalCase classes
- Type hints: modern syntax (`str | None`, not `Optional[str]`)
- Docstrings: NumPy-style (Args/Returns)
- Dataclasses for structured data
- Logging via `logging` module (DEBUG/INFO/WARNING)
- Private functions prefixed with `_`
- Constants in UPPER_CASE
- Lazy imports for optional deps (smplx, bpy) and for anything heavy on the Vercel import path

## Key Data Structures

- `KP` — IntEnum for 16 keypoint indices (keypoints.py)
- `Keypoints` — dataclass: xy coords + confidence per keypoint
- `SilhouetteMask` — dataclass: binary mask array
- `BodyMeasurements` — dataclass: all measurements in cm + warnings list
- `MeasurementEnvelope` (webui/schemas) — API response consumed by the frontend sizing + pattern engines

## Models

- `yolo26-pose.pt` — COCO-17 pose mapped to 16 points (default `coco` backend in prod)
- `pose-cus.pt` — optional custom 16-point pose (LV-MHP-v2 finetune), `custom` backend
- `yolo12l-person-seg-extended.pt` — person segmentation
- Missing configured weights raise at load (no silent fallback to other model families)
- `CircumferenceRegressor` — MLP: Linear(28→64)→ReLU→BN→Dropout(0.2)→Linear(64→32)→ReLU→BN→Linear(32→6)

## Notes

- `tests/test_snapshot.py` is the CI geometry gate: procedural bodies in `tests/fixtures/synthetic_bodies.py`
  vs `tests/fixtures/expected/synthetic_snapshot.json`. Intentional geometry change →
  `python tests/fixtures/synthetic_bodies.py --write-expected` and commit the JSON with the code.
  It does not cover pose/seg models; for those, also compare a before/after run over `supabase-dump/subjects.csv`
  locally (real photos — never commit or upload anything derived from them)
- UA sizes on the `ua_*` grids are Ukrainian sizes (UA = EU + 6); see `src/webui/docs/tailoring-config.md`
- `.gitignore` excludes: `.venv/`, `data/`, `models/*.pt`, `runs/`, `__pycache__/`
- `smpl-anthropometry` requires manual install from GitHub (not on PyPI)
- Blender required externally for synthetic pipeline rendering
