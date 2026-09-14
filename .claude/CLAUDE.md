# PointsX

Body measurement extraction from 2D photos using YOLO pose estimation + segmentation (FitMeasure AI).

**Input:** Front + side view photos + known height (cm)
**Output:** Body measurements (widths, lengths, circumferences) in cm

## Tech Stack

- Python 3.12+, PyTorch, Ultralytics (YOLO11), OpenCV, NumPy, SciPy
- Package manager: **uv** (`uv sync` to install)
- Build system: hatchling (PEP 517)
- Linter: Ruff (line-length: 120, target: py312)

## Project Structure

```
src/pointsx/
  cli.py              # CLI entry point (`pointsx` command)
  pipeline.py          # Top-level orchestrator (MeasurementPipeline)
  models.py            # YOLO model wrappers (pose + seg)
  schemas.py           # Dataclasses: Keypoints, SilhouetteMask, BodyMeasurements
  keypoints.py         # 16-point LV-MHP-v2 skeleton enum + geometric helpers
  calibration.py       # Pixel-to-cm scale from known height
  silhouette.py        # Width extraction from segmentation masks
  measurements.py      # Core measurement computation
  circumference.py     # Ellipse circumference (Ramanujan formula) or regression
  postprocess.py       # Validation, ratio checks, warnings
  train_pose.py        # YOLO pose finetuning script
  convert_pose.py      # LV-MHP-v2 .mat → YOLO format converter

  regression/          # MLP circumference regressor (28 features → 6 outputs)
    model.py           # CircumferenceRegressor architecture
    features.py        # 28-dim feature vector builder
    dataset.py         # PyTorch Dataset
    train.py           # Training with early stopping

  synthetic/           # Synthetic data generation (SMPL-X + Blender)
    pipeline.py        # Orchestrator
    body_generator.py  # SMPL-X body sampling
    landmarks.py       # 25 landmark definitions
    measurements_gt.py # Ground-truth measurements from meshes
    annotator.py       # 3D→2D projection + YOLO labels
    blender_render.py  # Headless Blender rendering

models/               # Pre-trained YOLO11n weights (pose + seg, ~6MB each)
data/LV-MHP-v2/       # Real dataset: 15k train + 5k val images
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
tests/                unit tests + geometry snapshot gate (procedural bodies, no real photos)
.github/workflows/    ci.yml (lint, test, vercel-smoke, docker, web → deploy), deploy.yml, keepalive.yml
```

- Push to `main` → `ci` → on green `deploy`: web tree committed on top of the `vercel` branch (Vercel builds it),
  Space tree committed on top of the HF Space git (`HF_TOKEN` secret) + smoke test. Nothing is force-pushed.
- `vercel` is a **generated deploy branch**, never a dev branch. Manual `git push origin <commit>:vercel` still deploys
  (hotfix path) but must also land on `main`, or the next deploy overlays it.
- Vercel function has no torch/cv2: keep heavy imports lazy on the path `api/index.py` imports (`vercel-smoke` enforces).
- `.vercelignore` patterns must be root-anchored (`/dataset`), CI fails if a deployable file would be dropped.
- Accuracy constants in `webui/envelope.py` are fitted against the ellipse; regressor only with `POINTSX_USE_REGRESSOR=1`.
- Input images are capped at 1280 px (`pointsx.pipeline.downscale_for_inference`) at every entry point.

## Commands

```bash
# Install
uv sync

# Run measurement pipeline
pointsx --front front.jpg --side side.jpg --height 175.0 --output table

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

1. **Pose estimation** — YOLO11n-pose → 16 keypoints per person (largest detected)
2. **Segmentation** — YOLO11n-seg → binary silhouette mask
3. **Calibration** — px_per_cm from head-to-ankle distance vs known height
4. **Width extraction** — horizontal mask extent at anatomical y-coordinates
5. **Circumference** — Ramanujan ellipse approximation (front + side widths) or regression
6. **Validation** — anthropometric ratio bounds, symmetry checks, warnings

## Code Conventions

- Snake_case functions/variables, PascalCase classes
- Type hints: modern syntax (`str | None`, not `Optional[str]`)
- Docstrings: NumPy-style (Args/Returns)
- Dataclasses for structured data
- Logging via `logging` module (DEBUG/INFO/WARNING)
- Private functions prefixed with `_`
- Constants in UPPER_CASE
- Lazy imports for optional deps (smplx, bpy)

## Key Data Structures

- `KP` — IntEnum for 16 keypoint indices (keypoints.py)
- `Keypoints` — dataclass: xy coords + confidence per keypoint
- `SilhouetteMask` — dataclass: binary mask array
- `BodyMeasurements` — dataclass: all measurements in cm + warnings list

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
- `.gitignore` excludes: `.venv/`, `data/`, `models/*.pt`, `runs/`, `__pycache__/`
- `smpl-anthropometry` requires manual install from GitHub (not on PyPI)
- Blender required externally for synthetic pipeline rendering
