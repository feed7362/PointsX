# PointsX

Body measurement extraction from 2D photos using YOLO11 pose estimation + segmentation.

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

- `yolo11n-pose.pt` — YOLO11 nano pose (16 keypoints, finetuned on LV-MHP-v2)
- `yolo11n-seg.pt` — YOLO11 nano segmentation (COCO pre-trained)
- `CircumferenceRegressor` — MLP: Linear(28→64)→ReLU→BN→Dropout(0.2)→Linear(64→32)→ReLU→BN→Linear(32→6)

## Notes

- No test suite yet — no `tests/` directory
- `.gitignore` excludes: `.venv/`, `data/`, `models/*.pt`, `runs/`, `__pycache__/`
- `smpl-anthropometry` requires manual install from GitHub (not on PyPI)
- Blender required externally for synthetic pipeline rendering
