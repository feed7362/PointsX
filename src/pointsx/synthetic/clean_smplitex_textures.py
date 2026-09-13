"""Clean SMPLitex texture artifacts (face + hand regions) for SMPL-X rendering.

SMPLitex textures are AI-generated for the SMPL UV layout. When applied to
SMPL-X meshes the face region wraps incorrectly (4-eye / blurred-eye artefacts)
and Stable-Diffusion-generated hands often have mangled fingers. This script
masks both regions and inpaints them with a clean skin tone, sampled from a
known-good chest/arm area in the same texture (so the colour matches each
person's tone).

Run once over the assets folder; output overwrites in place by default
(``--backup`` saves originals to ``smplitex_backup/`` first).

Usage:
    python -m pointsx.synthetic.clean_smplitex_textures \
        --dir Q:/Projects/KHNU/PointsX/assets/textures/smplitex \
        --backup
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# UV regions in normalized [0,1] coordinates for the SMPL/SMPLitex UV layout.
# Origin is top-left (cv2 / image convention).
#
# Visualised by overlaying these boxes on the example SMPLitex texture:
#
#  ┌───────────────┬──────────────┐
#  │ FACE_BBOX     │ legs (skip)  │
#  │ (top-left)    │              │
#  ├───────────────┤              │
#  │ torso (front) │              │
#  │ + back        │ arms + HANDS │
#  │ — sample skin │ (keep arm,   │
#  │   from here   │  mask hand)  │
#  └───────────────┴──────────────┘
# ---------------------------------------------------------------------------

# Face region — covers head, hair top, and a margin around eyes/mouth.
# Measured against the SMPLitex example you shared (face at top-left, ~30% wide).
FACE_BBOX = (0.00, 0.00, 0.42, 0.32)   # (x0, y0, x1, y1)

# Hand region — bottom strip of the right column where SMPLitex puts hands.
# Includes a generous margin because SD hand artefacts spill outward.
HAND_BBOX = (0.55, 0.78, 1.00, 1.00)

# Where to sample a clean skin tone. Upper torso (chest area) is most reliable
# because clothing patterns there are usually solid colours rather than text/logos.
SKIN_SAMPLE_BBOX = (0.05, 0.45, 0.35, 0.62)

# Edge softness — pixel radius for the alpha-fade when pasting fill back in.
FADE_PX = 12


def _bbox_to_pixels(bbox: tuple[float, float, float, float], h: int, w: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(round(bbox[0] * w)))
    y0 = max(0, int(round(bbox[1] * h)))
    x1 = min(w, int(round(bbox[2] * w)))
    y1 = min(h, int(round(bbox[3] * h)))
    return x0, y0, x1, y1


def _sample_skin_color(img: np.ndarray) -> np.ndarray:
    """Robust skin-tone estimate from the chest area: median of HSV S>20, V>30 pixels."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _bbox_to_pixels(SKIN_SAMPLE_BBOX, h, w)
    patch = img[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([180, 140, 120], dtype=np.uint8)  # mid-skin BGR fallback
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    valid = (hsv[..., 1] > 20) & (hsv[..., 2] > 30) & (hsv[..., 2] < 240)
    if not np.any(valid):
        return patch.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    return np.median(patch[valid].reshape(-1, 3), axis=0).astype(np.uint8)


def _make_fade_mask(h: int, w: int, bbox_px: tuple[int, int, int, int]) -> np.ndarray:
    """Soft alpha mask: 1.0 inside bbox, fades to 0 over FADE_PX along the edges."""
    x0, y0, x1, y1 = bbox_px
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    if FADE_PX > 0:
        kernel = 2 * FADE_PX + 1
        mask = cv2.GaussianBlur(mask, (kernel, kernel), FADE_PX / 2)
    return mask


def _fill_region_with_skin(img: np.ndarray, bbox_px: tuple[int, int, int, int],
                           skin_bgr: np.ndarray) -> np.ndarray:
    """Soft-blend a solid skin patch into the bbox of img."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox_px
    if x1 <= x0 or y1 <= y0:
        return img

    fill = np.full_like(img, skin_bgr.reshape(1, 1, 3))

    # Add subtle low-frequency noise so the patch doesn't read as "flat plastic".
    noise = (np.random.standard_normal((h, w, 3)) * 4.0).astype(np.float32)
    fill = np.clip(fill.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    alpha = _make_fade_mask(h, w, bbox_px)[..., None]   # (H, W, 1)
    out = (img.astype(np.float32) * (1.0 - alpha) + fill.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_one(in_path: Path) -> bool:
    img = cv2.imread(str(in_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"  skip (cannot read): {in_path.name}", file=sys.stderr)
        return False

    h, w = img.shape[:2]
    skin = _sample_skin_color(img)

    img = _fill_region_with_skin(img, _bbox_to_pixels(FACE_BBOX, h, w), skin)
    img = _fill_region_with_skin(img, _bbox_to_pixels(HAND_BBOX, h, w), skin)

    cv2.imwrite(str(in_path), img)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Mask face + hand regions in SMPLitex textures")
    parser.add_argument("--dir", required=True, type=Path,
                        help="Folder containing the .png textures (e.g. assets/textures/smplitex)")
    parser.add_argument("--backup", action="store_true",
                        help="Copy originals to <dir>_backup/ before modifying.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N files (debug).")
    args = parser.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"Not a directory: {args.dir}")

    pngs = sorted(args.dir.glob("*.png"))
    if not pngs:
        sys.exit(f"No .png files in {args.dir}")

    if args.backup:
        backup_dir = args.dir.with_name(args.dir.name + "_backup")
        if backup_dir.exists():
            print(f"Backup already exists: {backup_dir} (skipping copy)")
        else:
            print(f"Backing up originals to {backup_dir}…")
            shutil.copytree(args.dir, backup_dir)

    if args.limit is not None:
        pngs = pngs[: args.limit]

    n_ok = n_fail = 0
    try:
        from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn
        with Progress(
            TextColumn("[bold blue]Cleaning textures"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("clean", total=len(pngs))
            for p in pngs:
                if clean_one(p):
                    n_ok += 1
                else:
                    n_fail += 1
                progress.update(task, advance=1)
    except ImportError:
        for i, p in enumerate(pngs, 1):
            if clean_one(p):
                n_ok += 1
            else:
                n_fail += 1
            if i % 25 == 0:
                print(f"  {i}/{len(pngs)}")

    print(f"\nDone. Cleaned {n_ok}/{len(pngs)} textures (failed: {n_fail}).")
    print(f"Tip: open one cleaned PNG to confirm face/hand regions are smooth skin.")
    print(f"     If face/hand bboxes look off, edit the constants in {Path(__file__).name}")
    print(f"     and re-run from the backup.")


if __name__ == "__main__":
    main()
