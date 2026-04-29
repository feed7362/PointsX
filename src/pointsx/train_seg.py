"""Finetune YOLO11n-seg on the synthetic silhouette dataset.

Expects a dataset produced by:
  1. `blender_render.py --mode mask` (renders white-on-black body silhouettes)
  2. `masks_to_polygons.py --data <root>` (converts PNGs → YOLO polygon labels + dataset.yaml)

Usage:
    python -m pointsx.train_seg \
        --data   data/synthetic-seg/dataset.yaml \
        --epochs 30 \
        --imgsz  640
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Train YOLO11n-seg on synthetic body masks")
    parser.add_argument("--data", type=Path, required=True,
                        help="Path to dataset.yaml (produced by masks_to_polygons.py)")
    parser.add_argument("--model", type=Path,
                        default=project_root / "models" / "yolo11n-seg.pt",
                        help="Base YOLO-seg weights to finetune from")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", type=Path, default=project_root / "runs")
    parser.add_argument("--name", type=str, default="seg")
    parser.add_argument("--device", type=str, default=None,
                        help="'0' for GPU 0, 'cpu', or None for auto")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}\n"
            "Run `python -m pointsx.synthetic.masks_to_polygons --data <dir>` first."
        )
    if not args.model.exists():
        raise FileNotFoundError(
            f"Base model not found: {args.model}\n"
            "Download yolo11n-seg.pt from https://github.com/ultralytics/ultralytics"
        )

    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        workers=args.workers,
        device=args.device,
    )


if __name__ == "__main__":
    main()
