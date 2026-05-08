"""Command-line interface for PointsX body measurement pipeline."""

from __future__ import annotations

import argparse
import json
import logging

from pointsx.pipeline import MeasurementPipeline


def main():
    parser = argparse.ArgumentParser(
        description="PointsX: Extract body measurements from front + side photos.",
    )
    parser.add_argument("--front", required=True, help="Path to front-view photo")
    parser.add_argument("--side", required=True, help="Path to side/profile-view photo")
    parser.add_argument("--height", required=True, type=float, help="Known height in cm")
    parser.add_argument(
        "--pose-backend",
        choices=("custom", "coco"),
        default="coco",
        help="Pose model: custom 16-point (LV-MHP) or COCO-17 (converted to 16)",
    )
    parser.add_argument(
        "--pose-model-custom",
        default="models/pose-cus.pt",
        help="Weights for native 16-keypoint pose (used when --pose-backend=custom)",
    )
    parser.add_argument(
        "--pose-model-coco",
        default="models/yolo26-pose.pt",
        help="Weights for COCO 17-keypoint pose (used when --pose-backend=coco)",
    )
    parser.add_argument("--seg-model", default="models/yolo12l-person-seg-extended.pt", help="Path to segmentation model")
    parser.add_argument("--regression-model", default=None, help="Path to circumference regression model")
    parser.add_argument("--device", default="auto", help="Inference device: auto, cpu, cuda")
    parser.add_argument(
        "--output", choices=["table", "json", "csv"], default="table",
        help="Output format (default: table)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    pipeline = MeasurementPipeline(
        pose_custom_path=args.pose_model_custom,
        pose_coco_path=args.pose_model_coco,
        seg_model_path=args.seg_model,
        regression_model_path=args.regression_model,
        device=args.device,
    )

    result = pipeline(args.front, args.side, args.height, pose_backend=args.pose_backend)

    if args.output == "json":
        data = result.to_dict()
        if result.warnings:
            data["warnings"] = result.warnings
        print(json.dumps(data, indent=2, ensure_ascii=False))

    elif args.output == "csv":
        data = result.to_dict()
        print(",".join(data.keys()))
        print(",".join(str(v) for v in data.values()))

    else:  # table
        data = result.to_dict()
        print(f"\n{'Measurement':<30} {'Value':>8}")
        print("-" * 40)
        for key, val in data.items():
            label = key.replace("_cm", "").replace("_", " ").title()
            print(f"{label:<30} {val:>7.1f} cm")

        if result.warnings:
            print(f"\n{'Warnings':}")
            for w in result.warnings:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
