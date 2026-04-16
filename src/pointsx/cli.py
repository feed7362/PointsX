"""Command-line interface for PointsX body measurement pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pointsx.pipeline import MeasurementPipeline


def main():
    parser = argparse.ArgumentParser(
        description="PointsX: Extract body measurements from front + side photos.",
    )
    parser.add_argument("--front", required=True, help="Path to front-view photo")
    parser.add_argument("--side", required=True, help="Path to side/profile-view photo")
    parser.add_argument("--height", required=True, type=float, help="Known height in cm")
    parser.add_argument("--pose-model", default="models/yolo11n-pose.pt", help="Path to pose model")
    parser.add_argument("--seg-model", default="models/yolo11n-seg.pt", help="Path to segmentation model")
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
        pose_model_path=args.pose_model,
        seg_model_path=args.seg_model,
        regression_model_path=args.regression_model,
        device=args.device,
    )

    result = pipeline(args.front, args.side, args.height)

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
