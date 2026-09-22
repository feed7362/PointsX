"""Score the production pipeline on rendered bodies with exact mesh ground truth.

The app corpus cannot see below ~3 cm: its ground truth is self-measured tape (sd 1.8-6 cm per
measurement) and two photos of the same body minutes apart already differ by ~2 cm. This bench has
neither problem — the ground truth is the mesh the render came from, the silhouette is exact, and
the camera is known — so a systematic error of a centimetre is visible.

It found the two defects fixed on 2026-09-22: calibration treating a keypoint span as stature
(every width +7.5-13.7 %) and neck-base height measured to the ankle instead of the floor.

usage:
    .venv/Scripts/python scripts/synthetic/bench_pipeline.py [--bodies runs/synthetic/bench]
                                                             [--renders runs/synthetic/bench_renders]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

# GT id -> BodyMeasurements attribute produced by the pipeline
PAIRS = {
    "chest_circumference": "chest_circumference_cm",
    "waist_circumference": "waist_circumference_cm",
    "hip_circumference": "hip_circumference_cm",
    "thigh_circumference": "thigh_circumference_cm",
    "neck_circumference": "neck_circumference_cm",
    "calf_circumference": "calf_circumference_cm",
    "leg_length_inner_seam": "leg_length_inner_cm",
}


def composite(png: Path, jpg: Path) -> None:
    """Renders are transparent; the segmenter expects a photo, so put a plain wall behind."""
    im = Image.open(png).convert("RGBA")
    bg = Image.new("RGBA", im.size, (205, 205, 200, 255))
    bg.alpha_composite(im)
    bg.convert("RGB").save(jpg, quality=95)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", type=Path, default=REPO / "runs" / "synthetic" / "bench")
    ap.add_argument("--renders", type=Path, default=REPO / "runs" / "synthetic" / "bench_renders")
    args = ap.parse_args()

    from pointsx.pipeline import MeasurementPipeline
    bodies = {b["body_id"]: b for b in json.loads((args.bodies / "bodies.json").read_text(encoding="utf-8"))}
    pipe = MeasurementPipeline(device="cpu")

    errors: dict[str, list[float]] = {k: [] for k in PAIRS}
    rows = []
    for bid, body in sorted(bodies.items()):
        png = args.renders / f"{bid}_front_rgb.png"
        if not png.is_file():
            continue
        for view in ("front", "side"):
            composite(args.renders / f"{bid}_{view}_rgb.png", args.renders / f"{bid}_{view}.jpg")
        gt = body["measured"]
        got = pipe(str(args.renders / f"{bid}_front.jpg"), str(args.renders / f"{bid}_side.jpg"),
                   height_cm=gt["height_cm"]).to_dict()
        row = {"body_id": bid, "sex": body["sex"], "height_cm": gt["height_cm"]}
        for g, k in PAIRS.items():
            if gt.get(g) and got.get(k):
                e = got[k] - gt[g]
                errors[g].append(e)
                row[g] = round(e, 1)
        rows.append(row)

    if not rows:
        print("no rendered bodies found — run blender_render.py first")
        return 1
    print(f"pipeline vs exact mesh GT on {len(rows)} rendered bodies (perfect silhouette, known camera)\n")
    print(f"  {'measurement':24}{'n':>4}{'MAE':>8}{'bias':>8}{'worst':>8}")
    for g, v in errors.items():
        if not v:
            continue
        a = np.abs(v)
        print(f"  {g:24}{len(v):4}{a.mean():8.2f}{np.mean(v):+8.2f}{a.max():8.2f}")
    print("\nper body (error in cm):")
    keys = [g for g in PAIRS if errors[g]]
    print(f"  {'body':9}{'sex':7}{'height':>7}" + "".join(f"{g.split('_')[0][:7]:>9}" for g in keys))
    for r in rows:
        print(f"  {r['body_id']:9}{r['sex']:7}{r['height_cm']:7.1f}" +
              "".join(f"{r.get(g, float('nan')):9.1f}" for g in keys))
    return 0


if __name__ == "__main__":
    sys.exit(main())
