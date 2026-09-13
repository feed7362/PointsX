"""Procedural front/side silhouettes + keypoints for the CI geometry snapshot.

No photos and nothing derived from real people: every body is drawn from a
handful of centimetre parameters, so the fixture is safe in a public
repository. It exercises everything below YOLO — calibration, width extraction,
circumferences, validation, envelope corrections — but not the pose or
segmentation models (the post-deploy smoke test covers those).

Regenerate the expected output after an intentional geometry change:

    python tests/fixtures/synthetic_bodies.py --write-expected
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pointsx.calibration import calibrate  # noqa: E402
from pointsx.circumference import estimate_circumferences  # noqa: E402
from pointsx.keypoints import KP, NUM_KEYPOINTS  # noqa: E402
from pointsx.measurements import extract_measurements  # noqa: E402
from pointsx.postprocess import validate_measurements  # noqa: E402
from pointsx.schemas import Keypoints, SilhouetteMask  # noqa: E402
from pointsx.silhouette import extract_all_widths  # noqa: E402

EXPECTED_PATH = Path(__file__).resolve().parent / "expected" / "synthetic_snapshot.json"

IMG_H, IMG_W = 1200, 800
TOP_Y, BOTTOM_Y = 90, 1110  # head top / ankle line in pixels, identical for every body
CX = IMG_W // 2


@dataclass(frozen=True)
class BodySpec:
    """Body proportions in centimetres (widths = front view, depths = side view)."""

    body_id: str
    height_cm: float
    sex: str
    shoulder_breadth: float
    chest_width: float
    waist_width: float
    hip_width: float
    chest_depth: float
    waist_depth: float
    hip_depth: float
    thigh_width: float  # one leg
    neck_width: float
    neck_depth: float
    head_width: float
    head_depth: float


BODIES = (
    BodySpec("f_slim_160", 160.0, "female", 35, 27, 24, 33, 21, 17, 22, 15, 10.0, 10.5, 14.5, 18.5),
    BodySpec("f_avg_168", 168.0, "female", 37, 29, 27, 36, 23, 19, 24, 17, 10.5, 11.0, 15.0, 19.0),
    BodySpec("f_curvy_172", 172.0, "female", 38, 32, 30, 40, 26, 22, 27, 19, 11.0, 11.5, 15.0, 19.0),
    BodySpec("m_slim_175", 175.0, "male", 41, 31, 27, 33, 22, 19, 22, 15, 11.5, 12.0, 15.5, 19.5),
    BodySpec("m_avg_182", 182.0, "male", 44, 34, 31, 35, 25, 23, 24, 17, 12.5, 13.0, 16.0, 20.0),
    BodySpec("m_broad_190", 190.0, "male", 48, 38, 36, 38, 28, 27, 27, 19, 13.5, 14.0, 16.5, 20.5),
)


def _y(frac: float) -> float:
    """Pixel row at a fraction of the head-top → ankle span."""
    return TOP_Y + frac * (BOTTOM_Y - TOP_Y)


def _mask(polys: list[np.ndarray], head_axes: tuple[float, float], view: str) -> SilhouetteMask:
    canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    for poly in polys:
        cv2.fillPoly(canvas, [np.round(poly).astype(np.int32)], 255)
    head_center = (CX, int(round(_y(0.06))))
    cv2.ellipse(canvas, head_center, tuple(int(round(a)) for a in head_axes), 0, 0, 360, 255, -1)
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2) if contours else None
    return SilhouetteMask(mask=canvas > 0, contour=contour, view=view)


def _symmetric(rows: list[tuple[float, float]]) -> np.ndarray:
    """Polygon symmetric about CX from (y_frac, half_width_px) rows, top to bottom."""
    right = [(CX + w, _y(f)) for f, w in rows]
    left = [(CX - w, _y(f)) for f, w in reversed(rows)]
    return np.array(right + left, dtype=np.float64)


def _keypoints(xs: dict[KP, float], ys: dict[KP, float], conf: np.ndarray, view: str,
               nose_x: float) -> Keypoints:
    pts = np.zeros((NUM_KEYPOINTS, 2), dtype=np.float64)
    for k, f in ys.items():
        pts[k] = (xs.get(k, CX), _y(f))
    return Keypoints(points=pts, confidence=conf, view=view,
                     nose_xy=np.array([nose_x, _y(0.06)], dtype=np.float64), nose_conf=0.9)


KP_Y = {
    KP.HEAD_TOP: 0.0, KP.UPPER_NECK: 0.115, KP.THORAX: 0.170,
    KP.RIGHT_SHOULDER: 0.180, KP.LEFT_SHOULDER: 0.180,
    KP.RIGHT_ELBOW: 0.340, KP.LEFT_ELBOW: 0.340, KP.RIGHT_WRIST: 0.495, KP.LEFT_WRIST: 0.495,
    KP.PELVIS: 0.530, KP.RIGHT_HIP: 0.530, KP.LEFT_HIP: 0.530,
    KP.RIGHT_KNEE: 0.740, KP.LEFT_KNEE: 0.740, KP.RIGHT_ANKLE: 1.000, KP.LEFT_ANKLE: 1.000,
}


def front_view(b: BodySpec) -> tuple[Keypoints, SilhouetteMask]:
    ppc = (BOTTOM_Y - TOP_Y) / b.height_cm
    h = lambda cm: cm * ppc / 2.0  # noqa: E731 — half-width in px
    px = lambda cm: cm * ppc  # noqa: E731

    torso = _symmetric([
        (0.130, h(b.neck_width)), (0.175, h(b.shoulder_breadth)), (0.260, h(b.chest_width)),
        (0.400, h(b.waist_width)), (0.520, h(b.hip_width)), (0.560, h(b.hip_width)),
    ])
    neck = _symmetric([(0.090, h(b.neck_width)), (0.135, h(b.neck_width))])
    gap = px(1.0)
    legs, arms = [], []
    for s in (1, -1):
        legs.append(np.array([
            (CX + s * gap, _y(0.560)), (CX + s * h(b.hip_width), _y(0.560)),
            (CX + s * (gap + px(b.thigh_width)), _y(0.620)),
            (CX + s * (gap + px(0.75 * b.thigh_width)), _y(0.740)),
            (CX + s * (gap + px(0.45 * b.thigh_width)), _y(1.000)),
            (CX + s * (gap + px(0.05 * b.thigh_width)), _y(1.000)),
            (CX + s * (gap + px(0.10 * b.thigh_width)), _y(0.740)),
            (CX + s * gap, _y(0.620)),
        ]))
        sh = h(b.shoulder_breadth)
        arms.append(np.array([
            (CX + s * (sh - px(3)), _y(0.175)), (CX + s * (sh + px(3)), _y(0.175)),
            (CX + s * (sh + px(20)), _y(0.500)), (CX + s * (sh + px(14)), _y(0.505)),
        ]))
    mask = _mask([torso, neck, *legs, *arms], (h(b.head_width), _y(0.06) - TOP_Y), "front")

    sh = h(b.shoulder_breadth)
    xs = {
        KP.RIGHT_SHOULDER: CX - sh + px(2), KP.LEFT_SHOULDER: CX + sh - px(2),
        KP.RIGHT_ELBOW: CX - sh - px(7), KP.LEFT_ELBOW: CX + sh + px(7),
        KP.RIGHT_WRIST: CX - sh - px(14), KP.LEFT_WRIST: CX + sh + px(14),
        KP.RIGHT_HIP: CX - 0.55 * h(b.hip_width), KP.LEFT_HIP: CX + 0.55 * h(b.hip_width),
        KP.RIGHT_KNEE: CX - gap - px(0.4 * b.thigh_width), KP.LEFT_KNEE: CX + gap + px(0.4 * b.thigh_width),
        KP.RIGHT_ANKLE: CX - gap - px(0.25 * b.thigh_width), KP.LEFT_ANKLE: CX + gap + px(0.25 * b.thigh_width),
    }
    return _keypoints(xs, KP_Y, np.full(NUM_KEYPOINTS, 0.9), "front", CX), mask


def side_view(b: BodySpec) -> tuple[Keypoints, SilhouetteMask]:
    ppc = (BOTTOM_Y - TOP_Y) / b.height_cm
    h = lambda cm: cm * ppc / 2.0  # noqa: E731

    body = _symmetric([
        (0.130, h(b.neck_depth)), (0.175, h(0.85 * b.chest_depth)), (0.260, h(b.chest_depth)),
        (0.400, h(b.waist_depth)), (0.520, h(b.hip_depth)), (0.600, h(1.05 * b.thigh_width)),
        (0.740, h(0.70 * b.thigh_width)), (1.000, h(0.45 * b.thigh_width)),
    ])
    neck = _symmetric([(0.090, h(b.neck_depth)), (0.135, h(b.neck_depth))])
    mask = _mask([body, neck], (h(b.head_depth), _y(0.06) - TOP_Y), "side")

    conf = np.full(NUM_KEYPOINTS, 0.9)
    for k in (KP.LEFT_SHOULDER, KP.LEFT_ELBOW, KP.LEFT_WRIST, KP.LEFT_HIP, KP.LEFT_KNEE):
        conf[k] = 0.3  # far-side limbs are occluded in profile
    return _keypoints({}, KP_Y, conf, "side", CX + h(b.head_depth) * 0.8), mask


def snapshot() -> dict:
    """Run the geometry pipeline on every synthetic body and return a JSON-able dump."""
    from webui.envelope import body_to_envelope
    from webui.inference import InferenceResult

    def rnd(values):
        return [None if v is None else round(float(v), 3) for v in values]

    out: dict = {}
    for b in BODIES:
        fkp, fmask = front_view(b)
        skp, smask = side_view(b)
        cal = calibrate(fkp, skp, b.height_cm)
        bm = extract_measurements(fkp, skp, fmask, smask, cal)
        bm = estimate_circumferences(bm, None)
        bm = validate_measurements(bm)
        widths, selected_y = extract_all_widths(fmask, smask, fkp, skp)
        res = InferenceResult(body=bm, front_kp=fkp, side_kp=skp, front_mask=fmask, side_mask=smask,
                              cal=cal, has_regressor=False, pose_backend="coco")
        env = body_to_envelope(res, b.height_cm, b.sex, f"synthetic-{b.body_id}")
        out[b.body_id] = {
            "calibration": [round(cal.px_per_cm_front, 6), round(cal.px_per_cm_side, 6)],
            "widths": {k: rnd(v) for k, v in sorted(widths.items())},
            "selected_y": {k: rnd(v) for k, v in sorted(selected_y.items())},
            "body": {k: round(float(v), 4) for k, v in bm.to_dict().items()},
            "warnings": list(bm.warnings),
            "envelope": {it.id: it.value_cm for it in env.measurements},
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-expected", action="store_true", help=f"overwrite {EXPECTED_PATH.name}")
    args = ap.parse_args()
    data = snapshot()
    text = json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    if args.write_expected:
        EXPECTED_PATH.parent.mkdir(parents=True, exist_ok=True)
        EXPECTED_PATH.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {EXPECTED_PATH} ({len(data)} bodies)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
