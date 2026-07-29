"""Build a `pointsx-eval` subjects CSV from the local Supabase dump.

The dump's meta.json only carries the list-view fields, so GT measurements are
re-fetched from Supabase (full detail) and joined to the already-downloaded,
decrypted photos in supabase-dump/<id>/{front,side}.jpg.

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe scripts/build_eval_csv.py
    # -> supabase-dump/subjects.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "dataset-viewer"))

_key = REPO_ROOT / "keys" / "dataset_private_key.txt"
if _key.is_file():
    os.environ["DATASET_PRIVATE_KEY_FILE"] = str(_key)

# Supabase dataset-form measurement id -> canonical eval id (webui.envelope).
# Only unambiguous mappings are included. Deliberately EXCLUDED:
#   shoulder_width  -> the form measures full shoulder width (~52 cm) while the
#                      canonical `shoulder_slope_width` is the shoulder SLOPE
#                      (~15 cm) — different definitions, would fake a huge error.
#   sleeve_length   -> garment sleeve length, measured differently from the
#                      canonical `arm_length_shoulder_to_wrist` body length.
GT_MAP: dict[str, str] = {
    "chest_circumference": "chest_circumference",
    "waist_circumference": "waist_circumference",
    "hip_circumference": "hip_circumference",
    "thigh_circumference": "thigh_circumference",
    "neck_circumference": "neck_circumference",
    "arm_circumference_bicep": "upper_arm_circumference",
    "neck_base_height_from_floor": "neck_base_height",
    "chest_width": "chest_width_front",
    "back_width": "back_width_scapular",
    "front_length_to_waist": "front_length_to_waist",
    "back_length_to_waist": "back_length_to_waist",
    "outer_seam": "leg_length_outer_seam",
    "inner_seam": "leg_length_inner_seam",
    # These four sit in the envelope but were absent from DISPLAY_MEASUREMENT_IDS,
    # so the eval never scored them despite the corpus HAVING ground truth. Neck
    # in particular was ~35 cm out in production and invisible to every metric.
    "neck_circumference": "neck_circumference",
    "arm_circumference_bicep": "upper_arm_circumference",
    "back_width": "back_width_scapular",
    "front_length_to_waist": "front_length_to_waist",
}

# Plausible adult ranges (cm) per canonical id. The dataset contains TEST
# submissions where every field was filled with a placeholder (all 1s, all 9s) —
# ~half the rows. Scoring against those manufactures ~80 cm "errors" and destroys
# the aggregate, so implausible values are dropped and a subject with too few
# surviving measurements is excluded entirely.
GT_RANGES: dict[str, tuple[float, float]] = {
    "chest_circumference": (60, 170),
    "waist_circumference": (45, 170),
    "hip_circumference": (60, 180),
    "thigh_circumference": (30, 100),
    "neck_circumference": (25, 60),
    "upper_arm_circumference": (18, 60),
    "neck_base_height": (110, 180),
    "chest_width_front": (20, 55),
    "back_width_scapular": (25, 60),
    "front_length_to_waist": (25, 60),
    "back_length_to_waist": (25, 60),
    "leg_length_outer_seam": (70, 130),
    "leg_length_inner_seam": (55, 105),
    "neck_circumference": (25, 60),
    "upper_arm_circumference": (18, 60),
    "back_width_scapular": (25, 60),
    "front_length_to_waist": (25, 60),
}
# A subject needs at least this many plausible GT values to be worth scoring.
MIN_VALID_GT = 4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default=str(REPO_ROOT / "supabase-dump"),
                    help="Folder with <submission_id>/{front,side}.jpg")
    ap.add_argument("--out", default=None, help="CSV path (default: <dump>/subjects.csv)")
    args = ap.parse_args()

    dump = Path(args.dump)
    out = Path(args.out) if args.out else dump / "subjects.csv"

    from repository import DatasetRepository  # noqa: E402

    repo = DatasetRepository()
    subs = repo.list_submissions()
    print(f"[fetch] {len(subs)} submissions from Supabase")

    cols = ["subject_id", "front", "side", "height_cm", "sex", *GT_MAP.values()]
    rows: list[dict] = []
    n_skipped = 0

    for sub in subs:
        sid = sub["id"]
        front, side = dump / sid / "front.jpg", dump / sid / "side.jpg"
        if not (front.is_file() and side.is_file()):
            print(f"[skip] {sid[:8]}: photos not in dump", file=sys.stderr)
            n_skipped += 1
            continue
        try:
            detail = repo.get_submission(sid)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {sid[:8]}: {exc}", file=sys.stderr)
            n_skipped += 1
            continue

        height = detail.get("height_cm") or sub.get("height_cm")
        if not height:
            print(f"[skip] {sid[:8]}: no height", file=sys.stderr)
            n_skipped += 1
            continue

        row = {
            "subject_id": sid[:8],
            "front": str(front),
            "side": str(side),
            "height_cm": height,
            "sex": (detail.get("sex") or sub.get("sex") or "other").lower(),
        }
        n_implausible = 0
        for item in detail.get("measurements") or []:
            canonical = GT_MAP.get(item.get("id"))
            value = item.get("value")
            if not canonical or value in (None, ""):
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            lo, hi = GT_RANGES.get(canonical, (float("-inf"), float("inf")))
            if lo <= v <= hi:
                row[canonical] = v
            else:
                n_implausible += 1

        n_gt = sum(1 for c in GT_MAP.values() if c in row)
        if n_gt < MIN_VALID_GT:
            print(f"[skip] {sid[:8]}: only {n_gt} plausible GT values "
                  f"({n_implausible} out of range) — placeholder/test submission",
                  file=sys.stderr)
            n_skipped += 1
            continue
        rows.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"[done] wrote {out}  subjects={len(rows)} skipped={n_skipped}")
    if rows:
        per = [sum(1 for c in GT_MAP.values() if c in r) for r in rows]
        print(f"       GT measurements per subject: min={min(per)} max={max(per)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
