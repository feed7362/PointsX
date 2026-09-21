"""Full traceability report: every Supabase GT measurement vs the pipeline output.

Answers "where does this number come from?" — for each subject it prints every
one of the 16 fields the dataset form collects, the value the pipeline produced,
and the delta. Fields the eval deliberately skips are printed too, with the
reason, so nothing is silently dropped.

Usage:
    cd Q:/Projects/KHNU/PointsX
    .venv/Scripts/python.exe scripts/audit_measurements.py            # summary
    .venv/Scripts/python.exe scripts/audit_measurements.py --detail   # per subject
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "Pointx-backend" / "src"))

# Supabase form id -> canonical pipeline id, or a reason it is not comparable.
MAP: dict[str, tuple[str | None, str]] = {
    "height":                      (None, "INPUT — given to the pipeline, not predicted"),
    "neck_base_height_from_floor": ("neck_base_height", ""),
    "neck_circumference":          ("neck_circumference", ""),
    "chest_circumference":         ("chest_circumference", ""),
    "waist_circumference":         ("waist_circumference", ""),
    "hip_circumference":           ("hip_circumference", ""),
    "arm_circumference_bicep":     ("upper_arm_circumference", ""),
    "thigh_circumference":         ("thigh_circumference", ""),
    "shoulder_width":              (None, "SKIPPED — form measures full shoulder width (~52cm); "
                                          "canonical shoulder_slope_width is the SLOPE (~15cm)"),
    "shoulder_slope_width":        ("shoulder_slope_width", ""),  # form field since 2026-09-21
    "back_width":                  ("back_width_scapular", ""),
    "chest_width":                 ("chest_width_front", ""),
    "front_length_to_waist":       ("front_length_to_waist", ""),
    "back_length_to_waist":        ("back_length_to_waist", ""),
    "sleeve_length":               (None, "SKIPPED — garment sleeve length, measured differently "
                                          "from body arm_length_shoulder_to_wrist"),
    "outer_seam":                  ("leg_length_outer_seam", ""),
    "inner_seam":                  ("leg_length_inner_seam", ""),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detail", action="store_true", help="per-subject breakdown")
    ap.add_argument("--report", default=str(REPO / "supabase-dump" / "eval-final-check.csv"))
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "dataset-viewer"))
    key = REPO / "keys" / "dataset_private_key.txt"
    if key.is_file():
        os.environ["DATASET_PRIVATE_KEY_FILE"] = str(key)
    from repository import DatasetRepository  # noqa: E402

    repo = DatasetRepository()
    gt: dict[str, dict[str, float]] = {}
    meta: dict[str, dict] = {}
    for s in repo.list_submissions():
        d = repo.get_submission(s["id"])
        sid = s["id"][:8]
        meta[sid] = {"sex": d.get("sex"), "height": d.get("height_cm")}
        gt[sid] = {m["id"]: m["value"] for m in (d.get("measurements") or [])
                   if m.get("value") not in (None, "")}

    pred: dict[str, dict[str, float]] = defaultdict(dict)
    sec = None
    for r in csv.reader(open(args.report, encoding="utf-8")):
        if r and r[0].startswith("#"):
            sec = r[0]
            continue
        if sec and "subject, measurement" in sec and len(r) == 7 and r[0] == "coco+rama+off":
            pred[r[1]][r[3]] = float(r[4])

    scored = sorted(pred)  # subjects that survived the placeholder filter
    print(f"Supabase submissions: {len(gt)}   scored by the eval: {len(scored)}")
    print(f"(the rest are placeholder rows — every field 1 or 9)\n")

    print(f"{'form field':30} {'canonical id':28} {'n':>3} {'MAE':>6} {'bias':>7}  note")
    print("-" * 100)
    for form_id, (canon, note) in MAP.items():
        if canon is None:
            print(f"{form_id:30} {'—':28} {'—':>3} {'—':>6} {'—':>7}  {note}")
            continue
        errs = [pred[s][canon] - float(gt[s][form_id])
                for s in scored if canon in pred[s] and form_id in gt.get(s, {})]
        if not errs:
            print(f"{form_id:30} {canon:28} {0:>3} {'—':>6} {'—':>7}  not produced by the pipeline")
            continue
        a = np.array(errs)
        print(f"{form_id:30} {canon:28} {len(a):>3} {np.abs(a).mean():6.2f} {a.mean():+7.2f}")

    if args.detail:
        for s in scored:
            print(f"\n=== {s}  sex={meta[s]['sex']}  height={meta[s]['height']} cm ===")
            print(f"  {'measurement':28} {'GT':>7} {'predicted':>10} {'err':>8}")
            for form_id, (canon, _n) in MAP.items():
                if canon is None or form_id not in gt.get(s, {}):
                    continue
                p = pred[s].get(canon)
                g = float(gt[s][form_id])
                if p is None:
                    print(f"  {form_id:28} {g:7.1f} {'—':>10} {'—':>8}")
                else:
                    print(f"  {form_id:28} {g:7.1f} {p:10.1f} {p - g:+8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
