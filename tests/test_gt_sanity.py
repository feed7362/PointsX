"""pointsx.gt_sanity — cross-measurement gate on eval ground truth (synthetic values only)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pointsx.gt_sanity import check_gt  # noqa: E402

TYPICAL = {
    "leg_length_outer_seam": 108.0,
    "leg_length_inner_seam": 78.0,
    "neck_base_height": 150.0,
    "neck_circumference": 38.0,
    "chest_circumference": 98.0,
    "waist_circumference": 82.0,
    "hip_circumference": 100.0,
    "thigh_circumference": 56.0,
    "upper_arm_circumference": 31.0,
}


def test_typical_body_passes():
    check = check_gt(175.0, TYPICAL)
    assert check.dropped == {} and not check.exclude_subject


def test_single_implausible_value_is_dropped_subject_kept():
    check = check_gt(175.0, {**TYPICAL, "neck_circumference": 55.0})
    assert list(check.dropped) == ["neck_circumference"]
    assert not check.exclude_subject


def test_jointly_implausible_row_is_excluded():
    # Every value passes a per-measurement range; the combination does not.
    row = {**TYPICAL, "leg_length_outer_seam": 80.0, "leg_length_inner_seam": 60.0,
           "neck_circumference": 30.0, "thigh_circumference": 40.0, "hip_circumference": 92.0}
    check = check_gt(175.0, row)
    assert check.exclude_subject
    assert {"leg_length_outer_seam", "leg_length_inner_seam"} <= set(check.dropped)


def test_missing_values_are_not_violations():
    assert check_gt(170.0, {"waist_circumference": 80.0}).dropped == {}
