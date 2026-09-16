"""Cross-measurement sanity gate for ground-truth rows in the eval corpus.

Per-measurement ranges (``scripts/build_eval_csv.py:GT_RANGES``) catch placeholder
values such as all 1s or all 9s, but not a row where every value is individually
plausible and the set is not: an 80 cm outer seam is a normal number, yet not for a
175 cm man. This module checks ratios to height and between measurements.

Bounds are the observed range over the 16 consistent subjects of the app GT corpus
(2026-09-15) widened by roughly 10 %. They are deliberately loose: they must reject
impossible rows, not atypical bodies. Replace them with ANSUR II percentiles once
the ANSUR CSVs are back in ``scripts/ansur/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

HEIGHT = "height_cm"

# (measurement, denominator, lo, hi). A violation drops the measurement (the numerator).
# Observed range on the corpus is in the comment.
RATIO_BOUNDS: list[tuple[str, str, float, float]] = [
    ("leg_length_outer_seam", HEIGHT, 0.55, 0.72),                 # 0.607-0.669
    ("leg_length_inner_seam", HEIGHT, 0.37, 0.53),                 # 0.407-0.491
    ("neck_base_height", HEIGHT, 0.80, 0.95),                      # 0.846-0.938
    ("neck_circumference", HEIGHT, 0.18, 0.30),                    # 0.195-0.273
    ("thigh_circumference", "hip_circumference", 0.46, 0.66),      # 0.509-0.598
    ("upper_arm_circumference", "chest_circumference", 0.22, 0.40),  # 0.250-0.350
    ("waist_circumference", "hip_circumference", 0.60, 1.15),      # 0.677-1.036
]

# This many violations means the row as a whole is not a real person's measurements
# (test or placeholder submission): the subject is excluded, not just the values.
MAX_VIOLATIONS = 2


@dataclass
class GtCheck:
    dropped: dict[str, str] = field(default_factory=dict)  # measurement id -> reason
    exclude_subject: bool = False

    @property
    def reasons(self) -> list[str]:
        return [f"{mid}: {why}" for mid, why in self.dropped.items()]


def check_gt(height_cm: float, gt: dict[str, float]) -> GtCheck:
    """Check one subject's ground truth against the ratio bounds.

    Args:
        height_cm: Subject height in cm.
        gt: Measurement id -> value in cm.

    Returns:
        Which measurements to drop and whether to exclude the whole subject.
    """
    values = {HEIGHT: height_cm, **gt}
    result = GtCheck()
    for mid, denom, lo, hi in RATIO_BOUNDS:
        num, den = values.get(mid), values.get(denom)
        if num is None or not den:
            continue
        ratio = num / den
        if not lo <= ratio <= hi:
            result.dropped[mid] = f"{mid}/{denom}={ratio:.3f} outside [{lo}, {hi}]"
    result.exclude_subject = len(result.dropped) >= MAX_VIOLATIONS
    return result
