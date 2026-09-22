"""Bias corrections applied to pipeline outputs before they reach the envelope.

Fitted on the app ground-truth corpus; the comments are the provenance of every
value (corpus size, method, leave-one-out results). Keep them with the numbers.
"""
from __future__ import annotations

# Per-sex multiplicative bias correction for the four core circumferences.
# Values are PERCENT scale-factors applied as ``value *= 1 + pct/100``. They
# only apply to these four IDs (chest/waist/hip/thigh) — other measurements
# are not bias-corrected here.
#
# Refitted 2026-09-16 on the app GT corpus after the cross-measurement GT gate
# (pointsx/gt_sanity.py): n=16 (12 F / 4 M). Method: median(gt / predicted),
# the L1-optimal multiplicative correction, rounded to 0.5 %, via
# ``scripts/refit_corrections.py runs/eval/raw_gated.csv`` on a
# ``pointsx-eval --no-sex-offsets`` report (raw predictions, replayed exactly).
#
# A cell is kept only if (a) the bootstrap 95 % CI of the fitted ratio excludes
# 1.0 and (b) leave-one-out (LOO) MAE beats no-correction by > 0.2 cm.
# Numbers below are LOO (held-out), never in-sample:
#
#   cell             n   none   LOO    fitted   CI(ratio)
#   F chest         11   5.08   4.34   -2.5 %   [0.933, 0.997]
#   F waist         12  20.10   4.20  -20.0 %   [0.765, 0.832]
#   F hip           12   6.18   4.00   -5.5 %   [0.929, 0.978]
#   F thigh         12  10.52   3.60  -17.0 %   [0.817, 0.900]
#   M chest          4   3.73     —    +3.0 %   [0.963, 1.049]  EXCLUDED: CI spans 1.0
#   M waist          4  12.73   3.75  -13.0 %   [0.840, 0.929]
#   M hip            4  10.15   4.00   -7.5 %   [0.845, 0.955]
#   M thigh          4  12.33   3.29  -18.0 %   [0.762, 0.891]
#
# Male chest stays absent for the same reason as in the July fit (n=3 then, n=4
# now): the CI spans 1.0. CAVEAT: the male cells rest on n=4 (LOO on 3) —
# provisional until the corpus has ~8+ male subjects.
#
# Previous table (2026-07-20, n=11, 8 F / 3 M): F chest -4.5, waist -17.5,
# hip -5.0, thigh -17.0; M waist -15.5, hip -10.5, thigh -23.5. Replayed on the
# gated corpus it scored 3.73 cm on the 159 corrected observations, partly
# in-sample (8 of the 12 F were in its fit set); this table scores 3.69 held-out.
_SEX_CIRCUMFERENCE_SCALES_PCT: dict[str, dict[str, float]] = {
    # Refitted 2026-09-22 AFTER the calibration fix (pointsx/calibration.py: px_per_cm now comes
    # from the silhouette's head-to-floor extent, not the head_top->ankle keypoint span, which is
    # ~10 % short of stature and inflated every width by 7.5-13.7 %). The old table existed largely
    # to cancel that inflation, which is why it over-corrected on tight clothing.
    # Corpus: 43 photo pairs / 26 people. Method and gate unchanged (median(gt/pred), bootstrap CI
    # excludes 1.0, LOO gain > 0.2 cm); numbers below are LOO, never in-sample.
    #   cell        n   none    LOO   fitted   CI(ratio)
    #   F chest    34   5.49   4.19   +5.0 %   [1.017, 1.075]
    #   F waist    38  13.98   5.48  -16.5 %   [0.798, 0.856]
    #   F hip      38   6.77   6.10   +4.5 %   [1.021, 1.060]
    #   F thigh    38   5.73   3.92   -8.0 %   [0.903, 0.949]
    #   M chest     4  11.35   3.14  +13.0 %   [1.051, 1.160]
    #   M thigh     4   6.28   2.28   -9.0 %   [0.838, 0.950]
    #   M waist     4   4.55     —     -3.5 %  [0.900, 1.027]  EXCLUDED: CI spans 1.0
    #   M hip       4   5.43     —     +3.0 %  [0.903, 1.051]  EXCLUDED: CI spans 1.0
    # Male waist and hip now need NO correction — the constant they used to carry was the scale bug.
    "female": {  # n=38 pairs
        "chest_circumference":  +5.0,
        "waist_circumference": -16.5,
        "hip_circumference":    +4.5,
        "thigh_circumference":  -8.0,
    },
    "male": {  # n=4 pairs — provisional
        "chest_circumference": +13.0,
        "thigh_circumference":  -9.0,
    },
    "other": {
        "chest_circumference":  +9.0,
        "waist_circumference":  -8.0,
        "hip_circumference":    +2.0,
        "thigh_circumference":  -8.5,
    },
}

# Sex-INDEPENDENT corrections for the non-circumference measurements, same corpus and gate.
#   cell                    n   none    LOO   fitted   CI(ratio)
#   leg_length_inner_seam  42   3.57   2.61   +4.5 %   [1.027, 1.059]
#   leg_length_outer_seam  40   5.12   3.29   +4.5 %   [1.033, 1.055]
#   neck_base_height       42  18.72   2.94  +14.0 %   [1.132, 1.152]
#   chest_width_front      42   5.65   3.15  +16.0 %   [1.135, 1.219]
#   back_length_to_waist   38   5.57   1.92  +17.5 %   [1.148, 1.193]
#   front_length_to_waist  42   6.00   2.26  +15.5 %   [1.136, 1.176]
#
# CAVEAT — these are large and they encode a known defect, not anatomy: every vertical derivation
# measures to the ANKLE KEYPOINT rather than the floor, so it under-reads by the ankle height. The
# old too-small px_per_cm used to cancel it; with calibration fixed the gap is visible. The right
# repair is to measure those spans from the silhouette's floor line and refit again — see T4 in
# runs/tasks-2026-09-22.md. Until then these constants keep the output honest.
_LENGTH_SCALES_PCT: dict[str, float] = {
    "leg_length_inner_seam":  +4.5,
    "leg_length_outer_seam":  +4.5,
    "neck_base_height":      +14.0,
    "chest_width_front":     +16.0,
    "back_length_to_waist":  +17.5,
    "front_length_to_waist": +15.5,
}

# Set of IDs eligible for the multiplicative bias correction. Anything outside
# this set is left untouched by the sex-scale logic.
_SEX_SCALE_TARGET_IDS: set[str] = {
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "thigh_circumference",
}
