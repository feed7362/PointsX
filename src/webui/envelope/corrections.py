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
    "female": {  # n=12
        "chest_circumference":  -2.5,
        "waist_circumference": -20.0,   # %
        "hip_circumference":    -5.5,
        "thigh_circumference": -17.0,
    },
    "male": {  # n=4 — provisional
        "waist_circumference": -13.0,
        "hip_circumference":    -7.5,
        "thigh_circumference": -18.0,
    },
    # "other" averages male and female so an unknown-sex subject is biased
    # toward neither extreme (male chest counts as 0).
    "other": {
        "chest_circumference":  -1.0,
        "waist_circumference": -16.5,
        "hip_circumference":    -6.5,
        "thigh_circumference": -17.5,
    },
}

# Sex-INDEPENDENT bias corrections for the non-circumference measurements.
# Refitted 2026-09-16 on the same gated corpus (n=16), same method and gate.
# Sex-independent on purpose: 12 F / 4 M is too thin for per-sex length fits.
#
#   cell                    n   none   LOO   fitted   CI(ratio)
#   leg_length_inner_seam  16   7.60   4.25  -8.0 %   [0.884, 0.942]
#   leg_length_outer_seam  16   4.28   3.28  -2.5 %   [0.942, 0.994]
#   neck_base_height       16   6.35   3.05  +4.5 %   [1.027, 1.057]
#   back_length_to_waist   16   3.48   2.06  +8.0 %   [1.043, 1.135]  NEW (July: CI spanned 1.0 at n=11)
#   chest_width_front      16   4.66     —   +2.5 %   [0.986, 1.173]  EXCLUDED: CI spans 1.0
#   front_length_to_waist  16   3.87     —   +5.5 %   [0.957, 1.129]  EXCLUDED: CI spans 1.0
#
# Previous (2026-07-20): inner -9.5, outer -2.5, neck_base +4.0.
_LENGTH_SCALES_PCT: dict[str, float] = {
    "leg_length_inner_seam":  -8.0,   # %
    "leg_length_outer_seam":  -2.5,
    "neck_base_height":       +4.5,
    "back_length_to_waist":   +8.0,
}

# Set of IDs eligible for the multiplicative bias correction. Anything outside
# this set is left untouched by the sex-scale logic.
_SEX_SCALE_TARGET_IDS: set[str] = {
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "thigh_circumference",
}
