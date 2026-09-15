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
# Fit fresh values via ``pointsx-eval --fit-offsets`` and paste the printed dict back here when new
# ground-truth subjects arrive. Refitted 2026-07-20 on the app GT corpus (n=11
# real subjects with tape measurements, 8 F / 3 M), AFTER the thigh-width
# extraction fix. Method: median(gt / predicted), the L1-optimal multiplicative
# correction, rounded to 0.5 %.
#
# Male chest is deliberately absent (see the inline note on the female cell):
# its fitted +3.9 % improves in-sample but worsens leave-one-out — n=3
# overfitting. Leave-one-out MAE over the four circumferences: 6.50 cm with
# this table, 6.81 cm correcting all four, 11.0 cm with no correction at all.
#
# CAVEAT: the male cells rest on n=3 subjects — treat them as provisional and
# refit once the corpus has ~8+ male subjects.
_SEX_CIRCUMFERENCE_SCALES_PCT: dict[str, dict[str, float]] = {
    "female": {  # n=8
        # Chest RESTORED 2026-07-20 after a real-world report of ~10 cm
        # over-measurement. It had been gated to zero because its bootstrap CI
        # spans 1.0 — but "not significant" means UNCERTAIN, not "use zero": the
        # fitted point estimate is still the best single guess, and dropping it
        # measured worse (MAE 5.9 uncorrected vs 5.0 here; LOOCV 5.9 vs 5.6).
        # Male chest is deliberately still absent: its fitted +3.9 % improves
        # in-sample (4.6) but WORSENS leave-one-out (6.3) — n=3 overfitting.
        "chest_circumference":  -4.5,
        # Waist refitted 2026-07-29 for the new FRONT anchor (fixed 0.25 of the
        # pelvis->neck span instead of the narrowest row). MAE 5.98 -> 4.45.
        "waist_circumference": -17.5,   # %
        "hip_circumference":    -5.0,
        "thigh_circumference": -17.0,
    },
    "male": {  # n=3 — provisional
        "waist_circumference": -15.5,
        "hip_circumference":   -10.5,
        "thigh_circumference": -23.5,
    },
    # "other" averages male and female so an unknown-sex subject is biased
    # toward neither extreme.
    "other": {
        "waist_circumference": -14.0,
        "hip_circumference":    -8.0,
        "thigh_circumference": -20.0,
    },
}

# Sex-INDEPENDENT bias corrections for the non-circumference measurements.
# Fitted 2026-07-20 on the same n=11 app GT corpus, median(gt / predicted).
#
# Why these were the biggest remaining errors: the per-sex table above only ever
# covered the four circumferences, so lengths and heights carried their full
# systematic bias uncorrected. They were the WORST measurements in the pipeline
# (inner seam MAE 10.7 cm with bias +10.7 — i.e. essentially pure offset, no
# scatter), simply because nothing corrected them.
#
# Sex-INDEPENDENT on purpose: splitting these by sex measured WORSE in
# leave-one-out (5.45 vs 5.39 cm overall) — 8 female / 3 male is too thin to
# support per-sex length fits, so the split fits noise.
#
# Only measurements passing BOTH gates are listed: (a) the bootstrap 95 % CI on
# the fitted ratio excludes 1.0, and (b) leave-one-out MAE improves by >0.2 cm.
# Deliberately EXCLUDED by those gates:
#   chest_width_front     +8.9 % but CI [-1.0, +25.0] spans zero, LOOCV -0.3
#   back_length_to_waist  +6.8 % but CI [-0.7, +13.6] spans zero, LOOCV +0.7
#
# Leave-one-out overall MAE: 6.18 cm before -> 5.23 cm with this table.
_LENGTH_SCALES_PCT: dict[str, float] = {
    "leg_length_inner_seam":  -9.5,   # %   MAE 10.7 -> 6.8
    "leg_length_outer_seam":  -2.5,   #     MAE  8.7 -> 7.7
    "neck_base_height":       +4.0,   #     MAE  6.3 -> 2.8
}

# Set of IDs eligible for the multiplicative bias correction. Anything outside
# this set is left untouched by the sex-scale logic.
_SEX_SCALE_TARGET_IDS: set[str] = {
    "chest_circumference",
    "waist_circumference",
    "hip_circumference",
    "thigh_circumference",
}
