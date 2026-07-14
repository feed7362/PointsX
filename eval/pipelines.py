"""Pipelines under test by the BodyM eval orchestrator (`bodym.py`).

A pipeline is a pure transform: it takes the base-ellipse `raw` prediction dict
(computed once per subject by the shared silhouette adapter) + the subject's sex
+ the train-fitted per-(sex, measurement) scales, and returns a corrected dict.

The expensive silhouette work (calibrate → arm-clip → rows → widths → ellipse)
runs ONCE per subject; pipelines are cheap post-transforms replayed over it —
mirroring the backend `pointsx-eval`, which runs pose+seg once and replays combos.

Add a pipeline to `PIPELINES` and the orchestrator scores it automatically; no
change needed in `bodym.py`. The first entry (A) is the frozen baseline every
other pipeline is scored against — never edit it.
"""
from __future__ import annotations

from collections.abc import Callable

# A pipeline: (raw, sex, scales) -> corrected. `raw`/output map measurement id
# -> cm or None; `scales` is {sex: {mid: multiplicative_scale}}.
PipelineFn = Callable[[dict[str, float | None], str, dict[str, dict[str, float]]], dict[str, float | None]]


def raw_ellipse(raw, sex, scales):
    """A — the base two-widths→Ramanujan ellipse. Frozen baseline (identity)."""
    return dict(raw)


def corrected(raw, sex, scales):
    """B — apply the fitted per-sex median(gt/pred) scale to EVERY circumference."""
    s = scales.get(sex, {})
    return {m: (v * s.get(m, 1.0) if v is not None else None) for m, v in raw.items()}


def gated_corrected(raw, sex, scales, tau: float = 0.05):
    """C — B, but skip cells the fit deems already-good.

    If the fitted scale sits within `tau` of 1.0 (≈no systematic bias to remove),
    leave the raw value untouched — correcting an already-unbiased measurement can
    only add error (see testA chest/thigh under B). Same spirit as the backend
    `--fit-offsets` dropping sub-0.25% scales, at a coarser, honest threshold.
    """
    s = scales.get(sex, {})
    out: dict[str, float | None] = {}
    for m, v in raw.items():
        if v is None:
            out[m] = None
            continue
        sc = s.get(m, 1.0)
        out[m] = v * sc if abs(sc - 1.0) >= tau else v
    return out


# Registry — orchestrator scores each against PIPELINES[0]. Append to extend.
PIPELINES: list[tuple[str, PipelineFn]] = [
    ("A_raw", raw_ellipse),
    ("B_corrected", corrected),
    ("C_gated5%", gated_corrected),
]
