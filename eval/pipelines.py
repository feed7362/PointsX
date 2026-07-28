"""Pipelines under test by the BodyM eval orchestrator (`bodym.py`).

A pipeline is a pure transform: it takes the per-subject `raw` output of the
shared silhouette adapter + the subject's sex + the train-fitted `params`, and
returns {measurement id -> cm or None}.

`raw[m]` is a tuple `(circ, front_w_cm, side_w_cm)` or None:
  - `circ`     = the base two-widths→Ramanujan ellipse circumference (pipeline A).
  - `front_w`  = clipped front breadth in cm at that measurement's row.
  - `side_w`   = side depth in cm at the same anatomical fraction.
Post-transform pipelines (A/B/C) use `circ`; the geometric-model pipeline (D)
uses the widths directly, because it REPLACES the ellipse formula.

`params` (fit once on train, constant across subjects):
  - `params["scales"]` = {sex: {mid: multiplicative median(gt/circ) scale}}
  - `params["girth"]`  = {sex: {mid: [c0, c1, c2, c3]}}  learned width→girth coefs

The expensive silhouette work runs ONCE per subject; pipelines are cheap replays.
Add a pipeline to `PIPELINES` → the orchestrator scores it automatically. The
first entry (A) is the frozen baseline every other pipeline is scored against.

LEGAL GUARD (BodyM is CC BY-NC — evaluation only):
  • A — pure geometry (YOLO widths + ellipse formula). No data fit. Fully clean.
  • B/C/D — derive correction constants from BodyM GT (median / least-squares).
    A NON-COMMERCIAL RESEARCH BENCHMARK ONLY; these constants must never ship.
  • A trained ML MODEL (E+) may NOT be fit on BodyM. It must train on a legal
    source (synthetic / app GT) and use BodyM only as held-out test.
"""
from __future__ import annotations

from collections.abc import Callable

PipelineFn = Callable[[dict, str, dict], dict]


def _circ(v):
    return v[0] if v is not None else None


def raw_ellipse(raw, sex, params):
    """A — base two-widths→Ramanujan ellipse. Frozen baseline (identity on circ)."""
    return {m: _circ(v) for m, v in raw.items()}


def corrected(raw, sex, params):
    """B — A × per-(sex, measurement) median(gt/pred) scale on EVERY circumference."""
    s = params["scales"].get(sex, {})
    return {m: (v[0] * s.get(m, 1.0) if v is not None else None) for m, v in raw.items()}


def gated_corrected(raw, sex, params, tau: float = 0.05):
    """C — B, but skip cells whose fitted scale is within `tau` of 1.0 (already-good)."""
    s = params["scales"].get(sex, {})
    out: dict[str, float | None] = {}
    for m, v in raw.items():
        if v is None:
            out[m] = None
            continue
        sc = s.get(m, 1.0)
        out[m] = v[0] * sc if abs(sc - 1.0) >= tau else v[0]
    return out


def learned_girth(raw, sex, params):
    """D — replace the ellipse formula with a learned nonlinear width→girth map.

    circ = c0 + c1·front + c2·side + c3·(front·side)  — bilinear least-squares fit
    per (sex, measurement) on the train split. The cross term makes it nonlinear;
    it subsumes B's constant scale AND captures how the cross-section shape drifts
    from a true ellipse with size. Cheaper than a neural net, richer than the
    fixed Ramanujan ellipse. Falls back to the ellipse circ if no coefs fit.
    """
    coefs = params["girth"].get(sex, {})
    out: dict[str, float | None] = {}
    for m, v in raw.items():
        if v is None:
            out[m] = None
            continue
        circ, fw, sw = v
        c = coefs.get(m)
        out[m] = float(c[0] + c[1] * fw + c[2] * sw + c[3] * fw * sw) if c else circ
    return out


# Registry — orchestrator scores each against PIPELINES[0]. Append to extend.
PIPELINES: list[tuple[str, PipelineFn]] = [
    ("A_raw", raw_ellipse),
    ("B_corrected", corrected),
    ("C_gated5%", gated_corrected),
    ("D_learned_girth", learned_girth),
]
