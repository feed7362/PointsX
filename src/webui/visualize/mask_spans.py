"""Horizontal silhouette spans used to place the width lines."""
from __future__ import annotations

import numpy as np


def _mask_span_x(mask: np.ndarray, y: float) -> tuple[int, int] | None:
    h, _w = mask.shape
    yi = int(round(y))
    if yi < 0 or yi >= h:
        return None
    cols = np.where(mask[yi])[0]
    if len(cols) < 2:
        return None
    diffs = np.diff(cols)
    split_points = np.where(diffs > 3)[0] + 1
    segments = np.split(cols, split_points)
    best = max(segments, key=lambda s: (s[-1] - s[0]))
    if len(best) < 2:
        return None
    return int(best[0]), int(best[-1])


def _extreme_span_between_y(
    mask: np.ndarray, y0: float, y1: float, mode: str
) -> tuple[int, int, int] | None:
    """Return (x0, x1, y) for min/max continuous span between two y values."""
    lo = int(round(min(y0, y1)))
    hi = int(round(max(y0, y1)))
    best: tuple[int, int, int] | None = None
    for yi in range(lo, hi + 1):
        span = _mask_span_x(mask, float(yi))
        if span is None:
            continue
        x0, x1 = span
        w = x1 - x0
        if best is None:
            best = (x0, x1, yi)
            continue
        bw = best[1] - best[0]
        if (mode == "min" and w < bw) or (mode == "max" and w > bw):
            best = (x0, x1, yi)
    return best
