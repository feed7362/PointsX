"""Lightweight per-phase timing helper for the measurement endpoint.

Usage::

    tm = Timings()
    with tm("pose_front"):
        run_pose(...)
    with tm("seg_front"):
        run_seg(...)
    logger.warning("timings: %s", tm.format())

`format()` returns a compact string like ``pose_front=4.32s
pose_side=4.18s seg_front=2.81s total=11.31s`` ready to stuff into a
single log line or attach to the response envelope.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


class Timings:
    """Stopwatch keyed by phase name. Cheap, no third-party deps."""

    def __init__(self) -> None:
        self._t0 = time.perf_counter()
        self._phases: dict[str, float] = {}

    @contextmanager
    def __call__(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._phases[name] = self._phases.get(name, 0.0) + (time.perf_counter() - start)

    def mark(self, name: str, seconds: float) -> None:
        self._phases[name] = self._phases.get(name, 0.0) + float(seconds)

    def total_s(self) -> float:
        return time.perf_counter() - self._t0

    def as_dict(self) -> dict[str, float]:
        out = {k: round(v, 3) for k, v in self._phases.items()}
        out["total"] = round(self.total_s(), 3)
        return out

    def format(self) -> str:
        parts = [f"{k}={v:.2f}s" for k, v in self._phases.items()]
        parts.append(f"total={self.total_s():.2f}s")
        return " ".join(parts)
