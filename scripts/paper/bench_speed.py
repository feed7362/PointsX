"""Publishable speed benchmark for FitMeasureAI §3.3.4.

Per-stage wall-clock over N repeats x M subject pairs, median + IQR.
Cold start (import / model load / warm-up) reported separately.

Usage (repo root):
  .venv/Scripts/python <this> --repeats 20 --threads 8 --out speed.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics as st
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--pairs", type=int, default=5, help="how many subject pairs to cycle")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="speed.json")
    args = ap.parse_args()

    # Thread caps must be set BEFORE torch/OpenMP initialise; set_num_threads()
    # after import is silently ignored by the already-started OMP pool.
    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    t0 = time.perf_counter()
    import cv2
    import torch

    from webui._timing import Timings
    from webui.infrastructure.inference import WebuiPipeline
    import_s = time.perf_counter() - t0

    if args.threads:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)

    t0 = time.perf_counter()
    pipe = WebuiPipeline(
        None,
        str(REPO / "models/yolo26-pose.pt"),
        str(REPO / "models/yolo12l-person-seg-extended.pt"),
        device=args.device,
    )
    load_s = time.perf_counter() - t0

    warm = pipe.warmup() if hasattr(pipe, "warmup") else {}

    rows = list(csv.DictReader(open(REPO / "supabase-dump/subjects.csv", encoding="utf-8")))[: args.pairs]
    imgs = []
    for r in rows:
        f, s = cv2.imread(r["front"]), cv2.imread(r["side"])
        if f is None or s is None:
            continue
        imgs.append((f, s, float(r["height_cm"]), f.shape[:2], s.shape[:2]))
    if not imgs:
        print("no images", file=sys.stderr)
        return 1

    # Ultralytics resets the pool to min(8, cpu_count-1) on its first predict
    # (utils/__init__.py NUM_THREADS), so re-assert the cap after warm-up and
    # record what was actually in force during the timed loop.
    if args.threads:
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
    threads_in_loop = torch.get_num_threads()

    per_stage: dict[str, list[float]] = {}
    totals: list[float] = []
    for i in range(args.repeats):
        f, s, h, _, _ = imgs[i % len(imgs)]
        tm = Timings()
        t = time.perf_counter()
        pipe.measure(f, s, h, timings=tm)
        totals.append(time.perf_counter() - t)
        for k, v in tm.as_dict().items():
            if k == "total":
                continue
            per_stage.setdefault(k, []).append(v)

    def stats(xs: list[float]) -> dict:
        xs = sorted(xs)
        q1, q3 = st.quantiles(xs, n=4)[0], st.quantiles(xs, n=4)[2] if len(xs) >= 4 else (xs[0], xs[-1])
        return {
            "n": len(xs),
            "median_s": round(st.median(xs), 4),
            "mean_s": round(st.fmean(xs), 4),
            "iqr_lo_s": round(q1, 4),
            "iqr_hi_s": round(q3, 4),
            "min_s": round(xs[0], 4),
            "max_s": round(xs[-1], 4),
        }

    out = {
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cores": os.cpu_count(),
            "torch_threads_in_loop": threads_in_loop,
            "torch_threads_after": torch.get_num_threads(),
            "device": args.device,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "config": {
            "repeats": args.repeats,
            "pairs": len(imgs),
            "input_sizes": [f"{a[0]}x{a[1]}" for *_x, a, _b in imgs][:5],
            "img_size": getattr(getattr(pipe, "models", None), "img_size", None),
        },
        "cold_start_s": {
            "import": round(import_s, 3),
            "model_load": round(load_s, 3),
            "warmup": {k: round(v, 3) for k, v in warm.items()},
        },
        "per_stage": {k: stats(v) for k, v in sorted(per_stage.items())},
        "total": stats(totals),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["total"], indent=1))
    for k, v in out["per_stage"].items():
        print(f"{k:16} median={v['median_s']:.3f}s  IQR[{v['iqr_lo_s']:.3f},{v['iqr_hi_s']:.3f}]")
    print("cold:", out["cold_start_s"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
