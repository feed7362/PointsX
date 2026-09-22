"""Per-person and per-clothing breakdown of a pointsx-eval report.

A session where one person is photographed several times produces several rows for one body
(`<person>-<clothing>-<id>` subject ids, see build_eval_csv.py --links). Averaging over rows then
weights those people more heavily, and it hides the question the session was run to answer: how much
does clothing cost us on the same body?

usage: .venv/Scripts/python scripts/analyze_pairs.py runs/eval/track/NNN-*/app.csv
Local only: per-subject errors on real people.
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from webui.envelope import DISPLAY_MEASUREMENT_IDS  # noqa: E402

DISPLAYED = set(DISPLAY_MEASUREMENT_IDS)


def sections(path: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    name = header = None
    for row in csv.reader(path.open(encoding="utf-8")):
        if not row:
            continue
        if row[0].startswith("#"):
            name, header = row[0].lstrip("# ").strip(), None
            out[name] = []
        elif header is None:
            header = row
        else:
            out[name].append(dict(zip(header, row)))
    return out


def split_id(sid: str) -> tuple[str, str]:
    """`A-suit-bcc92531` -> ("A", "suit"); anything else -> (sid, "unknown")."""
    parts = sid.split("-", 2)          # the id itself may contain "-" (tg-A-1)
    if len(parts) == 3 and len(parts[0]) <= 2 and parts[1] in ("suit", "own", "mixed"):
        return parts[0], parts[1]
    return sid, "unknown"


def mae(v: list[float]) -> float:
    return sum(abs(x) for x in v) / len(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path)
    args = ap.parse_args()
    detail = sections(args.report)["per-(combo, subject, measurement)"]
    combo = detail[0]["combo"]
    rows = [r for r in detail if r["combo"] == combo]

    per_pair: dict[tuple[str, str, str], list[float]] = defaultdict(list)   # person, clothing, mid
    for r in rows:
        person, clothing = split_id(r["subject_id"])
        per_pair[(person, clothing, r["measurement"])].append(float(r["error_cm"]))

    print(f"combo {combo} — {len(rows)} observations")

    # ── per person (each pair counted once, then people averaged equally) ──
    by_person: dict[str, list[float]] = defaultdict(list)
    for (person, _c, mid), errs in per_pair.items():
        if mid in DISPLAYED:
            by_person[person] += errs
    print(f"\n=== per person, displayed measurements ({len(by_person)} people) ===")
    for p, errs in sorted(by_person.items(), key=lambda kv: -mae(kv[1])):
        print(f"  {p:10} n={len(errs):3d}  MAE {mae(errs):5.2f}")
    people_mae = [mae(e) for e in by_person.values()]
    print(f"  mean over people {st.mean(people_mae):.2f}   median {st.median(people_mae):.2f}"
          f"   row-weighted {mae([e for v in by_person.values() for e in v]):.2f}")

    # ── clothing, on the people who have both ──
    both = {p for p in {k[0] for k in per_pair}
            if {"suit", "own"} <= {k[1] for k in per_pair if k[0] == p}}
    print(f"\n=== suit vs own clothes, same bodies ({len(both)} people: {', '.join(sorted(both))}) ===")
    print(f"  {'measurement':30}{'suit':>8}{'own':>8}{'own-suit':>10}")
    tot: dict[str, list[float]] = defaultdict(list)
    for mid in DISPLAY_MEASUREMENT_IDS:
        cells = {}
        for cl in ("suit", "own"):
            errs = [e for (p, c, m), v in per_pair.items() if p in both and c == cl and m == mid for e in v]
            if errs:
                cells[cl] = mae(errs)
                tot[cl] += errs
        if len(cells) == 2:
            print(f"  {mid:30}{cells['suit']:8.2f}{cells['own']:8.2f}{cells['own'] - cells['suit']:+10.2f}")
    if len(tot) == 2:
        print(f"  {'ALL DISPLAYED':30}{mae(tot['suit']):8.2f}{mae(tot['own']):8.2f}"
              f"{mae(tot['own']) - mae(tot['suit']):+10.2f}")
        print("\n  positive = loose clothing costs accuracy; this is the number the synthetic")
        print("  body-under-clothing work has to beat (docs/synthetic-dataset-audit-plan.md §5, gate 3).")

    # ── repeatability: same person, same clothing, different captures ──
    print("\n=== capture-to-capture spread (same person and clothing, several pairs) ===")
    spreads = []
    for (person, clothing, mid), errs in sorted(per_pair.items()):
        if len(errs) > 1 and mid in DISPLAYED:
            spreads.append(max(errs) - min(errs))
    if spreads:
        print(f"  {len(spreads)} cells with repeats: mean range {st.mean(spreads):.2f} cm, "
              f"median {st.median(spreads):.2f}, max {max(spreads):.2f}")
        print("  (pure pipeline noise between two photos of the same body — a floor for any improvement)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
