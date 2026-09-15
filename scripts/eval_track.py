"""Accuracy ledger: re-measure after every pipeline change and compare with the last run.

Two benchmarks, two questions:
  app GT  (pointsx-eval, supabase-dump/subjects.csv) — real photos, real pose + seg,
          the product number. Sees every stage.
  BodyM   (eval/bodym.py, testA + testB) — perfect silhouettes, eval-side rows,
          production ellipse only. Sees the width -> circumference math, nothing else.

Usage (repo root):
  .venv/Scripts/python scripts/eval_track.py run --label "neck from prior"
  .venv/Scripts/python scripts/eval_track.py record --label baseline --app runs/eval/x.csv \
        --bodym eval/reports/run_019 eval/reports/run_020
  .venv/Scripts/python scripts/eval_track.py compare [OLD] [NEW]   # labels or indices, default last two
  .venv/Scripts/python scripts/eval_track.py list

The ledger (runs/eval/ledger.jsonl) and every report stay local: they hold per-subject
errors on real people. Only aggregate numbers go into docs or commit messages.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LEDGER = REPO / "runs" / "eval" / "ledger.jsonl"
TRACK_DIR = REPO / "runs" / "eval" / "track"
SUBJECTS = REPO / "supabase-dump" / "subjects.csv"
BODYM_REPORTS = REPO / "eval" / "reports"
PY = sys.executable

EPS_APP = 0.10    # cm: smaller moves are noise on 16-17 people
EPS_BODYM = 0.05  # cm: 87 / 400 subjects

# What each benchmark can and cannot see. "unchanged" on a blind benchmark is not "no regression".
DETECTION_MAP = """\
change in ...                                   app GT   BodyM
  GT corpus / sanity gate                         yes      no
  pose model, keypoint mapping (pose_coco.py)     yes      no
  segmentation model                              yes      no
  calibration (calibration.py)                    yes      no (eval-side calibrate)
  row placement (silhouette.py, derive.py)        yes      no (eval-side rows); also tests/test_snapshot.py
  ellipse / circumference.py                      yes      yes (chest/waist/hip/thigh only)
  envelope corrections / derivations              yes      no
  neck, back width, lengths                       yes      no (not scored)"""


# ── collecting ────────────────────────────────────────────────────────────────

def _git() -> dict:
    def out(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"sha": out("rev-parse", "--short", "HEAD"), "dirty": bool(out("status", "--porcelain", "--", "src", "eval", "scripts"))}


def _versions() -> dict:
    code = ("import json,torch,ultralytics,numpy,cv2;"
            "print(json.dumps({'torch':torch.__version__,'ultralytics':ultralytics.__version__,"
            "'numpy':numpy.__version__,'opencv':cv2.__version__}))")
    res = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    return json.loads(res.stdout.strip().splitlines()[-1]) if res.returncode == 0 else {}


def _sections(path: Path) -> dict[str, list[dict]]:
    """Split pointsx-eval's report CSV into its '# ...' sections."""
    out: dict[str, list[dict]] = {}
    name, header = None, None
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


def parse_app(path: Path) -> dict:
    from webui.envelope import DISPLAY_MEASUREMENT_IDS

    sec = _sections(path)
    detail = sec["per-(combo, subject, measurement)"]
    combo = detail[0]["combo"] if detail else ""
    errors: dict[str, dict[str, float]] = {}
    sexes: dict[str, str] = {}
    for r in detail:
        if r["combo"] != combo:
            continue
        errors.setdefault(r["subject_id"], {})[r["measurement"]] = float(r["error_cm"])
        sexes[r["subject_id"]] = r["sex"]
    return {"report": str(path.relative_to(REPO)), "combo": combo, "errors": errors, "sex": sexes,
            "summary": _summarize(errors, set(DISPLAY_MEASUREMENT_IDS))}


def _summarize(errors: dict[str, dict[str, float]], displayed: set[str], pairs: set | None = None) -> dict:
    per: dict[str, list[float]] = {}
    for sid, ms in errors.items():
        for mid, e in ms.items():
            if pairs is None or (sid, mid) in pairs:
                per.setdefault(mid, []).append(e)

    def agg(vals: list[float]) -> dict:
        return {"n": len(vals), "mae": sum(abs(v) for v in vals) / len(vals),
                "bias": sum(vals) / len(vals)} if vals else {"n": 0, "mae": None, "bias": None}

    flat = lambda keep: [e for mid, vs in per.items() if keep(mid) for e in vs]  # noqa: E731
    return {"overall": agg(flat(lambda m: True)), "displayed": agg(flat(lambda m: m in displayed)),
            "hidden": agg(flat(lambda m: m not in displayed)),
            "per_measurement": {mid: agg(vs) for mid, vs in sorted(per.items())}}


def parse_bodym(run_dir: Path) -> dict:
    m = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"run": run_dir.name, "subjects": m["subjects"], "overall": m["overall"],
            "per_measurement": {p: {k: {"n": v["n"], "mae": v["mae"], "bias": v["bias"]} for k, v in ms.items()}
                                for p, ms in m["pipelines"].items()}}


def _entry(label: str, app: Path | None, bodym_runs: list[Path]) -> dict:
    entry = {"label": label, "time": dt.datetime.now().isoformat(timespec="seconds"), "git": _git(),
             "versions": _versions(),
             "corpus_sha256": hashlib.sha256(SUBJECTS.read_bytes()).hexdigest()[:16] if SUBJECTS.is_file() else None}
    if app:
        entry["app"] = parse_app(app)
    entry["bodym"] = {}
    for rd in bodym_runs:
        b = parse_bodym(rd)
        split = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))["split"]
        entry["bodym"][split] = b
    return entry


def _append(entry: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[ledger] #{len(_load()) - 1} '{entry['label']}' @ {entry['git']['sha']}{' (dirty)' if entry['git']['dirty'] else ''}")


def _load() -> list[dict]:
    if not LEDGER.is_file():
        return []
    return [json.loads(line) for line in LEDGER.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_bodym_run() -> int:
    runs = [int(p.name.split("_")[1]) for p in BODYM_REPORTS.glob("run_*") if p.name.split("_")[1].isdigit()]
    return max(runs, default=0)


def cmd_run(args: argparse.Namespace) -> int:
    n = len(_load())
    out_dir = TRACK_DIR / f"{n:03d}-{''.join(c if c.isalnum() else '-' for c in args.label)[:40]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    app_csv = out_dir / "app.csv"
    print(f"[app] pointsx-eval -> {app_csv.relative_to(REPO)}")
    cmd = [PY, "-m", "pointsx.eval", "--subjects", str(SUBJECTS), "--pose-backend", "coco",
           "--device", args.device, "--output", str(app_csv)]
    with (out_dir / "app.txt").open("w", encoding="utf-8") as txt:
        if subprocess.run(cmd, cwd=REPO, stdout=txt, stderr=subprocess.STDOUT,
                          env={**os.environ, "PYTHONUTF8": "1"}).returncode != 0:
            print(f"[app] FAILED, see {out_dir / 'app.txt'}")
            return 2
    bodym_runs: list[Path] = []
    if not args.skip_bodym:
        for split in ("testA", "testB"):
            before = _latest_bodym_run()
            print(f"[bodym] {split}")
            with (out_dir / f"bodym_{split}.txt").open("w", encoding="utf-8") as txt:
                rc = subprocess.run([PY, "eval/bodym.py", "--split", split], cwd=REPO, stdout=txt,
                                    stderr=subprocess.STDOUT, env={**os.environ, "PYTHONUTF8": "1"}).returncode
            after = _latest_bodym_run()
            if rc != 0 or after == before:
                print(f"[bodym] {split} FAILED, see {out_dir / f'bodym_{split}.txt'}")
                return 2
            bodym_runs.append(BODYM_REPORTS / f"run_{after:03d}")
    _append(_entry(args.label, app_csv, bodym_runs))
    entries = _load()
    return compare(entries[-2], entries[-1]) if len(entries) > 1 else 0


def cmd_record(args: argparse.Namespace) -> int:
    _append(_entry(args.label, Path(args.app).resolve() if args.app else None,
                   [Path(p).resolve() for p in args.bodym]))
    return 0


# ── comparing ─────────────────────────────────────────────────────────────────

def _mark(delta: float | None, eps: float) -> str:
    if delta is None:
        return " "
    return "✓" if delta < -eps else ("✗" if delta > eps else "·")


def _fmt(v: float | None) -> str:
    return f"{v:6.2f}" if v is not None else "     —"


def compare(old: dict, new: dict) -> int:
    from webui.envelope import DISPLAY_MEASUREMENT_IDS

    regressions = 0
    print(f"\n=== '{old['label']}' ({old['git']['sha']}) -> '{new['label']}' ({new['git']['sha']}"
          f"{', dirty' if new['git']['dirty'] else ''}) ===")
    if old.get("versions") != new.get("versions"):
        print(f"  ! package versions changed: {old.get('versions')} -> {new.get('versions')}")
    if "app" in old and "app" in new:
        oe, ne = old["app"]["errors"], new["app"]["errors"]
        pairs_old = {(s, m) for s, ms in oe.items() for m in ms}
        pairs_new = {(s, m) for s, ms in ne.items() for m in ms}
        common = pairs_old & pairs_new
        print("\n[app GT] real photos, end to end")
        if pairs_old != pairs_new:
            gone = sorted({s for s, _ in pairs_old - pairs_new})
            added = sorted({s for s, _ in pairs_new - pairs_old})
            print(f"  ! scored set changed: {len(pairs_old)} -> {len(pairs_new)} observations "
                  f"(subjects losing values: {gone or '-'}, gaining: {added or '-'})")
            print("    Full-set numbers are NOT comparable. Deltas below use the "
                  f"{len(common)} observations present in both runs.")
        displayed = set(DISPLAY_MEASUREMENT_IDS)
        so = _summarize(oe, displayed, common)
        sn = _summarize(ne, displayed, common)
        print(f"  {'':34}{'n':>5} {'old':>6} {'new':>6} {'Δ':>6}")
        for key in ("displayed", "hidden", "overall"):
            d = sn[key]["mae"] - so[key]["mae"] if so[key]["mae"] is not None else None
            mark = _mark(d, EPS_APP)
            regressions += key == "displayed" and mark == "✗"
            print(f"  {key.upper():34}{sn[key]['n']:5d} {_fmt(so[key]['mae'])} {_fmt(sn[key]['mae'])} "
                  f"{d:+6.2f} {mark}" if d is not None else f"  {key.upper():34}    —")
        for mid in sorted(set(so["per_measurement"]) | set(sn["per_measurement"])):
            a, b = so["per_measurement"].get(mid), sn["per_measurement"].get(mid)
            if not a or not b or a["mae"] is None:
                continue
            d = b["mae"] - a["mae"]
            tag = "" if mid in displayed else " (hidden)"
            print(f"  {(mid + tag)[:34]:34}{b['n']:5d} {_fmt(a['mae'])} {_fmt(b['mae'])} {d:+6.2f} "
                  f"{_mark(d, EPS_APP)}  bias {a['bias']:+.1f} -> {b['bias']:+.1f}")
        full_o, full_n = old["app"]["summary"]["overall"], new["app"]["summary"]["overall"]
        print(f"  full sets: old n={full_o['n']} MAE {full_o['mae']:.2f} | new n={full_n['n']} MAE {full_n['mae']:.2f}"
              f" | new displayed n={new['app']['summary']['displayed']['n']} "
              f"MAE {new['app']['summary']['displayed']['mae']:.2f}")
    for split in sorted(set(old.get("bodym", {})) & set(new.get("bodym", {}))):
        bo, bn = old["bodym"][split], new["bodym"][split]
        print(f"\n[BodyM {split}] perfect silhouettes ({bo['run']} -> {bn['run']})")
        if bo["subjects"] != bn["subjects"]:
            print(f"  ! subject count changed {bo['subjects']} -> {bn['subjects']}: not comparable")
            continue
        for p in bn["overall"]:
            if p not in bo["overall"]:
                print(f"  {p:22} new pipeline, overall {bn['overall'][p]:.2f}")
                continue
            d = bn["overall"][p] - bo["overall"][p]
            mark = _mark(d, EPS_BODYM)
            regressions += mark == "✗"
            print(f"  {p:22} {bo['overall'][p]:6.2f} -> {bn['overall'][p]:6.2f} {d:+6.2f} {mark}")
    print("\nWhat each benchmark can see:\n" + DETECTION_MAP)
    print(f"\n{'REGRESSION' if regressions else 'no regression'} "
          f"(displayed app MAE or BodyM overall worse by more than {EPS_APP} / {EPS_BODYM} cm)")
    return 1 if regressions else 0


def _pick(entries: list[dict], key: str) -> dict:
    if key.lstrip("-").isdigit():
        return entries[int(key)]
    matches = [e for e in entries if e["label"] == key]
    if not matches:
        sys.exit(f"no ledger entry labelled {key!r}")
    return matches[-1]


def cmd_compare(args: argparse.Namespace) -> int:
    entries = _load()
    if len(entries) < 2:
        sys.exit("need at least two ledger entries")
    old = _pick(entries, args.old) if args.old else entries[-2]
    new = _pick(entries, args.new) if args.new else entries[-1]
    return compare(old, new)


def cmd_list(_: argparse.Namespace) -> int:
    for i, e in enumerate(_load()):
        app = e.get("app", {}).get("summary", {})
        d, o = app.get("displayed", {}), app.get("overall", {})
        bodym = " ".join(f"{s}:{b['overall'].get('A_raw', 0):.2f}" for s, b in sorted(e.get("bodym", {}).items()))
        print(f"#{i:<3} {e['time'][:16]} {e['git']['sha']}{'*' if e['git']['dirty'] else ' '} "
              f"displayed {d.get('mae') or 0:.2f} (n={d.get('n', 0)}) overall {o.get('mae') or 0:.2f} "
              f"(n={o.get('n', 0)})  BodyM A_raw {bodym or '-'}  {e['label']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run both benchmarks, append to the ledger, compare with the previous entry")
    r.add_argument("--label", required=True)
    r.add_argument("--device", default="cpu")
    r.add_argument("--skip-bodym", action="store_true", help="only when the change cannot affect BodyM")
    r.set_defaults(func=cmd_run)
    rec = sub.add_parser("record", help="append existing reports without re-running")
    rec.add_argument("--label", required=True)
    rec.add_argument("--app")
    rec.add_argument("--bodym", nargs="*", default=[])
    rec.set_defaults(func=cmd_record)
    c = sub.add_parser("compare")
    c.add_argument("old", nargs="?")
    c.add_argument("new", nargs="?")
    c.set_defaults(func=cmd_compare)
    sub.add_parser("list").set_defaults(func=cmd_list)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
