"""Figures for the FitMeasureAI thesis (Chapter 3).

Print-oriented: Times New Roman, Ukrainian labels, 300 dpi, grayscale-safe
(3-slot validated palette + direct value labels + hatching where colour alone
would carry identity).

Sources — all local, aggregates only leave this machine:
  eval_grid.csv          per-(subject, measurement) errors, production combo
  runs/eval/ledger.jsonl ablation steps + BodyM
  v3_cpu*.json           per-stage speed at 2/4/8/16 threads
  runs/{pose,seg}/results.csv  training curves
"""
from __future__ import annotations

import csv
import json
import random
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
S = Path(__file__).resolve().parent.parent.parent / "runs" / "paper"   # local inputs, gitignored
OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Validated 3-slot categorical palette (dataviz skill, all-pairs PASS)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d8d4"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.edgecolor": INK2,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def f(v, nd=2, sign=False):
    """Ukrainian decimal comma."""
    t = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return t.replace(".", ",")


def comma_axis(ax, which="x", nd=1):
    """Ukrainian decimal comma on tick labels. Trailing zeros are dropped only
    after the separator — never from the integer part (90 must stay 90)."""
    from matplotlib.ticker import FuncFormatter

    def _fmt(v, _p):
        t = f"{v:.{nd}f}"
        if "." in t:
            t = t.rstrip("0").rstrip(".")
        if t in ("", "-"):
            t = "0"
        return t.replace(".", ",")

    (ax.xaxis if which == "x" else ax.yaxis).set_major_formatter(FuncFormatter(_fmt))


UK = {
    "chest_circumference": "Обхват грудей",
    "waist_circumference": "Обхват талії",
    "hip_circumference": "Обхват стегон",
    "thigh_circumference": "Обхват стегна",
    "neck_circumference": "Обхват шиї",
    "upper_arm_circumference": "Обхват біцепса",
    "neck_base_height": "Висота основи шиї",
    "chest_width_front": "Ширина грудей спереду",
    "back_width_scapular": "Ширина спини",
    "shoulder_slope_width": "Ширина плечового ската",
    "front_length_to_waist": "Довжина переду до талії",
    "back_length_to_waist": "Довжина спини до талії",
    "leg_length_outer_seam": "Зовнішня довжина ноги",
    "leg_length_inner_seam": "Внутрішня довжина ноги",
}


def _person_by_subject() -> dict[str, str]:
    """subject_id -> person. subjects.csv has one row per photo PAIR, and several
    pairs can belong to the same volunteer, so the two are not interchangeable."""
    path = REPO / "supabase-dump" / "subjects.csv"
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {r["subject_id"]: r.get("person") or r["subject_id"] for r in csv.DictReader(fh)}


def load_rows() -> list[dict]:
    persons = _person_by_subject()
    rows, sec = [], None
    with open(S / "eval_grid.csv", encoding="utf-8") as f:
        for r in csv.reader(f):
            if r and r[0].startswith("#"):
                sec = r[0]
                continue
            if sec == "# per-(combo, subject, measurement)" and len(r) == 7 and r[0] == "coco+rama+off":
                rows.append({"subject": r[1], "person": persons.get(r[1], r[1]),
                             "sex": r[2], "mid": r[3],
                             "pred": float(r[4]), "gt": float(r[5]), "err": float(r[6])})
    return rows


def boot_ci(values_by_subject: dict[str, list[float]], stat, n_boot=5000, seed=7):
    """Bootstrap over PEOPLE (the independent unit), not observations and not photo
    pairs — two pairs of the same volunteer are not independent draws."""
    rng = random.Random(seed)
    subs = list(values_by_subject)
    out = []
    for _ in range(n_boot):
        pick = [rng.choice(subs) for _ in subs]
        pool = [v for s in pick for v in values_by_subject[s]]
        if pool:
            out.append(stat(pool))
    out.sort()
    return out[int(0.025 * len(out))], out[int(0.975 * len(out))]


# ── Fig 1: per-measurement MAE with bootstrap CI + bias ─────────────────────
def fig_mae(rows):
    by_mid: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_mid.setdefault(r["mid"], {}).setdefault(r["person"], []).append(r["err"])

    items = []
    for mid, per_sub in by_mid.items():
        errs = [e for v in per_sub.values() for e in v]
        mae = st.fmean(abs(e) for e in errs)
        bias = st.fmean(errs)
        lo, hi = boot_ci(per_sub, lambda p: st.fmean(abs(x) for x in p))
        items.append((mid, mae, bias, lo, hi, len(errs)))
    items.sort(key=lambda t: t[1])

    n_people = len({r["person"] for r in rows})
    n_pairs = len({r["subject"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.8), gridspec_kw={"width_ratios": [1.55, 1]})
    y = np.arange(len(items))
    labels = [f"{UK.get(m, m)}" for m, *_ in items]

    ax = axes[0]
    mae = [t[1] for t in items]
    lo = [t[1] - t[3] for t in items]
    hi = [t[4] - t[1] for t in items]
    ax.barh(y, mae, color=C1, height=0.62, zorder=3)
    ax.errorbar(mae, y, xerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=3, zorder=4)
    for i, t in enumerate(items):
        ax.text(t[4] + 0.12, i, f(t[1]), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"MAE, см  (95 % ДІ, бутстреп по {n_people} особах; {n_pairs} пар фото)")
    ax.set_xlim(0, max(t[4] for t in items) + 0.9)
    ax.grid(axis="y", visible=False)
    ax.set_title("а) Середня абсолютна похибка", fontsize=11.5, loc="left")

    ax = axes[1]
    bias = [t[2] for t in items]
    cols = [C2 if b < 0 else C3 for b in bias]
    ax.barh(y, bias, color=cols, height=0.62, zorder=3)
    ax.axvline(0, color=INK2, linewidth=1.0, zorder=4)
    for i, b in enumerate(bias):
        ax.text(b + (0.1 if b >= 0 else -0.1), i, f(b, sign=True), va="center",
                ha="left" if b >= 0 else "right", fontsize=9.5, color=INK)
    ax.set_yticks(y, [])
    ax.set_xlabel("Систематичне зміщення (bias), см")
    m = max(abs(min(bias)), abs(max(bias))) + 1.1
    ax.set_xlim(-m, m)
    ax.grid(axis="y", visible=False)
    ax.set_title("б) Зміщення: завищення / заниження", fontsize=11.5, loc="left")
    for a in axes:
        comma_axis(a, "x", 0)

    fig.savefig(OUT / "fig_mae_per_measurement.png")
    plt.close(fig)
    return items


# ── Fig 2: Bland–Altman for the three key circumferences ────────────────────
def fig_bland_altman(rows):
    keys = ["chest_circumference", "waist_circumference", "hip_circumference"]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9), sharey=False)
    for ax, mid, col in zip(axes, keys, (C1, C2, C3)):
        sel = [r for r in rows if r["mid"] == mid]
        mean = np.array([(r["pred"] + r["gt"]) / 2 for r in sel])
        diff = np.array([r["pred"] - r["gt"] for r in sel])
        md, sd = diff.mean(), diff.std(ddof=1)
        ax.scatter(mean, diff, s=34, facecolor=col, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(md, color=INK, linewidth=1.4, zorder=4)
        ax.axhline(md + 1.96 * sd, color=INK2, linewidth=1.0, linestyle="--", zorder=4)
        ax.axhline(md - 1.96 * sd, color=INK2, linewidth=1.0, linestyle="--", zorder=4)
        span = max(abs(diff).max(), abs(md) + 1.96 * sd)
        ax.set_ylim(-span * 1.45, span * 1.45)
        x1 = ax.get_xlim()[1]
        for val, txt in ((md + 1.96 * sd, "+1,96·SD"), (md, "середнє"), (md - 1.96 * sd, "−1,96·SD")):
            ax.text(x1, val, " " + txt, va="center", ha="left", fontsize=7.6, color=INK2, clip_on=False)
        ax.text(0.03, 0.03, "$\\bar d$ = " + f(md, sign=True) + " см\n1,96·SD = " + f(1.96 * sd) + " см",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.6, color=INK2,
                bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.3", linewidth=0.7))
        ax.set_title(f"{UK[mid]}  (n={len(sel)})", fontsize=11)
        ax.set_xlabel("Середнє (фото, стрічка), см")
        comma_axis(ax, "x", 0)
        comma_axis(ax, "y", 0)
    axes[0].set_ylabel("Різниця (фото − стрічка), см")
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(OUT / "fig_bland_altman.png")
    plt.close(fig)


# ── Fig 3: ablation ─────────────────────────────────────────────────────────
def fig_ablation():
    led = [json.loads(l) for l in open(REPO / "runs/eval/ledger.jsonl", encoding="utf-8")]
    steps = [
        ("Базова конфігурація", 0),
        ("+ гейт якості еталона", 1),
        ("+ пріори ANSUR (шия, спина)", 2),
        ("+ пріор ANSUR (біцепс)", 3),
        ("+ перефіт корекцій (LOO)", 4),
    ]
    vals = [led[i]["app"]["summary"]["overall"]["mae"] for _, i in steps]
    ns = [led[i]["app"]["summary"]["overall"]["n"] for _, i in steps]
    names = [n + "\n(n=" + str(k) + " мірок)" for (n, _), k in zip(steps, ns)]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=[C1] * (len(vals) - 1) + [C3], height=0.6, zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.09, yy, f(v), va="center", fontsize=10, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("MAE, см (корпус етапу розробки: 16 суб'єктів)")
    comma_axis(ax, "x", 0)
    ax.set_xlim(0, max(vals) + 0.8)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig_ablation.png")
    plt.close(fig)
    return vals


# ── Fig 4: speed per stage vs threads ───────────────────────────────────────
def fig_speed():
    data = {}
    for t in (2, 4, 8):
        p = S / f"v3_cpu{t}.json"
        if p.is_file():
            data[t] = json.loads(p.read_text(encoding="utf-8"))
    gp = S / "gpu.json"
    if gp.is_file():
        data["GPU"] = json.loads(gp.read_text(encoding="utf-8"))
    threads = [k for k in (2, 4, 8) if k in data] + (["GPU"] if "GPU" in data else [])
    stages = [("Поза (2 кадри)", ("pose_front", "pose_side"), C1, ""),
              ("Сегментація (2 кадри)", ("seg_front", "seg_side"), C2, "//"),
              ("Геометрія та обхвати", ("calibrate", "extract", "circumferences", "validate"), C3, "..")]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    y = np.arange(len(threads))[::-1]
    left = np.zeros(len(threads))
    for name, keys, col, hatch in stages:
        vals = np.array([sum(data[t]["per_stage"].get(k, {}).get("median_s", 0.0) for k in keys) for t in threads])
        ax.barh(y, vals, left=left, color=col, height=0.58, label=name, zorder=3,
                hatch=hatch, edgecolor="white", linewidth=1.2)
        left += vals
    for yy, t in zip(y, threads):
        tot = data[t]["total"]["median_s"]
        ax.text(tot + 0.05, yy, f"{f(tot)} с", va="center", fontsize=10, color=INK)
    names = {2: "CPU, 2 потоки", 4: "CPU, 4 потоки", 8: "CPU, 8 потоків", "GPU": "GPU (RTX 5070 Ti)"}
    ax.set_yticks(y, [names[t] for t in threads])
    ax.set_xlabel("Медіанний час повного циклу, с (вхід 576×576, n=15 на конфігурацію)")
    ax.set_xlim(0, max(data[t]["total"]["median_s"] for t in threads) + 0.55)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    comma_axis(ax, "x", 1)
    geo = st.fmean(sum(data[t]["per_stage"].get(k, {}).get("median_s", 0.0)
                       for k in ("calibrate", "extract", "circumferences", "validate")) for t in threads)
    ax.text(0.0, -0.30, f"Геометрія та обхвати — {geo*1000:.0f} мс (< 0,5 % циклу), на діаграмі не розрізняється.",
            transform=ax.transAxes, fontsize=8.8, color=INK2)
    fig.savefig(OUT / "fig_speed_stages.png")
    plt.close(fig)
    return data


# ── Fig 5: training curves (pose + seg) ─────────────────────────────────────
def fig_training():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, run, ptitle, mkey in (
        (axes[0], "pose", "а) Модель ключових точок (YOLO11n-pose)", "metrics/mAP50-95(P)"),
        (axes[1], "seg", "б) Модель сегментації (YOLO11n-seg)", "metrics/mAP50-95(M)"),
    ):
        rr = list(csv.DictReader(open(REPO / f"runs/{run}/results.csv")))
        rr = [{k.strip(): v for k, v in r.items()} for r in rr]
        ep = [int(r["epoch"]) for r in rr]
        tr = [sum(float(r[k]) for k in r if k.startswith("train/") and r[k]) for r in rr]
        va = [sum(float(r[k]) for k in r if k.startswith("val/") and r[k]) for r in rr]
        ax.plot(ep, tr, color=C1, linewidth=2, marker="o", markersize=4, label="Втрати, навчання")
        ax.plot(ep, va, color=C2, linewidth=2, marker="s", markersize=4, linestyle="--", label="Втрати, валідація")
        ax.set_xlabel("Епоха")
        ax.set_ylabel("Сумарна функція втрат")
        ax.set_title(ptitle, fontsize=11, loc="left")
        ax2 = ax.twiny()  # not a second y-scale: separate axis object only for the metric line
        ax2.remove()
        m = [float(r[mkey]) for r in rr]
        axm = ax.inset_axes([0.46, 0.5, 0.5, 0.42])
        axm.plot(ep, m, color=C3, linewidth=2)
        axm.set_title("mAP50-95", fontsize=8.5, color=INK2, pad=2)
        axm.tick_params(labelsize=7.5)
        axm.set_ylim(min(m) - 0.01, 1.003)
        axm.grid(True, color=GRID, linewidth=0.5)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        comma_axis(ax, "y", 1)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(OUT / "fig_training_curves.png")
    plt.close(fig)


# ── Fig 6: regressor training (from notebook log) ───────────────────────────
REG_LOG = [(1, 4382.38, 4406.35), (20, 1029.88, 965.48), (40, 67.86, 63.54), (60, 49.84, 52.69),
           (80, 49.30, 51.44), (100, 46.73, 49.72), (120, 47.15, 51.13), (140, 44.87, 48.10),
           (160, 45.43, 48.12), (180, 43.41, 49.17), (200, 43.82, 47.68), (220, 43.35, 48.04),
           (240, 42.61, 58.72), (260, 42.22, 47.98), (280, 42.65, 50.46), (300, 42.53, 48.89)]
REG_PER = {"Обхват шиї": (2.69, 4.25), "Обхват талії": (7.18, 9.20), "Обхват стегон": (5.88, 7.68),
           "Обхват стегна": (4.46, 7.38), "Обхват гомілки": (1.21, 1.70), "Обхват зап'ястя": (0.88, 1.58)}


def fig_regressor():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ep = [r[0] for r in REG_LOG]
    ax.plot(ep, [r[1] for r in REG_LOG], color=C1, linewidth=2, marker="o", markersize=4, label="Навчальна вибірка")
    ax.plot(ep, [r[2] for r in REG_LOG], color=C2, linewidth=2, marker="s", markersize=4,
            linestyle="--", label="Валідаційна вибірка")
    ax.set_yscale("log")
    ax.set_xlabel("Епоха")
    ax.set_ylabel("MSE, см² (лог. шкала)")
    ax.set_title("а) Навчання регресора обхватів (синтетика, n=3 395)", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.annotate("найкраща val RMSE = 6,84 см", xy=(200, 47.68), xytext=(120, 180),
                fontsize=9, color=INK2, arrowprops=dict(arrowstyle="->", color=INK2, linewidth=0.9))

    ax = axes[1]
    names = list(REG_PER)
    y = np.arange(len(names))[::-1]
    mae = [REG_PER[n][0] for n in names]
    ax.barh(y, mae, color=C1, height=0.6, zorder=3)
    for yy, v in zip(y, mae):
        ax.text(v + 0.12, yy, f(v), va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel("MAE на відкладеній вибірці, см")
    ax.set_xlim(0, max(mae) + 1.2)
    ax.grid(axis="y", visible=False)
    ax.set_title("б) Похибка регресора по обхватах", fontsize=11, loc="left")
    comma_axis(ax, "x", 0)
    fig.subplots_adjust(wspace=0.52)
    fig.savefig(OUT / "fig_regressor.png")
    plt.close(fig)


# ── Fig 7: BodyM ────────────────────────────────────────────────────────────
def fig_bodym():
    led = [json.loads(l) for l in open(REPO / "runs/eval/ledger.jsonl", encoding="utf-8")][-1]["bodym"]
    variants = [("Еліпс Рамануджана", "A_raw", C1, ""),
                ("Еліпс + корекції", "B_corrected", C2, "//"),
                ("Навчена модель обхвату\n(лише для оцінювання)", "D_learned_girth", C3, "..")]
    splits = [("testA (n=87)", "testA"), ("testB (n=400)", "testB")]
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    x = np.arange(len(splits))
    w = 0.26
    for i, (name, key, col, hatch) in enumerate(variants):
        vals = [led[s]["overall"][key] for _, s in splits]
        pos = x + (i - 1) * w
        ax.bar(pos, vals, width=w * 0.92, color=col, label=name, zorder=3,
               hatch=hatch, edgecolor="white", linewidth=1.1)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.12, f(v), ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x, [n for n, _ in splits])
    ax.set_ylabel("MAE, см")
    ax.set_ylim(0, 11)
    comma_axis(ax, "y", 0)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.savefig(OUT / "fig_bodym.png")
    plt.close(fig)


if __name__ == "__main__":
    rows = load_rows()
    items = fig_mae(rows)
    fig_bland_altman(rows)
    abl = fig_ablation()
    sp = fig_speed()
    fig_training()
    fig_regressor()
    fig_bodym()
    print("figures:", sorted(p.name for p in OUT.glob("*.png")))
    print("\n=== Таблиця 3.1 (регенеровано з того ж прогону) ===")
    for mid, mae, bias, lo, hi, n in sorted(items, key=lambda t: -t[1]):
        print(f"{UK.get(mid,mid):28} {mae:5.2f}  [{lo:.2f}; {hi:.2f}]  {bias:+5.2f}  n={n}")
    allerr = [r["err"] for r in rows]
    print(f"{'ЗАГАЛОМ':28} {st.fmean(abs(e) for e in allerr):5.2f}  bias={st.fmean(allerr):+.2f}  n={len(allerr)}")
