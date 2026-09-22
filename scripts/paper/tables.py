"""Chapter-3 tables, ready to paste into the .docx (Ukrainian, decimal comma)."""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
S = Path(__file__).resolve().parent.parent.parent / "runs" / "paper"   # local inputs, gitignored
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figures import UK, boot_ci, f, load_rows

out: list[str] = []
W = out.append


def md(rows: list[list[str]], head: list[str]) -> None:
    W("| " + " | ".join(head) + " |")
    W("|" + "|".join(["---"] * len(head)) + "|")
    for r in rows:
        W("| " + " | ".join(r) + " |")
    W("")


# ── Table 3.1 — measurement accuracy ────────────────────────────────────────
rows = load_rows()
by_mid: dict[str, dict[str, list[float]]] = {}
for r in rows:
    by_mid.setdefault(r["mid"], {}).setdefault(r["subject"], []).append(r["err"])
items = []
for mid, per_sub in by_mid.items():
    errs = [e for v in per_sub.values() for e in v]
    lo, hi = boot_ci(per_sub, lambda p: st.fmean(abs(x) for x in p))
    items.append((mid, st.fmean(abs(e) for e in errs), st.fmean(errs), lo, hi, len(errs)))
items.sort(key=lambda t: -t[1])

W("## Таблиця 3.1 — Похибки антропометричних параметрів")
W("")
W("Джерело: 16 суб'єктів, 206 парних спостережень «фото ↔ ручна мірка», продукційна")
W("конфігурація (COCO-поза + сегментація + еліпс Рамануджана з каліброваними корекціями).")
W("95 % довірчий інтервал — бутстреп (5 000 ресемплів) по **суб'єктах**, а не по спостереженнях.")
W("")
md([[UK.get(m, m), f(mae), f"[{f(lo)}; {f(hi)}]", f(bias, sign=True), str(n)] for m, mae, bias, lo, hi, n in items],
   ["Параметр", "MAE, см", "95 % ДІ для MAE", "Bias, см", "n"])
allerr = [r["err"] for r in rows]
W(f"**Загалом: MAE = {f(st.fmean(abs(e) for e in allerr))} см, "
  f"bias = {f(st.fmean(allerr), sign=True)} см, n = {len(allerr)} спостережень (16 суб'єктів).**")
W("")

# ── Table 3.2 — ablation ────────────────────────────────────────────────────
led = [json.loads(line) for line in open(REPO / "runs/eval/ledger.jsonl", encoding="utf-8")]
steps = [("Базова конфігурація", 0), ("+ гейт якості еталонних мірок", 1),
         ("+ антропометричні пріори ANSUR II (шия, ширина спини)", 2),
         ("+ пріор ANSUR II (обхват біцепса)", 3),
         ("+ перефіт корекцій зі схемою leave-one-out", 4)]
W("## Таблиця 3.2 — Внесок окремих складових методу (абляція)")
W("")
base = led[0]["app"]["summary"]["overall"]["mae"]
tr = []
for name, i in steps:
    o = led[i]["app"]["summary"]["overall"]
    tr.append([name, str(o["n"]), f(o["mae"]), f(o["bias"], sign=True),
               "—" if i == 0 else f(100 * (base - o["mae"]) / base, 1) + " %"])
md(tr, ["Конфігурація", "n мірок", "MAE, см", "Bias, см", "Зниження MAE відносно базової"])

# ── Table 3.3 — speed ───────────────────────────────────────────────────────
W("## Таблиця 3.3 — Швидкодія методу (повний цикл «два фото → MeasurementEnvelope»)")
W("")
data = {}
for t in (2, 4, 8):
    p = S / f"v3_cpu{t}.json"
    if p.is_file():
        data[t] = json.loads(p.read_text(encoding="utf-8"))
gpu = S / "gpu.json"
if gpu.is_file():
    data["GPU"] = json.loads(gpu.read_text(encoding="utf-8"))

STAGES = [("Локалізація ключових точок (2 кадри)", ("pose_front", "pose_side")),
          ("Сегментація силуету (2 кадри)", ("seg_front", "seg_side")),
          ("Калібрування, мірки, обхвати, валідація", ("calibrate", "extract", "circumferences", "validate"))]
tr = []
for k, d in data.items():
    ps = d["per_stage"]
    cells = []
    for _, keys in STAGES:
        v = sum(ps.get(kk, {}).get("median_s", 0.0) for kk in keys)
        cells.append(f"{v*1000:.0f}".replace(".", ",") if v < 0.1 else f(v, 2) + " с")
    tot = d["total"]
    label = "GPU (RTX 5070 Ti)" if k == "GPU" else f"CPU, {k} пот."
    cells = [c if c.endswith("с") else c + " мс" for c in cells]
    tr.append([label, *cells, f(tot['median_s']) + " с",
               f"[{f(tot['iqr_lo_s'])}; {f(tot['iqr_hi_s'])}]", str(tot["n"])])
md(tr, ["Конфігурація", *[n for n, _ in STAGES], "Повний цикл, медіана", "IQR, с", "n"])
d0 = next(iter(data.values()))
hw = d0["hardware"]
W(f"Обладнання: {hw['cpu']}, {hw['logical_cores']} логічних ядер, "
  f"torch {hw['torch']}, вхідні зображення {d0['config']['input_sizes'][0]} "
  f"(вхід обмежено 1280 px, інференс — {d0['config']['img_size']} px).")
cs = d0["cold_start_s"]
W(f"Холодний старт (одноразово при запуску контейнера): імпорт бібліотек {f(cs['import'])} с, "
  f"завантаження ваг {f(cs['model_load'])} с, прогрів моделей "
  + ", ".join(f"{k} {f(v)} с" for k, v in cs["warmup"].items()) + ".")
W("")
W("Конфігурація «CPU, 2 пот.» відповідає безкоштовному рівню Hugging Face Spaces, "
  "на якому розгорнуто дослідний зразок; «CPU, 8 пот.» — типовому користувацькому ПК.")
W("")

# ── Table 3.4 — comparison with related work ────────────────────────────────
W("## Таблиця 3.4 — Порівняння з найближчими рішеннями")
W("")
W("Пряме зіставлення коректне лише для однакових мірок і протоколів. Колонка «Що покриває")
W("число» вказує, яку саме частину циклу виміряно — без неї показники швидкодії різних робіт")
W("непорівнянні: 8,4 мс у [10] стосуються лише детектора, а не повного циклу.")
W("")
md([
    ["Foysal та ін. [6]", "front + side + зріст", "95,59 % accuracy (не MAE)", "не наведено", "—"],
    ["BMnet [7]", "2 силуети + метадані", "окремі мірки, інший протокол", "не наведено", "—"],
    ["MeasureNet [9]", "3 RGB + метадані", "MAE окремих обхватів, n=1200", "не наведено", "—"],
    ["Montazerian, Leymarie [10]", "front + side", "середня різниця < ±1 см", "8,4 мс", "лише детектор"],
    ["MeasureXpert [13]", "2 часткові 3D-скани", "0,28–1,55 см для окремих мірок", "не наведено", "—"],
    ["**FitMeasureAI**", "front + side + зріст",
     "**MAE 3,32 см** (13 мірок, n=206, 16 осіб)",
     "**2,16 с** (CPU, 2 пот.) / **0,10 с** (GPU)",
     "**повний цикл: 2 фото → 18 мірок**"],
], ["Метод", "Вхідні дані", "Точність", "Швидкодія", "Що покриває число"])
W("Для зіставних стадій: локалізація ключових точок у FitMeasureAI — 13–14 мс на кадр на GPU, "
  "що відповідає порядку 8,4 мс детектора в [10]. Решта часу циклу — сегментація силуету, "
  "тоді як увесь метричний блок (калібрування, мірки, обхвати, валідація) займає 3 мс, "
  "тобто менше 0,5 % циклу.")
W("")

path = REPO / "docs" / "figures" / "tables.md"
path.write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
