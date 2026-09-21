# Метрики для наукової роботи: що вже є, де лежить, як відтворити

> Запит (2026-09-21): графіки навчання регресора, швидкість пайплайну, «будь-які метрики».
> Правило: у роботу йдуть лише агрегати. `runs/eval/ledger.jsonl`, `runs/eval/track/*`, `supabase-dump/`
> містять пер-суб'єктні похибки реальних людей — не копіювати, не комітити (CLAUDE.md § Accuracy changes).

## 0. TL;DR

| Що | Є зараз | Дія |
|---|---|---|
| Графік навчання регресора | Лог 300 епох + scatter pred-vs-GT у `notebooks/05_train_regressor_colab.ipynb` (cells 13/15/17/18) | Витягти лог у CSV, перемалювати train/val loss (§2.3) |
| Графіки навчання pose / seg | `runs/pose/results.{csv,png}` (20 епох), `runs/seg/results.{csv,png}` (15 епох) | Готово; але **ці ваги не в продакшні** (§2.1–2.2) |
| Швидкість пайплайну | `scripts/ansur/time_pipe.py` → 1.0 с/пара на 8 потоках; per-stage `webui/_timing.py` у логах | Прогнати на 2 vCPU (HF Space) + таблиця по стадіях (§3) |
| Точність на реальних фото | `runs/eval/ledger.jsonl`: 8 записів, MAE 6.03 → 3.32 см | Таблиця абляцій + bootstrap CI (§1) |
| Точність на BodyM | ledger + `eval/reports/run_0xx`: testA (87) / testB (400), 4 варіанти | Таблиця (§1.2) |
| ANSUR II: бюджет похибки, чутливість | `scripts/ansur/{ell,reg,noise}.py`, `docs/accuracy-approaches-survey.md` §2 | Відтворити, взяти графіки (§4) |
| Якість GT | `pointsx/gt_sanity.py` + `[skip]`/`WARNING` у `runs/eval/track/007*/app.txt` | Таблиця виключень (§5) |
| Плечовий скат у формі датасету | **Додано 2026-09-21** (`shoulder_slope_width`, 8–25 см) | Завтра міряти; це перший GT для `shoulder_slope_width` (§6) |

## 1. Точність (головний науковий актив)

### 1.1 App-GT: реальні фото, повний пайплайн

Джерело: `runs/eval/ledger.jsonl` (кожен запис: label, git sha, версії torch/ultralytics/numpy/opencv,
sha256 корпусу, overall/displayed/hidden/per-measurement MAE + bias). Відтворення:
`.venv/Scripts/python scripts/eval_track.py list` / `compare`.

Корпус: **16 суб'єктів** (12 F / 4 M), 206 пар «GT — прогноз» по 13 вимірах (ті, що мають GT у формі).

| # | Крок | n | MAE, см | bias, см |
|---|---|---|---|---|
| 1 | baseline (до GT-гейту) | 219 | 6.03 | +1.41 |
| 2 | GT cross-measurement gate (`gt_sanity.py`) | 205 | 5.45 | +0.72 |
| 3 | + шия та ширина спини з ANSUR-пріорів | 205 | 3.56 | −0.28 |
| 4 | + обхват плеча з ANSUR-пріору | 206 | 3.52 | −0.49 |
| 5 | + перефіт корекцій, leave-one-out | 206 | **3.32** | +0.08 |
| 6–8 | A/B ряду грудей, ревєрт, прод-набір | 206 | 3.32 | +0.08 |

Пер-вимір (запис 8): back_length 1.96 · neck_circ 2.19 · back_width 2.61 · neck_base_height 2.71 ·
thigh 3.21 · outer_seam 3.30 · inner_seam 3.40 · hip 3.64 · waist 3.71 · front_length 3.87 ·
upper_arm 3.92 · chest 4.01 · chest_width 4.66 см.

Для статті: n завжди поруч з числом; bootstrap CI по суб'єктах (при n=16 інтервал буде широкий — краще
показати самим, ніж отримати питання від рецензента). Per-subject bootstrap: `eval.py` віддає
`ErrorRow` — CI рахувати з них локально, у doc — лише інтервал.

### 1.2 BodyM: ідеальні силуети, лише еліпс

Джерело: `eval/bodym.py`, звіти `eval/reports/run_033` (testA, 87) і `run_034` (testB, 400);
агрегати в ledger. Ліцензія CC BY-NC → **лише оцінка**, константи D_learned_girth не шиппаються
(memory: bodym-eval-only-license).

| Варіант | testA MAE | testB MAE |
|---|---|---|
| A_raw (Ramanujan) | 6.45 | 9.45 |
| B_corrected (прод. корекції) | 5.76 | 6.26 |
| C_gated5% | 5.82 | 6.83 |
| D_learned_girth (BodyM-fit, non-shippable) | 3.75 | 4.61 |

Читати як: BodyM бачить лише математику ширина→обхват (chest/waist/hip/thigh), не позу й не сегментацію.

### 1.3 Регресор vs еліпс — це і є «своє натреноване»

Регресор (MLP 28→64→32→6, `pointsx/regression/model.py`) навчений на **синтетиці** (3 395 тіл, `05_*` cell 13):
val RMSE 6.84 см; на held-out: neck 2.69 / waist 7.18 / hip 5.88 / thigh 4.46 / calf 1.21 / wrist 0.88 см MAE.
На реальних фото програв еліпсу з корекціями (3.32 см), тому в продакшні вимкнений (`POINTSX_USE_REGRESSOR` не встановлено → еліпс; `webui/config.py:90`). Це не «відсутність новизни», а **абляція з негативним результатом**: навчена голова
на синтетиці не переносить на реальні фото + одяг; аналітика + пріори ANSUR + LOO-корекції — переносять.
Так це й формулювати.

## 2. Криві навчання

### 2.1 Pose (YOLO11n-pose, LV-MHP-v2 → 16 kp, синтетика)

`runs/pose/results.csv` — 20 епох (`epoch, train/*_loss, val/*_loss, metrics/mAP50(-95)(B|P), lr`), `results.png`
— готовий графік. Фінал: Box mAP50-95 0.977, Pose mAP50-95 0.989 на val = 1 728 **синтетичних** кадрів.
Швидкість (T4): 2.7 мс inference/кадр. Ноутбук `02_train_pose_colab.ipynb` cells 18–22 містить curves,
`val_batch0_pred.jpg`, confusion matrix.

⚠ Продакшн бере `models/yolo26-pose.pt` (COCO-17, backend `coco`), а не `runs/pose/best.pt`
(`webui/config.py:85`; custom backend вимкнений, ваги `models/pose-cus.pt` відсутні). Число 0.989 —
на синтетичному val, переносу на реальні фото не міряли. У роботі: як експеримент, з цією застереженням.

### 2.2 Seg (YOLO11n-seg, синтетичні силуети)

`runs/seg/results.csv` — 15 епох, `results.png`. Фінал: Mask mAP50-95 0.944, Box 0.995. Та сама застереження:
продакшн — `models/yolo12l-person-seg-extended.pt`, не `runs/seg/best.pt`.

### 2.3 Регресор

Немає CSV — `regression/train.py` лише логує кожні 20 епох. Лог збережено у виводі cell 15 ноутбука
(16 точок: epoch, train_loss, val_loss, val_RMSE). Достатньо для графіка. Scatter pred-vs-GT по 6 обхватах —
cell 18 (`image/png` вбудовано в ipynb; витягти `nbformat`/`jupyter nbconvert --to markdown`).

Якщо потрібна повна крива (300 точок): додати в `train.py` запис history у CSV (`--history out.csv`)
і перезапустити на Colab (~хвилини на 3 395 × 28 фічах).

## 3. Швидкість

Виміряно (`docs/accuracy-approaches-survey.md` рядок 51, `scripts/ansur/time_pipe.py`):
CPU 24-core / 8 потоків / 576×576 — **1.0 с на пару фото**, import 6 с, завантаження моделей 0.4 с.
Warm-up у логах webui: `pose:coco=8.48s, seg=1.26s` (перший прогін, включно з JIT/аллокацією).

Що ще дістати:
- Per-stage: `webui/_timing.py` вже рахує `pose_front / pose_side / seg_front / seg_side / total`,
  логує `Measurement timings — …` (WARNING) на кожен запит. `time_pipe.py` → додати `tm.as_dict()` і
  прогнати 10 пар, звіт median ± IQR.
- Цільове залізо: HF Space (2 vCPU) — очікувано 3–5 с (survey §3), треба виміряти саме там
  (`scripts/ci/space_smoke.py` можна розширити таймером).
- Розмір моделей і параметри: pose 2.85M params / 7.3 GFLOPs (з логу), seg — з `ultralytics` summary.
- Вхід капнутий 1280 px (`pipeline.downscale_for_inference`) — вказати як частину протоколу.

## 4. ANSUR II: бюджет похибки (публічний домен, відтворюване)

`scripts/ansur/README.md`; дані `a.csv` / `m.csv` (1 986 F / 4 082 M).
- `ell.py` — bias еліпса Рамануджана і масштаб k per site×sex.
- `reg.py` — 5-fold CV: H | H+W+BMI | ellipse | ellipse+H+W. Ключовий висновок (memory: ansur-shippable-prior):
  вага — найбільший важіль; ширини додають ≈0 понад H+W.
- `noise.py` — чутливість до шуму ширин і «інфляції» від одягу.
- `priors.py` / `girth.py` — звідки взялися пріори шиї / спини / плеча (кроки 3–4 таблиці §1.1).
Результати вже зведені у survey §2.1–2.2 — брати звідти, графіки перемалювати зі скриптів.

## 5. Якість ground truth

- `pointsx/gt_sanity.py`: 7 ratio-bounds до зросту/між вимірами; >2 порушень → суб'єкт виключено.
- `build_eval_csv.py`: 27 надсилань → 16 суб'єктів, 11 відкинуто (placeholder-рядки з 1/9/тест).
  Приклад: `c90e3c9a` — шия 55 см (0.348 від зросту) відкинута.
- Для статті — таблиця: подано / виключено / причини; і пункт «похибка самого GT» (survey §2.3) — зараз
  без числа для нашого корпусу.

## 6. Що зібрати завтра (не відновлюється заднім числом)

1. **Плечовий скат** — поле `shoulder_slope_width` у формі (від основи шиї до плечової точки, один бік,
   8–25 см). Це єдине поле з прямим відповідником у envelope; `shoulder_width` (ширина плечей) далі не
   порівнюється. Після першої сесії: додати ratio-bound у `gt_sanity.py` за спостереженими значеннями і
   перевірити коефіцієнт 0.8 у `measurements.py:144`.
2. **Вага** — у формі немає. За ANSUR (§4) це найбільший важіль точності; без ваги регресія H+W недоступна
   на нашому корпусі. Хоча б записувати окремо.
3. **Повтор другим спостерігачем** на 5–10 людях (те саме, інша людина з сантиметром) — дає
   inter-observer розкид, тобто нижню межу досяжної MAE. Найсильніший «методичний» результат за один день.
4. Умови зйомки: відстань, камера, одяг (tight/loose) — як мітки до запису.

## 6.5 Експерименти, які можна прогнати вже зараз (відповідь на «нема новизни»)

Обидва перетворюють mAP на синтетиці (§2) у дельту на реальному корпусі через наявний harness:

1. **Seg A/B**: `POINTSX_SEG_MODEL=runs/seg/best.pt` → `scripts/eval_track.py run --label "custom seg (synthetic finetune)"`
   і порівняти з записом 8 (`yolo12l-person-seg-extended.pt`). Це і є «похибка на кастомному силуеті».
2. **Pose A/B**: `runs/pose/best.pt` — той самий 16-kp LV-MHP finetune, який очікує `POINTSX_POSE_MODEL_CUSTOM`
   (`models/pose-cus.pt`, зараз відсутній). Скопіювати → `pointsx-eval --pose-backend custom` / `eval_track.py run`.
3. **Регресор на реальних фото**: `POINTSX_USE_REGRESSOR=1 POINTSX_REGRESSION_MODEL=runs/reg/circumference_regressor.pt`
   → ще один рядок ledger; зараз «програв» задокументовано лише якісно.

Будь-який результат публікується: програш — той самий чесний негативний результат, що й для регресора (§1.3);
виграш — внесок, який можна шиппати. Ledger + `compare` дають таблицю автоматично.

## 7. Що НЕ можна показувати

- Пер-суб'єктні похибки, фото, будь-що з `supabase-dump/`, `runs/eval/track/*/app.csv`.
- Константи, підігнані на BodyM (`D_learned_girth`), як продакшн-числа.
