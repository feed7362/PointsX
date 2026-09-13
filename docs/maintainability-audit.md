# Аудит підтримуваності: SSOT, патерни, план міграції

> Продовження `architecture-review.md` (2026-07-20). Чотири незалежні ревʼю (reuse /
> simplification / efficiency / altitude) над вимірювальним ядром + правки, застосовані
> 2026-09-02 без зміни поведінки (snapshot 17 суб'єктів: widths, рядки, BodyMeasurements,
> envelope, warnings — **0 відмінностей** до/після). Прозою — українською, код — англійською.

---

## 0. TL;DR

- **Що зроблено зараз (safe, поведінка ідентична):** `silhouette.py` 701 → 562 рядків;
  видалено мертвий код (`_widest_segment_at_y`, `THIGH_HIP_KNEE_FRACTION`,
  `_extreme_continuous_width_between_y`, `_SEX_CIRCUMFERENCE_OFFSETS_CM`, закоментований
  виклик неіснуючої `_derive_outer_leg_to_floor`, невикористані `x_line/w_s/torso_x`); один
  хелпер замість 9 копій «зібрати валідні L/R y і усереднити» (`keypoints.mean_valid_y`);
  один `_neck_scan_start_y` замість двох копій; `extract_all_widths` збирає front/side у
  окремі словники і зшиває їх один раз (без ідіоми `existing = widths.get(...)` ×9);
  `_find_segments` — одна реалізація (було дві); `inference.py` бере хелпери з
  `MeasurementPipeline`, а не копіює; `postprocess` — один цикл перевірок; `circumference`
  — один список `ELLIPSE_PARTS` замість 12 рядків; `build_eval_csv` — прибрано 8 дубльованих
  ключів словників; названо `WAIST_SIDE_SEARCH_TOP_FRACTION`, `NOSE_MIN_CONFIDENCE`;
  застарілі твердження у коментарях/README/CLAUDE.md виправлено; `scripts/ansur` — спільний
  `_common.py` з **продакшн**-формулою еліпса (було 3 копії з іншою конвенцією півосей).
- **Що НЕ зроблено (зміна поведінки або велика міграція) і має стати наступними кроками:**
  реєстр вимірів (SSOT для id/діапазонів/полів), декларативна таблиця зрізів, винесення
  коефіцієнтів у версіонований JSON, обʼєднання двох оркестраторів, перенесення корекції
  з web-шару у вимірювальний, каталог правдоподібності з однією політикою, кеш/батчинг у
  eval і серверна ефективність. План — §4.

---

## 1. Статус пунктів попереднього аудиту (2026-07-20 → 2026-09-02)

| # | Пункт | Статус | Доказ |
|---|---|---|---|
| 1 | Два репозиторії — одна кодова база | **є** | 8/10 перевірених файлів байт-ідентичні з `../Pointx-backend/src`; `envelope.py`/`inference.py` розійшлися (backend має `warmup()`); `eval/app_pipelines.py`, `scripts/audit_measurements.py` роблять `sys.path.insert` у сусідній репозиторій |
| 2 | Конфліктні діапазони / словник id | **є, гірше** | талія у 4 місцях, 3 значення: `postprocess.py` 55–150, `envelope.py` 50–160, `build_eval_csv.py` 45–170, `synthetic/measurements_gt.py` 45–170; ще й `envelope.py::_derive_chest_circumference` має власне 60–150 при таблиці 60–160 у тому ж файлі |
| 3 | Анатомічні константи inline | **частково** | названо waist/thigh/hip-каппи; після цього аудиту — ще `WAIST_SIDE_SEARCH_TOP_FRACTION`, `NOSE_MIN_CONFIDENCE`; лишаються: `hip_search_y_range` 0.02/0.05/0.06/0.16, thigh 0.12/0.15, side thigh 0.5, calf 0.6, side torso 0.5, band pads 0.18/0.10, gap `>3` px, `max_jump=18`, leg 0.25/0.8, slope ×0.8, `envelope` 0.65 |
| 4 | Коефіцієнти у коді | **є** | `envelope.py` ручні словники; `pointsx-eval --fit-offsets` живе лише в backend-репозиторії; жодного fingerprint «на якій екстракції fit» |
| 5 | Фолбеки, що фабрикують дані | **частково** | `measure_width_in_band_at_y` і далі повертає необрізану ширину при порожньому бенді (свідомо — дивись docstring); інше виправлено |
| 6 | Дублювання eval A–D | **є** | `pipelines.py` vs `app_pipelines.py` vs backend `eval.py`: median-scale fit ×3, bilinear ×2 |
| 7 | Немає тестів | **частково** | `tests/` є (моделі, COCO-адаптер, синтетичні лендмарки), геометрії — 0. Компенсовано snapshot-скриптом (§5) |

---

## 2. Знайдені порушення SSOT (нові, поза попереднім аудитом)

| Факт | Де визначено | Наслідок |
|---|---|---|
| Словник id вимірів | `envelope.py` ×5 списків, `build_eval_csv.py` `GT_MAP`, `audit_measurements.py` `MAP`, `app_pipelines.py` `WIDTH_FIELDS`, `app.py` mock, `tailoring.js`, `dataset/measurements.js`, `dataset-viewer/labels.py` — **12 місць** | нове вимірювання = ~12 правок; уже гнило (дубльовані ключі, виключення як вільний текст) |
| «Де талія» | front 0.25 (`silhouette`), side search ≤0.4 (`silhouette`), fallback 0.4 (`measurements`), back-length anchor 0.65 від шиї (`envelope`), **min-search на фронті у `visualize.py`** | намальована лінія ≠ виміряна |
| Калібрування px→cm | `calibration.py` (head_top→ankle), `measurements.py` `height_cm` (повторно), `eval/bodym.py` (mask extent) | різна семантика prod vs eval |
| Коефіцієнт корекції застосовується лише в `body_to_envelope` | web `/api/measure` — так; CLI `pointsx --output json`, `app_pipelines.py`, `build_dataset.py` — **ні** | CLI-талія жінки на 17.5 % більша за web на тих самих фото |
| Регресор | завантажується в `pipeline.py` і `inference.py`, але обидва викликають `estimate_circumferences(regression_model=None)` | `--regression-model`, `POINTSX_REGRESSION_MODEL`, `has_regressor` — no-op; `source="ellipse-0.1"` завжди |
| `torso_width_front_cm == shoulder_width_cm` | `silhouette` (кп-дистанція) і `measurements` (та сама дистанція) | похідний стан під двома іменами; `envelope.chest_width_front` читає його |
| Формула еліпса | `circumference.py` (повні ширини) + 3 копії в `scripts/ansur` з півосями | **виправлено** — `_common.ellipse` імпортує продакшн |
| Хелпер сегментів | `silhouette._find_segments` ≡ `measurements._split_segments` ≡ inline у `visualize` | **виправлено** для перших двох; `visualize` — лишилось |
| Скелет рядкового усереднення | 5 функцій `for row in [y-m, y+m]: cols=np.where(...)` з різним редʼюсером | можна звести до `_mean_over_rows(mask, y, margin, row_fn)` — не робив (середній ризик, 1.2 мс/пару → не заради швидкості) |

---

## 3. Ефективність (виміряно профайлером, CPU 8 потоків)

| Етап | мс | Частка |
|---|---|---|
| seg ×2 (yolo12l, imgsz 640 при вході 576) | 857 | 75 % |
| pose ×2 (yolo26) | 101 | 9 % |
| **усе силуетне Python** (calibrate + extract + circ + validate) | **1.3** | 0.1 % |
| `body_to_envelope` з візуалізацією (на ендпоінті) | +289 | +25 % |
| — з них таблиця мірок (37× BGR→PIL→BGR) | 184 | |
| — 7× PNG → base64 (5.2 MB у відповіді) | ~120 | |
| import (torch 2.6 с + ultralytics→SAM→torchvision→dynamo 1.1 с) | ~5.5 с | старт |

Висновки: оптимізувати `silhouette.py` заради швидкості — безглуздо (1 мс). Реальні
важелі — **не застосовані** (міняють поведінку/контракт, потребують окремого рішення):
1. JPEG q85 замість PNG + опційні debug-зображення → −110 мс CPU, −90 % байтів (5.2 MB → <0.5 MB).
2. `await asyncio.to_thread(pipeline.measure, …)` — зараз 1.1–1.4 с блокують event loop для всіх.
3. Одна PIL-канва для таблиці мірок → −180 мс.
4. При падінні калібрування `preview()` повторно ганяє pose+seg (−1 с на цій гілці).
5. Батч `[front, side]` в одному виклику YOLO → −60 мс; `imgsz=576` → ~−150 мс (перевірити на eval — змінює числа).
6. Eval: `bodym.py` профіль ширин рядок-за-рядком — 66 з 68 мс/суб'єкт (~48 с/прогін); кеш `raw` per (split, sid) як у `app_cells.json`; `build_eval_csv` N+1 запитів до Supabase.

---

## 4. Цільова структура і міграція (числа eval ідентичні на кожному кроці)

Цільові модулі (≤12):

| Модуль | Відповідальність / що є SSOT |
|---|---|
| `pointsx/skeleton.py` | `KP` (+ опційний `NOSE`), `is_valid`, `mean_valid_y`, геометрія |
| `pointsx/pose_backends.py` | coco17→16 і custom адаптери + калібрування довіри per-backend; константи синтезу `HEAD_TOP` (0.75/1.10) |
| `pointsx/models.py` | лише load/infer/вибір персони |
| `pointsx/calibration.py` | px/cm з `Keypoints` **або** з mask extent (pluggable) |
| `pointsx/slicing.py` | `SliceSpec`, `SliceResult(width, y, x0, x1)`, `measure_slice`; локатори {fixed_fraction, extreme_search, gap_scan}, екстенти {torso_band, limb, none} — без анатомічних назв |
| `pointsx/anatomy.py` | `SLICE_SPECS[(id, view)]`, `LENGTH_SPECS`, `BODYM_SILHOUETTE_SPECS` — **усі** 0.25/0.4/0.5/0.6/0.65 тут |
| `pointsx/measure.py` | `extract(...) -> {id: Value(cm, y, x0, x1, flags)}` — замінює `measurements.py` + `envelope._derive_*` |
| `pointsx/girth.py` | еліпс / білінійна / регресор за спільним списком пар (front, side) |
| `pointsx/corrections.py` | `load(version)` з `params/*.json`, `apply`, `fit_median_scales`, `fit_bilinear` |
| `pointsx/catalog.py` | `MEASUREMENT_CATALOG` (id, label_uk, source, default conf, plausible range, correction-eligible, display order, form aliases + причина виключення) + `plausibility.check(values, policy)` |
| `pointsx/pipeline.py` | `InferenceResult`, `run_perception`, `run_estimation`, `MeasurementPipeline`; CLI і web — один шлях |
| `webui/envelope.py` | лише форматування у `MeasurementEnvelope`; `eval/` імпортує `slicing/anatomy/corrections` |

Міграція (кожен крок: snapshot §5 + `pointsx-eval` + `eval/bodym.py --split testA --n 200` +
`eval/app_pipelines.py --refresh`, diff CSV/metrics = 0 після округлення 0.1 см):

1. **`slicing.py` + `anatomy.py` механічно.** Кожний блок `extract_all_widths` → рядок spec з
   тим самим локатором/екстентом/margin/фолбеком (head на торсо-бенді; фолбек на необрізану
   ширину лише у `span_band`; waist front=fixed, side=search; side-значення лімбів
   розмножуються на `_right/_left` у крок зшивання). `visualize.py` малює з `SliceResult`
   (єдина видима зміна — фронтальна лінія талії збігається з виміряною).
2. **Один оркестратор.** `InferenceResult` + дві половини у `pointsx/pipeline.py`;
   `WebuiPipeline`, CLI, `build_dataset.py` делегують. Регресор: або прокинути
   `self.regressor` (одне слово ×2), або видалити шлях і прапор `--regression-model` — рішення
   користувача, не «виправляти» по дорозі.
3. **`_derive_*` і chest → `measure.py`/`girth.py`** з канонічними id; chest-drop 22–50/14–38/60–150
   і `π·w` single-view fallback лишаються там, де діють; `validate_measurements` бачить ті самі
   нескориговані значення. CLI JSON — тільки додаткові ключі.
4. **`corrections.py` + `params/corrections-2026-07-29.json`** з таблицями verbatim,
   семантикою `elif` (один прапор гейтить обидві таблиці), `max(0,…)`, provenance у metadata
   (текст із коментарів — інституційна памʼять, не викидати); `--fit-offsets` пише датований
   файл; `PipelineInfo.model_version` = версія файлу.
5. **Каталог + правдоподібність.** Спершу три колонки діапазонів verbatim (postprocess /
   envelope / gt) — diff 0; потім окремим комітом, **очікувано** зміщуючи числа, одна межа +
   політика (warn/tag/drop/exclude як аргумент) із записом дельти. Потім `bodym.py`-локатори
   як `BODYM_SILHOUETTE_SPECS` через `slicing.py` (metrics.json ідентичний) і pipeline E =
   продакшн-таблиця зрізів + калібрування по mask extent — BodyM нарешті оцінює продакшн-
   геометрію.

Що зберігати обовʼязково при узагальненні: асиметрію талії front-fixed/side-search
(виміряно +0.89 vs +0.77); фолбек `measure_width_in_band_at_y` на необрізану ширину;
`_derive_back_length` на пропорційному якорі (RMSE 7.87 із силуетною талією); female chest
−4.5 %, male chest відсутній; порядок «корекція → гейт»; вихідний порядок регресора
[neck, waist, hip, thigh, calf, wrist] зашитий у ваги.

---

## 5. Захисна сітка для рефакторингу

Тестів геометрії немає; замінник — snapshot-скрипт (`scratchpad/snapshot.py`, варто
перенести у `scripts/dev/snapshot.py`): для кожного суб'єкта `supabase-dump/subjects.csv`
дампить `extract_all_widths` (widths + selected_y), `BodyMeasurements.to_dict()`, `warnings` і
`envelope` items → JSON; diff до/після. Саме так верифіковано сьогоднішні правки (17/17
ідентичні). Перший юніт-тест, який варто написати, — саме цей snapshot як pytest із
зафіксованим еталоном.

## 6. Пропущено свідомо (потребує рішення користувача)

- `regression_model` мертвий наскрізь — прокинути чи видалити (змінює CLI-контракт).
- `_derive_chest_circumference` 60–150 vs таблиця 60–160 — уніфікація змінить, які значення
  падають у None.
- `visualize.py` дублікати (`_left_arm_contour_path`, leg-outer, thigh split, neck ladder,
  min-search талії на фронті) — лікується кроком 1 міграції, а не точковими правками.
- `_left_arm_contour_path`: 4 схожі, але не ідентичні вибори сегмента — обʼєднувати лише з
  тестом на еталон.
- Усе з §3 (JPEG, `to_thread`, батчинг, кеш eval).
