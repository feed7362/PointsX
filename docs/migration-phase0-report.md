# Фаза 0 — звіт розбіжностей і рішення для злиття

> Частина `monorepo-cicd-plan.md`. Дата 2026-09-13. Нічого не змінено у віддалених
> репозиторіях. Прозою — українською, код/шляхи — англійською.
> A = `PointsX@origin/vercel` (`b1a2f31`), B = `Pointx-backend@f5ce97b`, C = `Pointx-frontend@1391e3a`.

---

## 0. Стан після Фази 0

| Крок | Результат |
|---|---|
| Незакомічена робота | `c64349e` на локальній `migration/phase0-wip`; `vercel` не зачеплено |
| Дзеркала | `Q:/Projects/KHNU/_migration-backup/2026-09-13/mirrors/` — PointsX 439 MB, backend 1.6 MB, frontend 5.8 MB; fsck зелений |
| Baselines (17 суб'єктів) | B = A = рефакторинг, **0 відмінностей**; імпорти перевірено всередині кожного дерева |
| Звіт розбіжностей | цей документ |

Baselines лежать лише в папці бекапу (числа з фото реальних людей — не комітити).

---

## 1. Python: A ↔ B

### 1.1 Підсумок

| | к-сть |
|---|---|
| Спільні ідентичні | 34 (включно з **усією математикою вимірювань**) |
| Розійшлися | 14 (`pyproject.toml` + 13 у `src/`) |
| Лише в A | 54 (`api/index.py`, шрифт, 52 файли `webui/static/**`) |
| Лише в B | 6 (`webui/storage.py`, `webui/_timing.py`, `pointsx/eval.py`, 3 синтетичні) |

Константи корекції `_SEX_CIRCUMFERENCE_SCALES_PCT`, `_LENGTH_SCALES_PCT`, `_PLAUSIBLE_RANGE_CM` — **ідентичні**.

### 1.2 Пофайлове рішення (база = B)

| Файл | Дія |
|---|---|
| `pyproject.toml` | B + додати `supabase`, явний `httpx`, `pynacl>=1.6.2` |
| `webui/app.py` | B + повернути з A: proxy-режим Vercel, лінивий `cv2`/`numpy`, `/dataset.html`, `Permissions-Policy: camera=(self)`, `POINTSX_DATASET_DIR` + timestamp-імена + попередження збереження |
| `webui/envelope.py` | A (лише імпорти: `TYPE_CHECKING`, лінивий `visualize`) |
| `webui/inference.py` | B (`warmup()`, `timings`); політика регресора — **§4 рішення 1** |
| `webui/tts.py` | B + англійський голос з A (`tts_voice(text)`) |
| `webui/visualize.py` | B (попередження про фолбек шрифту) |
| `webui/__main__.py` | B (dotenv) + SSL-прапори з A |
| `webui/docs/tailoring-config.md` | A (v5) |
| `pointsx/models.py` | B (автозавантаження ваг); фолбек-перейменування — **§4 рішення 2** |
| `pointsx/cli.py` | B (дефолт `coco`) |
| `regression/build_dataset.py` | A (дефолт argparse `coco`, B його пропустив) |
| `synthetic/{blender_render,body_generator,pipeline}.py` | B |
| Лише в A: `api/index.py`, `static/**`, шрифт | лишити; `static/` виключити з Docker-контексту |
| Лише в B: `storage.py`, `_timing.py`, `eval.py`, синтетичні | лишити; `storage` імпортувати ліниво |

### 1.3 Чому «база B» без правок зламала б Vercel

1. Top-level `import cv2` в `app.py` — у Vercel-функції немає OpenCV → падає імпорт.
2. `envelope.py` тягне `inference` (torch/ultralytics) на верхньому рівні → падає `/api/measure/mock`.
3. Немає proxy-режиму → `/api/measure` повертає 503.
4. Зникають `/dataset.html`, SSL для LAN-зйомки з телефона, англійський голос, звіт про невдале збереження датасету.

А код A на Space: ваги не завантажуються (503), немає архіву, CORS, `/api/health`, інференс блокує event loop.

### 1.4 Відмінності, що впливають на числа (baseline їх не покрив)

| Відмінність | Вплив |
|---|---|
| B зменшує фото до 1280 px перед інференсом | Прод уже так робить (Vercel проксує на B); eval/локальний A — ні. Корпус 576×576 → baseline цього не бачив |
| B `models.py`: якщо `yolo26-pose.pt`/`yolo12l-...` відсутні, **тихо** підставляє `yolo11x-pose`/`yolo11n-seg` під очікуваним іменем | Виміри зміняться без помилки |
| Регресор: A жорстко вимкнений; B вмикається, якщо задано `POINTSX_REGRESSION_MODEL` | Сьогодні обидва дають еліпс; константи корекції підігнані саме під еліпс |

---

## 2. Статика/UI: A ↔ C

### 2.1 Підсумок

- 43 спільні: 31 ідентичний, 12 розійшлися. Лише в A — 8 (`dataset.html`, `camera.js`, `i18n.js`, 5 файлів `js/dataset/`, включно з `transfer.js` від MaksShu). Лише в C — `config.js`.
- C відгалужено від `origin/demo`, який **не** є предком `origin/vercel`, тому більшість A≠C — застарілий demo-код на боці C. Реально змінено в C лише 5 файлів.
- Ребрендинг «FitMeasure AI», PNG одягу, сітки розмірів, фолбек MediaPipe — **уже в A**. Жодного значення розмірів, яке є лише в C.
- Справжніх конфліктів немає.

### 2.2 Зміни, що є лише в C

| Коміт C | Зміна | Рішення |
|---|---|---|
| `85e9356` | `speech.js`: не відкликати blob URL при відхиленні autoplay | **Перенести** — в A звук ламається при відхиленні autoplay |
| `f2712ae` | `speech.js`: вимкнути серверний TTS на сесію після 503/мережевої помилки | **Перенести без тригера `AbortError`** — warmup-запит A може таймаутнути на холодному старті Vercel |
| `c2241e8` | `/api/tts` через `POINTSX_API_BASE` | Відкинути (лише для split-деплою) |
| `d69462e` | речення про HTTPS у підказці | Опційно — A вже показує `camera-https-warning` (uk+en) |
| база demo | `config.js`, `build_static.mjs`, `vercel.json` C, API_BASE | Відкинути |

Баг у C (не в A): `loadPoseLandmarkerImage` досі викликає видалений `MP_PKG` → ReferenceError. Ще одна причина архівувати C.

### 2.3 Модель Vercel-деплою в монорепозиторії

Лише модель A: same-origin, `vercel.json` → `api/index.py`, яка проксує `/api/measure` і сама
обслуговує `/api/measure/mock` та `/api/tts`. Модель C (`POINTSX_API_BASE` у браузері) обходить
mock і TTS — не використовувати.

---

## 3. Порядок злиття для Фази 1

1. `main` від `origin/vercel`; `git subtree add` для B і C з повною історією.
2. `src/` і `pyproject.toml` — з B; додати залежності з §1.2.
3. Ліниві важкі імпорти (`envelope.py` з A; `cv2`/`numpy` у функціях `app.py`; `storage`, `huggingface_hub` — ліниво).
4. Proxy-режим у `app.py` B: ранній вихід із `lifespan()` **до** `_pull()`, warmup і перевірки сховища; гілка проксі — першою в `measure()`.
5. Решта A-only з §1.2; два фікси `speech.js` з §2.2.
6. Рішення §4.
7. `api/index.py`, `static/`, шрифт з A; `static/` поза Docker-контекстом.
8. Коміт рефакторингу `c64349e` поверх.
9. Перевірка:
   - `snapshot2.py` = baselines;
   - імпорт `webui.app` з Vercel-змінними у venv лише з `requirements.txt`;
   - `docker build` + `/api/health`;
   - snapshot з зменшенням до 1280 і без.

---

## 4. Рішення, потрібні до Фази 1

1. **Регресор.** Один явний прапор, типово вимкнений (константи підігнані під еліпс) — чи підключити як у B?
2. **Тиха заміна ваг у `models.py`.** Заборонити (падати з помилкою, якщо немає очікуваних ваг) — чи лишити фолбек з попередженням?
3. **Зменшення до 1280 px.** Застосувати і в eval/локально, щоб eval міряв те саме, що прод?
4. **Речення про HTTPS** (C `d69462e`) — додати в i18n чи відкинути?
