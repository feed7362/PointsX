# План: монорепозиторій + CI/CD для Vercel і Hugging Face

> Планувальний документ. **Жодних змін у репозиторіях не зроблено** — це те, що
> треба затвердити перед першим комітом міграції. Стан зафіксовано 2026-09-03.
> Прозою — українською, код/шляхи — англійською.

---

## 0. TL;DR

Три репозиторії з двостороннім дрейфом → **один монорепозиторій під `feed7362`**, з
історією всіх трьох **без переписування SHA**, гілками-цілями деплою (`vercel` → Vercel, HF Space ← CI) і воротами
CI, які не дають зламати вимірювання. Порядок: спершу заморозити стан і звірити розбіжності,
потім міграція історії, потім CI, і лише потім прибирання.

**Знайдено при аудиті — вимагає рішення до міграції:** живий Vercel-деплой стоїть на
комміті `86c3c63`, це **18 комітів позаду** `origin/vercel`. У продакшні немає CDN-фолбеку
для libsodium/MediaPipe, виправлених сіток розмірів UA/US, якоря талії й відновленої
корекції грудей. Тобто «двічі задеплоєно» з попередніх сесій не доїхало.

---

## 1. Поточний стан (виміряно, не зі слів)

### 1.1 Репозиторії

| | файлів | гілки | `.git` (pack) | деплой |
|---|---|---|---|---|
| **PointsX** (чернетка + фактичне джерело Vercel) | 157 | 4 віддалені: `master`, `vercel`, `demo`, `ui-and-size-charts`; 23 брудних файли | **447 MB** | `vercel.json` у корені: `api/index.py` (@vercel/python) + `src/webui/static` |
| **Pointx-backend** | 62 | `main`, 30 комітів | 2 KB | 2 remotes: `github` + HF Space (`origin`); Dockerfile + front-matter у README |
| **Pointx-frontend** | 49 | `main`, 9 комітів, останній 2026-07-29 | ~0 | власний `vercel.json` + `scripts/build_static.mjs` → `public/` |

### 1.2 Дублювання і дрейф — головна причина багів

- **PointsX ↔ Pointx-backend**: 48 спільних шляхів у `src/` → **35 ідентичних, 13 розійшлися**:
  `webui/app.py`, `webui/envelope.py`, `webui/inference.py`, `webui/visualize.py`,
  `webui/tts.py`, `webui/__main__.py`, `pointsx/cli.py`, `pointsx/models.py`,
  `regression/build_dataset.py`, `synthetic/{pipeline,body_generator,blender_render}.py`,
  `webui/docs/tailoring-config.md`.
- **PointsX ↔ Pointx-frontend**: 43 спільних статичних файли → **31 ідентичний, 12 розійшлися**:
  `js/app.js`, `js/capture/{session,tailoring,ui,state,dom,poseGate,speech}.js`,
  `css/styles.css`, `index.html`, `data/tailoring_config.json`, тест `sizeEngine.consensus`.
- `pyproject.toml` теж різний: backend має `boto3`, `python-dotenv`, `huggingface_hub` і
  **entry point `pointsx-eval`, якого немає в PointsX**; PointsX має `supabase`.
- `eval/app_pipelines.py` і `scripts/audit_measurements.py` роблять `sys.path.insert` у
  **сусідній репозиторій** — тобто eval уже фізично залежить від двох чекаутів.

Це не косметика: у цій сесії правку синтетики зробили не в тому репозиторії, а константи
корекції довелося переносити разом з екстракцією, бо порізно вони роблять деплой **гіршим**.

### 1.3 Сміття до прибирання

- Гілки: `demo` (61 попереду / 64 позаду), `ui-and-size-charts` (влита в `master`).
- Відстежується `.DS_Store`; `__pycache__/` лежить у `tests/` і `dataset-viewer/`.
- 23 брудних файли в робочому дереві PointsX (8 видалених `docs/*.md`, 12 змінених
  вимірювальних модулів після сьогоднішнього `/simplify`, 3 нових).
- Історія: ~15 ревізій `uv.lock`, три копії `guide-geometry.json` (0.6 MB кожна),
  ноутбук 1.5 MB, шрифт DejaVu 0.7 MB.
- Немає `.github/` взагалі — CI зараз нульовий.

---

## 2. Затверджені рішення

| Питання | Рішення |
|---|---|
| Структура | **Один монорепозиторій**, деплой за фільтром шляхів |
| Власник | **Особистий акаунт `feed7362`**, без GitHub-org (org зламав би безкоштовний Vercel Hobby) |
| Історія | **Без переписування** (рішення 2026-09-13, є другий розробник): історія PointsX лишається як є; backend і frontend імпортуються `git subtree add` з повною історією — жоден SHA не змінюється, ніхто не переклоновує. `.git` лишається ~447 MB; у CI — `fetch-depth: 1` |
| Канонічний код (Python) | **Тристороннє злиття**: база `Pointx-backend@f5ce97b` → поверх окремим комітом рефакторинг `/simplify` (є лише в незакоміченому дереві PointsX) → snapshot = `baseline-backend.json` |
| Канонічний код (статика/UI) | `PointsX@origin/vercel` |
| Гілки лишити | `vercel`, `master`. Видалити `demo`, `ui-and-size-charts` (після збереження історії) |
| Вміст | Переносимо все: `eval/`, `scripts/`, `docs/`, `synthetic/`, `notebooks/`, `dataset-viewer/` |
| Ворота CI | геометричний snapshot + lint/tests/build + пост-деплой smoke. **Без** порогу на MAE |
| Дані для CI | **Зараз: кешовані маски + кейпоінти (npz), без фото**; рендери MPFB2/Anny — пізніше, коли запрацює генератор |
| Тригер деплою | merge у `main` → CI зелений → промоушен у `vercel` (UI) і в HF Space (backend) |
| Гілка `vercel` | **Лише файли Vercel-деплою**, генерується CI. `docs/`, `eval/`, `scripts/` та інше спільне — тільки в `main`. Прямі пуші у `vercel` припиняються (MaksShu працює через `main`) |
| Другий розробник | MaksShu (коміт `b1a2f31` у `origin/vercel`, 2026-08-03) — повідомити до Фази 3, що `vercel` стає CI-власною |
| Секрети | HF write-token; Vercel лишається git-driven (без токена); репозиторій публічний |

---

## 3. Цільова структура

```
PointsX/                          # той самий репозиторій, нове дерево
  packages/
    pointsx/                      # вимірювальне ядро (без web, без FastAPI)
    webui/                        # FastAPI app + envelope + visualize + static SPA
  apps/
    backend/                      # Dockerfile, README (HF front-matter), .env.example
    web/                          # vercel.json, api/index.py, build-крок
  eval/                           # BodyM, app-корпус, pipelines A-D
  scripts/                        # ansur/, supabase/, dev/
  tools/dataset-viewer/
  docs/
  notebooks/
  tests/
    fixtures/                     # маски + кейпоінти (npz) + еталонні виходи; рендери — пізніше
  .github/workflows/
  pyproject.toml                  # один, з optional-groups: [backend], [eval], [dev]
```

Чому `webui` у `packages/`, а не в `apps/`: **обидва** деплої його імпортують. Vercel зараз
крутить не «статику», а FastAPI через `api/index.py`, який проксює `/api/measure` на HF
(`POINTSX_INFERENCE_ENDPOINT`). Тобто `webui` — спільна бібліотека, а `apps/*` — лише
пакування під платформу.

`pyproject.toml` один, залежності розкладені групами: базові (torch, ultralytics, opencv)
у `[project.dependencies]`; `boto3`/`huggingface_hub`/`python-dotenv` у `[backend]`;
`supabase`/`pynacl` у `[eval]`. Entry points: `pointsx`, `pointsx-web`, `pointsx-eval`
(останній треба перенести з backend — у PointsX його немає, тому всі згадки
`pointsx-eval --fit-offsets` у коментарях `envelope.py` зараз ведуть у нікуди).

---

## 4. Топологія деплою і CI/CD

```
   PR  ──► ci-checks (lint, tests, snapshot, docker build, web build)
                    │ green
   merge to main ───┤
                    ├─ paths: packages/**, apps/backend/**  ──► deploy-backend
                    │        assemble Space tree → force-push → HF Space
                    │        → post-deploy smoke (/api/measure на фікстурі)
                    │
                    └─ paths: packages/webui/static/**, apps/web/**  ──► deploy-web
                             CI збирає дерево web-деплою → push у `vercel`
                             → Vercel будує сам (git-driven, без токена)
```

Зміна в `packages/pointsx/**` вважається backend-зміною **і** web-зміною: Vercel-функція
імпортує той самий код.

### 4.1 Workflows

| Файл | Тригер | Що робить |
|---|---|---|
| `ci.yml` | `pull_request`, `push: main` | ruff → pytest → geometry snapshot → `docker build` (без push) → `node scripts/build_static.mjs` |
| `deploy-backend.yml` | `push: main` + path filter | збирає дерево Space (Dockerfile, README front-matter, pyproject, `packages/`), комітить у тимчасовий репозиторій, `push --force` на `https://huggingface.co/spaces/Secret0123/Pointx-backend` через `HF_TOKEN`; чекає готовності Space; smoke-тест |
| `deploy-web.yml` | `push: main` + path filter | збирає лише те, що імпортує `api/index.py` (`vercel.json`, `api/`, `webui` + `static`, потрібні модулі `pointsx`), комітить поверх попереднього стану `vercel` (не `--force`, щоб історія деплоїв лишалась читабельною) і пушить. Vercel-проєкт лишається як є |
| `nightly-eval.yml` | `schedule` + `workflow_dispatch` | **не блокує** нічого: BodyM + app-корпус, публікує MAE як artifact/summary, щоб дрейф було видно |

Чому HF деплоїться складанням дерева, а не `git subtree push`: Space вимагає `Dockerfile`
і README-front-matter **у корені**, а монорепозиторій має їх у `apps/backend/`. Складання
дерева в CI — це те саме, що ви робили руками, лише детерміновано.

### 4.2 Ворота: геометричний snapshot

Сьогоднішній `/simplify` уже довів метод: скрипт дампить для кожного суб'єкта `widths`,
`selected_y`, `BodyMeasurements.to_dict()`, `warnings` і значення envelope у JSON; рефакторинг
приймається лише якщо diff порожній (17/17 збіглися). Це стає `tests/test_snapshot.py` з
еталоном у `tests/fixtures/expected/`.

Вхід для CI (рішення §7.1): **кешовані маски + кейпоінти** в `tests/fixtures/` — перевіряє
калібрування, екстракцію ширин, обхвати, корекції й envelope без жодного фото. Зміни в
pose/seg-моделях цей вхід не ловить — їх покриває пост-деплой smoke (§4.3).

### 4.3 Пост-деплой smoke

Після успішного білду Space: `POST /api/measure` з фікстурною парою + зростом, перевірка що
(а) 200, (б) чотири обхвати присутні, (в) значення в межах ±0.5 см від еталона. Це ловить
саме той клас поломок, який локальні тести не бачать: не докачалися ваги, немає шрифтів,
не ті змінні оточення. Падіння smoke → issue + повідомлення, автоматичного відкату немає
(відкат = revert коміту й повторний деплой).

---

## 5. План міграції (фази; кожна перевіряється до наступної)

### Фаза 0 — заморозка і звірка (перед будь-якою зміною)

0. Підготувати робоче дерево: `docs/` відновлено (2026-09-13); підтягнути `b1a2f31`;
   незакомічений рефакторинг, 3 нові документи й `scripts/ansur/` зберегти **на окремій
   локальній гілці** (не у `vercel`) — вони підуть у `main` монорепозиторію.
1. Записати поточні продакшн-значення: прогнати `snapshot.py` на 17 суб'єктах з
   **backend-кодом** і зберегти як `baseline-backend.json`; те саме з PointsX/vercel.
2. Зробити повні дзеркальні копії всіх трьох репозиторіїв (`git clone --mirror`) в
   окрему теку — точка відкату для всієї операції.
3. Звірити 13 розбіжних Python-файлів і 12 статичних: канон = backend для Python,
   `vercel` для статики; **знайти правки, які є лише в іншій стороні**, і перенести їх
   явно. Це єдиний крок, де можлива втрата роботи, тому — пофайловий звіт на затвердження.
4. Вирішити долю живого Vercel: він на 18 комітів позаду. Спершу довести його до
   `origin/vercel` (або свідомо лишити) — інакше «зелений CI» після міграції означатиме
   зовсім інший продакшн, ніж зараз.

*Перевірка:* `baseline-backend.json` і `baseline-vercel.json` існують; пофайловий звіт
затверджено.

### Фаза 1 — монорепозиторій без переписування історії

5. Створити `main` від `origin/vercel` (включає `b1a2f31`). `master` лишається як збережена гілка.
6. `git subtree add --prefix=_import/backend <Pointx-backend> main` і те саме для frontend —
   **повна** історія (без `--squash`), SHA обох репозиторіїв не змінюються.
7. Розкласти дерево за §3 через `git mv` (історія файлів відстежується `git log --follow`);
   застосувати тристороннє злиття Python (§2) і статики з `vercel`; один `pyproject.toml` з групами;
   прибрати `_import/`.
8. Окремим комітом — рефакторинг `/simplify`.

*Перевірка:* `snapshot.py` на новому дереві = `baseline-backend.json` байт у байт; `uv sync`
проходить; `pointsx`, `pointsx-web`, `pointsx-eval` запускаються; `git log --follow` на
`silhouette.py` показує стару історію; клон MaksShu після `git pull` працює без переклонування.

### Фаза 2 — CI без деплою

9. `ci.yml` (ruff + pytest + docker build + web build) — поки без snapshot-воріт.
10. Фікстури (§7.1) + `tests/test_snapshot.py` з еталоном.
11. Увімкнути snapshot як обов'язкову перевірку гілки `main`.

*Перевірка:* навмисно зламати константу в `envelope.py` у чернетковому PR → CI має впасти
саме на snapshot.

**Стан (2026-09-13):** кроки 9–10 зроблено, `.github/workflows/ci.yml`, 5 паралельних job:

| Job | Що перевіряє |
|---|---|
| `lint` | ruff, блокує лише клас помилок `E9,F63,F7,F82` (синтаксис, невизначені імена); повний звіт — без блокування, поки борг не сплачено |
| `test` | CPU-torch + `pip install -e .` → `pytest`: юніт-тести + `tests/test_snapshot.py` |
| `vercel-smoke` | venv лише з `requirements.txt` → `scripts/ci/vercel_smoke.py`: без torch/cv2, mock 200, `/` і `/dataset.html`, проксі → 502 |
| `docker` | `scripts/ci/stage_hf_space.sh` збирає дерево Space (без `static/`) → `docker build` + імпорт застосунку в контейнері; той самий скрипт піде в `deploy-backend.yml` |
| `web` | `node --check` усіх JS у `static/`, валідність `vercel.json`. `build_static.mjs` більше не існує (модель C відкинуто) |

Фікстури — **процедурні**, не кешовані маски реальних людей: `tests/fixtures/synthetic_bodies.py` малює
6 тіл (3 Ж / 3 Ч, 160–190 см) з параметрів у сантиметрах. Маски й кейпоінти реальних суб'єктів — теж
похідні персональні дані, у публічний репозиторій їм не можна. Покриття: усі 18 id envelope, у т. ч. чотири
core-обхвати (окремий тест це стереже). Еталон — `tests/fixtures/expected/synthetic_snapshot.json`;
після навмисної зміни геометрії: `python tests/fixtures/synthetic_bodies.py --write-expected`, JSON комітиться разом із кодом.

Перевірено локально: `pytest` 14/14; зсув жіночої корекції стегон −5,0 → −4,0 % валить snapshot рівно на
трьох жіночих `hip_circumference`; `vercel_smoke.py` зелений; actionlint без зауважень. Крок 11 (обов'язкова
перевірка для `main`) — налаштування репозиторію, робить власник після першого зеленого прогону.

### Фаза 3 — деплой

12. `deploy-web.yml` (push у `vercel`). Перевірка: тривіальна зміна тексту доїжджає.
13. `deploy-backend.yml` + `HF_TOKEN` у секретах. Перевірка: та сама зміна доїжджає в Space
    і smoke зелений.
14. Один навмисно зламаний деплой (наприклад, прибрати шрифт із Dockerfile) — щоб
    переконатися, що smoke це ловить.

*Перевірка:* обидва деплої відтворювані, ручні `git push` більше не потрібні.

**Стан (2026-09-13):** `.github/workflows/deploy.yml` — один воркфлоу, дві job (`web`, `backend`), запускається
як останній job `ci.yml` після всіх зелених перевірок на `main` (`workflow_call`) або вручну (`workflow_dispatch` з прапорцями «деплоїти попри відсутність змін»).

| | `web` → гілка `vercel` | `backend` → Space `Secret0123/Pointx-backend` |
|---|---|---|
| Що деплоїться | `scripts/ci/stage_vercel.sh`: `vercel.json`, `requirements.txt`, `.vercelignore`, `api/`, `src/webui/`, `src/pointsx/` без synthetic/regression/train/eval; **без** `pyproject.toml` (з ним Vercel тягне torch — див. старий `deploy-vercel.sh`) | `scripts/ci/stage_hf_space.sh`: `apps/backend/*` у корені + `pyproject.toml` + `src/` без `static/` |
| Коли | змінились ці шляхи від коміту, записаного як `source: <sha>` в останньому деплой-коміті гілки | те саме, маркер читається з HEAD git-репозиторію Space |
| Як | коміт **поверх** поточного `vercel` (без `--force`) через `GITHUB_TOKEN` | коміт **поверх** історії Space, push через `HF_TOKEN` |
| Після | Vercel деплоїть гілку сам | очікування `runtime.stage == RUNNING` з тим самим sha (до 30 хв), потім `scripts/ci/space_smoke.py`: `pipeline_ready`, `"coco" in pose_backends`, не proxy-режим, mock → 18 мірок |

**Ручний push у `vercel` лишається другим методом** (рішення 2026-09-13): гілка без protection, CI ніколи не
переписує історію, тож `git push origin <коміт>:vercel` деплоїться як є; наступний деплой з `main` накладе своє дерево
поверх. Правки, зроблені лише там, при цьому губляться — їх треба донести й у `main`. README у гілці це пояснює.
Відмінність від сьогоднішнього стану: `vercel` більше не гілка розробки з повним деревом, а згенероване дерево деплою.

Smoke Space не робить справжній `/api/measure`: для цього потрібна закомічена пара фото (§7.1, варіант 1 або 2), поки її немає.
`HF_TOKEN` доданий власником у Secrets (2026-09-13).

Перший запуск (2026-09-13) не стартував: `workflow_run`/`workflow_dispatch` GitHub читає лише з гілки за замовчуванням, а нею досі є `master`. Тому деплой викликається з `ci.yml` напряму (`workflow_call`); ручний запуск через `workflow_dispatch` запрацює після зміни гілки за замовчуванням на `main` (Settings → General → Default branch).

**Vercel, 2026-09-14.** Git-інтеграція проєкту `fitmeasure-ai` не працювала з 2026-06-28 (жоден push у `vercel` не деплоївся).
Перепідключено до `feed7362/PointsX`, Production Branch = `vercel`; `3086d51` піднято в Production — живий сайт тепер з `main`.
Опційно: Deploy Hook для гілки `vercel` → секрет `VERCEL_DEPLOY_HOOK`, деплой-job викликає його після push.

**Моніторинг і keep-alive (безкоштовно на Hobby):**

- Web Analytics + Speed Insights: скрипти `/_vercel/insights` і `/_vercel/speed-insights` у трьох HTML (не на localhost).
  Лише перегляди сторінок і Web Vitals; custom events — Pro. Увімкнути в дашборді: Analytics → Enable, Speed Insights → Enable.
  В аналітику не йдуть фото, мірки, id заявок.
- Keep-alive Space: Vercel cron раз на добу (ліміт Hobby) → `GET /api/keepalive` (з `CRON_SECRET`, якщо задано) пінгує `/api/health` Space;
  плюс `.github/workflows/keepalive.yml` кожні 12 год запускає `scripts/ci/space_smoke.py` (будить і перевіряє, що міряє).
- `Permissions-Policy: camera=(self)` тепер ставиться й на статичних HTML через `headers` у маршрутах `vercel.json`.

### Фаза 4 — прибирання

15. Видалити віддалені `demo`, `ui-and-size-charts` (їхні коміти недосяжні з `main` — перед видаленням
    поставити теги `archive/demo`, `archive/ui-and-size-charts`, щоб нічого не зникло).
16. Заархівувати `Pointx-frontend` і `Pointx-backend` на GitHub у режимі read-only з
    README «переїхало в …».
17. Прибрати `.DS_Store`, `__pycache__`, розібратися з 23 брудними файлами.
18. Оновити `.claude/CLAUDE.md` під нову структуру.

**Стан (2026-09-14):**

- Кроки 15: теги `archive/master`, `archive/demo`, `archive/ui-and-size-charts` запушено, віддалені гілки видалено (2026-09-13).
  На GitHub лишились лише `main` (гілка за замовчуванням) і згенерована `vercel`.
- Крок 16: в `Pointx-backend` і `Pointx-frontend` запушено README з позначкою «архівовано, переїхало в PointsX».
  Обидві гілки `main` (`f5ce97b`, `1391e3a`) уже в історії монорепозиторію; локальні клони без незбережених змін.
  **Архівація (Settings → Danger Zone → Archive this repository) — дія власника.**
- Крок 17: з `main` прибрано `.DS_Store` (+ `.gitignore`). Тимчасових `__pycache__`/`.pyc` у git немає.
  Лишаються на рішення власника: застарілі інструкції доби двох репозиторіїв (`START.md`, `RUN-LOCAL.md`, `RUN-WEBUI.md`,
  `DATASET_COLLECTION.md`, `deploy/`, `run-local.sh`) і `deploy-vercel.sh` (ручний деплой через Vercel CLI; тепер
  ручний шлях — `git push` у `vercel`).
- Крок 18: `.claude/CLAUDE.md` — розділ про структуру монорепозиторію й деплой, актуальні ваги моделей.
- Локальна тека `Q:/Projects/KHNU/Pointx-backend` має remote `origin` = HF Space: не пушити звідти — Space тепер
  оновлює лише CI (push буде відхилено як non-fast-forward, але краще видалити клон після архівації).

---

## 6. Ризики

| Ризик | Чому реальний | Пом'якшення |
|---|---|---|
| Втрата правки при звірці 25 розбіжних файлів | обидві сторони містять унікальні виправлення | Фаза 0 крок 3: пофайловий звіт + `baseline` snapshot до/після |
| Прямий пуш у `vercel` перезапишеться CI | MaksShu пушить у `vercel` напряму | Повідомити до Фази 3; branch protection на `vercel` (пушить лише CI) |
| Великий `.git` | історію не переписуємо | `fetch-depth: 1` у CI; стрип блобів — окремим рішенням, коли всі погодяться переклонувати |
| Vercel/HF перестають бачити джерело | шляхи в `vercel.json` і корінь Space змінюються | `vercel` лишається тією ж гілкою того ж репозиторію; Space отримує зібране дерево з тим самим коренем |
| Маски+кейпоінти не ловлять зміни pose/seg | вхід нижче YOLO | пост-деплой smoke; рендери — пізніше |
| Space збирається довго / впаде після push | free CPU, ~1.5 GB образ | smoke з ретраями; деплой backend лише за зміни у `packages/**` або `apps/backend/**` |
| Публічний репозиторій + приватні дані | `supabase-dump/`, `keys/`, `.env` | Уже в `.gitignore`; **перед публікацією прогнати сканер історії на секрети** (окремий пункт Фази 1) |

---

## 7. Що треба вирішити до старту

### 7.1 Звідки взяти синтетичні фікстури

Snapshot-ворота потребують зображень, на яких YOLO щось знаходить. Варіанти:

1. **Фото автора/колеги, явно зняті для CI** — найшвидше, згода однозначна, покриває весь
   ланцюг pose→seg→geometry. Мінус: у публічному репозиторії лежить фото людини.
2. **Рендери MPFB2/MakeHuman (виходи CC0) або Anny (Apache 2.0)** — правильне рішення, збігається
   з рекомендацією `accuracy-approaches-survey.md` §E2, дає скільки завгодно пар і жодних
   персональних даних. Мінус: спершу треба підняти рендер (Blender + ассети).
3. **Кешовані маски+кейпоінти без фото** — тестує лише геометрію нижче YOLO; не ловить
   зміни в pose/seg, зате нульовий ризик і працює вже завтра.

**Рішення 2026-09-13: (3) зараз, (2) пізніше.** Пропозиція була: почати з (3) як воріт уже в Фазі 2, паралельно робити (2) і замінити. (1) —
лише якщо потрібне повне покриття раніше, ніж запрацює рендер.

### 7.2 Що робити зі стеком Vercel-функції

`api/index.py` піднімає весь FastAPI на Vercel лише щоб проксувати на HF. Якщо це справді
лише проксі — дешевше зробити статику + rewrite, і тоді Vercel взагалі не залежить від
Python-ядра (менше збірки, менше площі поломки). Треба перевірити, чи Vercel-функція не
робить чогось ще (TTS? dataset?) — і вирішити до Фази 1, бо це змінює межу `packages` ↔ `apps`.

### 7.3 Дрібне

- `pointsx-eval` живе лише в backend — підтвердити, що в монорепозиторії він канонічний.
- `Pointx-frontend` після архівації: підтвердити, що там немає нічого, чого немає у
  `vercel`-гілці (розбіжних файлів — звірка в Фазі 0 покриє).
- Нейм: у комітах фронтенду є ребрендинг «FitMeasure AI», у деплої — домен
  `fitmeasureai.pp.ua`, а репозиторій зветься PointsX. Лишаємо як є чи перейменовуємо
  монорепозиторій?

---

## 8. Перевірка готовності (2026-09-13)

**Вердикт: Фаза 0 — так. Фаза 1 (переписування історії) — ні, поки не закрито блокери нижче.**

Нові факти, що змінюють план:

| Факт | Наслідок |
|---|---|
| `origin/vercel` має коміт `b1a2f31` від **MaksShu** (2026-08-03, 8 файлів, +327, новий `js/dataset/transfer.js`); локальний `vercel` на 1 позаду | Є **другий розробник**. Мітигація «один клон на машині» (§6) — хибна. `git-filter-repo` змінить усі SHA і зламає його клон. Потрібен узгоджений cutover або варіант без переписування |
| `transfer.js` відсутній у `Pointx-frontend`; `session.js`, `tailoring.js`, `index.html`, `styles.css` з того ж коміту розійшлися ще більше | Звіт розбіжностей (§1.2) треба перерахувати від `origin/vercel`, не від `2e89a39` |
| Рефакторинг `/simplify` існує **лише в незакоміченому дереві PointsX**; `Pointx-backend@f5ce97b` старший | «Backend виграє для Python» тихо відкотив би рефакторинг, а snapshot усе одно пройшов би. Злиття — **тристороннє**: база `f5ce97b` → поверх коміт рефакторингу → snapshot = `baseline-backend.json` |
| 8 файлів `docs/*.md` (2131 рядок) видалені в робочому дереві, але є в HEAD | Не комітити, поки не підтверджено, що це свідомо |
| Vercel-функція обслуговує `/`, `/dataset.html`, `/api/tts`, `/api/measure/mock`; проксує лише `/api/measure` | §7.2 закрито: це не чистий проксі, функцію лишаємо. `Pointx-frontend` шле `/api/tts` на HF — поведінкова, не лише файлова, розбіжність |
| Сканер історії: у жодному з трьох репозиторіїв не комітилися `.env`, ключі, `supabase-dump`, `bucket-dump` | Друга перезапис заради секретів не знадобиться |
| `Pointx-backend`: `origin/main` (HF) і `github/main` — 0/0 | Проблема «Space на 7 комітів позаду» вирішена |
| `gh` не встановлено; `git filter-repo --version` повертає хеш | Архівація/видалення гілок — вручну або після встановлення `gh`; filter-repo спершу перевірити на одноразовому дзеркалі |
| `HF_TOKEN` | Користувач додає сам у GitHub → Settings → Secrets; у сесію не передається |
| Фаза 0 (2026-09-13): baseline `f5ce97b` = `origin/vercel` = рефакторинг `c64349e`, 0 відмінностей на 17 суб'єктах; імпорти перевірено всередині кожного дерева | 13 розбіжних Python-файлів **не змінюють числа** в локальному шляху вимірювання; розбіжності — у проксі/сховищі/boot. Proxy-режим Vercel і HF-boot цей snapshot не покриває |
| `Pointx-backend/.gitattributes` — стандартний HF-шаблон LFS (`*.bin`, `*.pt`…), але LFS-об'єктів в історії 0 | `git subtree add` безпечний; `.gitattributes` переносити лише в `apps/backend/`, щоб LFS-правила не зачепили весь монорепозиторій |

Згідно з правилом «Reversibility: One-way → red-team перед комітом»: переписування історії
незворотне для всіх клонів — перед Фазою 1 прогнати `/red-team docs/monorepo-cicd-plan.md`.

**Закрито того ж дня:** переписування історії скасовано → блокер MaksShu знято (його клон не
ламається), red-team для переписування не потрібен. `docs/` відновлено. Фікстури — маски+кейпоінти.
`vercel` — лише файли деплою. Лишається: Фаза 0 крок 3 (пофайловий звіт, перерахований від
`origin/vercel`) і повідомлення MaksShu до Фази 3.

## 9. Наступний крок

**Фаза 0** — нічого не змінює у віддалених репозиторіях: локальна гілка для незакоміченої
роботи, дзеркальні копії, `baseline-backend.json` / `baseline-vercel.json`, пофайловий звіт
розбіжностей від `origin/vercel`. Після затвердження звіту — Фаза 1.
