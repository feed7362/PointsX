# План: декомпозиція застосунку за зразком Video_streaming

> Плановий документ, 2026-09-14. Код ще не змінено. Прозою — українською, код/шляхи — англійською.
> Зразок: `Q:/Projects/Video_streaming/backend` (`main.py` на 83 рядки, шари `app/` + `src/{api,services,schemas,infrastructure,core,errors}`).

---

## 0. TL;DR

`src/webui/app.py` має 1042 рядки і сім відповідальностей в одному файлі. Розкладаємо `webui` на шари, як у
Video_streaming: тонка фабрика `create_app()`, окремі роутери, сервіси, схеми, інфраструктура, один
механізм помилок. **Числа вимірювань, API-контракт і обидва деплої не змінюються ні на одному кроці.**
Фронтенд лишається vanilla JS без збирача, але отримує єдиний API-клієнт і розбиття великих модулів.

Що **не** копіюємо з Video_streaming (зайве для одного сервісу): Vault, RabbitMQ/`events/`, Redis,
Elasticsearch, helm, monitoring, nginx gateway, окремий клас налаштувань на кожну область,
Prometheus/request-id middleware, React.

---

## 1. Що зараз не так (виміряно)

| Файл | Рядків | Що змішано |
|---|---|---|
| `webui/app.py` | 1042 | env/налаштування; переклад помилок українською; Pydantic-схеми envelope; mock-envelope; префетч ваг і lifespan; валідація аплоаду й збереження пар у датасет; 8 маршрутів, з них `measure` — 180 рядків разом із проксі на Space і архівацією |
| `webui/storage.py` | 714 | S3-клієнт, Supabase REST, локальна архівація, шифрування sealed box, ретраї |
| `webui/visualize.py` | 541 | одна функція `_draw_measure_lines` на 310 рядків |
| `webui/envelope.py` | 510 | каталог 18 мірок, таблиці корекцій з історією підгонки, `_derive_*`, збирання envelope |
| `static/js/capture/session.js` | 967 | камера, завантаження MediaPipe, стабільність пози, автозйомка, таймер (28 експортів) |
| `static/js/capture/tailoring.js` | 843 | каталог одягу, вкладки розмірів, рендер, обробник заміру, `fetch` на API |
| `static/js/capture/i18n.js` | 1225 | переважно словники uk/en |

Від `webui.app` залежать лише `api/index.py` (`from webui.app import app`), `__main__.py`
(`"webui.app:app"`), CI-перевірка docker і лінивий імпорт у `envelope.py`. Тож `webui.app:app` можна
лишити як сумісну точку входу — жоден деплой не зачіпається.

---

## 2. Цільова структура бекенду

Назву пакета `webui` зберігаємо (інакше міняються Dockerfile, `api/index.py`, entry points і `stage_*.sh`).

```
src/webui/
  app.py                    # сумісність: `from webui.bootstrap import create_app; app = create_app()`
  __main__.py               # без змін (uvicorn "webui.app:app")
  config.py                 # Settings: один dataclass з env + get_settings() з кешем
                            #   (без pydantic-settings — не додаємо залежність у requirements.txt Vercel)
  errors.py                 # AppError(code, status, message_uk) + підкласи; мапа pipeline ValueError → UK

  bootstrap/                # = app/ у Video_streaming: лише складання
    factory.py              # create_app(use_lifespan=True) -> FastAPI
    routers.py              # include_routers(app)
    middleware.py           # CORS, Permissions-Policy
    exceptions.py           # AppError / RequestValidationError (UK) / 500
    lifespan.py             # proxy-режим: одразу yield; інакше weights.prefetch → pipeline → warmup → storage probe

  api/                      # тонкі роутери: розпарсити запит → викликати сервіс → повернути схему
    pages.py                # GET /, /dataset.html
    health.py               # GET /api/health, /api/keepalive
    measure.py              # POST /api/measure, /api/measure/mock
    tts.py                  # POST /api/tts
    dependencies.py         # get_settings, get_pipeline (503 якщо не завантажено), get_measure_service

  schemas/
    envelope.py             # MeasurementEnvelope, MeasurementItem, PipelineInfo, SubjectInfo, Capture*
    tts.py                  # TtsRequest

  services/                 # бізнес-логіка, без FastAPI-типів у сигнатурах
    measurement.py          # MeasureService.run(uploads, height, sex, backend) — decode → downscale →
                            #   save pair → pipeline → envelope → archive; calibration-failed гілка
    proxy.py                # forward_measure_to_space() (httpx), keepalive_ping()
    mock.py                 # build_mock_measurement_envelope
    uploads.py              # перевірка magic bytes / розміру / content-type, decode (cv2 ліниво)
    dataset_capture.py      # unique stem, збереження пари, UK-попередження

  envelope/                 # доменна частина envelope (зараз envelope.py)
    catalog.py              # CANONICAL_MEASUREMENTS, DISPLAY ids, default confidence, plausible ranges
    corrections.py          # _SEX_CIRCUMFERENCE_SCALES_PCT, _LENGTH_SCALES_PCT + коментарі-provenance дослівно
    derive.py               # _derive_* (chest, back/front length, neck base, upper arm, ankle)
    build.py                # body_to_envelope
    __init__.py             # реекспорт: body_to_envelope, CANONICAL_MEASUREMENTS, DISPLAY_MEASUREMENT_IDS

  infrastructure/
    inference.py            # WebuiPipeline, InferenceResult (зараз inference.py)
    weights.py              # _prefetch_weights (bucket mount → HF Hub → S3)
    tts_edge.py             # edge-tts (зараз tts.py)
    storage/
      __init__.py           # is_enabled, archive_measurement(_local), download_to_path, local_model_path
      config.py             # S3Config, _load_config, префікси env
      s3.py                 # _LazyClient, _put_with_retry, помилки
      supabase.py           # REST backend
      local.py              # LOCAL_DATA_DIR
      crypto.py             # sealed box
    visualize/
      primitives.py         # шрифти, текст, pose/seg overlay, png_b64
      measure_lines.py      # _draw_measure_lines, розбита по мірках
      __init__.py           # pipeline_visualizations_b64
```

Правила (записати в `CLAUDE.md` після завершення):

1. `api/*` не містить логіки: лише парсинг, `Depends`, виклик сервісу.
2. `services/*` не імпортують `fastapi` (крім `UploadFile` через тонкий адаптер у `api/`), тестуються без HTTP.
3. Важкі імпорти (`cv2`, `torch`, `ultralytics`, `boto3`) — лише всередині функцій у `services/uploads.py`,
   `infrastructure/*`. Шлях `api/index.py → bootstrap → api/* → services/proxy|mock` мусить імпортуватись у venv
   лише з `requirements.txt` (`vercel-smoke` це вже перевіряє).
4. Помилки: сервіс кидає `AppError`, один обробник у `bootstrap/exceptions.py` перетворює на `{"detail": ...}`
   (формат відповіді як зараз — фронтенд читає `detail`).
5. `__init__.py` кожного пакета реекспортує публічні імена.

---

## 3. Цільова структура фронтенду (vanilla, без збирача)

```
src/webui/static/js/
  api/
    client.js               # єдиний fetch: base URL (same-origin), таймаут/AbortController, розбір {detail},
                            #   translateBackendError; жоден інший модуль не викликає fetch напряму
    measure.js              # measure(formData, {withViz}), measureMock(height, sex)
    tts.js                  # synthesize(text) (+ сесійний kill-switch зі speech.js)
  i18n/
    index.js                # initI18n, setLang, t, translatePage (API як зараз)
    uk.js, en.js            # словники
  capture/
    camera.js               # (є)
    poseModel.js            # loadPoseLandmarker(Image) — з session.js
    poseStability.js        # стабільна поза, hold, countdown — з session.js
    autoCapture.js          # автозйомка/таймер, readiness — з session.js
    session.js              # лише оркестрація кроків (ціль < 300 рядків)
    poseGate.js, overlay.js, speech.js, ui.js, dom.js, state.js   # без змін або дрібні правки
  sizing/
    catalog.js              # loadTailoringCatalog, translateGarment
    sizeTabs.js             # selectSizeTab, ensureSizeTabsWired
    render.js               # refreshTailoringView, renderGarmentStrip, escapeHtml
    measureFlow.js          # attachMeasureHandler (через api/measure.js)
  dataset/ …                # розбиття app.js (648) — окремим кроком, після MaksShu
  sizeEngine.js, patternEngine.js, guideGeometry.js              # без змін
```

Шляхи статики зміняться → оновити `?v=` у `index.html`/`dataset.html`; `check_vercelignore.py` і
`node --check` у CI вже покривають нові файли.

---

## 4. Порядок міграції (кожен крок — окремий коміт/PR, деплой лише після зеленого CI)

Ворота на **кожному** кроці:
- `pytest` (unit + geometry snapshot, 0 відмінностей);
- новий `tests/test_api_contract.py` (крок 1);
- `vercel-smoke`, `docker` (імпорт застосунку в образі);
- локально: `pointsx-eval --pose-backend coco` на 17 суб'єктах — звіт байт у байт як до кроку;
- для фронтенд-кроків: ручна перевірка в браузері (камера → замір → розміри; dataset) + `node --check`.

| # | Крок | Ризик |
|---|---|---|
| 1 | **Спершу тести контракту** (`TestClient`, без lifespan): `/api/health` форма; mock → 18 мірок; `/api/measure` без pipeline → 503 з UK-текстом; невалідна форма → 422 з UK-текстом; порожній/не-зображення файл → 400; proxy-режим → 502 на недоступному upstream; keepalive 401/502; `Permissions-Policy`. Фіксує поведінку **до** рефакторингу | низький |
| 2 | `config.py`, `errors.py`, `schemas/` — перенесення без зміни логіки; `envelope.py` імпортує схеми звідти (прибирається циклічний `_envelope_models()`) | низький |
| 3 | `services/{uploads,dataset_capture,mock,proxy}.py`, `infrastructure/weights.py` | низький |
| 4 | `services/measurement.py` + `api/*` + `bootstrap/*`; `app.py` → 5 рядків. `create_app(use_lifespan=False)` для тестів | середній (180-рядковий `measure`) |
| 5 | `infrastructure/storage/` (розбиття 714 рядків), `infrastructure/inference.py`, `tts_edge.py` | низький; архів перевірити вручну з `LOCAL_DATA_DIR` |
| 6 | `envelope/` (catalog / corrections / derive / build) — коментарі з історією підгонки **дослівно** | низький (snapshot + eval ловлять будь-яку зміну) |
| 7 | `visualize/` — розбиття `_draw_measure_lines` | низький; порівняти PNG до/після побайтово на 3 синтетичних тілах |
| 8 | Фронтенд: `api/client.js` + `measure.js` + `tts.js`; `i18n/` на `uk.js`/`en.js` | середній (браузер) |
| 9 | Фронтенд: розбиття `session.js`, `tailoring.js` | середній |
| 10 | `CLAUDE.md`: правила шарів; `docs/` оновити | — |

Орієнтовно: кроки 1–4 — одна робоча сесія; 5–7 — ще одна; 8–9 — окремо, з ручним тестом на телефоні.

Поза цим планом (окремий трек, уже описаний у `docs/maintainability-audit.md` §4): розбиття ядра
`pointsx/` на `slicing/anatomy/measure/girth/corrections/catalog` — там змінюється вимірювальна
логіка і потрібні інші ворота (BodyM + app-корпус).

---

## 5. Рішення, які потрібні від власника

| # | Питання | Рекомендація |
|---|---|---|
| 1 | Назва пакета `webui` лишається? | **Так** — інакше зачіпаються обидва деплої |
| 2 | Налаштування: dataclass + env чи `pydantic-settings`? | **dataclass** — без нової залежності на Vercel |
| 3 | Фронтенд: лишаємось на vanilla чи переходимо на Vite/React як у Video_streaming? | **vanilla зараз**: перехід на збирач міняє модель деплою Vercel і не зменшує файли сам по собі; переглянути пізніше |
| 4 | `dataset/app.js` (код MaksShu) розбивати зараз? | **Пізніше**, після узгодження з MaksShu |
| 5 | Робити через PR (з CI-перевіркою до merge) чи прямо в `main`? | **PR на кожен крок** — деплой лише після merge |

**Затверджено 2026-09-14:** кроки 1–4 (бекенд) зараз; фронтенд лишається vanilla, розбиття — пізніше;
один PR на крок (гілки `refactor/webui-NN-*`, деплой лише після merge у `main`);
`dataset/app.js` — після розмови з MaksShu. Рішення 1 і 2 — за рекомендацією.

**Журнал:**

- Крок 1 (`refactor/webui-01-contract-tests`): `tests/test_api_contract.py`, 21 тест контракту, зелені на старому `app.py`.
- Крок 2 (`refactor/webui-02-config-errors-schemas`): `webui/config.py` (`Settings.from_env`, `get_settings`),
  `webui/errors.py`, `webui/schemas/`; `envelope.py` імпортує схеми напряму (циклічний імпорт через `app.py` прибрано);
  `app.py` 1042 → 890 рядків. Перевірено: pytest 45/45 (+ `tests/test_config.py`), `vercel_smoke` у venv лише з
  `requirements.txt`, `pointsx-eval` на 17 суб'єктах байт у байт як до міграції, реальний старт lifespan з вагами
  (`pipeline_ready`, `coco`).
- Крок 3 (`refactor/webui-03-services`): `webui/services/{uploads,dataset_capture,mock,proxy}.py`,
  `webui/infrastructure/weights.py`, `AppError` у `webui/errors.py` з одним обробником у `app.py` (та сама форма
  `{"detail": ...}`); сервіси не імпортують FastAPI і мають власні тести (`tests/test_services.py`).
  `app.py` 890 → ~590 рядків. Перевірено: pytest 51/51, `vercel_smoke`, `pointsx-eval` байт у байт,
  реальний старт lifespan з вагами.
- Крок 4 (`refactor/webui-04-routers-factory`): `webui/bootstrap/{factory,lifespan,routers,middleware,exceptions}.py`,
  `webui/api/{pages,health,measure,tts,dependencies}.py`, `webui/services/measurement.py`; `app.py` — 27 рядків
  (`app = create_app()`). Перевірка «модель не завантажена» лишилась у сервісі, не в `Depends`, щоб 422 форми
  спрацьовував раніше за 503 (контракт). Новий `tests/test_bootstrap.py` фіксує точний список маршрутів.
  Перевірено: pytest 53/53, `vercel_smoke`, `pointsx-eval` байт у байт, реальний старт з вагами,
  `/api/measure` на 3 реальних фото через HTTP — JSON старого `app.py` і нового застосунку ідентичний.
- Крок 5 (`refactor/webui-05-infrastructure`): `storage.py` (714 рядків) → `webui/infrastructure/storage/`
  {`s3`, `supabase`, `local`, `crypto`, `common`} + `__init__` з публічним API (диспетчер Supabase → S3);
  `inference.py` → `infrastructure/inference.py`, `tts.py` → `infrastructure/tts_edge.py` (`git mv`, імпортери оновлено,
  шимів немає). Спершу `tests/test_storage.py` (11 тестів: конфіг і префікси env, локальний архів, Supabase plain/sealed,
  S3 через fake boto3 з ретраєм) зафіксовано на старому модулі — зелені й після розбиття.
  Перевірено: pytest 64/64, `vercel_smoke`, `pointsx-eval` байт у байт, реальний старт з вагами.
- Крок 6 (`refactor/webui-06-envelope`): `envelope.py` → пакет `webui/envelope/` {`catalog`, `corrections`, `derive`,
  `build`}; нарізка за діапазонами рядків з перевірками-якорями, тож таблиці корекцій і коментарі з історією підгонки
  перенесені дослівно; `webui.envelope` імпортується як раніше (`__init__` реекспортує). Перевірено: pytest 64/64,
  `vercel_smoke`, `pointsx-eval` байт у байт, хеші оверлеїв і envelope на синтетичних тілах = базові,
  `/api/measure` з оверлеями на 3 реальних фото = базові (значення й PNG).
- Крок 7 (`refactor/webui-07-visualize`): `visualize.py` → пакет `webui/visualize/` {`primitives`, `mask_spans`,
  `measure_lines`}; 310-рядкова `_draw_measure_lines` → 9 функцій-секцій над спільним `_Canvas` (стан розміщення
  підписів), викликаються в тому самому порядку; тіла нарізані дослівно. Шлях до вбудованого шрифту виправлено на
  `parents[1]` (модуль на рівень глибше). Перевірено: pytest 64/64, хеші всіх 7 PNG на 6 синтетичних тілах = базові,
  `/api/measure` з оверлеями на 3 реальних фото = базові, `vercel_smoke`.
- Крок 8 (`refactor/webui-08-frontend-api-i18n`): `static/js/api/{client,measure,tts}.js` — жоден модуль потоку зйомки не
  викликає `fetch` напряму (виняток: `guideGeometry.js` і `dataset/*`); кеш і kill-switch TTS перенесено дослівно, таймаут
  8 с так само покриває й тіло відповіді. `capture/i18n.js` (1225) → `static/js/i18n/{uk,en,index}.js`, згенеровано з
  обчисленого об'єкта з коментарями секцій; `capture/i18n.js` лишився реекспортом для `dataset/app.js`.
  Перевірено: 1986 перевірок i18n (312 ключів × uk/en/інша мова, 21 одяг, 7 помилок бекенду) = старий модуль; усі JS
  парсяться; побічно старий і новий білд у браузері (без моделей, TTS вимкнено): текст сторінки uk/en, результати mock,
  зміна одягу, вкладки й фокус, перемикання мови, тексти помилок uk/en, один запит `/api/tts` і спрацювання kill-switch —
  ідентичні; імпорт усіх 25 модулів без помилок. Порівняння одразу виявило втрачений `stopActiveAudio` у `speech.js` —
  виправлено до коміту. Два падіння в `sizeEngine.consensus.test.js` існують і на кроці 7 (не пов'язані).
- Крок 9 (`refactor/webui-09-frontend-split`): `capture/session.js` 967 → 474 + `poseModel` (161), `imageOps` (64),
  `uploads` (258), `autoCapture` (132); `capture/tailoring.js` 843 → фасад на 33 рядки + `static/js/sizing/`
  {`catalog`, `text`, `measurementOrder`, `render` (302), `sizeTabs`, `view`, `measureFlow` (258)}. Оголошення перенесено
  дослівно скриптом; імпорти згенеровано за фактичним використанням; фасади `session.js` / `tailoring.js` реекспортують
  рівно ті самі 28 / 8 імен (перевірено скриптом і в браузері), тож `app.js` і `dataset/app.js` не змінено.
  Перевірено: старий (крок 7) і новий білди в браузері — 24 перевірки ідентичні (текст uk/en, mock, одяг, вкладки, фокус,
  мова, помилки uk/en, TTS, експорти фасадів, завантаження фото з квадратною обрізкою, відмова за типом і розміром,
  таймер без камери, підпис кнопки) + сторінка датасету; імпорт усіх 36 модулів; окрема статична перевірка областей
  видимості (шляхи камери, які не запускаються в браузерній панелі); `node --check`; `sizeEngine` без змін (ті самі 2 падіння).
