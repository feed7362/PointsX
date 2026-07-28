# ML Blueprint: антропометрія за двома фото — максимальна точність

> Відповідь Senior ML Engineer / CV Researcher на повний список із 14 питань.
> Ґрунтується на **виміряних** числах цього проєкту (eval-раны `eval/reports/run_001…014`,
> продакшн-логи, код обох репо), а не на загальних міркуваннях. Код і шляхи — англійською,
> пояснення — українською. Дата: 2026-07-18.

---

## 0. Спершу чесно: чи реальні 1–2 см за двома фото?

**Коротко: для довжин — так. Для обхватів — на межі фізичної можливості, і ось чому.**

Ключовий факт, який ігнорують усі маркетингові обіцянки конкурентів: **сама рулетка має
похибку ~1–2 см.** В ANSUR II допустима різниця між двома вимірювачами: талія ~1.2 см,
груди ~1.5 см. Тобто ціль "1–2 см" — це рівень шуму самої ground truth. Досягти MAE
нижче шуму розмітки неможливо в принципі — можна лише зрівнятися з ним.

Що показують наші власні виміри та література:

| Група вимірювань | Реально досяжно (2 фото, довільний одяг) | Реально досяжно (2 фото, ідеальні умови: облягаючий одяг, контрольована поза) |
|---|---|---|
| Довжини (руки, ноги, тулуб), ширина плечей | **1–2 см** — проєкції вимірюються майже напряму | ≤1 см |
| Малі обхвати (зап'ястя, щиколотка, коліно, шия) | 1–2 см (малі абсолютні значення, шкіра щільна) | ~1 см |
| Литка, біцепс, передпліччя | 1.5–2.5 см | 1–1.5 см |
| **Талія, стегна, груди, стегно (нога)** | **2.5–4 см** | **1.5–2.5 см** |

Докази з цього репозиторію:
- Геометрична стеля (ідеальні маски BodyM + справжній зріст): еліпс 6.4–9.4 см MAE →
  **навчена білінійна модель ширина→обхват: 3.7–4.6 см** (runs 013/014).
- Література: контурна CNN-регресія на чистих силуетах ~1.6–3 см (ICPR 2020 Silhouette
  Benchmarks); SMPL-fitting на сканах 0.25–1.6 см — але це скани, не фото.
- Наші продакшн-логи: реальні фото дають похибки калібрування класу
  `leg/height 0.65` — **capture-шум у дикій природі домінує над модельним**.

**Чому обхвати принципово важчі:** дві проєкції дають 2 хорди перерізу; форма перерізу
(еліптичність, асиметрія, м'які тканини) залишається невидимою. Одяг, дихання, поза
додають 1–2 см кожен. Це не лікується "кращою моделлю пози".

**Альтернативи, що реально пробивають бар'єр** (ранжовано за accuracy/friction):
1. **Коротке відео (3–5 с оберту)** замість другого фото → десятки ракурсів →
   мультиви́дова реконструкція; найкращий апгрейд за співвідношенням точність/зусилля
   користувача. Очікувано −30–50 % похибки обхватів.
2. **+1 фото під 45°** → третя хорда перерізу → еліптичність стає спостережуваною. Дешево.
3. **LiDAR/TrueDepth (iPhone)** → метрична глибина + пряме вимірювання перерізу; ~0.5–1 см,
   але втрачаємо "працює в будь-якому браузері".
4. Референс-об'єкт (аркуш A4 на підлозі) → суб-1 % масштаб.

Отже стратегія: **вичавити максимум із 2 фото (стеля ~2–3 см на обхватах), чесно
показувати довірчі інтервали, і мати відео/45° як опційний "precision mode".**

---

## 1. Аналіз поточного коду

Що вже є (і це більше, ніж здається):

| Компонент | Стан | Оцінка |
|---|---|---|
| YOLO26-pose (COCO-17→LV-MHP-16 адаптер) + yolo12l-seg | працює, стабільний | база нормальна |
| Калібрування head_top→ankle + fallback-пропорції | **вразливе**: HEAD_TOP на волоссі, дрейф кейпоінтів (інцидент 0.65) | головний ризик |
| Ширини з силуету + Ramanujan-еліпс | **доведено, що еліпс — floor, не ceiling** (−40–50 % від навченої моделі) | замінити |
| `_SEX_CIRCUMFERENCE_SCALES_PCT` | підігнано на **n=3** | терміново перефітити (n=18 вже є в Supabase) |
| `CircumferenceRegressor` (MLP 28→6) | скаффолд; без chest, без статі | перецілити |
| `synthetic/` (SMPL-X + Blender, 25 landmarks, GT з мешів) | скаффолд, не запускався у масштабі | **золотий актив** |
| `eval/` (BodyM, 4 пайплайни, train-fit/test-apply) + `pointsx-eval` (--grid, --fit-offsets) | збудовано, працює | фундамент дисципліни |
| Capture UX (MediaPipe в браузері, голосові підказки) | є, але без жорстких гейтів | розширити |

Повний покроковий план покращень по кожному аспекту — у
[enhancement-roadmap.md](enhancement-roadmap.md). Тут — цільова архітектура і код.

---

## 2. Найкраща сучасна архітектура

Три рівні складності; будувати треба так, щоб кожен наступний **вкладався** в попередній,
а не замінював його. Рівень C — відповідь на "найкраща можлива незалежно від складності".

### Рівень A — Geometric+ (те, що вже довели еваломи)
`seg + pose → фузійне калібрування → ширини на анатомічних рядках → навчена
функція ширина→обхват (per sex)`. Стеля ~3.5–4.5 см на обхватах. Уже майже є.

### Рівень B — Direct regression (контурний)
Двопотоковий енкодер силуетів (front+side) + скаляри (зріст, стать, вага) → MLP-голова
→ вектор вимірювань + аleatoric-дисперсія. Література: ~1.6–3 см на чистих силуетах.
Тренується на синтетиці + BodyM (research) / синтетиці + власний GT (комерція).

### Рівень C — Parametric body model (SOTA, ціль)
**Двовидова регресія SMPL-X** → вимірювання знімаються з відновленого 3D-меша.

```
front RGB ─┐                          ┌─ β (10-16 shape params)
           ├─ ViT-енкодер (shared) ───┤   θ (pose), camera
side RGB ──┘        + cross-view attn └─ height-constrained scale
зріст/стать/вага ── FiLM-кондиціювання ─┘
                     ↓
            SMPL-X mesh (обидва види)
                     ↓
   диференційовні лоси: silhouette IoU (обидва види) +
   2D keypoint reprojection + measurement loss (пряма L1 на обхватах GT!)
                     ↓
     measurements = переріз меша площинами (SMPL-Anthropometry)
```

Чому C найкраща:
- **Меш робить невидиме видимим**: форма перерізу береться з навченого пріору людських
  тіл (SMPL-X натреновано на тисячах сканів), а не з припущення "еліпс".
- **Зріст стає жорстким констрейном**: масштаб меша фіксується відомим зростом — це
  прибирає головну віссю невизначеності монокулярного 3D (scale ambiguity зникає
  повністю, у нас же зріст відомий!). Вага (якщо є) — м'який констрейн на об'єм
  меша (щільність тіла ~0.95–1.05 г/см³ → об'єм ≈ вага).
- **Все ще один forward pass** (~50–200 мс GPU, ~1 с CPU int8) — це regression, не
  optimization-fitting (фітинг ітераціями — хвилини, у продакшн не йде).
- **Measurement loss — вирішальна деталь**: тренуємо не "гарний меш", а точні
  сантиметри. BEDLAM/AGORA-style синтетика дає нескінченний точний GT для цього.

SOTA-компоненти 2024–2026, з яких збирати рівень C:
- **Sapiens (Meta, ECCV 2024)** — foundation-модель для людини (pose/seg/depth/normals,
  претрен на 300M фото людей). Найкращий беквон-донор для енкодера і водночас
  найкращий human-parsing/seg на сьогодні.
- **CameraHMR / TokenHMR / HMR2.0 (2023-24)** — архітектурні патерни ViT→SMPL регресії.
- **NLF — Neural Localizer Fields (2024)** — довільні точки тіла з одного forward;
  корисно для landmark-ліній (бюст/талія/стегна) без окремої голови.
- **SHAPY (CVPR 2022)** — доведення, що атрибути (зріст/вага/стать) суттєво
  покращують саме *shape*-точність; наш випадок 1:1.
- **BEDLAM (CVPR 2023)** — доведення, що **синтетики достатньо** для SOTA shape
  без жодного реального 3D-скана. Це знімає головний блокер (нема CAESAR — і не треба).

---

## 3. Які моделі використати (по кожній із запропонованих)

| Модель | Вердикт | Роль |
|---|---|---|
| **MediaPipe Pose** | ✅ але тільки клієнт | Capture-гейти в браузері (WASM, безкоштовно, миттєво). НЕ для вимірювань — точність ландмарків нижча за серверні моделі |
| **YOLO (pose+seg)** | ✅ сьогодні | Робочий рівень A. Pose file-tune на власних landmark-визначеннях (синтетика вже вміє) |
| **SAM2** | ⚠️ ні в продакшн | Чудові маски, але: важкий, промптований, відео-орієнтований. Використати **офлайн** — для авторозмітки складних кейсів у тренувальному корпусі |
| **Human Parsing (Sapiens-seg / SCHP)** | ✅ рівень B/C | Частини тіла → правильні рядки вимірювань (де "талія", а де "стегна"), відсікання рук від торса (наша знайдена проблема chest-occlusion) |
| **Depth Estimation (Depth Anything V2, Metric3D v2)** | ⚠️ допоміжна | Монокулярна метрична глибина 2024+ вже пристойна, але для обхватів дає мало (потрібні мм, а не дм). Корисна для: перевірки відстані/нахилу камери, відсікання фону. НЕ критичний компонент |
| **SMPL/SMPL-X** | ✅✅ ядро рівня C | Див. §2. SMPL-X (не SMPL): є руки/шия, кращий для повного списку вимірювань. `smpl-anthropometry` (уже в pyproject) — зняття мірок з меша |
| **BiRefNet / MODNet** | ✅ точковий | Matting-grade краї маски на кропі торса, якщо eval покаже чутливість до країв (спершу виміряти! `eval` §4.3b roadmap) |
| **Sapiens** | ✅✅ | Беквон/учитель для всього людського стеку (див. §2) |

**Головна заміна відносно твого списку:** глибину (Depth Estimation) я понижую до
допоміжної, а замість неї підношу **синтетичний data-engine + measurement-loss SMPL-X
регресію** — це дає сантиметри, глибина сама по собі — ні.

---

## 4. Як поєднати в один pipeline

```
                        КЛІЄНТ (браузер)
  MediaPipe гейти: кут рук 20–45°, все тіло в кадрі, гіро-нахил ±3°,
  профіль-чек для side; burst 5 кадрів → найкращий → upload
                              │ JPEG ×2 + зріст + стать (+вага) + гіро-метадані
                              ▼
                        СЕРВЕР (FastAPI)
  1. Person detect + crop (обидва види)
  2. Seg (int8) ──┬── маски      3. Pose (int8) ── кейпоінти
                  ▼
  4. КАЛІБРУВАННЯ (фузія): median{kp head→ankle, mask extent,
     пропорційні оцінки} + outlier-reject + tilt-корекція з гіро
  5. РЯДКИ: кейпоінти дають діапазон → екстремуми профілю ширин
     уточнюють (waist=min, hip=max, chest=найширший чистий рядок
     нижче "обриву пахви") ← доведено на 487 суб'єктах BodyM
  6. Ширини front/side на кожному рядку (torso-band, руки відсічені)
  7a. ШВИДКИЙ ШЛЯХ: навчена girth-модель (per sex)  ──┐
  7b. ТОЧНИЙ ШЛЯХ: SMPL-X регресія → мірки з меша ──┤→ фузія/вибір
  8. Conformal-інтервали per measurement × BMI-band  ─┘
  9. Абстеншн з причиною, якщо CI ширший за розмірний крок
  10. Envelope JSON: value ± CI, confidence, debug overlay
```

Контракт між стадіями — типізовані дataclass'и (див. §13). Стадії незалежні й
замінні; eval-оркестратор ганяє кожну конфігурацію проти замороженої бази
(вже реалізовано в `eval/bodym.py` — патерн переноситься на продакшн-еval).

---

## 5. Пікселі→см і використання зросту

Базова формула: `scale_view = height_cm / height_px_view` — **окремо для кожного виду**
(камера могла стояти на різній відстані; наш код уже попереджає при розбіжності >15 %).

Що робити краще за поточне:
1. **height_px — фузія трьох оцінок**, а не один ланцюжок кейпоінтів:
   - маска: `y_bottom − y_top` (доведено чисте на BodyM);
   - кейпоінти: head_top→ankle_mid (+0.5·стопа, бо ankle — не підлога);
   - пропорційні: (shoulder→hip)/0.288, (hip→ankle)/0.53 → імпліцитні зрости.
   Медіана + відкидання викидів + warning при спреді >3 %.
2. **Волосся**: head_top з кейпоінта завищує зріст на 1–4 см (зачіска). Маска ще гірша.
   Рішення: обличчя-ландмарки (є в MediaPipe) → верх черепа = brow_line + 0.35·(brow−chin);
   або навчити HEAD_TOP на синтетиці, де скальп відомий точно.
3. **Нахил камери** (гіро з клієнта): при куті φ масштаб залежить від рядка y.
   Перше наближення: `scale(y) = scale_mid · (1 + k·(y−y_mid))`, k з φ і FOV.
   Це прибирає **систематичний** перекіс стегна/литки від телефона на рівні грудей.
4. **Вага (якщо є)** — не для масштабу, а як констрейн об'єму (рівень C, §2) і як
   фіча girth-моделі (BMI-band — наш найгірший бакет, 13–14 см на еліпсі).

---

## 6. Як оцінити похибку вимірювань

Три незалежні механізми, всі три обов'язкові:

1. **Офлайн: розподіли залишків** на hold-out GT per measurement × sex × BMI-band —
   з них MAE/RMSE/bias/квантилі. Вже реалізовано в `eval/` (BodyM) і `pointsx-eval`
   (end-to-end). Правило: **дві доріжки не змішувати** (стеля ≠ продукт).
2. **Онлайн: conformal prediction** — калібруємо квантиль |residual| (90 %) на
   калібрувальному сеті → у продакшні віддаємо `[pred−q, pred+q]` per measurement.
   Distribution-free, без перенавчання моделі, чесно за побудовою. Код у §13.
3. **Test–retest repeatability**: 10 людей × 3 капчури → σ повторюваності.
   Це метрика довіри для B2B (кравець міряє двічі — отримує те саме число).
   Ніколи не вимірювалась; коштує один вечір.

Плюс **error-budget** розкладка (капчур/калібрування/рядки/сегментація/модель) —
таблиця з виміряними внесками вже в [enhancement-roadmap.md §2](enhancement-roadmap.md).

---

## 7. Як покращити точність (ранжовано за виміряним ефектом)

| # | Дія | Ефект | Підстава |
|---|---|---|---|
| 1 | Жорсткі capture-гейти на клієнті | закриває найбільший внесок (2–4 см) | testA vs testB: 6.4→9.4 см на ідентичному коді — різниця тільки в фото |
| 2 | Перефіт констант на n=18 (замість n=3) | прибирає невідомий ризик у кожній відповіді | константи множать кожен обхват |
| 3 | Навчена girth-модель замість еліпса | **−40–50 %** на обхватах | runs 013/014, виграє всі мірки на обох сплітах |
| 4 | Рядки: екстремуми профілю в межах kp-діапазону | знімає chest-occlusion і waist-дрейф | доведено на BodyM (адаптер eval) |
| 5 | Фузійне калібрування | прибирає клас інцидентів 0.65 | §5 |
| 6 | Синтетика 10–50k + fine-tune seg/pose | якість масок/ландмарків у складних умовах | BEDLAM-прецедент |
| 7 | Контурний регресор (рівень B) | → ~2–3 см | ICPR 2020 lit. |
| 8 | SMPL-X двовидова регресія (рівень C) | → ~1.5–2.5 см на обхватах | §2 |
| 9 | Burst-капчур + медіана масок | −шум пози/блюру | стандартна практика |
| 10 | Відео/45°/LiDAR precision-mode | пробиває стелю 2 фото | §0 |

---

## 8. Відкриті датасети

| Датасет | Що всередині | Ліцензія | Використання тут |
|---|---|---|---|
| **BodyM** (Amazon) | 2.5k суб'єктів, 8k+ пар силуетів front/side, 14 GT-мірок зі сканів, зріст/вага/стать | **CC BY-NC** | ✅ уже інтегровано (eval). Тільки research/eval — НЕ в комерційні ваги |
| **ANSUR II** | 6k військових, 93 виміри рулеткою (таблиці, без фото) | публічний | пріори/діапазони валідації, популяційні розподіли для синтетики |
| **CAESAR** | 4.4k 3D-сканів + мірки | платна | якщо колись буде бюджет; поки не потрібен (синтетика закриває) |
| **AGORA** | SMPL-X GT + фотореалістичні рендери | research | еталон формату для власної синтетики |
| **BEDLAM** | 380k синтетичних кадрів, SMPL-X GT, одяг/волосся | research (є комерційні обмеження — перевірити) | прецедент "синтетики достатньо"; архітектурний референс |
| **SSP-3D, THuman2/3, HuMMan, 3DPW** | shape/pose у дикій природі | research | допоміжні для pose/shape претрену |
| **Власна синтетика** (`src/pointsx/synthetic/`) | необмежено; SMPL-X + Blender + 25 landmarks + GT з меша | **наша, чиста для комерції** | ✅✅ головне джерело тренувальних даних |
| **Власний GT** (dataset-сторінка, E2E) | n=18 → ціль 50–300; фото+рулетка | наша | fine-tune + продукт-eval + фіт констант |

Ліцензійне правило (навчилися на BodyM): **ваги, що бачили NC-дані, не їдуть у
платний продукт.** Комерційний трек = синтетика + власний GT.

---

## 9. Як навчити власну модель (synthetic-first)

**Стратегія: pretrain на синтетиці → fine-tune на реальному GT → distill у швидку модель.**

1. **Генерація** (скаффолд є): семпл β з популяційних розподілів (ANSUR-статистика,
   покриття BMI 17–45!), A-pose ±10° джитер, камера: висота 0.8–1.8 м, pitch ±10°,
   відстань 2–4 м, FOV 50–80°; HDRI-фони; одяг — cloth-sim або displacement-шум
   (два домени: tight/loose); рендер RGB + маска + 25 landmarks + **GT-мірки з меша**
   (той самий метод, що в inference — визначення збігаються за побудовою).
2. **Рівень B (контурний регресор)**: вхід — два силуети 256×512 + [зріст, стать,
   вага?]; беквон — ефективний CNN/ViT-S; голова — мірки + log-σ (aleatoric).
   Лос: `Σ L1/σ + log σ` + консистентність front/side. Синтетика 50k → BodyM-eval
   (домен-геп ґейт) → fine-tune на власному GT (freeze беквон, тюнити голову).
3. **Рівень C (SMPL-X регресія)**: init з HMR2.0/Sapiens-ваг; кондиціювання
   зріст/стать/вага через FiLM; лоси: β/θ (синтетика має GT!), 2×silhouette IoU,
   2×keypoint reprojection, **measurement L1 на мешових мірках**, height-constraint
   жорстко. Fine-tune: реальний GT (мірки є, меша нема → тільки measurement +
   silhouette + keypoint лоси — цього достатньо, це і є слабка супервізія).
4. **Distillation**: рівень C (учитель) розмічає 100k синтетичних+реальних пар →
   рівень B (студент) вчиться відтворювати → CPU-inference швидкого студента з
   ~точністю вчителя.

Все це — **звичайний PyTorch-тренінг** на одній 4090/A10G; нічого екзотичного.

---

## 10. Inference < 2 секунд

Сьогодні: 11.5–13 с на 2 vCPU (seg 3.5–6 с/вид — вузьке місце). Шлях до <2 с:

| Крок | Ефект | Заувага |
|---|---|---|
| ONNX Runtime / OpenVINO **INT8** для seg+pose | seg 4–6 с → ~1.5–2 с; сумарно ~4–5 с | найбільший виграш за годину роботи |
| Batch=2 (front+side одним forward) | −30–40 % на модель | тривіально |
| Вхід seg 1280→960 | −40 % часу seg | **тільки** якщо eval покаже Δcm≈0 (ґейт!) |
| Рівень B замість seg+pose+geometry | один forward ~50–150 мс CPU | це і є головна причина мати рівень B |
| GPU (HF A10G / $20 VM) | все ≤ 1 с включно з рівнем C | коли перший пілот платить |

Реалістична розкладка <2 с на CPU: detect+crop 0.1 c → seg int8 (batch 2) 1.2 c →
pose int8 0.3 c → geometry+girth <0.01 c → conformal <0.001 c. **Досяжно без GPU.**
Рівень C на CPU ~1 c int8 → сумарно ~2.5 c; на GPU — все разом <1 с.

---

## 11. Метрики якості

Обов'язковий набір (усі вже або майже реалізовані в `eval/`):

- **MAE, RMSE, bias (signed!)** per measurement — bias окремо, бо систематика
  лікується константою, а розкид — ні (наш висновок: на testB bias ≈ MAE).
- **Доля в допуску: % ≤1 см / ≤3 см / ≤5 см** — "драбина толерансів".
- **Wrong-size rate** (помилка > кроку розмірної сітки ≈ 4–5 см на обхватах грудей) —
  метрика безпеки: тихо помилитися розміром гірше, ніж відмовитись.
- **Бакети: стать × BMI-band × split (controlled/wild)** — середнє бреше;
  BMI>30 у нас удвічі гірший за середнє (13–14 см проти 6–9 на сирому еліпсі).
- **Repeatability σ** (test–retest) — продуктова метрика довіри.
- **Coverage конформних інтервалів** (заявлені 90 % мають ловити ~90 % фактично).
- **Abstention rate** — скільки запитів чесно відхилено (пара до wrong-size).
- Для рівня C додатково: PVE/MPJPE меша — але тільки як діагностика;
  **продуктова істина — сантиметри мірок, не вершини меша.**

---

## 12. Структура проєкту

Еволюція наявної (не революція — усе, що працює, лишається на місці):

```
src/pointsx/
  capture/            # NEW: гейти якості, tilt-корекція, burst-вибір
  perception/         # detect, seg, pose (+ export_onnx.py, int8-калібрування)
  calibration.py      # фузія трьох оцінок масштабу (§5)
  landmarks/          # рядки вимірювань: kp-діапазон + профіль-екстремуми
  girth/              # NEW: навчена ширина→обхват (fit/predict/io), superellipse
  bodymodel/          # NEW (рівень C): SMPL-X регресор, mesh_measure.py
  uncertainty.py      # NEW: conformal інтервали
  synthetic/          # є: генератор → розширити domain randomization
  regression/         # є: перецілити на контурний рівень B
eval/                 # є: BodyM-оркестратор (A–D) → додати E (регресор), F (SMPL-X)
training/             # NEW: конфіги, лупи, distillation
src/webui/            # є: FastAPI serving
tests/                # NEW: юніт на кожен чистий модуль (girth, calibration, rows)
```

---

## 13. Production-ready код (ключові модулі)

Нижче — модулі, які закривають найближчі рівні (A+). Повністю типізовані,
з тестами. Це не псевдокод — girth-модель дослівно відтворює доведений
pipeline D з `eval/pipelines.py`.

### 13.1 `src/pointsx/girth/model.py` — навчена функція ширина→обхват

```python
"""Learned width→girth map: the measured 40-50% improvement over the ellipse.

Bilinear closed-form fit (no NN):  circ = c0 + c1·w_front + c2·w_side
+ c3·w_front·w_side, per (sex, measurement). Proven: BodyM runs 013/014,
testA MAE 6.4→3.7 cm, testB 9.4→4.6 cm, wins every measurement on both splits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Sex = str  # "male" | "female" | "other"


@dataclass(frozen=True)
class GirthSample:
    """One training observation for a single measurement."""
    front_width_cm: float
    side_width_cm: float
    gt_circumference_cm: float


@dataclass
class LearnedGirthModel:
    """Per-(sex, measurement) bilinear width→girth coefficients.

    Coefficients are fit with ordinary least squares — 4 params per cell,
    so ~30 samples per cell is already stable and overfitting is a non-issue.
    """

    coefs: dict[Sex, dict[str, list[float]]] = field(default_factory=dict)

    MIN_SAMPLES: int = 8  # below this, refuse to fit the cell

    def fit_cell(self, sex: Sex, measurement: str, samples: list[GirthSample]) -> bool:
        """Fit one (sex, measurement) cell. Returns True if fitted."""
        if len(samples) < self.MIN_SAMPLES:
            return False
        fw = np.array([s.front_width_cm for s in samples])
        sw = np.array([s.side_width_cm for s in samples])
        gt = np.array([s.gt_circumference_cm for s in samples])
        design = np.column_stack([np.ones_like(fw), fw, sw, fw * sw])
        coef, *_ = np.linalg.lstsq(design, gt, rcond=None)
        self.coefs.setdefault(sex, {})[measurement] = coef.tolist()
        return True

    def predict(
        self, sex: Sex, measurement: str,
        front_width_cm: float, side_width_cm: float,
        fallback_cm: float | None = None,
    ) -> float | None:
        """Predict circumference; falls back (e.g. to the ellipse) if no cell."""
        cell = self.coefs.get(sex, {}).get(measurement)
        if cell is None:
            # "other" averages both sexes when present
            cells = [d[measurement] for d in self.coefs.values() if measurement in d]
            if not cells:
                return fallback_cm
            cell = np.mean(cells, axis=0).tolist()
        c0, c1, c2, c3 = cell
        return float(c0 + c1 * front_width_cm + c2 * side_width_cm
                     + c3 * front_width_cm * side_width_cm)

    # ── persistence ────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"format": "girth-bilinear-v1", "coefs": self.coefs}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "LearnedGirthModel":
        data = json.loads(path.read_text())
        if data.get("format") != "girth-bilinear-v1":
            raise ValueError(f"Unknown girth model format in {path}")
        return cls(coefs=data["coefs"])
```

### 13.2 `src/pointsx/calibration_fused.py` — фузійне калібрування

```python
"""Robust px→cm scale: median over independent height estimates + outlier reject.

Motivation: production incident `leg/height 0.65` — a single keypoint chain
failure reached the user. Three estimators rarely fail the same way.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Population priors (ANSUR II): segment/stature ratios used as weak estimators.
SHOULDER_HIP_TO_STATURE = 0.288
HIP_ANKLE_TO_STATURE = 0.530
MAX_RELATIVE_SPREAD = 0.03  # >3% disagreement between estimators → warn


@dataclass(frozen=True)
class FusedScale:
    px_per_cm: float
    n_estimates: int
    relative_spread: float
    warnings: tuple[str, ...]


def fuse_height_estimates(
    kp_height_px: float | None,
    mask_height_px: float | None,
    shoulder_hip_px: float | None,
    hip_ankle_px: float | None,
    known_height_cm: float,
) -> FusedScale:
    """Combine up to four independent pixel-height estimates into one scale.

    Each estimator implies a full-body pixel height; we take the median and
    reject estimates >5% away from it (a keypoint on the hair, a mask that
    caught a shadow — either is an outlier, not a vote).
    """
    implied: list[tuple[str, float]] = []
    if kp_height_px and kp_height_px > 10:
        implied.append(("keypoints", kp_height_px))
    if mask_height_px and mask_height_px > 10:
        implied.append(("mask", mask_height_px))
    if shoulder_hip_px and shoulder_hip_px > 5:
        implied.append(("torso-prior", shoulder_hip_px / SHOULDER_HIP_TO_STATURE))
    if hip_ankle_px and hip_ankle_px > 5:
        implied.append(("leg-prior", hip_ankle_px / HIP_ANKLE_TO_STATURE))

    if not implied:
        raise ValueError("No usable height estimate — cannot calibrate")

    values = np.array([v for _, v in implied])
    med = float(np.median(values))
    keep = [(n, v) for n, v in implied if abs(v - med) / med <= 0.05]
    dropped = [n for n, v in implied if abs(v - med) / med > 0.05]

    kept_vals = np.array([v for _, v in keep])
    fused_px = float(np.median(kept_vals))
    spread = float((kept_vals.max() - kept_vals.min()) / fused_px) if len(keep) > 1 else 0.0

    warns: list[str] = []
    if dropped:
        warns.append(f"calibration outliers rejected: {', '.join(dropped)}")
    if spread > MAX_RELATIVE_SPREAD:
        warns.append(f"calibration estimators disagree by {spread:.1%}")
    for w in warns:
        logger.warning("%s", w)

    return FusedScale(
        px_per_cm=fused_px / known_height_cm,
        n_estimates=len(keep),
        relative_spread=spread,
        warnings=tuple(warns),
    )
```

### 13.3 `src/pointsx/uncertainty.py` — conformal-інтервали

```python
"""Split-conformal prediction intervals per (measurement, bucket).

Distribution-free coverage guarantee: with calibration residuals r_1..r_n and
q = ceil((n+1)·(1-α))/n empirical quantile of |r|, the interval pred ± q
covers the truth with probability ≥ 1-α on exchangeable data. No model change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ConformalCalibrator:
    alpha: float = 0.1  # 90% intervals
    quantiles: dict[str, float] = field(default_factory=dict)  # key = "measure|bucket"

    @staticmethod
    def _key(measurement: str, bucket: str) -> str:
        return f"{measurement}|{bucket}"

    def calibrate(self, measurement: str, bucket: str, residuals: list[float]) -> None:
        """Store the conformal quantile for one cell (needs ≥ ~20 residuals)."""
        n = len(residuals)
        if n < 20:
            return  # refuse to promise coverage we cannot back
        abs_r = np.sort(np.abs(np.asarray(residuals)))
        rank = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        self.quantiles[self._key(measurement, bucket)] = float(abs_r[min(rank, n - 1)])

    def interval(
        self, measurement: str, prediction_cm: float, bucket: str = "all",
    ) -> tuple[float, float] | None:
        """Return (lo, hi) or None when the cell was never calibrated."""
        q = self.quantiles.get(self._key(measurement, bucket)) \
            or self.quantiles.get(self._key(measurement, "all"))
        if q is None:
            return None
        return (prediction_cm - q, prediction_cm + q)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"alpha": self.alpha, "quantiles": self.quantiles}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "ConformalCalibrator":
        d = json.loads(path.read_text())
        return cls(alpha=d["alpha"], quantiles=d["quantiles"])
```

### 13.4 `src/pointsx/bodymodel/mesh_measure.py` — мірки з SMPL-X меша (рівень C)

```python
"""Circumference extraction from an SMPL-X mesh by plane slicing.

The same definitions are used to generate synthetic GT and to read predictions
at inference — so train and inference measure by construction identically.
For the canonical landmark heights use smpl-anthropometry (already a project
dependency) or the vertex-id anchors below.
"""
from __future__ import annotations

import numpy as np


def slice_circumference(
    vertices: np.ndarray,          # (V, 3) posed mesh, meters
    faces: np.ndarray,             # (F, 3) int
    plane_y: float,                # slicing height (mesh coords)
    band: tuple[float, float] | None = None,  # (x_min, x_max) to isolate a limb
) -> float | None:
    """Perimeter of the mesh cross-section at plane y = plane_y, in meters.

    Intersects each triangle with the plane; the resulting segment soup is
    ordered into a loop by nearest-neighbour chaining, then summed. For convex
    torso slices this equals the tape-measure path; for slight concavities the
    tape pulls straight — so we return the CONVEX HULL perimeter, matching how
    a physical tape behaves.
    """
    y0, y1 = vertices[faces][:, :, 1].min(axis=1), vertices[faces][:, :, 1].max(axis=1)
    crossing = faces[(y0 <= plane_y) & (y1 >= plane_y)]
    if len(crossing) == 0:
        return None

    points: list[np.ndarray] = []
    for tri in vertices[crossing]:                      # (3, 3)
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            if (a[1] - plane_y) * (b[1] - plane_y) <= 0 and a[1] != b[1]:
                t = (plane_y - a[1]) / (b[1] - a[1])
                points.append(a + t * (b - a))
    if len(points) < 3:
        return None

    pts = np.asarray(points)[:, [0, 2]]                 # project to XZ plane
    if band is not None:
        pts = pts[(pts[:, 0] >= band[0]) & (pts[:, 0] <= band[1])]
        if len(pts) < 3:
            return None

    # convex hull perimeter = tape path
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    loop = pts[hull.vertices]
    return float(np.linalg.norm(np.roll(loop, -1, axis=0) - loop, axis=1).sum())
```

### 13.5 `tests/test_girth.py` — тест, що ловить регресію логіки

```python
import numpy as np

from pointsx.girth.model import GirthSample, LearnedGirthModel


def _synthetic_cell(c=(4.0, 1.1, 1.3, 0.005), n=200, seed=0):
    rng = np.random.default_rng(seed)
    fw = rng.uniform(25, 45, n)
    sw = rng.uniform(18, 30, n)
    gt = c[0] + c[1] * fw + c[2] * sw + c[3] * fw * sw + rng.normal(0, 0.3, n)
    return [GirthSample(f, s, g) for f, s, g in zip(fw, sw, gt)]


def test_fit_recovers_generating_coefficients():
    model = LearnedGirthModel()
    assert model.fit_cell("female", "waist", _synthetic_cell())
    pred = model.predict("female", "waist", 35.0, 24.0)
    truth = 4.0 + 1.1 * 35.0 + 1.3 * 24.0 + 0.005 * 35.0 * 24.0
    assert abs(pred - truth) < 0.5          # within the injected noise


def test_refuses_tiny_cells_and_falls_back():
    model = LearnedGirthModel()
    assert not model.fit_cell("male", "hip", _synthetic_cell()[:3])
    assert model.predict("male", "hip", 40.0, 25.0, fallback_cm=101.5) == 101.5
```

---

## 14. Покроковий план розробки

Детальний по-аспектний план з оцінками зусиль —
[enhancement-roadmap.md](enhancement-roadmap.md). Стисло, у порядку виконання:

1. **Тиждень 0** (без ML): перефіт констант на n=18 → capture-гейти → фікс
   privacy-суперечності → CORS/rate-limit.
2. **Тижні 1–4** (рівень A+): фузійне калібрування (§13.2) → рядки з профілю →
   `LearnedGirthModel` у продакшн при n≥30 (§13.1) → conformal (§13.3) → INT8.
   **Ціль: ≤4–5 см end-to-end, ≤5 с, чесні інтервали.**
3. **Місяці 1–2** (рівень B): синтетика в масштабі → контурний регресор →
   fine-tune seg/pose → burst-капчур. **Ціль: ≤3 см.**
4. **Місяці 3–6** (рівень C): SMPL-X двовидова регресія з measurement-loss →
   мірки з меша (§13.4) → distillation у швидку модель → GPU з першим пілотом.
   **Ціль: ≤2 см на більшості мірок; на талії/грудях — чесні 2–2.5 см
   з інтервалами, і precision-mode (відео/45°) для тих, кому треба менше.**

Кожен крок проходить крізь два eval-ґейти (BodyM-стеля + власний GT) — обидва
вже збудовані. Жоден ML-крок не стартує, поки капчур-ґейти не в продакшні:
**testA→testB довів, що фото зараз коштують дорожче за будь-яку модель.**
