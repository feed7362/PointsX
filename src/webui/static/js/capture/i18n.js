export let currentLang = "uk";

const translations = {
  // Static HTML translations
  "title": {
    uk: "FitMeasure AI — мірки",
    en: "FitMeasure AI — measurements"
  },
  "header-sub": {
    uk: "Дві фотографії, зріст, стать → результат з сервера, розміри за сітками. Поза перевіряється в браузері.",
    en: "Two photos, height, sex → server result, size grids. Pose is checked in browser."
  },
  "btn-dataset-cta": {
    uk: "Долучитись до збору датасету",
    en: "Join Dataset Collection"
  },
  "btn-send-to-dataset": {
    uk: "Надіслати мірки до форми датасету",
    en: "Send measurements to dataset form"
  },
  "err-dataset-transfer-failed": {
    uk: "Не вдалося підготувати дані для датасету: {msg}",
    en: "Could not prepare data for the dataset form: {msg}"
  },
  "ref-heading": {
    uk: "Приклади правильної пози",
    en: "Proper Pose Examples"
  },
  "ref-intro": {
    uk: "<strong>Анфас:</strong> ноги приблизно на ширині плечей, пахви відкриті, руки злегка відведені на ~15–20° від корпусу.<br><strong>Профіль:</strong> стоїте боком, голова вздовж тіла, дивіться в той самий бік, куди звернене тіло, руки мають бути вздовж тіла.<br>Відстань до камери ~2&nbsp;м, повний зріст у кадрі.",
    en: "<strong>Front:</strong> feet shoulder-width apart, armpits open, arms slightly spread ~15–20° from body.<br><strong>Profile:</strong> stand sideways, head aligned, look in the direction the body is facing, arms should be along the body.<br>Camera distance ~2&nbsp;m, full height in frame."
  },
  "front": {
    uk: "Анфас",
    en: "Front"
  },
  "profile": {
    uk: "Профіль",
    en: "Profile"
  },
  "btn-to-capture": {
    uk: "Продовжити до зйомки",
    en: "Continue to Capture"
  },
  "params-heading": {
    uk: "Параметри",
    en: "Parameters"
  },
  "label-height": {
    uk: "Зріст (см)",
    en: "Height (cm)"
  },
  "label-sex": {
    uk: "Стать",
    en: "Sex"
  },
  "sex-male": {
    uk: "Чоловік",
    en: "Male"
  },
  "sex-female": {
    uk: "Жінка",
    en: "Female"
  },
  "sex-other": {
    uk: "Інше",
    en: "Other"
  },
  "capture-heading": {
    uk: "Зйомка",
    en: "Capture"
  },
  "preview-idle-title": {
    uk: "Камера вимкнена",
    en: "Camera is Off"
  },
  "preview-idle-hint": {
    uk: "Натисніть «Увімкнути камеру», щоб побачити превʼю й ескіз пози для заміру. Або завантажте фото.",
    en: "Click \"Turn on camera\" to see preview and pose outline for measurement. Or upload photos."
  },
  "btn-start": {
    uk: "Увімкнути камеру",
    en: "Turn on camera"
  },
  "btn-capture": {
    uk: "Зробити фото",
    en: "Take photo"
  },
  "btn-capture-timer": {
    uk: "Фото через 10 с",
    en: "Photo in 10s"
  },
  "btn-stop": {
    uk: "Зупинити камеру",
    en: "Stop camera"
  },
  "upload-intro": {
    uk: "Або завантажте готові фото (JPEG, PNG, WebP; до 5 МБ)",
    en: "Or upload ready photos (JPEG, PNG, WebP; up to 5 MB)"
  },
  "upload-front-label": {
    uk: "Завантажити анфас",
    en: "Upload Front"
  },
  "upload-side-label": {
    uk: "Завантажити профіль",
    en: "Upload Profile"
  },
  "upload-debug-span": {
    uk: "Не перевіряти позу при завантаженні файлу",
    en: "Skip pose check on file upload"
  },
  "btn-retake-front": {
    uk: "Перезняти анфас",
    en: "Retake Front"
  },
  "btn-retake-side": {
    uk: "Перезняти профіль",
    en: "Retake Profile"
  },
  "btn-measure": {
    uk: "Розрахувати мірки",
    en: "Calculate Measurements"
  },
  "btn-measure-test": {
    uk: "Тест результатів без фото",
    en: "Test results without photo"
  },
  "results-heading": {
    uk: "Результат",
    en: "Results"
  },
  "model-viz-heading": {
    uk: "Візуалізація моделей",
    en: "Model Visualizations"
  },
  "model-viz-hint": {
    uk: "YOLO pose та сегментація по кожному знімку (як бачить сервер).",
    en: "YOLO pose and segmentation for each photo (as seen by server)."
  },
  "garment-strip-label": {
    uk: "Тип одягу",
    en: "Garment Type"
  },
  "tailoring-measures-sub": {
    uk: "Мірки для обраного виробу",
    en: "Measurements for Selected Garment"
  },
  "table-th-measurement": {
    uk: "Показник",
    en: "Measurement"
  },
  "table-th-cm": {
    uk: "см",
    en: "cm"
  },
  "size-grids-sub": {
    uk: "Розмірні сітки (довідково)",
    en: "Size Grids (reference)"
  },
  "tab-ua": {
    uk: "Україна (орієнт.)",
    en: "Ukraine (approx.)"
  },
  "tab-eu": {
    uk: "Європа (орієнт.)",
    en: "Europe (approx.)"
  },
  "tab-us": {
    uk: "США (орієнт.)",
    en: "USA (approx.)"
  },
  "pattern-summary": {
    uk: "Індивідуальні мірки для пошиття",
    en: "Custom Tailoring Measurements"
  },
  "pattern-raw-heading": {
    uk: "Базові розміри (тіло + припуски на вільність)",
    en: "Base dimensions (body + ease allowance)"
  },
  "pattern-seam-heading": {
    uk: "З припусками на шви",
    en: "With seam allowance"
  },
  "pattern-hint": {
    uk: "Ці величини використовуються для індивідуального пошиття. Всі значення приблизні.",
    en: "These values are used for individual tailoring. All values are approximate."
  },
  "tailoring-disclaimer": {
    uk: "Розміри на вкладках — орієнтовні",
    en: "Sizes in tabs are approximate"
  },
  "all-measures-summary": {
    uk: "Усі мірки (повна таблиця)",
    en: "All Measurements (full table)"
  },
  "table-th-val-cm": {
    uk: "Значення (см)",
    en: "Value (cm)"
  },

  // Dynamic strings from JS (app.js, ui.js, camera.js, tailoring.js, poseGate.js, session.js, speech.js)
  "pose-model-ready": {
    uk: "Модель пози готова. Увімкніть камеру.",
    en: "Pose model ready. Turn on camera."
  },
  "pose-model-failed": {
    uk: "Модель пози не завантажена (офлайн?).",
    en: "Pose model not loaded (offline?)."
  },
  "both-photos-preview": {
    uk: "Обидва знімки в превʼю",
    en: "Both photos in preview"
  },
  "step-1-label": {
    uk: "Крок 1 з 2: анфас — пахви відкриті, ноги приблизно на ширині плечей",
    en: "Step 1 of 2: front — armpits open, feet shoulder-width apart"
  },
  "step-2-label": {
    uk: "Крок 2 з 2: профіль — боком до камери, руки мають бути вздовж тіла",
    en: "Step 2 of 2: profile — sideways to camera, arms should be along the body"
  },
  "loading-state": {
    uk: "Завантаження…",
    en: "Loading..."
  },
  "countdown-three": {
    uk: "три",
    en: "three"
  },
  "countdown-two": {
    uk: "два",
    en: "two"
  },
  "countdown-one": {
    uk: "один",
    en: "one"
  },
  "countdown-ready": {
    uk: "готово",
    en: "ready"
  },
  "camera-https-warning": {
    uk: "Камера в браузері потребує захищеного з'єднання (HTTPS) або localhost. Зараз відкрито: {url}. На телефоні в локальній мережі запустіть сервер з HTTPS (див. RUN-WEBUI.md) або завантажте фото з галереї замість камери.",
    en: "Browser camera access requires HTTPS or localhost. Current origin: {url}. Run local HTTPS server (see RUN-WEBUI.md) or upload photos from gallery instead."
  },
  "camera-browser-unsupported": {
    uk: "Цей браузер не підтримує доступ до камери. Спробуйте Chrome або Safari або завантажте фото.",
    en: "This browser does not support camera access. Try Chrome or Safari, or upload photos."
  },
  "camera-permission-denied": {
    uk: "Доступ до камери заборонено. Дозвольте камеру для цього сайту в налаштуваннях браузера (іконка замка / «Дозволи сайту») і натисніть «Увімкнути камеру» знову.",
    en: "Camera access denied. Please grant camera access in your browser settings (lock icon / Site Permissions) and click \"Turn on camera\" again."
  },
  "camera-not-found": {
    uk: "Камеру не знайдено на цьому пристрої.",
    en: "No camera found on this device."
  },
  "camera-busy": {
    uk: "Камера зайнята іншим застосунком або недоступна. Закрийте інші програми з камерою.",
    en: "Camera is busy or unavailable. Please close other apps using the camera."
  },
  "camera-params-unsupported": {
    uk: "Камера не підтримує обрані параметри. Спробуйте ще раз — застосунок спробує простіший режим.",
    en: "Camera does not support requested parameters. Try again — the app will try a simpler mode."
  },
  "camera-unknown-error": {
    uk: "Невідома помилка камери.",
    en: "Unknown camera error."
  },
  "failed-load-config": {
    uk: "Не вдалося завантажити tailoring_config.json",
    en: "Failed to load tailoring_config.json"
  },
  "sex-male-label": {
    uk: "чоловік",
    en: "male"
  },
  "sex-female-label": {
    uk: "жінка",
    en: "female"
  },
  "sex-other-label": {
    uk: "інше",
    en: "other"
  },
  "viz-front-pose": {
    uk: "Анфас — поза",
    en: "Front — pose"
  },
  "viz-front-seg": {
    uk: "Анфас — силует",
    en: "Front — silhouette"
  },
  "viz-front-measures": {
    uk: "Анфас — лінії зняття мірок",
    en: "Front — measurement lines"
  },
  "viz-side-pose": {
    uk: "Профіль — поза",
    en: "Profile — pose"
  },
  "viz-side-seg": {
    uk: "Профіль — силует",
    en: "Profile — silhouette"
  },
  "viz-side-measures": {
    uk: "Профіль — лінії зняття мірок",
    en: "Profile — measurement lines"
  },
  "verdict-unanimous": {
    uk: "Однозначно",
    en: "Unanimous"
  },
  "verdict-unanimous-edge": {
    uk: "Однозначно (крайній розмір)",
    en: "Unanimous (extreme size)"
  },
  "verdict-majority": {
    uk: "Більшість 2/3",
    en: "Majority 2/3"
  },
  "verdict-majority-edge": {
    uk: "Більшість (крайній розмір)",
    en: "Majority (extreme size)"
  },
  "verdict-no-consensus": {
    uk: "Неоднозначно",
    en: "Ambiguous"
  },
  "verdict-insufficient": {
    uk: "Недостатньо даних",
    en: "Insufficient data"
  },
  "approx-verdict": {
    uk: "орієнт. {code}",
    en: "approx. {code}"
  },
  "borderline-size": {
    uk: "На межі розмірів: {size1} / {size2}",
    en: "Borderline: {size1} / {size2}"
  },
  "no-table": {
    uk: "немає таблиці",
    en: "no grid"
  },
  "no-measurement": {
    uk: "немає мірки",
    en: "no measurement"
  },
  "sizing-failed": {
    uk: "Помилка: {msg}",
    en: "Error: {msg}"
  },

  // Bodice, sleeves, pants labels in UA & EN
  "armscye_depth": { uk: "Глибина пройми", en: "Armscye depth" },
  "sleeve_cap_height": { uk: "Висота окату рукава", en: "Sleeve cap height" },
  "sleeve_cap_width": { uk: "Ширина окату", en: "Sleeve cap width" },
  "chest_pattern": { uk: "Обхват грудей (лекало)", en: "Chest pattern circumference" },
  "waist_pattern": { uk: "Обхват талії (лекало)", en: "Waist pattern circumference" },
  "hip_pattern": { uk: "Обхват стегон (лекало)", en: "Hip pattern circumference" },
  "back_length_cb": { uk: "Довжина спини", en: "Back length (CB)" },
  "hip_line_depth": { uk: "Рівень лінії стегон", en: "Hip line depth" },
  "cross_back": { uk: "Ширина спини", en: "Cross back width" },
  "cross_front": { uk: "Ширина переду", en: "Cross front width" },
  "shoulder_seam": { uk: "Довжина плечового шва", en: "Shoulder seam length" },
  "neck_width": { uk: "Ширина горловини", en: "Neck width" },
  "neck_depth_front": { uk: "Глибина горловини (перед)", en: "Neck depth (front)" },
  "neck_depth_back": { uk: "Глибина горловини (спинка)", en: "Neck depth (back)" },
  "dart_intake_total": { uk: "Загальна виточка", en: "Total dart intake" },
  "dart_back": { uk: "Виточка спинки", en: "Back dart" },
  "dart_side": { uk: "Виточка бокова", en: "Side dart" },
  "dart_front": { uk: "Виточка переду", en: "Front dart" },
  "sleeve_underarm_length": { uk: "Довжина рукава (від пахви)", en: "Sleeve underarm length" },
  "bicep_pattern": { uk: "Обхват біцепса (лекало)", en: "Bicep pattern circumference" },
  "cuff_pattern": { uk: "Обхват манжета", en: "Cuff pattern circumference" },
  "crotch_depth": { uk: "Глибина сидіння", en: "Crotch depth" },
  "knee_pattern": { uk: "Обхват коліна (лекало)", en: "Knee pattern circumference" },
  "hem_width_pants": { uk: "Ширина низу штанини", en: "Pants hem width" },

  // Measurements labels
  "chest_circumference": { uk: "Обхват грудей", en: "Chest circumference" },
  "waist_circumference": { uk: "Обхват талії", en: "Waist circumference" },
  "hip_circumference": { uk: "Обхват стегон", en: "Hip circumference" },
  "neck_circumference": { uk: "Обхват шиї", en: "Neck circumference" },
  "neck_base_height": { uk: "Висота точки основи шиї", en: "Neck base height" },
  "shoulder_slope_width": { uk: "Ширина плечового ската", en: "Shoulder slope width" },
  "back_width_scapular": { uk: "Ширина спини (між лопатками)", en: "Back width (scapular)" },
  "chest_width_front": { uk: "Ширина грудей (між пахвами спереду)", en: "Chest width (front)" },
  "back_length_to_waist": { uk: "Довжина спини до талії (по хребту)", en: "Back length to waist" },
  "front_length_to_waist": { uk: "Довжина переду до талії (через найвищу точку грудей)", en: "Front length to waist" },
  "arm_length_shoulder_to_wrist": { uk: "Довжина руки (від плеча до зап'ястя)", en: "Arm length (shoulder to wrist)" },
  "upper_arm_circumference": { uk: "Обхват плеча (біцепс)", en: "Upper arm circumference" },
  "wrist_circumference": { uk: "Обхват зап'ястя", en: "Wrist circumference" },
  "leg_length_inner_seam": { uk: "Довжина ноги по внутрішньому шву", en: "Inseam (inner leg length)" },
  "leg_length_outer_seam": { uk: "Довжина ноги по зовнішньому шву", en: "Outseam (outer leg length)" },
  "thigh_circumference": { uk: "Обхват стегна", en: "Thigh circumference" },
  "calf_circumference": { uk: "Обхват гомілки (литки)", en: "Calf circumference" },
  "ankle_circumference": { uk: "Обхват щиколотки", en: "Ankle circumference" },

  // Sizing report intro
  "sizing-report-intro": {
    uk: "Зріст: {height} см. Стать: {sex}. Оберіть тип одягу для орієнтовних розмірів (Україна / Європа / США).",
    en: "Height: {height} cm. Sex: {sex}. Choose garment type for approximate sizes (Ukraine / Europe / USA)."
  },
  "sizing-pattern-error": {
    uk: "Пошив і сітки: {msg}",
    en: "Sizing & Pattern: {msg}"
  },

  // Loading Calculation Texts
  "loading-test-title": {
    uk: "Тестовий розрахунок...",
    en: "Test calculation..."
  },
  "loading-measure-title": {
    uk: "Обчислення мірок...",
    en: "Calculating measurements..."
  },
  "loading-test-step": {
    uk: "Генерація демо-даних...",
    en: "Generating demo data..."
  },
  "loading-measure-step": {
    uk: "Підготовка даних...",
    en: "Preparing data..."
  },
  "step-send-photos": {
    uk: "Надсилання знімків на сервер...",
    en: "Sending photos to server..."
  },
  "step-yolo-pose": {
    uk: "Локалізація ключових точок...",
    en: "Localizing keypoints..."
  },
  "step-yolo-seg": {
    uk: "Сегментація силуету тіла...",
    en: "Segmenting body silhouette..."
  },
  "step-calc-anthropometry": {
    uk: "Розрахунок обʼємів та антропометрії...",
    en: "Calculating volumes & anthropometry..."
  },
  "step-finish": {
    uk: "Завершення обчислень...",
    en: "Finishing calculations..."
  },

  // Errors / alerts in tailoring.js
  "err-need-both-photos": {
    uk: "Потрібні обидва знімки — анфас і профіль.",
    en: "Both photos (front and profile) are required."
  },
  "err-missing-results-container": {
    uk: "Помилка: немає контейнера результатів у розмітці.",
    en: "Error: results container not found in layout."
  },
  "err-height-range": {
    uk: "Вкажіть зріст від 100 до 250 см.",
    en: "Enter a height between 100 and 250 cm."
  },
  "status-test-calc": {
    uk: "Тестовий розрахунок…",
    en: "Test calculation..."
  },
  "status-calculating": {
    uk: "Обчислення…",
    en: "Calculating..."
  },
  "err-empty-measurements": {
    uk: "Сервер повернув порожній список мірок. Спробуйте інші знімки або перевірте позу й освітлення.",
    en: "Server returned empty measurements. Try other photos or check pose and lighting."
  },
  "status-done": {
    uk: "Готово.",
    en: "Done."
  },
  "err-request-failed": {
    uk: "Помилка запиту: {msg}",
    en: "Request error: {msg}"
  },

  // PoseGate reasons
  "pose-show-right-elbow": {
    uk: "Покажіть правий лікоть",
    en: "Show right elbow"
  },
  "pose-lower-arm": {
    uk: "Опустіть руку вздовж тіла",
    en: "Lower your arm along your body"
  },
  "pose-show-face-shoulders": {
    uk: "Покажіть обличчя і плечі в кадрі.",
    en: "Show face and shoulders in frame."
  },
  "pose-align-shoulders": {
    uk: "Вирівняйте плечі (не нахиляйте корпус).",
    en: "Align shoulders (do not tilt body)."
  },
  "pose-face-camera": {
    uk: "Поверніться обличчям до камери",
    en: "Face the camera"
  },
  "pose-show-head": {
    uk: "Покажіть всю голову в кадрі",
    en: "Show full head in frame"
  },
  "pose-lower-camera": {
    uk: "Опустіть камеру або відійдіть трохи, вся голова має бути в кадрі",
    en: "Lower the camera or step back, the entire head must be in the frame"
  },
  "pose-elbows-in-frame": {
    uk: "Лікті мають бути в кадрі",
    en: "Elbows must be in frame"
  },
  "pose-spread-arms": {
    uk: "Відведіть руки на ~15–20° від тіла",
    en: "Spread arms ~15-20° from body"
  },
  "pose-arms-too-wide": {
    uk: "Не розводьте руки занадто широко (достатньо ~15–20°).",
    en: "Do not spread arms too wide (approx. 15-20° is enough)."
  },
  "pose-open-armpits": {
    uk: "Лікті трохи вбік від корпусу, пахви мають залишатися відкритими",
    en: "Elbows slightly out, armpits must remain open"
  },
  "pose-stand-wider": {
    uk: "Спробуйте поставити ноги трохи ширше — приблизно на ширині плечей",
    en: "Try standing with feet slightly wider — about shoulder-width apart"
  },
  "pose-show-hips": {
    uk: "Має бути видно зону стегон і паху",
    en: "Hips and groin area must be visible"
  },
  "pose-dont-squeeze-thighs": {
    uk: "Не зводьте стегна",
    en: "Do not squeeze thighs"
  },
  "pose-show-feet": {
    uk: "Покажіть повний зріст, стопи повинні бути в кадрі",
    en: "Show full height, feet must be in frame"
  },
  "pose-feet-fully-in-frame": {
    uk: "Стопи мають бути повністю в кадрі",
    en: "Feet must be fully in frame"
  },
  "pose-dont-squat": {
    uk: "Станьте повним зростом на прямих ногах, не присідайте",
    en: "Stand full height on straight legs, do not squat"
  },
  "pose-cannot-see": {
    uk: "Не видно {parts}",
    en: "Cannot see {parts}"
  },
  "pose-out-of-frame": {
    uk: "Поза кадром {parts}",
    en: "Out of frame: {parts}"
  },
  "pose-larger-in-frame": {
    uk: "Покажіть себе крупніше в кадрі",
    en: "Position yourself larger in the frame"
  },
  "pose-turn-sideways": {
    uk: "Поверніться боком на ~90°",
    en: "Turn sideways ~90°"
  },
  "pose-stand-right-side": {
    uk: "У профілі стійте правим боком до камери",
    en: "In profile, stand with your right side to the camera"
  },
  "pose-dont-turn-head": {
    uk: "Не повертайте голову до камери",
    en: "Do not turn head to camera"
  },
  "pose-dont-turn-head-alt": {
    uk: "Не розвертайте голову до камери",
    en: "Do not turn head to camera"
  },
  "pose-look-same-direction": {
    uk: "Дивіться в той самий бік, куди звернене тіло .",
    en: "Look in the same direction the body is facing."
  },
  "pose-dont-tilt-torso": {
    uk: "Не нахиляйте корпус у профіль",
    en: "Do not tilt torso in profile"
  },
  "pose-lower-arm-torso": {
    uk: "Опустіть руку вздовж тулуба",
    en: "Lower your arm along your body"
  },
  "pose-straight-legs": {
    uk: "Станьте на рівних ногах",
    en: "Stand on straight legs"
  },
  "pose-show-full-height": {
    uk: "Покажіть повний зріст",
    en: "Show full height"
  },

  // Body parts names for visibility gate
  "part-head": { uk: "голову", en: "head" },
  "part-left-shoulder": { uk: "ліве плече", en: "left shoulder" },
  "part-right-shoulder": { uk: "праве плече", en: "right shoulder" },
  "part-left-elbow": { uk: "лівий лікоть", en: "left elbow" },
  "part-right-elbow": { uk: "правий лікоть", en: "right elbow" },
  "part-left-wrist": { uk: "ліву кисть", en: "left wrist" },
  "part-right-wrist": { uk: "праву кисть", en: "right wrist" },
  "part-left-hip": { uk: "таз зліва", en: "left hip" },
  "part-right-hip": { uk: "таз справа", en: "right hip" },
  "part-left-knee": { uk: "ліве коліно", en: "left knee" },
  "part-right-knee": { uk: "праве коліно", en: "right knee" },
  "part-left-ankle": { uk: "ліву щиколотку", en: "left ankle" },
  "part-right-ankle": { uk: "праву щиколотку", en: "right ankle" },

  // Session notifications
  "sess-camera-inactive": {
    uk: "Камера не активна. Увімкніть камеру.",
    en: "Camera is off. Turn on camera."
  },
  "sess-video-size-failed": {
    uk: "Не вдалося прочитати розмір відео. Зачекайте або перезапустіть камеру.",
    en: "Failed to read video size. Please wait or restart the camera."
  },
  "sess-low-resolution": {
    uk: "Занадто низька роздільна здатність ({vw}×{vh}). Потрібно щонайменше {min}px по кожній стороні.",
    en: "Resolution too low ({vw}x{vh}). Need at least {min}px on each side."
  },
  "sess-photo-timer-btn": {
    uk: "Фото через {sec} с",
    en: "Photo in {sec}s"
  },
  "sess-pose-model-not-ready": {
    uk: "Модель пози ще не готова. Зачекайте або оновіть сторінку.",
    en: "Pose model not ready. Please wait or refresh the page."
  },
  "sess-no-person": {
    uk: "На фото не видно людину",
    en: "No person detected in photo"
  },
  "sess-pose-invalid": {
    uk: "Поза не відповідає вимогам",
    en: "Pose does not meet requirements"
  },
  "sess-front-ready-both": {
    uk: "Анфас завантажено. Обидва знімки готові — натисніть «Розрахувати мірки».",
    en: "Front photo uploaded. Both photos ready — click \"Calculate Measurements\"."
  },
  "sess-front-ready-need-side": {
    uk: "Анфас завантажено. Додайте профіль (завантаження або камера).",
    en: "Front photo uploaded. Add profile photo (upload or camera)."
  },
  "sess-mirrored-auto": {
    uk: " Фото віддзеркалено автоматично.",
    en: " Photo mirrored automatically."
  },
  "sess-side-ready-both": {
    uk: "Профіль завантажено.{flipHint} Обидва знімки готові — натисніть «Розрахувати мірки».",
    en: "Profile photo uploaded.{flipHint} Both photos ready — click \"Calculate Measurements\"."
  },
  "sess-side-ready-need-front": {
    uk: "Профіль завантажено.{flipHint} Додайте анфас (завантаження або камера).",
    en: "Profile photo uploaded.{flipHint} Add front photo (upload or camera)."
  },
  "sess-image-format-invalid": {
    uk: "Оберіть зображення JPEG, PNG або WebP.",
    en: "Please select a JPEG, PNG, or WebP image."
  },
  "sess-file-too-large": {
    uk: "Файл завеликий (максимум 5 МБ).",
    en: "File is too large (maximum 5 MB)."
  },
  "sess-read-image-failed": {
    uk: "Не вдалося прочитати зображення.",
    en: "Failed to read image."
  },
  "sess-crop-square-failed": {
    uk: "Не вдалося привести зображення до квадрата.",
    en: "Failed to crop image to square."
  },
  "sess-file-accepted-no-check": {
    uk: "Файл прийнято без перевірки пози",
    en: "File accepted without pose check"
  },
  "sess-mediapipe-failed": {
    uk: "Не вдалося завантажити MediaPipe. Перевірте мережу та оновіть сторінку.",
    en: "Failed to load MediaPipe. Check your connection and refresh the page."
  },
  "sess-pose-valid-photo": {
    uk: "Поза на фото підходить",
    en: "Pose in photo is valid"
  },
  "sess-taking-photo-in": {
    uk: "Знімок через… {n}",
    en: "Taking photo in... {n}"
  },
  "sess-waiting-pose-model": {
    uk: "Очікування моделі пози…",
    en: "Waiting for pose model..."
  },
  "sess-person-not-seen": {
    uk: "Людину не видно",
    en: "No person detected"
  },
  "sess-hold-pose": {
    uk: "Утримайте позу для зйомки",
    en: "Hold pose for photo"
  },
  "sess-pose-ok": {
    uk: "Поза підходить",
    en: "Pose is correct"
  },
  "sess-adjust-pose": {
    uk: "Виправте позу.",
    en: "Adjust your pose."
  },
  "sess-mediapipe-load-failed-msg": {
    uk: "MediaPipe не завантажився: {msg}",
    en: "MediaPipe loading failed: {msg}"
  },
  "sess-restart-camera-btn": {
    uk: "Перезапустити камеру",
    en: "Restart camera"
  },
  "sess-analyzing-pose": {
    uk: "Аналіз пози…",
    en: "Analyzing pose..."
  },
  "sess-camera-access-failed": {
    uk: "Не вдалося отримати доступ до камери: {msg}",
    en: "Failed to access camera: {msg}"
  },
  "sess-video-not-ready": {
    uk: "Відео ще не готове.",
    en: "Video is not ready yet."
  },
  "sess-frame-invalid": {
    uk: "Кадр не підходить.",
    en: "Frame is not valid."
  },
  "sess-pose-invalid-snap": {
    uk: "На кадрі не видно людину — повторіть знімок.",
    en: "No person detected in frame — take photo again."
  },
  "sess-pose-invalid-snap-requirements": {
    uk: "Поза на мить знімка не відповідає вимогам.",
    en: "Pose at moment of photo did not meet requirements."
  },
  "sess-canvas-context-failed": {
    uk: "Не вдалося отримати контекст.",
    en: "Failed to get canvas context."
  },
  "sess-take-photo-failed": {
    uk: "Не вдалося створити знімок.",
    en: "Failed to take photo."
  },
  "sess-front-updated-calculate": {
    uk: "Анфас оновлено. Можна «Розрахувати мірки» або перезняти кадр.",
    en: "Front photo updated. You can \"Calculate Measurements\" or take photo again."
  },
  "sess-turn-on-camera-side": {
    uk: "Увімкніть камеру знову для знімка в профіль.",
    en: "Turn on camera again for profile photo."
  },
  "sess-both-ready-calculate": {
    uk: "Обидва знімки готові. Натисніть «Розрахувати мірки».",
    en: "Both photos ready. Click \"Calculate Measurements\"."
  },
  "sess-turn-on-camera-timer": {
    uk: "Увімкніть камеру перед запуском таймера.",
    en: "Turn on camera before starting the timer."
  },
  "sess-timer-cancelled": {
    uk: "Таймер скасовано.",
    en: "Timer cancelled."
  },
  "Покажіть правий лікоть": {
    uk: "Покажіть правий лікоть",
    en: "Show right elbow"
  },
  "Опустіть руку вздовж тіла": {
    uk: "Опустіть руку вздовж тіла",
    en: "Lower your arm along your body"
  },
  "Покажіть обличчя і плечі в кадрі.": {
    uk: "Покажіть обличчя і плечі в кадрі.",
    en: "Show face and shoulders in frame."
  },
  "Вирівняйте плечі (не нахиляйте корпус).": {
    uk: "Вирівняйте плечі (не нахиляйте корпус).",
    en: "Align shoulders (do not tilt body)."
  },
  "Поверніться обличчям до камери": {
    uk: "Поверніться обличчям до камери",
    en: "Face the camera"
  },
  "Покажіть всю голову в кадрі": {
    uk: "Покажіть всю голову в кадрі",
    en: "Show full head in frame"
  },
  "Опустіть камеру або відійдіть трохи, вся голова має бути в кадрі": {
    uk: "Опустіть камеру або відійдіть трохи, вся голова має бути в кадрі",
    en: "Lower the camera or step back, the entire head must be in the frame"
  },
  "Лікті мають бути в кадрі": {
    uk: "Лікті мають бути в кадрі",
    en: "Elbows must be in frame"
  },
  "Відведіть руки на ~15–20° від тіла": {
    uk: "Відведіть руки на ~15–20° від тіла",
    en: "Spread arms ~15-20° from body"
  },
  "Не розводьте руки занадто широко (достатньо ~15–20°).": {
    uk: "Не розводьте руки занадто широко (достатньо ~15–20°).",
    en: "Do not spread arms too wide (approx. 15-20° is enough)."
  },
  "Лікті трохи вбік від корпусу, пахви мають залишатися відкритими": {
    uk: "Лікті трохи вбік від корпусу, пахви мають залишатися відкритими",
    en: "Elbows slightly out, armpits must remain open"
  },
  "Спробуйте поставити ноги трохи ширше — приблизно на ширині плечей": {
    uk: "Спробуйте поставити ноги трохи ширше — приблизно на ширині плечей",
    en: "Try standing with feet slightly wider — about shoulder-width apart"
  },
  "Має бути видно зону стегон і паху": {
    uk: "Має бути видно зону стегон і паху",
    en: "Hips and groin area must be visible"
  },
  "Не зводьте стегна": {
    uk: "Не зводьте стегна",
    en: "Do not squeeze thighs"
  },
  "Покажіть повний зріст, стопи повинні бути в кадрі": {
    uk: "Покажіть повний зріст, стопи повинні бути в кадрі",
    en: "Show full height, feet must be in frame"
  },
  "Стопи мають бути повністю в кадрі": {
    uk: "Стопи мають бути повністю в кадрі",
    en: "Feet must be fully in frame"
  },
  "Станьте повним зростом на прямих ногах, не присідайте": {
    uk: "Станьте повним зростом на прямих ногах, не присідайте",
    en: "Stand full height on straight legs, do not squat"
  },
  "Покажіть себе крупніше в кадрі": {
    uk: "Покажіть себе крупніше в кадрі",
    en: "Position yourself larger in the frame"
  },
  "Поверніться боком на ~90°": {
    uk: "Поверніться боком на ~90°",
    en: "Turn sideways ~90°"
  },
  "У профілі стійте правим боком до камери": {
    uk: "У профілі стійте правим боком до камери",
    en: "In profile, stand with your right side to the camera"
  },
  "Не повертайте голову до камери": {
    uk: "Не повертайте голову до камери",
    en: "Do not turn head to camera"
  },
  "Не розвертайте голову до камери": {
    uk: "Не розвертайте голову до камери",
    en: "Do not turn head to camera"
  },
  "Дивіться в той самий бік, куди звернене тіло .": {
    uk: "Дивіться в той самий бік, куди звернене тіло .",
    en: "Look in the same direction the body is facing."
  },
  "Не нахиляйте корпус у профіль": {
    uk: "Не нахиляйте корпус у профіль",
    en: "Do not tilt torso in profile"
  },
  "Опустіть руку вздовж тулуба": {
    uk: "Опустіть руку вздовж тулуба",
    en: "Lower your arm along your body"
  },
  "Станьте на рівних ногах": {
    uk: "Станьте на рівних ногах",
    en: "Stand on straight legs"
  },
  "Покажіть повний зріст": {
    uk: "Покажіть повний зріст",
    en: "Show full height"
  },
  "sess-cancel-timer-btn": {
    uk: "Скасувати ({sec} с)",
    en: "Cancel ({sec}s)"
  },
  "sess-timer-countdown-pose": {
    uk: "Автозйомка через {sec} с… Станьте в правильну позу.",
    en: "Auto-capture in {sec}s... Stand in the correct pose."
  },
  "sess-timer-countdown": {
    uk: "Автозйомка через {sec} с…",
    en: "Auto-capture in {sec}s..."
  },
  "sess-taking-photo": {
    uk: "Знімаю фото…",
    en: "Taking photo..."
  },
  "sess-retake-front-status": {
    uk: "Перезйомка анфасу: увімкніть камеру та встаньте в позу.",
    en: "Retaking front: turn on camera and stand in pose."
  },
  "sess-retake-side-status": {
    uk: "Перезйомка профілю: увімкніть камеру та встаньте в позу.",
    en: "Retaking profile: turn on camera and stand in pose."
  },

  // Dataset collection page
  "dataset-title": {
    uk: "FitMeasure AI — збір датасету",
    en: "FitMeasure AI — dataset collection"
  },
  "dataset-header-title": {
    uk: "Долучитись до збору датасету",
    en: "Join Dataset Collection"
  },
  "dataset-header-sub": {
    uk: "Допоможіть покращити точність нейромережі, надавши свої фото та мірки",
    en: "Help improve neural network accuracy by providing your photos and measurements"
  },
  "dataset-guidelines-heading": {
    uk: "Гайдлайн і згоди",
    en: "Guidelines & Consent"
  },
  "dataset-guidelines-purpose": {
    uk: "<strong>Мета збору:</strong> Фотографії та мірки збираються виключно для навчання нейромережевого класифікатора визначення розмірів одягу. Ваші фото будуть зашифровані перед завантаженням і зберігатимуться в захищеному вигляді.",
    en: "<strong>Collection purpose:</strong> Photos and measurements are collected exclusively for training a neural network classifier for clothing size determination. Your photos will be encrypted before upload and stored securely."
  },
  "dataset-guidelines-privacy": {
    uk: "<strong>Конфіденційність:</strong> Всі фотографії шифруються на вашому пристрої перед відправкою (end-to-end encryption). Ми зберігаємо зашифровані файли, які можна розшифрувати тільки офлайн-ключем. Мірки зберігаються як анонімні дані без прив'язки до особистості.",
    en: "<strong>Privacy:</strong> All photos are encrypted on your device before sending (end-to-end encryption). We store encrypted files that can only be decrypted with an offline key. Measurements are stored as anonymous data without personal identification."
  },
  "dataset-consent-18plus": {
    uk: "Мені є 18 років або більше",
    en: "I am 18 years old or older"
  },
  "dataset-consent-terms": {
    uk: "Я погоджуюсь з умовами збору даних і розумію, що мої зашифровані фото та мірки будуть використані для навчання нейромережі",
    en: "I agree to the data collection terms and understand that my encrypted photos and measurements will be used for neural network training"
  },
  "dataset-label-dob": {
    uk: "Дата народження",
    en: "Date of birth"
  },
  "dataset-dob-placeholder": {
    uk: "ДД.ММ.РРРР",
    en: "DD.MM.YYYY"
  },
  "dataset-dob-calendar": {
    uk: "Відкрити календар",
    en: "Open calendar"
  },
  "dataset-measurements-heading": {
    uk: "Ваші мірки",
    en: "Your Measurements"
  },
  "dataset-measurements-intro": {
    uk: "Введіть реальні мірки вашого тіла в сантиметрах. Ви можете редагувати кожне значення. Якщо якісь мірки виглядають нетиповими, поля будуть позначені жовтим — перевірте їх і натисніть «Надіслати дані» ще раз.",
    en: "Enter your real body measurements in centimeters. You can edit each value. If any measurements look atypical, fields will be marked yellow — double-check them and click \"Submit Data\" again."
  },
  "dataset-measure-placeholder": {
    uk: "Введіть значення",
    en: "Enter value"
  },
  "dataset-measure-height": {
    uk: "Зріст",
    en: "Height"
  },
  "dataset-measure-neck-base-height": {
    uk: "Висота точки основи шиї (від підлоги)",
    en: "Neck base height (from floor)"
  },
  "dataset-measure-neck-circumference": {
    uk: "Обхват шиї",
    en: "Neck circumference"
  },
  "dataset-measure-chest-circumference": {
    uk: "Обхват грудей (ОГ)",
    en: "Chest circumference"
  },
  "dataset-measure-waist-circumference": {
    uk: "Обхват талії (ОТ)",
    en: "Waist circumference"
  },
  "dataset-measure-hip-circumference": {
    uk: "Обхват стегон (ОС)",
    en: "Hip circumference"
  },
  "dataset-measure-arm-circumference": {
    uk: "Обхват плеча / руки (в біцепсі)",
    en: "Upper arm circumference (bicep)"
  },
  "dataset-measure-thigh-circumference": {
    uk: "Обхват одного стегна (вгорі ноги)",
    en: "Thigh circumference (upper leg)"
  },
  "dataset-measure-shoulder-width": {
    uk: "Ширина плечей",
    en: "Shoulder width"
  },
  "dataset-measure-back-width": {
    uk: "Ширина спини (ШС)",
    en: "Back width"
  },
  "dataset-measure-chest-width": {
    uk: "Ширина грудей (ШГ)",
    en: "Chest width"
  },
  "dataset-measure-front-length-to-waist": {
    uk: "Довжина переду до талії (ДПТ)",
    en: "Front length to waist"
  },
  "dataset-measure-back-length-to-waist": {
    uk: "Довжина спини до талії (ДСТ)",
    en: "Back length to waist"
  },
  "dataset-measure-sleeve-length": {
    uk: "Довжина рукава (ДР)",
    en: "Sleeve length"
  },
  "dataset-measure-outer-seam": {
    uk: "Зовнішній шов",
    en: "Outer seam"
  },
  "dataset-measure-inner-seam": {
    uk: "Внутрішній шов",
    en: "Inner seam"
  },
  "dataset-btn-submit": {
    uk: "Надіслати дані",
    en: "Submit Data"
  },
  "dataset-submitting": {
    uk: "Відправка даних...",
    en: "Submitting data..."
  },
  "dataset-encrypting": {
    uk: "Шифрування фотографій...",
    en: "Encrypting photos..."
  },
  "dataset-uploading": {
    uk: "Завантаження до сховища...",
    en: "Uploading to storage..."
  },
  "dataset-finalizing": {
    uk: "Завершення...",
    en: "Finalizing..."
  },
  "dataset-success-heading": {
    uk: "Дякуємо!",
    en: "Thank you!"
  },
  "dataset-prefill-loaded": {
    uk: "Мірки та фото завантажено з результатів заміру. Перевірте значення перед відправкою.",
    en: "Measurements and photos loaded from your measurement results. Review values before submitting."
  },
  "dataset-prefill-failed": {
    uk: "Не вдалося завантажити дані з результатів: {error}",
    en: "Could not load data from measurement results: {error}"
  },
  "dataset-success-message": {
    uk: "Ваші дані успішно надіслані і допоможуть покращити точність системи. Ваші фото зашифровані та захищені.",
    en: "Your data has been successfully submitted and will help improve system accuracy. Your photos are encrypted and protected."
  },
  "dataset-success-id": {
    uk: "ID запису: {id}",
    en: "Submission ID: {id}"
  },
  "dataset-btn-home": {
    uk: "← Повернутись на головну",
    en: "← Back to home"
  },
  "dataset-err-need-consent": {
    uk: "Ви повинні підтвердити всі згоди та бути віком 18+ для відправки даних",
    en: "You must confirm all consents and be 18+ to submit data"
  },
  "dataset-err-need-photos": {
    uk: "Необхідно зробити або завантажити обидва фото (анфас та профіль)",
    en: "Both photos are required (front and profile)"
  },
  "dataset-err-need-measurements": {
    uk: "Будь ласка, заповніть всі 16 мірок",
    en: "Please fill in all 16 measurements"
  },
  "dataset-err-measurement-non-positive": {
    uk: "Мірка не може бути нулем або від'ємним числом. Перевірте позначені поля.",
    en: "A measurement cannot be zero or negative. Check the marked fields."
  },
  "dataset-err-age-check": {
    uk: "Згідно з вказаною датою народження, вам менше 18 років",
    en: "According to the provided date of birth, you are under 18"
  },
  "dataset-err-dob-invalid": {
    uk: "Введіть дату народження у форматі ДД.ММ.РРРР",
    en: "Enter date of birth as DD.MM.YYYY"
  },
  "dataset-warn-outliers": {
    uk: "Деякі мірки виглядають нетиповими (позначено жовтим). Перевірте позначені поля та натисніть «Надіслати дані» ще раз для підтвердження.",
    en: "Some measurements look atypical (marked yellow). Check the marked fields and click \"Submit Data\" again to confirm."
  },
  "dataset-err-submission": {
    uk: "Помилка відправки: {error}",
    en: "Submission error: {error}"
  }
};

const GARMENT_LABELS_MAP = {
  "shirt": { uk: "Сорочка", en: "Shirt" },
  "tshirt": { uk: "Футболка", en: "T-Shirt" },
  "polo": { uk: "Поло", en: "Polo" },
  "blouse": { uk: "Блузка", en: "Blouse" },
  "jacket": { uk: "Піджак", en: "Jacket" },
  "coat": { uk: "Пальто", en: "Coat" },
  "vest": { uk: "Жилет", en: "Vest" },
  "hoodie": { uk: "Худі", en: "Hoodie" },
  "cardigan": { uk: "Кардиган", en: "Cardigan" },
  "sport_top": { uk: "Спортивний топ", en: "Sport Top" },
  "raincoat": { uk: "Дощовик", en: "Raincoat" },
  "pajama_top": { uk: "Піжамний топ", en: "Pajama Top" },
  "pants": { uk: "Штани", en: "Pants" },
  "jeans": { uk: "Джинси", en: "Jeans" },
  "shorts": { uk: "Шорти", en: "Shorts" },
  "pajama_bottom": { uk: "Піжамні штани", en: "Pajama Bottom" },
  "skirt": { uk: "Спідниця", en: "Skirt" },
  "pencil_skirt": { uk: "Спідниця-олівець", en: "Pencil Skirt" },
  "dress": { uk: "Сукня", en: "Dress" },
  "dress_fitted": { uk: "Приталена сукня", en: "Fitted Dress" },
  "overalls": { uk: "Комбінезон", en: "Overalls" }
};

const BACKEND_ERRORS_MAP = {
  "На знімку анфасу не виявлено людину": "No person detected in the front photo. Make sure the figure is fully in the frame and the pose meets requirements.",
  "На знімку профілю не виявлено людину": "No person detected in the profile photo. Make sure the figure is fully in the frame and the pose meets requirements.",
  "На анфасі не вдалося виділити силует тіла": "Failed to segment the body silhouette on the front view. Try different lighting or background.",
  "На профілі не вдалося виділити силует тіла": "Failed to segment the body silhouette on the profile view. Try different lighting or background.",
  "Недостатньо видимих ключових точок на анфасі для калібровки за зростом": "Not enough visible keypoints on the front photo for height calibration. Make sure feet and head are in the frame.",
  "Недостатньо видимих ключових точок на профілі для калібровки за зростом": "Not enough visible keypoints on the profile photo for height calibration. Make sure feet and head are in the frame.",
  "Некоректне значення статі для пайплайну": "Invalid sex value for the pipeline."
};

// Initialize language from localStorage or default to Ukrainian
export function initI18n() {
  try {
    const saved = localStorage.getItem("pointsx.lang");
    if (saved === "uk" || saved === "en") {
      currentLang = saved;
    }
  } catch {
    /* ignore private mode */
  }
  translatePage();
}

export function setLang(lang) {
  if (lang !== "uk" && lang !== "en") return;
  currentLang = lang;
  try {
    localStorage.setItem("pointsx.lang", lang);
  } catch {
    /* ignore private mode */
  }
  translatePage();
  
  // Dispatch custom event to let other components know the language changed
  window.dispatchEvent(new CustomEvent("langchanged", { detail: { lang } }));
}

export function t(key, vars = {}) {
  const trans = translations[key];
  if (!trans) return key;
  let text = trans[currentLang] || trans["uk"] || key;
  
  // Interpolate variables
  for (const [k, v] of Object.entries(vars)) {
    text = text.replace(new RegExp(`{${k}}`, "g"), v);
  }
  return text;
}

export function translateGarment(garmentId, labelUk) {
  const item = GARMENT_LABELS_MAP[garmentId];
  if (!item) return labelUk;
  return item[currentLang] || labelUk;
}

export function translateBackendError(errText) {
  if (currentLang !== "en") return errText;
  for (const [ukKey, enVal] of Object.entries(BACKEND_ERRORS_MAP)) {
    if (errText.includes(ukKey)) {
      return errText.replace(ukKey, enVal);
    }
  }
  return errText;
}

export function translatePage() {
  document.documentElement.lang = currentLang;
  
  // Set title (dataset page uses data-page-title on <body>)
  const pageTitleKey = document.body?.dataset?.pageTitle;
  document.title = t(pageTitleKey || "title");
  
  // Translate static elements with data-i18n
  const elements = document.querySelectorAll("[data-i18n]");
  for (const el of elements) {
    const key = el.getAttribute("data-i18n");
    const htmlKey = el.getAttribute("data-i18n-html");
    if (htmlKey === "true") {
      el.innerHTML = t(key);
    } else {
      el.textContent = t(key);
    }
  }
  
  // Update toggle state visually
  const btns = document.querySelectorAll(".lang-btn");
  for (const btn of btns) {
    const active = btn.dataset.lang === currentLang;
    btn.classList.toggle("active", active);
    btn.setAttribute("aria-pressed", active ? "true" : "false");
  }
}
