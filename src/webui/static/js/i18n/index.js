/**
 * UI localisation (uk / en): language state, lookup with {var} interpolation,
 * garment names, English renderings of backend error texts, and page translation.
 *
 * Strings live in uk.js / en.js (same keys). Missing English falls back to Ukrainian.
 */

import uk from "./uk.js";
import en from "./en.js";

export let currentLang = "uk";

const DICTS = { uk, en };

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
  if (!Object.hasOwn(uk, key) && !Object.hasOwn(en, key)) return key;
  let text = DICTS[currentLang]?.[key] || uk[key] || key;
  
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
