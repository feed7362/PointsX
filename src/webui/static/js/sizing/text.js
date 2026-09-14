/**
 * Small text helpers for the sizing UI.
 */

import { t } from "../i18n/index.js";


// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

export function sexLabel(sex) {
  if (sex === "male")   return t("sex-male-label");
  if (sex === "female") return t("sex-female-label");
  return t("sex-other-label");
}


export function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
