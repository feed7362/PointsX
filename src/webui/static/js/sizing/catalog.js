/**
 * Loads the tailoring catalog (garments, size grids) once per page.
 */

import { captureState } from "../capture/state.js";
import { t } from "../i18n/index.js";
import { apiFetch } from "../api/client.js";

// ---------------------------------------------------------------------------
// Catalog loading
// ---------------------------------------------------------------------------

export async function loadTailoringCatalog() {
  if (captureState.tailoringCatalog) return captureState.tailoringCatalog;
  const res = await apiFetch("/static/data/tailoring_config.json?v=3");
  if (!res.ok) throw new Error(t("failed-load-config"));
  captureState.tailoringCatalog = await res.json();
  return captureState.tailoringCatalog;
}
