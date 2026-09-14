/**
 * Tailoring UI — garment strip, regional size tabs, and measurement table.
 *
 * /api/measure → MeasurementEnvelope v2
 * → sizeAndPatternHandler → SizingReport
 * → renderConsensusBlock (UA / EU / US panels)
 *
 * Facade: implementation is split into the modules re-exported below; existing
 * `import * as ...` users keep the same names.
 */

export {
  attachMeasureHandler,
} from "../sizing/measureFlow.js";

export {
  ensureSizeTabsWired,
  selectSizeTab,
} from "../sizing/sizeTabs.js";

export {
  escapeHtml,
  sexLabel,
} from "../sizing/text.js";

export {
  loadTailoringCatalog,
} from "../sizing/catalog.js";

export {
  refreshTailoringView,
  renderGarmentStrip,
} from "../sizing/view.js";
