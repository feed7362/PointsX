/**
 * UA / EU / US size tabs: selection and keyboard navigation.
 */

import { captureState } from "../capture/state.js";
import { getCaptureDom } from "../capture/dom.js";


// ---------------------------------------------------------------------------
// Tab wiring
// ---------------------------------------------------------------------------

export function selectSizeTab(which) {
  const { tabUa, tabEu, tabUs, panelUa, panelEu, panelUs } = getCaptureDom();
  if (!tabUa || !tabEu || !tabUs || !panelUa || !panelEu || !panelUs) return;
  const tabs = [
    { id: "ua", tab: tabUa, panel: panelUa },
    { id: "eu", tab: tabEu, panel: panelEu },
    { id: "us", tab: tabUs, panel: panelUs },
  ];
  for (const { id, tab, panel } of tabs) {
    const sel = id === which;
    tab.setAttribute("aria-selected", sel ? "true" : "false");
    tab.tabIndex = sel ? 0 : -1;
    panel.hidden = !sel;
  }
}


// ---------------------------------------------------------------------------
// Tab keyboard wiring
// ---------------------------------------------------------------------------

export function ensureSizeTabsWired() {
  const { tabUa, tabEu, tabUs } = getCaptureDom();
  if (captureState.sizeTabsWired || !tabUa || !tabEu || !tabUs) return;
  captureState.sizeTabsWired = true;
  const order    = ["ua", "eu", "us"];
  const tabById  = { ua: tabUa, eu: tabEu, us: tabUs };
  for (const id of order) {
    tabById[id].addEventListener("click", () => selectSizeTab(id));
  }
  for (let i = 0; i < order.length; i++) {
    const id  = order[i];
    const tab = tabById[id];
    tab.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const next = order[(i + 1) % order.length];
        selectSizeTab(next);
        tabById[next].focus();
      }
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = order[(i - 1 + order.length) % order.length];
        selectSizeTab(prev);
        tabById[prev].focus();
      }
    });
  }
}
