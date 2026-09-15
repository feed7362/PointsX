/**
 * Measure buttons: request (real or mock), results table, sizing view, hand-off to the dataset form.
 */

import { captureState } from "../capture/state.js";
import { getCaptureDom } from "../capture/dom.js";
import { setStatus, updateUiStep } from "../capture/ui.js";
import { t, translateBackendError } from "../i18n/index.js";
import { ApiError } from "../api/client.js";
import { requestMeasurement } from "../api/measure.js";
import { envelopeToDatasetMeasurements, saveDatasetPrefill } from "../dataset/transfer.js";
import { loadTailoringCatalog } from "./catalog.js";
import { escapeHtml, sexLabel } from "./text.js";
import { orderMeasurementsManual } from "./measurementOrder.js";
import { renderPipelineModelViz } from "./render.js";
import { ensureSizeTabsWired, selectSizeTab } from "./sizeTabs.js";
import { refreshTailoringView, renderGarmentStrip } from "./view.js";


function setResultsDatasetActionVisible(visible) {
  const actionsEl = document.getElementById("results-dataset-actions");
  if (actionsEl) actionsEl.hidden = !visible;
}


async function sendResultsToDatasetPage() {
  const envelope = captureState.lastMockResponse;
  const btnSend = document.getElementById("btn-send-to-dataset");
  if (!envelope || !captureState.frontBlob || !captureState.sideBlob) {
    setStatus(t("err-need-both-photos"), true);
    return;
  }
  const { heightInput, sexSelect } = getCaptureDom();
  const heightCm = Number(envelope.subject?.height_cm ?? heightInput.value);
  const sex = envelope.subject?.sex ?? sexSelect.value;
  const measurements = envelopeToDatasetMeasurements(envelope, heightCm);
  try {
    if (btnSend) btnSend.disabled = true;
    await saveDatasetPrefill({
      heightCm,
      sex,
      measurements,
      frontBlob: captureState.frontBlob,
      sideBlob: captureState.sideBlob,
    });
    window.location.href = "/dataset.html?prefill=1";
  } catch (err) {
    if (btnSend) btnSend.disabled = false;
    setStatus(t("err-dataset-transfer-failed", { msg: err?.message ?? String(err) }), true);
  }
}


// ---------------------------------------------------------------------------
// Measure button handler
// ---------------------------------------------------------------------------

export function attachMeasureHandler() {
  const {
    btnMeasure,
    btnMeasureTest,
    resultsSection,
    resultsBody,
    heightInput,
    sexSelect,
    tailoringIntro,
    garmentStripWrap,
    tailoringPanels,
    tailoringDisclaimer,
    measureLoading,
    measureLoadingTitle,
    measureLoadingStep,
  } = getCaptureDom();

  const btnSendToDataset = document.getElementById("btn-send-to-dataset");
  if (btnSendToDataset && !btnSendToDataset.dataset.wired) {
    btnSendToDataset.dataset.wired = "1";
    const label = t("btn-send-to-dataset");
    btnSendToDataset.setAttribute("aria-label", label);
    btnSendToDataset.setAttribute("title", label);
    btnSendToDataset.addEventListener("click", () => void sendResultsToDatasetPage());
  }

  async function runMeasureRequest(mode = "capture") {
    const useTestImages = mode === "test";
    if (!useTestImages && (!captureState.frontBlob || !captureState.sideBlob)) {
      setStatus(t("err-need-both-photos"), true);
      return;
    }
    if (!resultsSection || !resultsBody) {
      setStatus(t("err-missing-results-container"), true);
      return;
    }
    const heightCmNum = Number(String(heightInput.value).replace(",", "."));
    if (!Number.isFinite(heightCmNum) || heightCmNum < 100 || heightCmNum > 250) {
      setStatus(t("err-height-range"), true);
      return;
    }
    setStatus(useTestImages ? t("status-test-calc") : t("status-calculating"));
    resultsSection.hidden = true;
    setResultsDatasetActionVisible(false);
    const modelVizEl = document.getElementById("model-viz");
    if (modelVizEl) modelVizEl.hidden = true;

    // Disable action buttons during active computation
    btnMeasure.disabled = true;
    if (btnMeasureTest) btnMeasureTest.disabled = true;

    // Show loading spinner and details
    if (measureLoading) {
      measureLoading.hidden = false;
      measureLoading.scrollIntoView({ behavior: "smooth", block: "center" });
      if (measureLoadingTitle) {
        measureLoadingTitle.textContent = useTestImages ? t("loading-test-title") : t("loading-measure-title");
      }
      if (measureLoadingStep) {
        measureLoadingStep.textContent = useTestImages ? t("loading-test-step") : t("loading-measure-step");
      }
    }

    let progressTimer = null;
    if (!useTestImages && measureLoadingStep) {
      const steps = [
        { time:  1000, text: t("step-send-photos") },
        { time:  7000, text: t("step-yolo-pose") },
        { time: 14000, text: t("step-yolo-seg") },
        { time: 21000, text: t("step-calc-anthropometry") },
        { time: 28000, text: t("step-finish") }
      ];
      let currentStep = 0;
      const updateStep = () => {
        if (currentStep < steps.length) {
          measureLoadingStep.textContent = steps[currentStep].text;
          const nextDelay = currentStep === 0 ? steps[0].time : (steps[currentStep].time - steps[currentStep-1].time);
          currentStep++;
          progressTimer = setTimeout(updateStep, nextDelay);
        }
      };
      updateStep();
    }

    const fd = new FormData();
    fd.append("height_cm", String(heightCmNum));
    fd.append("sex",       sexSelect.value);
    if (!useTestImages) {
      fd.append("pose_backend", "coco");
      const frontName =
        captureState.frontBlob instanceof File && captureState.frontBlob.name
          ? captureState.frontBlob.name
          : "front.jpg";
      const sideName =
        captureState.sideBlob instanceof File && captureState.sideBlob.name
          ? captureState.sideBlob.name
          : "side.jpg";
      fd.append("front", captureState.frontBlob, frontName);
      fd.append("side", captureState.sideBlob, sideName);
    }

    try {
      let data;
      try {
        data = await requestMeasurement(fd, { mock: useTestImages });
      } catch (err) {
        // Server detail texts are Ukrainian; show them in the UI language.
        throw err instanceof ApiError ? new Error(translateBackendError(err.message)) : err;
      }
      console.log("[FitMeasure AI] Full model output:", data);
      const measurements = orderMeasurementsManual(data.measurements ?? []);
      if (!measurements.length) {
        captureState.lastMockResponse = null;
        const mvEmpty = document.getElementById("model-viz");
        if (mvEmpty) mvEmpty.hidden = true;
        resultsBody.innerHTML = "";
        if (tailoringIntro) tailoringIntro.hidden = true;
        if (garmentStripWrap) garmentStripWrap.hidden = true;
        if (tailoringPanels) tailoringPanels.hidden = true;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = true;
        const patternDetailsEl = document.getElementById("pattern-details");
        if (patternDetailsEl) patternDetailsEl.hidden = true;
        resultsSection.hidden = false;
        setResultsDatasetActionVisible(false);
        const apiWarn =
          Array.isArray(data.warnings) && data.warnings.length
            ? " " + data.warnings.join(" ")
            : "";
        setStatus(t("err-empty-measurements") + apiWarn, true);
        return;
      }

      captureState.lastMockResponse = data;
      renderPipelineModelViz(data.derived);

      const sex      = data.subject?.sex ?? sexSelect.value;
      const heightCm = data.subject?.height_cm ?? heightCmNum;
      resultsBody.innerHTML = "";
      for (const row of measurements) {
        const tr = document.createElement("tr");
        const label = t(row.id) !== row.id ? t(row.id) : (row.label_uk ?? row.id);
        tr.innerHTML =
          "<td>" + escapeHtml(label) +
          "</td><td>" + escapeHtml(String(row.value_cm)) + "</td>";
        resultsBody.appendChild(tr);
      }

      ensureSizeTabsWired();
      selectSizeTab("ua");

      try {
        await loadTailoringCatalog();
        captureState.selectedGarmentId = captureState.tailoringCatalog.garments[0]?.id || "shirt";

        if (tailoringIntro) {
          tailoringIntro.textContent = t("sizing-report-intro", { height: heightCm, sex: sexLabel(sex) });
          tailoringIntro.hidden = false;
        }
        if (garmentStripWrap)   garmentStripWrap.hidden   = false;
        if (tailoringPanels)    tailoringPanels.hidden    = false;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = false;

        renderGarmentStrip();
        refreshTailoringView();
      } catch (cfgErr) {
        if (tailoringIntro) {
          tailoringIntro.textContent = t("sizing-pattern-error", { msg: cfgErr?.message ?? String(cfgErr) });
          tailoringIntro.hidden = false;
        }
        if (garmentStripWrap)   garmentStripWrap.hidden   = true;
        if (tailoringPanels)    tailoringPanels.hidden    = true;
        if (tailoringDisclaimer) tailoringDisclaimer.hidden = true;
      }

      resultsSection.hidden = false;
      setResultsDatasetActionVisible(
        !useTestImages && Boolean(captureState.frontBlob && captureState.sideBlob)
      );
      setStatus(t("status-done"));
    } catch (e) {
      const mv = document.getElementById("model-viz");
      if (mv) mv.hidden = true;
      setStatus(t("err-request-failed", { msg: e?.message ?? String(e) }), true);
    } finally {
      if (progressTimer) clearTimeout(progressTimer);
      if (measureLoading) measureLoading.hidden = true;
      updateUiStep();
      if (btnMeasureTest) btnMeasureTest.disabled = false;
    }
  }

  btnMeasure.addEventListener("click", async () => {
    await runMeasureRequest("capture");
  });

  if (btnMeasureTest) {
    btnMeasureTest.addEventListener("click", async () => {
      await runMeasureRequest("test");
    });
  }
}
