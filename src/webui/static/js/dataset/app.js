/**
 * Dataset collection page entry point
 * 
 * Reuses capture modules for camera/pose/thumbnails.
 * Handles consent gating, measurement input, outlier validation, and submission.
 */

// Import capture modules (reuse from main app)
import { initI18n, setLang, t } from '../capture/i18n.js';
import { getCaptureDom } from '../capture/dom.js';
import { captureState } from '../capture/state.js';
import * as session from '../capture/session.js';
import * as overlay from '../capture/overlay.js';
import * as ui from '../capture/ui.js';
import {
  guideGeom,
  applyReloadGuideQueryParam,
  loadGuideGeometry,
  loadGuideGeometryFromOptionalStaticFile,
  drawGuideSilhouetteOnCanvas
} from '../guideGeometry.js';

// Import dataset-specific modules
import {
  MEASUREMENTS,
  renderMeasurementGrid,
  getMeasurementValues,
  validateMeasurements,
  analyzeMeasurements,
  markMeasurementFieldStates,
} from './measurements.js';
import { uploadAndInsert } from './supabaseClient.js';

// Dataset-specific DOM elements
const datasetDom = {
  consent18plus: null,
  consentTerms: null,
  dob: null,
  dobPicker: null,
  dobCalendarBtn: null,
  heightInput: null,
  sexSelect: null,
  measurementsContainer: null,
  btnSubmit: null,
  datasetStatus: null,
  datasetLoading: null,
  datasetLoadingTitle: null,
  datasetLoadingStep: null,
  datasetSuccess: null,
  datasetSuccessId: null
};

// Two-stage submission state
let pendingConfirm = false;

const THUMB_PLACEHOLDER_SILHOUETTE = {
  strokeStyle: 'rgba(113, 168, 255, 0.5)',
  fillStyle: 'rgba(118, 170, 255, 0.07)',
  shadowColor: 'transparent',
  shadowBlur: 0,
  lineWidthScale: 0.72,
  profileInteriorStroke: 'rgba(113, 168, 255, 0.42)',
};

/** Render pose example silhouettes (same logic as main capture page). */
function renderReferenceGuides() {
  const canvases = [
    { id: 'ref-canvas-front', step: 1 },
    { id: 'ref-canvas-profile', step: 2 },
  ];
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  for (const item of canvases) {
    const canvas = document.getElementById(item.id);
    if (!canvas) continue;
    const cssW = Math.max(1, canvas.clientWidth || 260);
    const cssH = Math.max(1, canvas.clientHeight || Math.round(cssW * 1.6));
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) continue;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = cssW * 0.07;
    const box = {
      left: pad,
      top: cssH * 0.02,
      width: cssW - pad * 2,
      height: cssH * 0.94,
    };
    drawGuideSilhouetteOnCanvas(ctx, box, item.step, guideGeom);
  }
}

/** Empty photo slots: muted guide silhouettes like the main capture page. */
function renderThumbPlaceholders() {
  const items = [
    { id: 'thumb-placeholder-front', step: 1 },
    { id: 'thumb-placeholder-side', step: 2 },
  ];
  const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
  for (const item of items) {
    const canvas = document.getElementById(item.id);
    if (!canvas) continue;
    const slot = canvas.closest('.thumb-slot');
    const cssW = Math.max(1, slot?.clientWidth ?? canvas.clientWidth ?? 220);
    const cssH = Math.max(1, slot?.clientHeight ?? Math.round((cssW * 160) / 100));
    const bw = Math.round(cssW * dpr);
    const bh = Math.round(cssH * dpr);
    if (canvas.width !== bw || canvas.height !== bh) {
      canvas.width = bw;
      canvas.height = bh;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) continue;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const pad = cssW * 0.07;
    const box = {
      left: pad,
      top: cssH * 0.02,
      width: cssW - pad * 2,
      height: cssH * 0.94,
    };
    drawGuideSilhouetteOnCanvas(ctx, box, item.step, guideGeom, THUMB_PLACEHOLDER_SILHOUETTE);
  }
}

function refreshGuideArtwork() {
  renderReferenceGuides();
  renderThumbPlaceholders();
}

/**
 * Initialize dataset DOM references
 */
function initDatasetDom() {
  datasetDom.consent18plus = document.getElementById('consent-18plus');
  datasetDom.consentTerms = document.getElementById('consent-terms');
  datasetDom.dob = document.getElementById('dob');
  datasetDom.dobPicker = document.getElementById('dob-picker');
  datasetDom.dobCalendarBtn = document.getElementById('dob-calendar');
  datasetDom.heightInput = document.getElementById('height');
  datasetDom.sexSelect = document.getElementById('sex');
  datasetDom.measurementsContainer = document.getElementById('measurements-container');
  datasetDom.btnSubmit = document.getElementById('btn-dataset-submit');
  datasetDom.datasetStatus = document.getElementById('dataset-status');
  datasetDom.datasetLoading = document.getElementById('dataset-loading');
  datasetDom.datasetLoadingTitle = document.getElementById('dataset-loading-title');
  datasetDom.datasetLoadingStep = document.getElementById('dataset-loading-step');
  datasetDom.datasetSuccess = document.getElementById('dataset-success');
  datasetDom.datasetSuccessId = document.getElementById('dataset-success-id');
}

/**
 * Set status message
 */
function setStatus(message, isError = false) {
  datasetDom.datasetStatus.textContent = message;
  datasetDom.datasetStatus.className = isError ? 'status err' : 'status';
}

/**
 * Clear status message
 */
function clearStatus() {
  datasetDom.datasetStatus.textContent = '';
  datasetDom.datasetStatus.className = 'status';
}

/**
 * Calculate age from ISO date of birth (YYYY-MM-DD).
 */
function calculateAge(dobIso) {
  const [year, month, day] = dobIso.split('-').map(Number);
  const dob = new Date(year, month - 1, day);
  const today = new Date();
  let age = today.getFullYear() - dob.getFullYear();
  const monthDiff = today.getMonth() - dob.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < dob.getDate())) {
    age--;
  }
  return age;
}

function isoFromLocalDate(date) {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const d = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${d}`;
}

function getMaxDobDate() {
  const max = new Date();
  max.setFullYear(max.getFullYear() - 18);
  return max;
}

function formatDobDisplay(iso) {
  const [year, month, day] = iso.split('-');
  return `${day}.${month}.${year}`;
}

function formatDobInput(raw) {
  const digits = raw.replace(/\D/g, '').slice(0, 8);
  if (digits.length <= 2) return digits;
  if (digits.length <= 4) return `${digits.slice(0, 2)}.${digits.slice(2)}`;
  return `${digits.slice(0, 2)}.${digits.slice(2, 4)}.${digits.slice(4)}`;
}

function parseDobText(text) {
  const match = text.trim().match(/^(\d{1,2})[./](\d{1,2})[./](\d{4})$/);
  if (!match) return null;

  const day = Number(match[1]);
  const month = Number(match[2]);
  const year = Number(match[3]);
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;

  const parsed = new Date(year, month - 1, day);
  if (
    parsed.getFullYear() !== year ||
    parsed.getMonth() !== month - 1 ||
    parsed.getDate() !== day
  ) {
    return null;
  }

  return isoFromLocalDate(parsed);
}

function getDobIso() {
  const fromText = parseDobText(datasetDom.dob?.value || '');
  if (fromText) return fromText;
  if (datasetDom.dobPicker?.value) return datasetDom.dobPicker.value;
  return '';
}

function syncDobPickerFromText() {
  const iso = parseDobText(datasetDom.dob.value);
  if (iso) datasetDom.dobPicker.value = iso;
}

function syncDobTextFromPicker() {
  if (!datasetDom.dobPicker.value) return;
  datasetDom.dob.value = formatDobDisplay(datasetDom.dobPicker.value);
}

function openDobCalendar() {
  syncDobPickerFromText();
  const picker = datasetDom.dobPicker;
  if (!picker) return;

  if (typeof picker.showPicker === 'function') {
    try {
      picker.showPicker();
      return;
    } catch (_) {
      // Safari may throw if not triggered from a direct user gesture chain.
    }
  }

  picker.focus({ preventScroll: true });
  picker.click();
}

function refreshDobFieldLocale() {
  if (!datasetDom.dob) return;
  datasetDom.dob.placeholder = t('dataset-dob-placeholder');
  datasetDom.dobCalendarBtn?.setAttribute('aria-label', t('dataset-dob-calendar'));
}

function initDateOfBirthField() {
  const max = getMaxDobDate();
  const min = new Date();
  min.setFullYear(min.getFullYear() - 120);

  datasetDom.dobPicker.max = isoFromLocalDate(max);
  datasetDom.dobPicker.min = isoFromLocalDate(min);
  refreshDobFieldLocale();

  datasetDom.dob.addEventListener('input', (event) => {
    const input = event.target;
    const formatted = formatDobInput(input.value);
    if (formatted !== input.value) {
      input.value = formatted;
    }
    syncDobPickerFromText();
  });

  datasetDom.dobPicker.addEventListener('change', syncDobTextFromPicker);
  datasetDom.dobCalendarBtn.addEventListener('click', openDobCalendar);
}

/**
 * Validate consents and age
 */
function validateConsents() {
  if (!datasetDom.consent18plus.checked || !datasetDom.consentTerms.checked) {
    setStatus(t('dataset-err-need-consent'), true);
    return false;
  }

  const dobIso = getDobIso();
  if (!dobIso) {
    setStatus(t('dataset-err-dob-invalid'), true);
    return false;
  }

  const age = calculateAge(dobIso);
  if (age < 18) {
    setStatus(t('dataset-err-age-check'), true);
    return false;
  }
  
  return true;
}

/**
 * Validate photos captured
 */
function validatePhotos() {
  if (!captureState.frontBlob || !captureState.sideBlob) {
    setStatus(t('dataset-err-need-photos'), true);
    return false;
  }
  return true;
}

/**
 * Scroll to the first marked measurement field.
 */
function scrollToFirstMeasurementField(selector) {
  const field = datasetDom.measurementsContainer.querySelector(selector);
  if (field) {
    field.scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

/**
 * Handle dataset submission (two-stage with outlier confirmation)
 */
async function handleDatasetSubmit() {
  clearStatus();
  
  // Stage 1: Basic validation
  if (!validateConsents()) return;
  if (!validatePhotos()) return;
  
  const analysis = analyzeMeasurements(datasetDom.measurementsContainer);
  const { missing, nonPositive, values: measurementValues } = analysis;

  markMeasurementFieldStates(datasetDom.measurementsContainer, {
    errorIds: [...missing, ...nonPositive],
  });

  if (missing.length > 0) {
    setStatus(t('dataset-err-need-measurements'), true);
    scrollToFirstMeasurementField('.field-err');
    return;
  }

  if (nonPositive.length > 0) {
    setStatus(t('dataset-err-measurement-non-positive'), true);
    scrollToFirstMeasurementField('.field-err');
    return;
  }
  
  // Stage 2: Outlier validation
  const outliers = validateMeasurements(measurementValues);
  
  if (outliers.length > 0 && !pendingConfirm) {
    // First click with outliers: warn and require confirmation
    markMeasurementFieldStates(datasetDom.measurementsContainer, { warnIds: outliers });
    setStatus(t('dataset-warn-outliers'), true);
    pendingConfirm = true;
    
    // Scroll to first outlier
    const firstOutlier = datasetDom.measurementsContainer.querySelector('.field-warn');
    if (firstOutlier) {
      firstOutlier.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    return;
  }
  
  // Stage 3: Submit (no outliers or user confirmed)
  pendingConfirm = false;
  markMeasurementFieldStates(datasetDom.measurementsContainer);
  
  try {
    // Disable button and show loading
    datasetDom.btnSubmit.disabled = true;
    datasetDom.datasetLoading.hidden = false;
    datasetDom.datasetLoadingTitle.textContent = t('dataset-submitting');
    datasetDom.datasetLoadingStep.textContent = '';
    
    // Scroll to loading indicator
    datasetDom.datasetLoading.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Prepare submission data
    const dobIso = getDobIso();
    const age = calculateAge(dobIso);
    const heightCm = parseFloat(datasetDom.heightInput.value);
    const sex = datasetDom.sexSelect.value;
    
    // Update loading steps
    datasetDom.datasetLoadingStep.textContent = t('dataset-encrypting');
    
    // Submit to Supabase
    const result = await uploadAndInsert({
      frontBlob: captureState.frontBlob,
      sideBlob: captureState.sideBlob,
      consent18plus: true,
      consentTerms: true,
      dateOfBirth: dobIso,
      ageYears: age,
      heightCm: heightCm,
      sex: sex,
      measurements: measurementValues
    });
    
    // Success!
    datasetDom.datasetLoading.hidden = true;
    datasetDom.btnSubmit.hidden = true;
    datasetDom.datasetSuccess.hidden = false;
    datasetDom.datasetSuccessId.textContent = t('dataset-success-id', { id: result.id });
    
    // Scroll to success message
    datasetDom.datasetSuccess.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    console.log('Dataset submission successful:', result);
    
  } catch (error) {
    console.error('Dataset submission failed:', error);
    setStatus(t('dataset-err-submission', { error: error.message }), true);
    datasetDom.datasetLoading.hidden = true;
    datasetDom.btnSubmit.disabled = false;
  }
}

function refreshPoseStatusMessage() {
  if (captureState.poseLandmarker) {
    ui.setPoseStatusVisual(t('pose-model-ready'), 'ok');
  } else if (captureState.poseLoadError) {
    ui.setPoseStatusVisual(t('pose-model-failed'), 'bad');
  }
}

/** Refresh JS-driven strings after language change (static HTML uses translatePage). */
function refreshDatasetLocale() {
  ui.updateUiStep();
  session.resetTimerButtonLabel();
  refreshPoseStatusMessage();
  refreshDobFieldLocale();

  const values = getMeasurementValues(datasetDom.measurementsContainer);
  renderMeasurementGrid(datasetDom.measurementsContainer, t, values);
  prefillHeight();
}

/**
 * Prefill height in measurement grid from params input
 */
function prefillHeight() {
  const heightInput = datasetDom.measurementsContainer.querySelector('#measure-height');
  if (heightInput && datasetDom.heightInput.value) {
    heightInput.value = datasetDom.heightInput.value;
  }
}

/**
 * Sync height between params and measurements
 */
function syncHeight() {
  datasetDom.heightInput.addEventListener('input', () => {
    prefillHeight();
  });
}

/**
 * Main initialization
 */
async function init() {
  console.log('Dataset collection app initializing...');
  
  // Initialize i18n (uses data-page-title on <body> for document title)
  initI18n();
  
  // Initialize DOM references
  const dom = getCaptureDom();
  initDatasetDom();
  initDateOfBirthField();
  
  // Load guide geometry and draw pose example silhouettes
  applyReloadGuideQueryParam();
  loadGuideGeometry();
  refreshGuideArtwork();
  void loadGuideGeometryFromOptionalStaticFile().then(() => {
    refreshGuideArtwork();
  });
  window.addEventListener('resize', refreshGuideArtwork);
  const thumbsWrap = document.getElementById('thumbs-wrap');
  if (thumbsWrap && typeof ResizeObserver !== 'undefined') {
    new ResizeObserver(() => renderThumbPlaceholders()).observe(thumbsWrap);
  }
  requestAnimationFrame(() => {
    renderThumbPlaceholders();
  });
  
  // Wire language toggle — setLang() dispatches langchanged; handler refreshes dynamic UI
  const langBtns = document.querySelectorAll('.lang-btn');
  langBtns.forEach(btn => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang));
  });
  window.addEventListener('langchanged', () => {
    refreshDatasetLocale();
    refreshGuideArtwork();
  });
  
  // Wire scroll navigation
  const btnToCapture = document.getElementById('btn-to-capture');
  if (btnToCapture) {
    btnToCapture.addEventListener('click', () => {
      const target = document.getElementById('section-params');
      if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  }
  
  // Wire camera/capture controls (reuse from main app)
  dom.btnStart.addEventListener('click', () => void session.startCamera());
  dom.btnStop.addEventListener('click', () => session.stopCamera());
  dom.btnCapture.addEventListener('click', () => {
    session.clearCaptureTimer();
    session.resetAutoCaptureUi();
    session.captureFrameToBlob(session.onCaptureReady);
  });
  dom.btnCaptureTimer.addEventListener('click', () => void session.startCaptureTimer());
  
  // Wire retake buttons
  const btnRetakeFront = document.getElementById('btn-retake-front');
  const btnRetakeSide = document.getElementById('btn-retake-side');
  if (btnRetakeFront) {
    btnRetakeFront.addEventListener('click', () => {
      session.retakeFrontPhoto();
    });
  }
  if (btnRetakeSide) {
    btnRetakeSide.addEventListener('click', () => {
      session.retakeSidePhoto();
    });
  }
  
  // Wire upload inputs (query directly as they're not in getCaptureDom())
  const uploadFront = document.getElementById('upload-front');
  const uploadSide = document.getElementById('upload-side');
  if (uploadFront) {
    uploadFront.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) void session.applyUploadedImage('front', file);
    });
  }
  if (uploadSide) {
    uploadSide.addEventListener('change', (e) => {
      const file = e.target.files?.[0];
      if (file) void session.applyUploadedImage('side', file);
    });
  }
  
  // Render measurement grid
  renderMeasurementGrid(datasetDom.measurementsContainer, t);
  
  // Prefill height and sync
  prefillHeight();
  syncHeight();
  
  // Wire submit button
  datasetDom.btnSubmit.addEventListener('click', () => void handleDatasetSubmit());
  
  // Clear pending confirm when user edits any measurement
  datasetDom.measurementsContainer.addEventListener('input', () => {
    if (pendingConfirm) {
      pendingConfirm = false;
      clearStatus();
    }
  });

  ui.updateUiStep();

  void session.loadPoseLandmarker().then(() => {
    refreshPoseStatusMessage();
  });
  
  console.log('Dataset collection app ready!');
}

// Start when DOM ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  void init();
}
