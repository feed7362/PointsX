/**
 * Dataset measurement definitions and validation
 * 
 * Defines the 17 manual measurements users must enter, with plausible ranges
 * for outlier detection (yellow field warnings).
 */

/**
 * 17 manual measurement definitions
 * 
 * Each measurement has:
 * - id: JSON key for storage
 * - labelKey: i18n key (dataset-measure-{id})
 * - unit: 'cm' for all body measurements
 * - plausibleMin/Max: Heuristic ranges for outlier detection
 */
export const MEASUREMENTS = [
  {
    id: 'height',
    labelKey: 'dataset-measure-height',
    unit: 'cm',
    plausibleMin: 140,
    plausibleMax: 220
  },
  {
    id: 'neck_base_height_from_floor',
    labelKey: 'dataset-measure-neck-base-height',
    unit: 'cm',
    plausibleMin: 110,
    plausibleMax: 180
  },
  {
    id: 'neck_circumference',
    labelKey: 'dataset-measure-neck-circumference',
    unit: 'cm',
    plausibleMin: 28,
    plausibleMax: 50
  },
  {
    id: 'chest_circumference',
    labelKey: 'dataset-measure-chest-circumference',
    unit: 'cm',
    plausibleMin: 70,
    plausibleMax: 150
  },
  {
    id: 'waist_circumference',
    labelKey: 'dataset-measure-waist-circumference',
    unit: 'cm',
    plausibleMin: 55,
    plausibleMax: 150
  },
  {
    id: 'hip_circumference',
    labelKey: 'dataset-measure-hip-circumference',
    unit: 'cm',
    plausibleMin: 70,
    plausibleMax: 160
  },
  {
    id: 'arm_circumference_bicep',
    labelKey: 'dataset-measure-arm-circumference',
    unit: 'cm',
    plausibleMin: 20,
    plausibleMax: 50
  },
  {
    id: 'thigh_circumference',
    labelKey: 'dataset-measure-thigh-circumference',
    unit: 'cm',
    plausibleMin: 35,
    plausibleMax: 80
  },
  {
    id: 'shoulder_width',
    labelKey: 'dataset-measure-shoulder-width',
    unit: 'cm',
    plausibleMin: 30,
    plausibleMax: 55
  },
  {
    // Shoulder slope (Шп): neck-base point -> shoulder point, one side. Same id and
    // 8–25 cm band as the envelope's `shoulder_slope_width` so this is the only form
    // field with a direct GT counterpart for it (shoulder_width is the full width).
    id: 'shoulder_slope_width',
    labelKey: 'dataset-measure-shoulder-slope-width',
    unit: 'cm',
    plausibleMin: 8,
    plausibleMax: 25
  },
  {
    id: 'back_width',
    labelKey: 'dataset-measure-back-width',
    unit: 'cm',
    plausibleMin: 28,
    plausibleMax: 50
  },
  {
    id: 'chest_width',
    labelKey: 'dataset-measure-chest-width',
    unit: 'cm',
    plausibleMin: 25,
    plausibleMax: 50
  },
  {
    id: 'front_length_to_waist',
    labelKey: 'dataset-measure-front-length-to-waist',
    unit: 'cm',
    plausibleMin: 35,
    plausibleMax: 65
  },
  {
    id: 'back_length_to_waist',
    labelKey: 'dataset-measure-back-length-to-waist',
    unit: 'cm',
    plausibleMin: 35,
    plausibleMax: 65
  },
  {
    id: 'sleeve_length',
    labelKey: 'dataset-measure-sleeve-length',
    unit: 'cm',
    plausibleMin: 50,
    plausibleMax: 90
  },
  {
    id: 'outer_seam',
    labelKey: 'dataset-measure-outer-seam',
    unit: 'cm',
    plausibleMin: 80,
    plausibleMax: 130
  },
  {
    id: 'inner_seam',
    labelKey: 'dataset-measure-inner-seam',
    unit: 'cm',
    plausibleMin: 60,
    plausibleMax: 100
  }
];

/**
 * Render measurement input grid
 * 
 * @param {HTMLElement} container - Container element to render into
 * @param {function} t - i18n translation function
 * @param {Object} initialValues - Optional initial values {id: value}
 */
export function renderMeasurementGrid(container, t, initialValues = {}) {
  container.innerHTML = '';
  
  const grid = document.createElement('div');
  grid.className = 'measurement-grid';
  
  MEASUREMENTS.forEach(measurement => {
    const field = document.createElement('div');
    field.className = 'field measurement-field';
    field.dataset.measurementId = measurement.id;
    
    const label = document.createElement('label');
    label.htmlFor = `measure-${measurement.id}`;
    label.textContent = t(measurement.labelKey);
    
    const inputWrapper = document.createElement('div');
    inputWrapper.className = 'measurement-input-wrapper';
    
    const input = document.createElement('input');
    input.type = 'number';
    input.id = `measure-${measurement.id}`;
    input.name = measurement.id;
    input.step = '0.1';
    input.min = '0';
    input.max = '300';
    input.required = true;
    input.placeholder = t('dataset-measure-placeholder');
    
    if (initialValues[measurement.id] !== undefined) {
      input.value = initialValues[measurement.id];
    }
    
    // Auto-clear validation highlight when user edits the field
    input.addEventListener('input', () => {
      field.classList.remove('field-warn', 'field-err');
    });
    
    const unitSpan = document.createElement('span');
    unitSpan.className = 'measurement-unit';
    unitSpan.textContent = measurement.unit;
    
    inputWrapper.appendChild(input);
    inputWrapper.appendChild(unitSpan);
    
    field.appendChild(label);
    field.appendChild(inputWrapper);
    grid.appendChild(field);
  });
  
  container.appendChild(grid);
}

/**
 * Get all measurement values from the rendered grid
 * 
 * @param {HTMLElement} container - Container with measurement inputs
 * @returns {Object} - {id: value} map
 */
export function getMeasurementValues(container) {
  const values = {};
  
  MEASUREMENTS.forEach(measurement => {
    const input = container.querySelector(`#measure-${measurement.id}`);
    if (input && input.value) {
      values[measurement.id] = parseFloat(input.value);
    }
  });
  
  return values;
}

/**
 * Validate measurements and identify outliers
 * 
 * Returns array of measurement IDs that are outside plausible ranges.
 * 
 * @param {Object} values - {id: value} map
 * @returns {string[]} - Array of outlier measurement IDs
 */
export function validateMeasurements(values) {
  const outliers = [];
  
  MEASUREMENTS.forEach(measurement => {
    const value = values[measurement.id];
    
    if (value === undefined || value === null) {
      return; // Skip missing values (handled by required attribute)
    }
    
    if (value < measurement.plausibleMin || value > measurement.plausibleMax) {
      outliers.push(measurement.id);
    }
  });
  
  return outliers;
}

/**
 * Analyze measurement inputs: missing, non-positive, and valid values.
 *
 * @param {HTMLElement} container
 * @returns {{ missing: string[], nonPositive: string[], values: Object }}
 */
export function analyzeMeasurements(container) {
  const missing = [];
  const nonPositive = [];
  const values = {};

  MEASUREMENTS.forEach((measurement) => {
    const input = container.querySelector(`#measure-${measurement.id}`);
    if (!input) return;

    const raw = input.value.trim();
    if (raw === '') {
      missing.push(measurement.id);
      return;
    }

    const value = parseFloat(raw);
    if (!Number.isFinite(value) || value <= 0) {
      nonPositive.push(measurement.id);
      return;
    }

    values[measurement.id] = value;
  });

  return { missing, nonPositive, values };
}

/**
 * Mark measurement fields with error (invalid/missing) or warning (outlier) state.
 *
 * @param {HTMLElement} container
 * @param {{ errorIds?: string[], warnIds?: string[] }} states
 */
export function markMeasurementFieldStates(container, { errorIds = [], warnIds = [] } = {}) {
  container.querySelectorAll('.measurement-field').forEach((field) => {
    field.classList.remove('field-err', 'field-warn');
  });

  errorIds.forEach((id) => {
    const field = container.querySelector(`[data-measurement-id="${id}"]`);
    if (field) field.classList.add('field-err');
  });

  warnIds.forEach((id) => {
    const field = container.querySelector(`[data-measurement-id="${id}"]`);
    if (field) field.classList.add('field-warn');
  });
}

/**
 * Mark outlier fields with warning class
 * 
 * @param {HTMLElement} container - Container with measurement inputs
 * @param {string[]} outlierIds - Array of outlier measurement IDs
 */
export function markOutlierFields(container, outlierIds) {
  markMeasurementFieldStates(container, { warnIds: outlierIds });
}

/**
 * Validate all measurements are filled
 * 
 * @param {Object} values - {id: value} map
 * @returns {boolean} - true if all 17 measurements present and numeric
 */
export function areAllMeasurementsFilled(values) {
  if (Object.keys(values).length !== MEASUREMENTS.length) {
    return false;
  }
  
  return MEASUREMENTS.every(measurement => {
    const value = values[measurement.id];
    return typeof value === 'number' && !isNaN(value) && value > 0;
  });
}
