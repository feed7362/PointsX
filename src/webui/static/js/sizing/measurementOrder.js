/**
 * Display order of measurements and the ids hidden from the results tables.
 */


const MEASUREMENT_MANUAL_ORDER = [
  "chest_circumference",
  "waist_circumference",
  "hip_circumference",
  "thigh_circumference",
 
  "neck_base_height",
  "chest_width_front",
  "back_width_scapular",
  "shoulder_slope_width",

  "arm_length_shoulder_to_wrist",
  "leg_length_outer_seam",
  "leg_length_inner_seam",
  
  "back_length_to_waist",
  "front_length_to_waist",

  "neck_circumference",
  "upper_arm_circumference", 
  "wrist_circumference",
  "calf_circumference",
  "ankle_circumference",
];


export const HIDDEN_MEASUREMENT_IDS = new Set([
  "neck_circumference",
  "upper_arm_circumference",
  "wrist_circumference",
  "calf_circumference",
  "ankle_circumference",
  "back_width_scapular",
  "front_length_to_waist",
]);


export function orderMeasurementsManual(list) {
  const rows = (Array.isArray(list) ? list : []).filter(
    (r) => !HIDDEN_MEASUREMENT_IDS.has(String(r?.id ?? ""))
  );
  const byId = new Map(rows.map((r) => [r?.id, r]));
  const ordered = [];
  for (const id of MEASUREMENT_MANUAL_ORDER) {
    const row = byId.get(id);
    if (row) ordered.push(row);
  }
  for (const row of rows) {
    if (!MEASUREMENT_MANUAL_ORDER.includes(row?.id)) ordered.push(row);
  }
  return ordered;
}
