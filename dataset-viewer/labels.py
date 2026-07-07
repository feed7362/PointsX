"""Human-readable labels for dataset measurement keys (English)."""

MEASUREMENT_LABELS: dict[str, str] = {
    "height": "Height",
    "neck_base_height_from_floor": "Neck base height (from floor)",
    "neck_circumference": "Neck circumference",
    "chest_circumference": "Chest circumference",
    "waist_circumference": "Waist circumference",
    "hip_circumference": "Hip circumference",
    "arm_circumference_bicep": "Upper arm circumference (bicep)",
    "thigh_circumference": "Thigh circumference (upper leg)",
    "shoulder_width": "Shoulder width",
    "back_width": "Back width",
    "chest_width": "Chest width",
    "front_length_to_waist": "Front length to waist",
    "back_length_to_waist": "Back length to waist",
    "sleeve_length": "Sleeve length",
    "outer_seam": "Outer seam",
    "inner_seam": "Inner seam",
}

MEASUREMENT_ORDER: list[str] = list(MEASUREMENT_LABELS.keys())
