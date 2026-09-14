"""Deterministic demo envelope for the «без фото» UI button (no images, no ML)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from webui.schemas import (
    CaptureInfo,
    CaptureQuality,
    MeasurementEnvelope,
    MeasurementItem,
    PipelineInfo,
    SubjectInfo,
)


def build_mock_measurement_envelope(
    height_cm: float,
    sex: Literal["male", "female", "other"],
) -> MeasurementEnvelope:
    """Deterministic demo envelope for the «без фото» UI button (no ML)."""
    from webui.envelope import CANONICAL_MEASUREMENTS

    h_scale = height_cm / 175.0
    if sex == "female":
        sex_scale = 0.94
    elif sex == "male":
        sex_scale = 1.0
    else:
        sex_scale = 0.97

    base_cm: dict[str, float] = {
        "chest_circumference": 102.0,
        "waist_circumference": 86.0,
        "hip_circumference": 100.0,
        "neck_circumference": 39.0,
        "neck_base_height": 148.0,
        "shoulder_slope_width": 46.0,
        "back_width_scapular": 38.0,
        "chest_width_front": 34.0,
        "back_length_to_waist": 44.0,
        "front_length_to_waist": 42.0,
        "arm_length_shoulder_to_wrist": 60.0,
        "upper_arm_circumference": 30.0,
        "wrist_circumference": 17.0,
        "leg_length_inner_seam": 78.0,
        "leg_length_outer_seam": 102.0,
        "thigh_circumference": 58.0,
        "calf_circumference": 38.0,
        "ankle_circumference": 24.0,
    }

    measurements: list[MeasurementItem] = []
    for mid, label_uk, src in CANONICAL_MEASUREMENTS:
        raw = base_cm.get(mid, 50.0) * h_scale * sex_scale
        val = round(max(1.0, raw), 1)
        measurements.append(
            MeasurementItem(
                id=mid,
                label_uk=label_uk,
                value_cm=val,
                uncertainty_cm=round(max(0.5, val * 0.04), 1),
                confidence=0.55,
                source=src,
                quality_flags=["mock"],
            )
        )

    capture = CaptureInfo(
        front=CaptureQuality(quality=0.55, pose_ok=True, occlusions=[]),
        side=CaptureQuality(quality=0.55, pose_ok=True, occlusions=[]),
    )
    return MeasurementEnvelope(
        request_id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        pipeline=PipelineInfo(source="mock", model_version="mock-0.1", unit_system="metric"),
        subject=SubjectInfo(height_cm=height_cm, sex=sex, age_band="adult", posture_flags=[]),
        capture=capture,
        measurements=measurements,
        derived={},
        warnings=["Тестовий режим: зображення й моделі не використовувалися."],
    )
