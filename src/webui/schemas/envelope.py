"""MeasurementEnvelope v2 — the JSON contract consumed by the frontend sizing + pattern engine."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MeasurementItem(BaseModel):
    id: str
    label_uk: str
    value_cm: float = Field(..., description="Body measurement in centimetres, one decimal")
    uncertainty_cm: float = Field(..., ge=0.0, description="1σ estimate from the pipeline")
    confidence: float = Field(..., ge=0.0, le=1.0)
    source: Literal["front", "side", "fused", "manual"] = "fused"
    quality_flags: list[str] = Field(default_factory=list)


class PipelineInfo(BaseModel):
    source: Literal["mock", "mediapipe", "regression"] = "regression"
    model_version: str = "regression-0.1"
    unit_system: Literal["metric"] = "metric"
    pose_backend: Literal["custom", "coco"] | None = None


class SubjectInfo(BaseModel):
    height_cm: float = Field(..., ge=100, le=250)
    sex: Literal["male", "female", "other"]
    age_band: Literal["adult", "teen", "child"] | None = None
    posture_flags: list[str] = Field(default_factory=list)


class CaptureQuality(BaseModel):
    quality: float = Field(..., ge=0.0, le=1.0)
    pose_ok: bool = True
    occlusions: list[str] = Field(default_factory=list)


class CaptureInfo(BaseModel):
    front: CaptureQuality
    side: CaptureQuality


class MeasurementEnvelope(BaseModel):
    """Schema v2 — consumed by the frontend sizing + pattern engine."""
    schema_id: str = Field("pointsx.measurement.envelope", alias="schema")
    schema_version: int = 2
    request_id: str
    created_at: str
    pipeline: PipelineInfo
    subject: SubjectInfo
    capture: CaptureInfo
    measurements: list[MeasurementItem]
    derived: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}
