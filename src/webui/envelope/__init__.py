"""Convert a `BodyMeasurements` (+ keypoints, calibration) to a `MeasurementEnvelope`.

The envelope is the public API contract consumed by the frontend size + pattern
engines. It exposes 18 canonical measurement IDs. The `BodyMeasurements`
dataclass maps directly to 11 of them; the other 7 are derived from keypoints +
widths (no extra ML required).

    catalog.py      ids, labels, display subset, default confidence, plausible ranges
    corrections.py  per-sex circumference and length bias corrections (with provenance)
    derive.py       chest, back/front length, neck base, upper arm, ankle
    priors.py       neck and back width from height, sex and chest (ANSUR II linear models)
    build.py        body_to_envelope
"""
from webui.envelope.build import body_to_envelope
from webui.envelope.catalog import CANONICAL_MEASUREMENTS, DISPLAY_MEASUREMENT_IDS

__all__ = ["CANONICAL_MEASUREMENTS", "DISPLAY_MEASUREMENT_IDS", "body_to_envelope"]
