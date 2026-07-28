"""Headless wrapper around the vendored SMPL-Anthropometry (MIT, David Bojanić).

The upstream modules use bare intra-package imports and a plotly-based
visualizer. This wrapper adds the vendored dir to sys.path so those imports
resolve, stubs the optional visualizer (no plotly needed for measuring), and
exposes a single `measure_smplx()` that drives the vetted measurement
definitions from our own SMPL-X verts/joints/faces — no model .pkl files or
joint regressor required.

Measurement method (upstream): the mesh is sliced by a plane through anatomical
landmarks; the slice is filtered to the relevant BODY PART via face
segmentation (so arms never contaminate the waist/hip), then a convex-hull
perimeter is taken — exactly how a tape measure behaves.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# visualize.py imports plotly and is only used for 3D viz — stub it so the
# measurement code imports cleanly in a headless pipeline.
for _name in ("plotly", "plotly.graph_objects"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
if "visualize" not in sys.modules:
    _v = types.ModuleType("visualize")
    _v.Visualizer = object
    sys.modules["visualize"] = _v

import measure as _M  # noqa: E402  (vendored)

_seg = _M.load_face_segmentation(str(_HERE / "data" / "smplx" / "smplx_body_parts_2_faces.json"))
_defs = _M.SMPLXMeasurementDefinitions()

# Vetted SMPL-X measurement name → our BodyMeasurementsGT field.
NAME_MAP: dict[str, str] = {
    "height": "height_cm",
    "chest circumference": "chest_circumference_cm",
    "waist circumference": "waist_circumference_cm",
    "hip circumference": "hips_circumference_cm",
    "thigh left circumference": "thigh_circumference_cm",
    "calf left circumference": "calf_circumference_cm",
    "neck circumference": "neck_circumference_cm",
    "wrist right circumference": "wrist_circumference_cm",
    "shoulder breadth": "shoulder_width_cm",
    "arm left length": "arm_length_cm",
    "inside leg height": "inseam_length_cm",
}
# Only request names the vendored definitions actually know.
_REQUEST = [n for n in NAME_MAP if n in _defs.possible_measurements]


def measure_smplx(verts: np.ndarray, joints: np.ndarray, faces: np.ndarray) -> dict[str, float]:
    """Measure an SMPL-X body. Returns {our_field: cm} for every measurement
    that computed to a positive value. Missing / non-positive results are
    simply absent — the caller decides whether the body is usable. No fake
    constants are ever substituted.
    """
    m = object.__new__(_M.MeasureSMPLX)
    for attr in ("measurements", "height_normalized_measurements", "labeled_measurements",
                 "height_normalized_labeled_measurements", "labels2names"):
        setattr(m, attr, {})
    m.model_type = "smplx"
    m.num_points = 10475
    m.gender = None
    m.faces = np.asarray(faces)
    m.face_segmentation = _seg
    m.landmarks = _M.SMPLX_LANDMARK_INDICES
    m.measurement_types = _M.MEASUREMENT_TYPES
    m.length_definitions = _defs.LENGTHS
    m.circumf_definitions = _defs.CIRCUMFERENCES
    m.circumf_2_bodypart = _defs.CIRCUMFERENCE_TO_BODYPARTS
    m.all_possible_measurements = _defs.possible_measurements
    m.joint2ind = _M.SMPLX_JOINT2IND
    m.num_joints = _M.SMPLX_NUM_JOINTS
    m.verts = np.asarray(verts).squeeze()
    m.joints = np.asarray(joints).squeeze()

    m.measure(_REQUEST)
    return {
        NAME_MAP[name]: float(val)
        for name, val in m.measurements.items()
        if name in NAME_MAP and val is not None and float(val) > 0.0
    }
