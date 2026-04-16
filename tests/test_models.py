from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointsx.models import _resolve_device, _select_primary_person_index


class TestModelSelection(unittest.TestCase):
    def test_selects_largest_without_reference(self):
        areas = np.array([1200.0, 1800.0, 1500.0], dtype=np.float32)
        centers = np.array([[50.0, 80.0], [180.0, 120.0], [300.0, 200.0]], dtype=np.float32)
        idx = _select_primary_person_index(areas, centers, reference_point=None)
        self.assertEqual(idx, 1)

    def test_selects_closest_with_reference(self):
        areas = np.array([1400.0, 1700.0], dtype=np.float32)
        centers = np.array([[48.0, 62.0], [260.0, 260.0]], dtype=np.float32)
        idx = _select_primary_person_index(areas, centers, reference_point=(52.0, 58.0))
        self.assertEqual(idx, 0)

    def test_single_candidate_is_always_zero(self):
        areas = np.array([999.0], dtype=np.float32)
        centers = np.array([[10.0, 10.0]], dtype=np.float32)
        idx = _select_primary_person_index(areas, centers, reference_point=(100.0, 100.0))
        self.assertEqual(idx, 0)

    def test_resolve_device_auto_returns_valid_runtime(self):
        value = _resolve_device("auto")
        self.assertIn(value, ("cpu", "cuda"))
        self.assertEqual(_resolve_device("cpu"), "cpu")


if __name__ == "__main__":
    unittest.main()
