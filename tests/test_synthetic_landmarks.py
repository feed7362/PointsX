from __future__ import annotations

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointsx.synthetic.landmarks import (
    LANDMARK_NAMES,
    POINTSX16_FLIP_IDX,
    POINTSX16_NAMES,
    select_pointsx16,
)


class TestSyntheticLandmarkExport(unittest.TestCase):
    def test_pointsx16_shape_and_flip(self):
        self.assertEqual(len(POINTSX16_NAMES), 16)
        self.assertEqual(len(POINTSX16_FLIP_IDX), 16)

    def test_select_pointsx16_order(self):
        by_name = {name: [float(i), float(i + 1), float(i + 2)] for i, name in enumerate(LANDMARK_NAMES)}
        selected = select_pointsx16(by_name)
        self.assertEqual(len(selected), 16)

        right_ankle_idx = LANDMARK_NAMES.index("right_ankle")
        left_ankle_idx = LANDMARK_NAMES.index("left_ankle")
        self.assertEqual(selected[0], by_name[LANDMARK_NAMES[right_ankle_idx]])
        self.assertEqual(selected[5], by_name[LANDMARK_NAMES[left_ankle_idx]])

    def test_select_pointsx16_requires_all_required_names(self):
        by_name = {name: [0.0, 0.0, 0.0] for name in LANDMARK_NAMES}
        by_name.pop("right_shoulder")
        with self.assertRaises(KeyError):
            select_pointsx16(by_name)


if __name__ == "__main__":
    unittest.main()
