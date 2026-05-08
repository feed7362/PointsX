from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pointsx.keypoints import KP
from pointsx.pose_coco import coco17_to_lv_mhp16


class TestCoco17ToLvMhp16(unittest.TestCase):
    def test_shape_and_view(self):
        xy = np.arange(17 * 2, dtype=np.float32).reshape(17, 2)
        conf = np.linspace(0.1, 0.9, 17).astype(np.float32)
        kp = coco17_to_lv_mhp16(xy, conf, view="front")
        self.assertEqual(kp.points.shape, (16, 2))
        self.assertEqual(kp.confidence.shape, (16,))
        self.assertEqual(kp.view, "front")

    def test_wrong_count_raises(self):
        xy = np.zeros((16, 2), dtype=np.float32)
        conf = np.ones(16, dtype=np.float32)
        with self.assertRaises(ValueError):
            coco17_to_lv_mhp16(xy, conf, "front")

    def test_shoulders_match_coco(self):
        xy = np.zeros((17, 2), dtype=np.float32)
        conf = np.ones(17, dtype=np.float32) * 0.8
        xy[5] = [10.0, 20.0]  # left_shoulder
        xy[6] = [30.0, 22.0]  # right_shoulder
        xy[0] = [20.0, 5.0]  # nose
        # minimal fill for hips/knees/ankles/elbows/wrists
        for i in range(1, 17):
            if i in (5, 6):
                continue
            xy[i] = [float(i), float(i)]

        kp = coco17_to_lv_mhp16(xy, conf, "side")
        np.testing.assert_array_equal(kp.points[KP.LEFT_SHOULDER], xy[5])
        np.testing.assert_array_equal(kp.points[KP.RIGHT_SHOULDER], xy[6])
        thorax = (xy[5] + xy[6]) * 0.5
        np.testing.assert_array_almost_equal(kp.points[KP.THORAX], thorax)

    def test_head_top_above_nose_in_image_coords(self):
        """y increases downward; head top should have smaller y than nose."""
        xy = np.zeros((17, 2), dtype=np.float32)
        conf = np.ones(17, dtype=np.float32)
        xy[0] = [100.0, 50.0]  # nose
        xy[5] = [90.0, 80.0]
        xy[6] = [110.0, 82.0]
        for i in range(1, 17):
            if i in (5, 6):
                continue
            xy[i] = [100.0, 100.0 + float(i)]

        kp = coco17_to_lv_mhp16(xy, conf, "front")
        self.assertLess(kp.points[KP.HEAD_TOP, 1], xy[0, 1])

    def test_all_finite(self):
        rng = np.random.default_rng(0)
        xy = rng.standard_normal((17, 2)).astype(np.float32) * 50 + 128
        conf = rng.random(17).astype(np.float32)
        kp = coco17_to_lv_mhp16(xy, conf, "front")
        self.assertTrue(np.all(np.isfinite(kp.points)))
        self.assertTrue(np.all(np.isfinite(kp.confidence)))


if __name__ == "__main__":
    unittest.main()
