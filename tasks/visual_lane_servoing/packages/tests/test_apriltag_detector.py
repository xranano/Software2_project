import unittest

import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.apriltag_detector import AprilTagDetector
from tasks.visual_lane_servoing.packages import april_tag
from tasks.visual_lane_servoing.packages.sign_behavior_config import SignBehaviorConfig


class _Ctx:
    config = SignBehaviorConfig(tag_10_11_swap=False)


@unittest.skipUnless(hasattr(cv2, "aruco"), "OpenCV aruco module is unavailable")
class TestAprilTagDetector(unittest.TestCase):
    def test_detects_tag36h11_id(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        marker = cv2.aruco.generateImageMarker(dictionary, 21, 200)
        frame = np.full((480, 640), 255, dtype=np.uint8)
        frame[140:340, 220:420] = marker
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        detections = AprilTagDetector(min_area=50).detect(frame_bgr)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].tag_id, 21)
        self.assertGreater(detections[0].area, 30000)
        self.assertAlmostEqual(detections[0].center[0], 319.5, delta=1.0)

    def test_returns_no_detections_for_blank_frame(self):
        frame = np.full((480, 640, 3), 255, dtype=np.uint8)
        self.assertEqual(AprilTagDetector().detect(frame), [])


class TestAprilTagConfirm(unittest.TestCase):
    def test_requires_two_frames_by_default(self):
        fsm = _Ctx()
        fsm._tag_buffer = {}
        raw = [{"tag_id": 26, "corners": np.zeros((4, 2))}]
        self.assertEqual(april_tag.confirm_tags(fsm, raw), [])
        self.assertEqual(april_tag.confirm_tags(fsm, raw), [26])
        self.assertEqual(april_tag.confirm_tags(fsm, []), [])


if __name__ == "__main__":
    unittest.main()
