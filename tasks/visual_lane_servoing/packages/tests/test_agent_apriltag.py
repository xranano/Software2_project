import unittest
from unittest.mock import patch

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.visual_lane_servoing.packages.apriltag_detector import AprilTagDetection


class _FakeAprilTagDetector:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, _frame_bgr):
        return [AprilTagDetection(
            tag_id=21,
            corners=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
            center=(20.0, 20.0),
            area=400.0,
        )]


class TestAgentAprilTag(unittest.TestCase):
    @patch("tasks.visual_lane_servoing.packages.agent.AprilTagDetector", _FakeAprilTagDetector)
    def test_stop_detection_is_published_to_debug_info(self):
        agent = LaneServoingAgent()
        agent.apriltag_interval = 1

        left, right = agent.compute_commands(np.zeros((48, 64, 3), dtype=np.uint8))

        self.assertEqual((left, right), (0.0, 0.0))
        self.assertEqual(agent.last_debug_info["frame_count"], 1)
        self.assertEqual(agent.last_debug_info["apriltags"][0]["id"], 21)
        self.assertIsNone(agent.last_debug_info["apriltag_error"])


if __name__ == "__main__":
    unittest.main()
