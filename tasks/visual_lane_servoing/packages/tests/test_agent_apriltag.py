import unittest
from unittest.mock import patch

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.sign_detection.packages.sign_behavior_config import TagID


class TestAgentAprilTag(unittest.TestCase):
    def test_lane_follows_without_tags_or_red_line(self):
        agent = LaneServoingAgent()
        with patch(
            "tasks.sign_detection.packages.sign_behavior.detect_tags",
            return_value=[],
        ), patch(
            "tasks.sign_detection.packages.sign_behavior.detect_red_line",
            return_value=False,
        ):
            left, right = agent.compute_commands(np.zeros((48, 64, 3), dtype=np.uint8))

        self.assertEqual(agent.sign_state, "MOVING")
        self.assertFalse(agent.sign_debug.get("red_line", True))
        self.assertIsNone(agent.sign_debug.get("saved_tag"))
        self.assertEqual(agent.last_debug_info["frame_count"], 1)

    def test_tag_alone_does_not_stop_robot(self):
        agent = LaneServoingAgent()
        agent._sign_fsm.config.tag_confirm_frames = 1

        tag = {"tag_id": 26, "corners": np.zeros((4, 2), dtype=np.float32)}
        with patch(
            "tasks.sign_detection.packages.sign_behavior.detect_tags",
            return_value=[tag],
        ), patch(
            "tasks.sign_detection.packages.sign_behavior.detect_red_line",
            return_value=False,
        ):
            left, right = agent.compute_commands(np.zeros((48, 64, 3), dtype=np.uint8))

        self.assertEqual(agent.sign_state, "MOVING")
        self.assertEqual(agent.sign_debug.get("saved_tag"), int(TagID.STOP))
        self.assertNotIn(agent.sign_state, ("SLOWING", "STOPPED", "CHECKPATH"))

    def test_red_line_alone_does_not_stop_robot(self):
        agent = LaneServoingAgent()
        with patch(
            "tasks.sign_detection.packages.sign_behavior.detect_tags",
            return_value=[],
        ), patch(
            "tasks.sign_detection.packages.sign_behavior.detect_red_line",
            return_value=True,
        ):
            left, right = agent.compute_commands(np.zeros((48, 64, 3), dtype=np.uint8))

        self.assertEqual(agent.sign_state, "MOVING")
        self.assertTrue(agent.sign_debug.get("red_line"))
        self.assertIsNone(agent.sign_debug.get("saved_tag"))


if __name__ == "__main__":
    unittest.main()
