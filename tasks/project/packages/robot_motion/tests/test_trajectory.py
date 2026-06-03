"""Tests for trajectory helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from tasks.project.packages.robot_motion.config import INTERSECTION_RESET_POSE
from tasks.project.packages.robot_motion.trajectory import (
    drive_start,
    has_turn_section,
    load_all_trajectories,
    straight_distance_from_reset,
    turn_start_index,
)

_TRAJECTORIES = Path(__file__).resolve().parent.parent / "trajectories"


class TestTrajectoryHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajs = load_all_trajectories(_TRAJECTORIES)

    def test_left_has_turn_after_straight(self) -> None:
        left = self.trajs["left"]
        idx = turn_start_index(left)
        reset_y = INTERSECTION_RESET_POSE["y"]
        self.assertGreater(idx, 0)
        self.assertAlmostEqual(float(left["track"][idx, 1]), 0.0, places=2)
        self.assertAlmostEqual(straight_distance_from_reset(left, reset_y), -reset_y, places=2)

    def test_right_has_shorter_straight(self) -> None:
        right = self.trajs["right"]
        reset_y = INTERSECTION_RESET_POSE["y"]
        self.assertTrue(has_turn_section(right))
        self.assertLess(
            straight_distance_from_reset(right, reset_y),
            straight_distance_from_reset(self.trajs["left"], reset_y),
        )

    def test_straight_has_no_turn_section(self) -> None:
        self.assertFalse(has_turn_section(self.trajs["straight"]))

    def test_drive_start_for_left_is_at_intersection(self) -> None:
        _, x, y, yaw = drive_start(self.trajs["left"])
        self.assertAlmostEqual(x, 0.0, places=2)
        self.assertAlmostEqual(y, 0.0, places=2)
        self.assertAlmostEqual(yaw, 1.5707963267948966, delta=0.02)


if __name__ == "__main__":
    unittest.main()
