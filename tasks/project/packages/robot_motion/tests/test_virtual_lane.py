"""Unit tests for robot_motion virtual lane planner."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from tasks.project.packages.robot_motion.config import END_CONDITIONS, INTERSECTION_RESET_POSE
from tasks.project.packages.robot_motion.trajectory import load_trajectory
from tasks.project.packages.robot_motion.virtual_lane import compute_lane_pose, global_path_index

_TRAJECTORIES = Path(__file__).resolve().parent.parent / "trajectories"


class TestComputeLanePose(unittest.TestCase):
    """Verify lane errors on the straight reference path."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.traj = load_trajectory(_TRAJECTORIES / "straight.yaml", "straight")
        cls.end = END_CONDITIONS["straight"]

    def test_on_path_near_origin(self) -> None:
        """Robot at reset pose on straight path → d≈0, phi≈0, curv=0."""
        d, phi, curv, done = compute_lane_pose(
            x=INTERSECTION_RESET_POSE["x"],
            y=INTERSECTION_RESET_POSE["y"],
            yaw=INTERSECTION_RESET_POSE["yaw"],
            traj=self.traj,
            end_distance=self.end["distance"],
            end_angle_deg=self.end["angle_deg"],
        )
        self.assertLess(abs(d), 0.06)
        self.assertAlmostEqual(phi, 0.0, places=1)
        self.assertAlmostEqual(curv, 0.0, places=2)
        self.assertFalse(done)

    def test_mid_path_not_done(self) -> None:
        """Mid-intersection pose should not trigger completion."""
        # Index ~60 corresponds to y≈0 on the straight reference path.
        d, phi, curv, done = compute_lane_pose(
            x=0.0,
            y=0.0,
            yaw=math.pi / 2,
            traj=self.traj,
            end_distance=self.end["distance"],
            end_angle_deg=self.end["angle_deg"],
        )
        _ = global_path_index(0.0, 0.0, self.traj)
        self.assertAlmostEqual(d, 0.0, places=2)
        self.assertAlmostEqual(phi, 0.0, places=2)
        self.assertFalse(done)


if __name__ == "__main__":
    unittest.main()
