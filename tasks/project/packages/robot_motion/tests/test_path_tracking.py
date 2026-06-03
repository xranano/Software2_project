"""Integration tests for path tracking + lane controller."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from tasks.project.packages.robot_motion.config import (
    END_CONDITIONS,
    INTERSECTION_RESET_POSE,
    LANE_CONTROLLER,
    OPEN_LOOP_YAW_DAMPING,
)
from tasks.project.packages.robot_motion.lane_controller import LaneController
from tasks.project.packages.robot_motion.odometry import apply_yaw_damping, integrate_pose
from tasks.project.packages.robot_motion.trajectory import load_all_trajectories
from tasks.project.packages.robot_motion.virtual_lane import (
    compute_lane_pose,
    compute_lane_pose_forward,
    global_path_index,
)

_TRAJECTORIES = Path(__file__).resolve().parent.parent / "trajectories"
_DT = 0.05


class TestPathTracking(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.trajs = load_all_trajectories(_TRAJECTORIES)
        cls.lc = LaneController(**LANE_CONTROLLER)

    def test_straight_stays_near_zero_curvature(self) -> None:
        traj = self.trajs["straight"]
        ec = END_CONDITIONS["straight"]
        x, y, yaw = (
            INTERSECTION_RESET_POSE["x"],
            INTERSECTION_RESET_POSE["y"],
            INTERSECTION_RESET_POSE["yaw"],
        )
        for _ in range(20):
            d, phi, curv, done = compute_lane_pose(
                x, y, yaw, traj, ec["distance"], ec["angle_deg"]
            )
            path_index = global_path_index(x, y, traj)
            self.assertAlmostEqual(curv, 0.0, places=1)
            vl, vr, v, omega = self.lc.compute(d, phi, curv, _DT)
            x, y, yaw = integrate_pose(x, y, yaw, v, omega, _DT)
            self.assertFalse(done)

    def _simulate_turn(self, name: str) -> tuple[float, float, float, bool]:
        traj = self.trajs[name]
        ec = END_CONDITIONS[name]
        x, y, yaw = (
            INTERSECTION_RESET_POSE["x"],
            INTERSECTION_RESET_POSE["y"],
            INTERSECTION_RESET_POSE["yaw"],
        )
        path_index = 0
        self.lc.reset()
        done = False
        for _ in range(800):
            d, phi, curv, done = compute_lane_pose(
                x, y, yaw, traj, ec["distance"], ec["angle_deg"]
            )
            path_index = global_path_index(x, y, traj)
            _vl, _vr, v, omega = self.lc.compute(d, phi, curv, _DT)
            prev_yaw = yaw
            x, y, yaw = integrate_pose(x, y, yaw, v, omega, _DT)
            yaw = apply_yaw_damping(prev_yaw, yaw, omega_factor=OPEN_LOOP_YAW_DAMPING)
            if done:
                break
        return x, y, yaw, done

    def test_left_exit_yaw_near_pi(self) -> None:
        x, y, yaw, done = self._simulate_turn("left")
        self.assertTrue(done)
        self.assertGreater(yaw, 2.4)
        self.assertAlmostEqual(yaw, math.pi, delta=0.55)

    def test_right_exit_yaw_near_zero(self) -> None:
        x, y, yaw, done = self._simulate_turn("right")
        self.assertTrue(done)
        self.assertLess(abs(yaw), 0.85)

    def test_spec_compute_lane_pose_matches_forward_at_origin(self) -> None:
        traj = self.trajs["straight"]
        ec = END_CONDITIONS["straight"]
        d1, phi1, curv1, done1 = compute_lane_pose(
            INTERSECTION_RESET_POSE["x"],
            INTERSECTION_RESET_POSE["y"],
            INTERSECTION_RESET_POSE["yaw"],
            traj,
            ec["distance"],
            ec["angle_deg"],
        )
        d2, phi2, curv2, done2, _ = compute_lane_pose_forward(
            INTERSECTION_RESET_POSE["x"],
            INTERSECTION_RESET_POSE["y"],
            INTERSECTION_RESET_POSE["yaw"],
            traj,
            ec["distance"],
            ec["angle_deg"],
        )
        self.assertAlmostEqual(d1, d2, places=3)
        self.assertAlmostEqual(phi1, phi2, places=3)
        self.assertAlmostEqual(curv1, curv2, places=3)
        self.assertEqual(done1, done2)


if __name__ == "__main__":
    unittest.main()
