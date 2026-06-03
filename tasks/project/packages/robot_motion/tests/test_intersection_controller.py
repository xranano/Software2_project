"""Tests for proj-lfi intersection FSM."""

from __future__ import annotations

import unittest

from tasks.project.packages.sign_constants import RegulatorySign
from tasks.project.packages.sign_tracker import IntersectionPlan
from tasks.project.packages.robot_motion.config import STOP_LINE_CROSS_M, STOP_TIME_S
from tasks.project.packages.robot_motion.intersection_controller import (
    IntersectionController,
    MotionPhase,
)
from tasks.project.packages.robot_motion.trajectory import drive_start, load_all_trajectories


def _plan(regulatory: RegulatorySign = RegulatorySign.NONE) -> IntersectionPlan:
    return IntersectionPlan(
        allowed_turns=("left",),
        regulatory=regulatory,
        topology_tag_id=None,
        regulatory_tag_id=None,
    )


class TestIntersectionController(unittest.TestCase):
    def test_stop_waits_two_seconds(self) -> None:
        ctrl = IntersectionController(stop_time_s=STOP_TIME_S)
        ctrl.enter_wait(_plan())
        self.assertFalse(ctrl.tick_wait(1.0, []))
        self.assertFalse(ctrl.tick_wait(0.5, []))
        self.assertTrue(ctrl.tick_wait(0.5, []))

    def test_begin_drive_uses_drive_start_pose(self) -> None:
        trajs = load_all_trajectories(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "trajectories"
        )
        sliced, sx, sy, syaw = drive_start(trajs["left"])
        ctrl = IntersectionController()
        ctrl.begin_drive(sliced, sx, sy, syaw)
        self.assertEqual(ctrl.phase, MotionPhase.INTERSECTION_DRIVE)
        self.assertAlmostEqual(ctrl.pose.x, sx)
        self.assertAlmostEqual(ctrl.pose.y, sy)
        self.assertAlmostEqual(ctrl.pose.yaw, syaw)
        self.assertIs(ctrl.drive_traj, sliced)

    def test_cross_straight_then_drive(self) -> None:
        trajs = load_all_trajectories(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "trajectories"
        )
        ctrl = IntersectionController()
        ctrl.begin_cross(STOP_LINE_CROSS_M)
        self.assertEqual(ctrl.phase, MotionPhase.CROSS_STRAIGHT)
        while not ctrl.tick_cross(0.05, 0.23):
            pass
        sliced, sx, sy, syaw = drive_start(trajs["left"])
        ctrl.begin_drive(sliced, sx, sy, syaw)
        self.assertEqual(ctrl.phase, MotionPhase.INTERSECTION_DRIVE)
        self.assertAlmostEqual(ctrl.pose.y, 0.0, places=2)

    def test_enter_wait_keeps_preferred_turn(self) -> None:
        ctrl = IntersectionController(default_turn="left")
        ctrl.trajectory_name = "left"
        plan = IntersectionPlan(
            allowed_turns=("left", "right"),
            regulatory=RegulatorySign.NONE,
            topology_tag_id=8,
            regulatory_tag_id=None,
        )
        ctrl.enter_wait(plan, preferred_turn="left")
        self.assertEqual(ctrl.trajectory_name, "left")

    def test_finish_drive_returns_to_lane_follow(self) -> None:
        trajs = load_all_trajectories(
            __import__("pathlib").Path(__file__).resolve().parent.parent / "trajectories"
        )
        sliced, sx, sy, syaw = drive_start(trajs["straight"])
        ctrl = IntersectionController()
        ctrl.begin_drive(sliced, sx, sy, syaw)
        ctrl.finish_drive()
        self.assertEqual(ctrl.phase, MotionPhase.LANE_FOLLOW)


if __name__ == "__main__":
    unittest.main()
