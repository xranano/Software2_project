"""proj-lfi intersection FSM — stop, wait, reset pose, virtual-lane path tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tasks.project.packages.sign_constants import RegulatorySign
from tasks.project.packages.sign_tracker import IntersectionPlan, crossing_traffic_visible
from tasks.project.packages.apriltag_detector import TagDetection
from tasks.project.packages.robot_motion.config import (
    INTERSECTION_RESET_POSE,
    RESET_Y_BY_TRAJECTORY,
    STOP_TIME_S,
)

logger = logging.getLogger(__name__)

STOP_RIGHT_OF_WAY_MAX_S = 8.0
YIELD_MIN_WAIT_S = 0.75


@dataclass
class Pose:
    x: float
    y: float
    yaw: float


class MotionPhase:
    LANE_FOLLOW = "LANE_FOLLOW"
    INTERSECTION_WAIT = "INTERSECTION_WAIT"
    CROSS_STRAIGHT = "CROSS_STRAIGHT"
    INTERSECTION_DRIVE = "INTERSECTION_DRIVE"


class IntersectionController:
    """
    proj-lfi FSM (proj_lfi.yaml):
      LANE_FOLLOW → at_stop_line → INTERSECTION_WAIT (2 s)
      → CROSS_STRAIGHT (ω=0, roll onto stop line)
      → INTERSECTION_DRIVE (arc slice from drive_start)
      → done → LANE_FOLLOW
    """

    def __init__(self, default_turn: str = "straight", stop_time_s: float = STOP_TIME_S):
        self.default_turn = default_turn
        self.stop_time_s = stop_time_s
        self.phase = MotionPhase.LANE_FOLLOW
        self.trajectory_name = default_turn
        self.pose = Pose(
            x=INTERSECTION_RESET_POSE["x"],
            y=INTERSECTION_RESET_POSE["y"],
            yaw=INTERSECTION_RESET_POSE["yaw"],
        )
        self.plan: IntersectionPlan | None = None
        self.wait_elapsed = 0.0
        self.path_index = 0
        self.cross_remaining_m = 0.0
        self.drive_traj: dict | None = None
        self._regulatory_hint = RegulatorySign.NONE

    def enter_wait(self, plan: IntersectionPlan, preferred_turn: str | None = None) -> None:
        self.plan = plan
        if preferred_turn and preferred_turn in plan.allowed_turns:
            self.trajectory_name = preferred_turn
        elif self.trajectory_name in plan.allowed_turns:
            pass
        else:
            self.trajectory_name = plan.choose_turn()
        self.phase = MotionPhase.INTERSECTION_WAIT
        self.wait_elapsed = 0.0
        logger.info(
            "FSM → INTERSECTION_WAIT (%.1fs) trajectory=%s allowed=%s",
            self.stop_time_s,
            self.trajectory_name,
            plan.allowed_turns,
        )

    def tick_wait(self, dt: float, detections: list[TagDetection]) -> bool:
        if self.plan is None:
            return False

        self.wait_elapsed += dt
        if self.wait_elapsed < self.stop_time_s:
            return False

        plan = self.plan
        crossing = crossing_traffic_visible(detections)
        if plan.regulatory == RegulatorySign.STOP and crossing:
            if self.wait_elapsed < STOP_RIGHT_OF_WAY_MAX_S:
                return False
        if plan.regulatory == RegulatorySign.YIELD:
            if self.wait_elapsed < YIELD_MIN_WAIT_S:
                return False
            if crossing:
                return False
        return True

    def begin_cross(self, distance_m: float) -> None:
        """Drive straight across the stop line before path tracking."""
        self.phase = MotionPhase.CROSS_STRAIGHT
        self.cross_remaining_m = max(0.0, distance_m)
        self.drive_traj = None
        self.path_index = 0
        reset_y = RESET_Y_BY_TRAJECTORY.get(
            self.trajectory_name, INTERSECTION_RESET_POSE["y"]
        )
        self.pose = Pose(
            x=INTERSECTION_RESET_POSE["x"],
            y=reset_y,
            yaw=INTERSECTION_RESET_POSE["yaw"],
        )
        logger.info(
            "FSM → CROSS_STRAIGHT trajectory=%s distance=%.2f m",
            self.trajectory_name,
            self.cross_remaining_m,
        )

    def tick_cross(self, dt: float, speed_mps: float) -> bool:
        """Advance straight creep; True when the stop line has been crossed."""
        if speed_mps > 0.0:
            self.cross_remaining_m = max(0.0, self.cross_remaining_m - speed_mps * dt)
        return self.cross_remaining_m <= 0.0

    def begin_drive(self, drive_traj: dict, x: float, y: float, yaw: float) -> None:
        """Start virtual-lane tracking on the arc slice (after CROSS_STRAIGHT)."""
        self.phase = MotionPhase.INTERSECTION_DRIVE
        self.path_index = 0
        self.drive_traj = drive_traj
        self.pose = Pose(x, y, yaw)
        logger.info(
            "FSM → INTERSECTION_DRIVE trajectory=%s pose=(%.2f, %.2f, %.2f rad)",
            self.trajectory_name,
            self.pose.x,
            self.pose.y,
            self.pose.yaw,
        )

    def finish_drive(self) -> None:
        logger.info("FSM → LANE_FOLLOW (maneuver complete)")
        self.phase = MotionPhase.LANE_FOLLOW
        self.plan = None
        self.wait_elapsed = 0.0
        self.path_index = 0
        self.drive_traj = None
        self.cross_remaining_m = 0.0

    def set_trajectory(self, name: str) -> None:
        if name not in ("straight", "left", "right"):
            raise ValueError(f"Unknown trajectory: {name!r}")
        self.trajectory_name = name

    def yield_speed_factor(self) -> float:
        if self.phase != MotionPhase.LANE_FOLLOW:
            return 1.0
        return 0.55 if self._regulatory_hint == RegulatorySign.YIELD else 1.0

    def note_regulatory(self, regulatory: RegulatorySign) -> None:
        self._regulatory_hint = regulatory

    @property
    def wait_remaining(self) -> float:
        return max(0.0, self.stop_time_s - self.wait_elapsed)
