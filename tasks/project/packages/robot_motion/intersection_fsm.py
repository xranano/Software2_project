"""proj-lfi intersection FSM — see ``intersection_controller`` for implementation."""

from __future__ import annotations

from tasks.project.packages.robot_motion.intersection_controller import (
    IntersectionController,
    MotionPhase,
    Pose,
)

# Backward-compatible aliases used by hardware and tests.
IntersectionFSM = IntersectionController
State = MotionPhase

LANE_FOLLOW = MotionPhase.LANE_FOLLOW
INTERSECTION_WAIT = MotionPhase.INTERSECTION_WAIT
INTERSECTION_DRIVE = MotionPhase.INTERSECTION_DRIVE

__all__ = [
    "IntersectionController",
    "IntersectionFSM",
    "MotionPhase",
    "State",
    "Pose",
    "LANE_FOLLOW",
    "INTERSECTION_WAIT",
    "INTERSECTION_DRIVE",
]
