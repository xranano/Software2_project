"""Project agent — sign tracking and non-blocking intersection motion."""

from __future__ import annotations

import logging
import os

from tasks.project.packages.lane_follow import patch_lane_servoing_detector
from tasks.project.packages.robot_motion import hardware
from tasks.project.packages.robot_motion import status as motion_status
from tasks.project.packages.robot_motion.config import DEFAULT_TRAJECTORY
from tasks.project.packages.robot_motion.main_loop import run

logger = logging.getLogger(__name__)


def main(camera, wheels, leds, stop_event) -> None:
    """
    Entry point called by the project server thread.

    Args:
        camera: Camera driver instance.
        wheels: Wheel driver instance.
        leds: LED driver instance (optional, unused).
        stop_event: ``threading.Event`` set on shutdown.
    """
    trajectory = os.environ.get("ROBOT_TRAJECTORY", DEFAULT_TRAJECTORY)
    if trajectory not in ("straight", "left", "right"):
        logger.warning("Invalid ROBOT_TRAJECTORY=%r, using %s", trajectory, DEFAULT_TRAJECTORY)
        trajectory = DEFAULT_TRAJECTORY

    patch_lane_servoing_detector()
    hardware.init(wheels=wheels, camera=camera, stop_event=stop_event)
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting project agent (fallback_trajectory=%s)", trajectory)

    try:
        run(trajectory_name=trajectory)
    finally:
        hardware.stop_motors()
        motion_status.update(running=False)
