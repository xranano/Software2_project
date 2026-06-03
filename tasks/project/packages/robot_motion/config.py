"""Motion and lane-controller constants (proj-lfi default.yaml + commands.sh)."""

from __future__ import annotations

import math

END_CONDITIONS = {
    "straight": {"distance": 0.2, "angle_deg": 30},
    "left": {"distance": 0.3, "angle_deg": 15},
    "right": {"distance": 0.25, "angle_deg": 18},
}

# proj-lfi localization_node reset — 20 cm behind approach stopline centre.
INTERSECTION_RESET_POSE = {"x": 0.0, "y": -0.20, "yaw": math.pi / 2}

# Roll forward after wait so virtual pose reaches stop line (y=0) before arc tracking.
STOP_LINE_CROSS_M = 0.20

RESET_Y_BY_TRAJECTORY = {
    "straight": -0.20,
    "left": -0.20,
    "right": -0.20,
}

STOP_TIME_S = 2.0
CONTROL_HZ = 20
CONTROL_DT = 1.0 / CONTROL_HZ

# Open-loop odom during INTERSECTION_DRIVE — 1.0 = no yaw suppression.
YAW_DAMPING_FACTOR = 1.0

# Open-loop sim / diagnostics (no Godot fusion).
OPEN_LOOP_YAW_DAMPING = 1.0

# Pose fusion weights (optional Layer 1 — not wired in main_loop by default).
LOCALIZER_GODOT_WEIGHT = 0.65
LOCALIZER_STOPLINE_WEIGHT = 0.25
LOCALIZER_ODOM_WEIGHT = 0.10

USE_FORWARD_PATH_SEARCH = False
INTERSECTION_DRIVE_DEBUG_LOG = False

DEBUG_OPEN_LOOP_TWIST = False
DEBUG_OPEN_LOOP_SECONDS = 3.0
DEBUG_OPEN_LOOP_V = 0.23
DEBUG_OPEN_LOOP_OMEGA = 1.5

LANE_CONTROLLER = {
    "v_bar": 0.23,
    "k_d": -3.5,
    "k_theta": -1.0,
    "k_Id": 1.0,
    "wheel_base": 0.103,
    "omega_max": 4.0,
    "use_feedforward": True,
}

DEFAULT_TRAJECTORY = "straight"

# ROBOT_MOTION_DEBUG=1  ROBOT_SKIP_TO_DRIVE=1  ROBOT_USE_FORWARD_SEARCH=1  ROBOT_DIAGNOSE=1
