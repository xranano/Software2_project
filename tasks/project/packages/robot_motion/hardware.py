"""Hardware hooks — motors, lane follow, and stopline detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from tasks.project.packages.robot_motion.config import DEFAULT_TRAJECTORY
from tasks.project.packages.robot_motion.intersection_controller import (
    IntersectionController,
    MotionPhase,
)
from tasks.project.packages.robot_motion import status as motion_status
from tasks.project.packages.robot_motion.stopline_detector import StoplineDetector

if TYPE_CHECKING:
    from duckiebot.wheel_driver.wheels_driver_abs import WheelsDriverAbs

logger = logging.getLogger(__name__)


@dataclass
class HardwareContext:
    wheels: Optional["WheelsDriverAbs"] = None
    camera: Optional[object] = None
    stop_event: Optional[object] = None
    lane_agent: Optional[object] = None
    stopline: StoplineDetector = field(default_factory=StoplineDetector)


_ctx = HardwareContext()
_controller: Optional[IntersectionController] = None
_running = True
_last_lane_debug: dict = {}
_last_pwm: tuple[float, float] = (0.0, 0.0)


def init(
    wheels: "WheelsDriverAbs",
    camera: object,
    stop_event: object,
    lane_agent: Optional[object] = None,
) -> None:
    global _ctx
    _ctx = HardwareContext(
        wheels=wheels,
        camera=camera,
        stop_event=stop_event,
        lane_agent=lane_agent,
    )


def attach_controller(controller: IntersectionController) -> None:
    global _controller
    _controller = controller
    motion_status.update(state=controller.phase, trajectory=controller.trajectory_name)


attach_fsm = attach_controller


def set_running(running: bool) -> None:
    global _running
    _running = running
    motion_status.update(running=running)
    if not running:
        stop_motors()


def is_running() -> bool:
    return _running


def set_trajectory(name: str) -> None:
    if _controller is not None:
        _controller.set_trajectory(name)
    motion_status.update(trajectory=name)


def reset_motion(trajectory: str = DEFAULT_TRAJECTORY) -> None:
    _ctx.stopline.reset()
    if _controller is not None:
        _controller.trajectory_name = trajectory
        _controller.finish_drive()
    motion_status.update(
        state=MotionPhase.LANE_FOLLOW,
        trajectory=trajectory,
        d=0.0,
        phi=0.0,
        curvature=0.0,
    )
    stop_motors()


def get_ui_data() -> dict:
    data = motion_status.snapshot()
    if _controller is not None:
        data["state"] = _controller.phase
        data["trajectory"] = _controller.trajectory_name
        data["wait_remaining"] = _controller.wait_remaining
        data["pose_x"] = _controller.pose.x
        data["pose_y"] = _controller.pose.y
        data["pose_yaw"] = _controller.pose.yaw
    return data


def get_lane_debug() -> tuple[dict, float, float]:
    """Latest lane-detection debug from the control loop (for web overlay)."""
    return _last_lane_debug, _last_pwm[0], _last_pwm[1]


def get_lane_agent():
    return _get_lane_agent()


def analyze_lane_frame(frame_rgb: np.ndarray) -> tuple[dict, float, float]:
    """
    Run lane detection on a frame for display only (does not drive motors).

    Used by the web UI when the agent thread has not refreshed debug recently.
    """
    agent = _get_lane_agent()
    left, right = agent.compute_commands(frame_rgb)
    return agent.last_debug_info, left, right


def read_frame_rgb() -> Optional[np.ndarray]:
    if _ctx.camera is None:
        return None
    if hasattr(_ctx.camera, "read_rgb"):
        ok, frame = _ctx.camera.read_rgb()
    else:
        ok, frame = _ctx.camera.read()
        if ok and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if not ok or frame is None:
        return None
    return frame


def read_frame_bgr() -> Optional[np.ndarray]:
    rgb = read_frame_rgb()
    if rgb is None:
        return None
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _get_lane_agent():
    if _ctx.lane_agent is None:
        from tasks.project.packages.lane_follow import patch_lane_servoing_detector
        from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

        patch_lane_servoing_detector()
        _ctx.lane_agent = LaneServoingAgent()
    return _ctx.lane_agent


def _ms_to_normalized(v_left: float, v_right: float) -> tuple[float, float]:
    """Convert linear wheel speeds (m/s) to normalized PWM [-1, 1]."""
    wheels = _ctx.wheels
    if wheels is None:
        return 0.0, 0.0
    v_max = getattr(wheels, "v_max", 0.5)
    if v_max <= 0:
        return 0.0, 0.0
    return (
        max(-1.0, min(1.0, v_left / v_max)),
        max(-1.0, min(1.0, v_right / v_max)),
    )


def set_wheel_speeds(v_left: float, v_right: float) -> None:
    if _ctx.wheels is None:
        return
    left, right = _ms_to_normalized(v_left, v_right)
    _ctx.wheels.set_wheels_speed(left, right)


def set_twist(v: float, omega: float) -> None:
    """
    Apply (v, omega) from LaneController.

    Prefer wheels.set_velocity — same scaling as Godot sim / Duckiebot driver.
    Fallback: convert to per-wheel m/s then divide by v_max (not v_max/radius).
    """
    wheels = _ctx.wheels
    if wheels is None:
        return
    if hasattr(wheels, "set_velocity"):
        wheels.set_velocity(v, omega)
        return
    baseline = getattr(wheels, "baseline", 0.103)
    half = baseline / 2.0
    set_wheel_speeds(v - omega * half, v + omega * half)


def get_executed_pwm() -> tuple[float, float]:
    """Last normalized wheel PWM [-1, 1] after set_twist / set_wheels_speed."""
    wheels = _ctx.wheels
    if wheels is None:
        return _last_pwm
    left = float(getattr(wheels, "left_pwm", _last_pwm[0]))
    right = float(getattr(wheels, "right_pwm", _last_pwm[1]))
    return left, right


def wheel_baseline() -> float:
    wheels = _ctx.wheels
    if wheels is None:
        return 0.103
    return float(getattr(wheels, "baseline", 0.103))


def wheel_v_max() -> float:
    wheels = _ctx.wheels
    if wheels is None:
        return 0.5
    return float(getattr(wheels, "v_max", 0.5))


def godot_distance_traveled() -> float | None:
    """Godot sim odometer (m) — None if unavailable. Compare to virtual pose y for sync."""
    wheels = _ctx.wheels
    if wheels is None:
        return None
    gs = getattr(wheels, "game_state", None)
    if gs is None:
        return None
    return float(getattr(gs, "distance_traveled", 0.0))


def request_godot_state() -> None:
    """Poll Godot for pose + odom (no-op on real hardware)."""
    wheels = _ctx.wheels
    if wheels is None:
        return
    if hasattr(wheels, "request_state"):
        wheels.request_state()
    elif hasattr(wheels, "transport") and hasattr(wheels.transport, "send_get_state"):
        wheels.transport.send_get_state()


def godot_pose() -> tuple[float | None, float | None, float | None]:
    """Godot world (x, z, yaw) — None if unavailable."""
    wheels = _ctx.wheels
    if wheels is None:
        return None, None, None
    gs = getattr(wheels, "game_state", None)
    if gs is None:
        return None, None, None
    return (
        getattr(gs, "position_x", None),
        getattr(gs, "position_z", None),
        getattr(gs, "yaw", None),
    )


def get_diagnostics_snapshot() -> dict:
    """Hardware snapshot for troubleshooting /status or logs."""
    left, right = get_executed_pwm()
    v_act, omega_act = twist_from_executed_pwm()
    return {
        "running": _running,
        "phase": _controller.phase if _controller else None,
        "trajectory": _controller.trajectory_name if _controller else None,
        "wheel_baseline": wheel_baseline(),
        "wheel_v_max": wheel_v_max(),
        "pwm_left": left,
        "pwm_right": right,
        "v_act": v_act,
        "omega_act": omega_act,
        "godot_distance_m": godot_distance_traveled(),
        "godot_pose": godot_pose(),
        "stopline_armed": _ctx.stopline.armed,
        "stopline_latched": _ctx.stopline.latched,
        "camera_ok": _ctx.camera is not None,
        "wheels_ok": _ctx.wheels is not None,
    }


def twist_from_executed_pwm() -> tuple[float, float]:
    """Return (v, omega) from the last executed normalized wheel PWM."""
    wheels = _ctx.wheels
    if wheels is None:
        return 0.0, 0.0
    left = float(getattr(wheels, "left_pwm", 0.0))
    right = float(getattr(wheels, "right_pwm", 0.0))
    v_max = getattr(wheels, "v_max", 0.5)
    baseline = getattr(wheels, "baseline", 0.103)
    v = (left + right) * v_max / 2.0
    omega = (right - left) * v_max / baseline if baseline > 0 else 0.0
    return v, omega


def stop_motors() -> None:
    set_twist(0.0, 0.0)


def should_stop() -> bool:
    return _ctx.stop_event is not None and _ctx.stop_event.is_set()


def lane_follow_step(dt: float, speed_factor: float = 1.0) -> None:
    global _last_lane_debug, _last_pwm
    if should_stop():
        stop_motors()
        return
    if _ctx.camera is None or _ctx.wheels is None:
        return
    frame = read_frame_rgb()
    if frame is None:
        stop_motors()
        return
    agent = _get_lane_agent()
    left, right = agent.compute_commands(frame)
    _last_lane_debug = dict(agent.last_debug_info)
    _last_pwm = (left * speed_factor, right * speed_factor)
    _ctx.wheels.set_wheels_speed(left * speed_factor, right * speed_factor)


def update_stopline(frame_rgb: np.ndarray) -> bool:
    return _ctx.stopline.update(frame_rgb)


def at_stop_line(frame_rgb: np.ndarray) -> bool:
    """Spec hook: True once when the robot reaches the red stop line."""
    return update_stopline(frame_rgb)


def disarm_stopline() -> None:
    _ctx.stopline.disarm_until_reset()


def reset_stopline() -> None:
    _ctx.stopline.reset()
