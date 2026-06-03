"""Main control loop — proj-lfi virtual-lane intersection FSM."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path

from tasks.project.packages.apriltag_detector import AprilTagDetector
from tasks.project.packages.sign_tracker import SignTracker
from tasks.project.packages.robot_motion.config import (
    CONTROL_HZ,
    DEBUG_OPEN_LOOP_OMEGA,
    DEBUG_OPEN_LOOP_SECONDS,
    DEBUG_OPEN_LOOP_TWIST,
    DEBUG_OPEN_LOOP_V,
    DEFAULT_TRAJECTORY,
    END_CONDITIONS,
    INTERSECTION_DRIVE_DEBUG_LOG,
    LANE_CONTROLLER,
    STOP_LINE_CROSS_M,
    USE_FORWARD_PATH_SEARCH,
    YAW_DAMPING_FACTOR,
)
from tasks.project.packages.robot_motion import hardware
from tasks.project.packages.robot_motion import status as motion_status
from tasks.project.packages.sign_constants import TrafficState
from tasks.project.packages.robot_motion.intersection_controller import (
    IntersectionController,
    MotionPhase,
)
from tasks.project.packages.robot_motion.intersection_drive import (
    CSV_HEADER,
    DriveTickInput,
    compute_drive_commands,
    format_drive_csv_row,
)
from tasks.project.packages.robot_motion.lane_controller import LaneController
from tasks.project.packages.robot_motion.odometry import apply_yaw_damping, integrate_pose
from tasks.project.packages.robot_motion.trajectory import drive_start, load_all_trajectories

logger = logging.getLogger(__name__)

_TRAJECTORIES_DIR = Path(__file__).resolve().parent / "trajectories"
_DRIVE_LOG_PATH = Path(__file__).resolve().parent / "drive_log.csv"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _traffic_state(controller: IntersectionController, tracker: SignTracker) -> str:
    if controller.phase == MotionPhase.LANE_FOLLOW:
        if tracker.last_topology_tag_id or tracker.last_regulatory_tag_id:
            return TrafficState.SIGN_DETECTED.value
        return TrafficState.DRIVING.value
    if controller.phase == MotionPhase.INTERSECTION_WAIT:
        return TrafficState.WAITING.value
    if controller.phase == MotionPhase.CROSS_STRAIGHT:
        return TrafficState.MANEUVERING.value
    if controller.phase == MotionPhase.INTERSECTION_DRIVE:
        return TrafficState.MANEUVERING.value
    return TrafficState.DRIVING.value


def _publish_status(controller: IntersectionController, tracker: SignTracker) -> None:
    plan = controller.plan
    motion_status.update(
        traffic_state=_traffic_state(controller, tracker),
        state=controller.phase,
        trajectory=controller.trajectory_name,
        regulatory_sign=(plan.regulatory.value if plan else tracker.regulatory.value),
        allowed_turns=(plan.allowed_turns if plan else tracker.allowed_turns),
        last_tag_id=tracker.last_topology_tag_id or tracker.last_regulatory_tag_id,
    )


def _init_drive_log(enabled: bool) -> None:
    if not enabled:
        return
    _DRIVE_LOG_PATH.write_text(CSV_HEADER + "\n", encoding="utf-8")
    logger.info("Writing drive log to %s", _DRIVE_LOG_PATH)


def _append_drive_log(
    enabled: bool,
    elapsed_s: float,
    trajectory: str,
    out,
    v_act: float,
    omega_act: float,
    pwm_left: float,
    pwm_right: float,
    godot_dist: float | None,
) -> None:
    if not enabled:
        return
    row = format_drive_csv_row(elapsed_s, trajectory, out)
    extra = f",{v_act:.4f},{omega_act:.4f},{pwm_left:.4f},{pwm_right:.4f},{godot_dist or ''}"
    with _DRIVE_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(row + extra + "\n")


def _log_drive_tick(
    *,
    elapsed_s: float,
    trajectory: str,
    out,
    v_act: float,
    omega_act: float,
    pwm_left: float,
    pwm_right: float,
    godot_dist: float | None,
    search_mode: str,
) -> None:
    godot_dead = abs(out.v_cmd) > 0.05 and abs(v_act) < 0.02 and abs(pwm_left) < 0.02
    line = (
        f"DRIVE {elapsed_s:5.2f}s | {trajectory:7s} | "
        f"pose=({out.x:+.3f},{out.y:+.3f},{math.degrees(out.yaw):+6.1f}°) | "
        f"idx={out.path_index:3d} | d={out.d:+.3f} phi={math.degrees(out.phi):+5.1f}° "
        f"curv={out.curv:+.2f} | "
        f"v_cmd={out.v_cmd:.3f} ω_cmd={out.omega_cmd:+.3f} | "
        f"v_act={v_act:.3f} ω_act={omega_act:+.3f} | "
        f"pwm=({pwm_left:+.3f},{pwm_right:+.3f}) | "
        f"godot_m={godot_dist if godot_dist is not None else 'n/a'} | "
        f"dist_end={out.dist_end:.3f} ang_end={out.ang_end_deg:.1f}° done={out.done} | {search_mode}"
    )
    if godot_dead:
        logger.warning("%s | GODOT_NOT_MOVING", line)
    else:
        logger.info(line)


def run(
    trajectories_dir: str | Path | None = None,
    trajectory_name: str = DEFAULT_TRAJECTORY,
) -> None:
    if _env_flag("ROBOT_DIAGNOSE"):
        from tasks.project.packages.robot_motion.diagnostics import main as run_diagnostics

        code = run_diagnostics([])
        if code != 0:
            logger.error("ROBOT_DIAGNOSE failed — see output above")

    dt = 1.0 / CONTROL_HZ
    traj_dir = Path(trajectories_dir) if trajectories_dir else _TRAJECTORIES_DIR
    trajs = load_all_trajectories(traj_dir)
    lane_controller = LaneController(**LANE_CONTROLLER)
    lane_controller.wheel_base = hardware.wheel_baseline()
    controller = IntersectionController(default_turn=trajectory_name)
    tracker = SignTracker(default_turn=trajectory_name)
    tag_detector = AprilTagDetector()

    use_forward_search = _env_flag("ROBOT_USE_FORWARD_SEARCH", USE_FORWARD_PATH_SEARCH)
    drive_debug_log = _env_flag("ROBOT_MOTION_DEBUG", INTERSECTION_DRIVE_DEBUG_LOG)
    debug_open_loop = _env_flag("ROBOT_DEBUG_OPEN_LOOP", DEBUG_OPEN_LOOP_TWIST)
    skip_to_drive = _env_flag("ROBOT_SKIP_TO_DRIVE")
    search_mode = "forward" if use_forward_search else "global"

    hardware.attach_controller(controller)
    motion_status.update(trajectory=trajectory_name, running=hardware.is_running())
    _init_drive_log(drive_debug_log)

    drive_elapsed = 0.0
    debug_open_loop_until = 0.0

    if skip_to_drive:
        controller.trajectory_name = trajectory_name
        sliced, sx, sy, syaw = drive_start(trajs[trajectory_name])
        controller.begin_drive(sliced, sx, sy, syaw)
        lane_controller.reset()
        logger.warning(
            "ROBOT_SKIP_TO_DRIVE: jumped to INTERSECTION_DRIVE (%s) — no stop line",
            trajectory_name,
        )

    logger.info(
        "Motion loop: turn=%s hz=%d search=%s wheel_base=%.3f v_max=%.3f "
        "debug=%s skip_to_drive=%s open_loop_debug=%s",
        trajectory_name,
        CONTROL_HZ,
        search_mode,
        lane_controller.wheel_base,
        hardware.wheel_v_max(),
        drive_debug_log,
        skip_to_drive,
        debug_open_loop,
    )
    logger.info("Hardware: %s", hardware.get_diagnostics_snapshot())

    while True:
        if hardware.should_stop():
            hardware.stop_motors()
            break

        if not hardware.is_running():
            hardware.stop_motors()
            time.sleep(dt)
            continue

        if controller.phase == MotionPhase.LANE_FOLLOW:
            frame_bgr = hardware.read_frame_bgr()
            if frame_bgr is not None:
                detections = tag_detector.detect(frame_bgr)
                tracker.observe(detections)
                controller.note_regulatory(tracker.regulatory)

            _publish_status(controller, tracker)

            frame_rgb = hardware.read_frame_rgb()
            if frame_rgb is not None and hardware.at_stop_line(frame_rgb):
                hardware.stop_motors()
                hardware.disarm_stopline()
                plan = tracker.freeze_plan(fallback_turn=controller.trajectory_name)
                controller.enter_wait(plan, preferred_turn=controller.trajectory_name)
                logger.info("FSM: stop line latched → INTERSECTION_WAIT")
                time.sleep(dt)
                continue

            hardware.lane_follow_step(dt, speed_factor=controller.yield_speed_factor())

        elif controller.phase == MotionPhase.INTERSECTION_WAIT:
            hardware.stop_motors()
            frame_bgr = hardware.read_frame_bgr()
            detections = tag_detector.detect(frame_bgr) if frame_bgr is not None else []

            _publish_status(controller, tracker)
            if controller.tick_wait(dt, detections):
                controller.begin_cross(STOP_LINE_CROSS_M)
                logger.info("FSM: wait complete → CROSS_STRAIGHT (%.2f m)", STOP_LINE_CROSS_M)
                time.sleep(dt)
                continue

        elif controller.phase == MotionPhase.CROSS_STRAIGHT:
            v_cross = LANE_CONTROLLER["v_bar"]
            hardware.set_twist(v_cross, 0.0)
            x, y, yaw = controller.pose.x, controller.pose.y, controller.pose.yaw
            x, y, yaw = integrate_pose(x, y, yaw, v_cross, 0.0, dt)
            controller.pose.x, controller.pose.y, controller.pose.yaw = x, y, yaw
            _publish_status(controller, tracker)
            if controller.tick_cross(dt, v_cross):
                sliced, sx, sy, syaw = drive_start(trajs[controller.trajectory_name])
                controller.begin_drive(sliced, sx, sy, syaw)
                lane_controller.reset()
                drive_elapsed = 0.0
                logger.info(
                    "FSM: crossed stop line → INTERSECTION_DRIVE at (%.2f, %.2f)",
                    sx,
                    sy,
                )
                if debug_open_loop:
                    debug_open_loop_until = DEBUG_OPEN_LOOP_SECONDS
                    logger.warning(
                        "DEBUG_OPEN_LOOP: set_twist(%.2f, %.2f) for %.1fs",
                        DEBUG_OPEN_LOOP_V,
                        DEBUG_OPEN_LOOP_OMEGA,
                        DEBUG_OPEN_LOOP_SECONDS,
                    )

        elif controller.phase == MotionPhase.INTERSECTION_DRIVE:
            traj = controller.drive_traj or trajs[controller.trajectory_name]
            ec = END_CONDITIONS[controller.trajectory_name]
            x, y, yaw = controller.pose.x, controller.pose.y, controller.pose.yaw
            drive_elapsed += dt

            if debug_open_loop_until > 0.0:
                v_cmd, omega_cmd = DEBUG_OPEN_LOOP_V, DEBUG_OPEN_LOOP_OMEGA
                hardware.set_twist(v_cmd, omega_cmd)
                prev_yaw = yaw
                x, y, yaw = integrate_pose(x, y, yaw, v_cmd, omega_cmd, dt)
                yaw = apply_yaw_damping(prev_yaw, yaw, omega_factor=YAW_DAMPING_FACTOR)
                controller.pose.x, controller.pose.y, controller.pose.yaw = x, y, yaw
                pwm_left, pwm_right = hardware.get_executed_pwm()
                v_act, omega_act = hardware.twist_from_executed_pwm()
                if drive_debug_log:
                    logger.info(
                        "OPEN_LOOP %.2fs pose=(%.3f,%.3f,%.1f°) "
                        "v_cmd=%.3f ω_cmd=%.3f v_act=%.3f ω_act=%.3f pwm=(%.3f,%.3f)",
                        drive_elapsed,
                        x,
                        y,
                        math.degrees(yaw),
                        v_cmd,
                        omega_cmd,
                        v_act,
                        omega_act,
                        pwm_left,
                        pwm_right,
                    )
                debug_open_loop_until -= dt
            else:
                out = compute_drive_commands(
                    DriveTickInput(
                        x=x,
                        y=y,
                        yaw=yaw,
                        path_index=controller.path_index,
                        trajectory_name=controller.trajectory_name,
                        traj=traj,
                        end_distance=ec["distance"],
                        end_angle_deg=ec["angle_deg"],
                        use_forward_search=use_forward_search,
                        dt=dt,
                    ),
                    lane_controller,
                )
                controller.path_index = out.path_index

                hardware.set_twist(out.v_cmd, out.omega_cmd)
                hardware.request_godot_state()
                pwm_left, pwm_right = hardware.get_executed_pwm()
                v_act, omega_act = hardware.twist_from_executed_pwm()
                godot_dist = hardware.godot_distance_traveled()

                prev_yaw = yaw
                x, y, yaw = integrate_pose(x, y, yaw, v_act, omega_act, dt)
                yaw = apply_yaw_damping(prev_yaw, yaw, omega_factor=YAW_DAMPING_FACTOR)
                controller.pose.x, controller.pose.y, controller.pose.yaw = x, y, yaw
                out.x, out.y, out.yaw = x, y, yaw
                out.pose_source = "odom"

                if drive_debug_log:
                    logger.info(
                        "pose x=%.3f y=%.3f yaw=%.3f d=%.3f phi=%.3f src=%s",
                        out.x,
                        out.y,
                        out.yaw,
                        out.d,
                        out.phi,
                        out.pose_source,
                    )
                    _log_drive_tick(
                        elapsed_s=drive_elapsed,
                        trajectory=controller.trajectory_name,
                        out=out,
                        v_act=v_act,
                        omega_act=omega_act,
                        pwm_left=pwm_left,
                        pwm_right=pwm_right,
                        godot_dist=godot_dist,
                        search_mode=search_mode,
                    )
                    _append_drive_log(
                        True,
                        drive_elapsed,
                        controller.trajectory_name,
                        out,
                        v_act,
                        omega_act,
                        pwm_left,
                        pwm_right,
                        godot_dist,
                    )

                motion_status.update(
                    pose_x=out.x,
                    pose_y=out.y,
                    pose_yaw=out.yaw,
                    d=out.d,
                    phi=out.phi,
                    curvature=out.curv,
                    state=controller.phase,
                    trajectory=controller.trajectory_name,
                )

                if out.done:
                    hardware.stop_motors()
                    yaw_ok = 2.64 <= out.yaw <= math.pi + 0.1
                    logger.info(
                        "EXIT_POSE %s: x=%.3f y=%.3f yaw=%.3f (%.1f°) d=%.3f phi=%.3f | "
                        "yaw_near_pi=%s x_negative=%s d_near_0=%s",
                        controller.trajectory_name,
                        out.x,
                        out.y,
                        out.yaw,
                        math.degrees(out.yaw),
                        out.d,
                        out.phi,
                        yaw_ok,
                        out.x < 0.0,
                        abs(out.d) < 0.05,
                    )
                    controller.finish_drive()
                    tracker.clear()
                    hardware.reset_stopline()
                    snap = motion_status.snapshot()
                    motion_status.update(
                        intersection_done_count=snap["intersection_done_count"] + 1,
                    )
                    logger.info(
                        "Intersection complete (%s) after %.2fs",
                        controller.trajectory_name,
                        drive_elapsed,
                    )

            _publish_status(controller, tracker)

        time.sleep(dt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
