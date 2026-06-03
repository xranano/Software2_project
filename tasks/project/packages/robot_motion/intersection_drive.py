"""Pure intersection path-tracking step — testable without Godot or camera."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tasks.project.packages.robot_motion.lane_controller import LaneController
from tasks.project.packages.robot_motion.odometry import apply_yaw_damping, integrate_pose
from tasks.project.packages.robot_motion.virtual_lane import (
    compute_lane_pose,
    compute_lane_pose_forward,
    end_pose_metrics,
    global_path_index,
)


@dataclass
class DriveTickInput:
    x: float
    y: float
    yaw: float
    path_index: int
    trajectory_name: str
    traj: dict
    end_distance: float
    end_angle_deg: float
    use_forward_search: bool
    dt: float


@dataclass
class DriveTickOutput:
    x: float
    y: float
    yaw: float
    path_index: int
    d: float
    phi: float
    curv: float
    done: bool
    dist_end: float
    ang_end_deg: float
    v_cmd: float
    omega_cmd: float
    v_left: float
    v_right: float
    pose_source: str = ""


def resolve_lane_pose(
    x: float,
    y: float,
    yaw: float,
    traj: dict,
    end_distance: float,
    end_angle_deg: float,
    path_index: int,
    use_forward_search: bool,
) -> tuple[float, float, float, bool, int]:
    if use_forward_search:
        return compute_lane_pose_forward(
            x, y, yaw, traj, end_distance, end_angle_deg, path_index=path_index
        )
    d, phi, curv, done = compute_lane_pose(x, y, yaw, traj, end_distance, end_angle_deg)
    return d, phi, curv, done, global_path_index(x, y, traj)


def compute_drive_commands(
    state: DriveTickInput,
    lane_controller: LaneController,
) -> DriveTickOutput:
    """Virtual lane + lane PID (Layer 2–3). Pose is integrated open-loop by the caller."""
    d, phi, curv, _done_hint, path_index = resolve_lane_pose(
        state.x,
        state.y,
        state.yaw,
        state.traj,
        state.end_distance,
        state.end_angle_deg,
        state.path_index,
        state.use_forward_search,
    )
    dist_end, ang_end_deg, done = end_pose_metrics(
        state.x,
        state.y,
        state.yaw,
        state.traj,
        state.end_distance,
        state.end_angle_deg,
    )
    v_left, v_right, v_cmd, omega_cmd = lane_controller.compute(d, phi, curv, state.dt)
    return DriveTickOutput(
        x=state.x,
        y=state.y,
        yaw=state.yaw,
        path_index=path_index,
        d=d,
        phi=phi,
        curv=curv,
        done=done,
        dist_end=dist_end,
        ang_end_deg=ang_end_deg,
        v_cmd=v_cmd,
        omega_cmd=omega_cmd,
        v_left=v_left,
        v_right=v_right,
    )


def drive_tick(
    state: DriveTickInput,
    lane_controller: LaneController,
    *,
    yaw_damping_factor: float = 1.0,
) -> DriveTickOutput:
    """Open-loop sim helper: commanded odom + damping (no Godot / stopline fusion)."""
    out = compute_drive_commands(state, lane_controller)
    prev_yaw = state.yaw
    x, y, yaw = integrate_pose(state.x, state.y, state.yaw, out.v_cmd, out.omega_cmd, state.dt)
    yaw = apply_yaw_damping(prev_yaw, yaw, omega_factor=yaw_damping_factor)
    out.x, out.y, out.yaw = x, y, yaw
    return out


def simulate_maneuver(
    trajectory_name: str,
    traj: dict,
    end_distance: float,
    end_angle_deg: float,
    *,
    reset_x: float = 0.0,
    reset_y: float = -0.20,
    reset_yaw: float = math.pi / 2,
    use_forward_search: bool = False,
    dt: float = 0.05,
    yaw_damping_factor: float = 1.0,
    lane_controller: LaneController | None = None,
    max_ticks: int = 800,
) -> list[DriveTickOutput]:
    """Open-loop sim — returns per-tick log for diagnostics."""
    from tasks.project.packages.robot_motion.config import LANE_CONTROLLER

    lc = lane_controller or LaneController(**LANE_CONTROLLER)
    lc.reset()
    x, y, yaw = reset_x, reset_y, reset_yaw
    path_index = 0
    history: list[DriveTickOutput] = []

    for _ in range(max_ticks):
        out = drive_tick(
            DriveTickInput(
                x=x,
                y=y,
                yaw=yaw,
                path_index=path_index,
                trajectory_name=trajectory_name,
                traj=traj,
                end_distance=end_distance,
                end_angle_deg=end_angle_deg,
                use_forward_search=use_forward_search,
                dt=dt,
            ),
            lc,
            yaw_damping_factor=yaw_damping_factor,
        )
        history.append(out)
        x, y, yaw, path_index = out.x, out.y, out.yaw, out.path_index
        if out.done:
            break
    return history


def format_drive_csv_row(t_s: float, trajectory: str, out: DriveTickOutput) -> str:
    """CSV row for sharing with another agent."""
    return (
        f"{t_s:.2f},{trajectory},"
        f"{out.x:.4f},{out.y:.4f},{math.degrees(out.yaw):.2f},"
        f"{out.path_index},{out.d:.4f},{math.degrees(out.phi):.2f},{out.curv:.4f},"
        f"{out.v_cmd:.4f},{out.omega_cmd:.4f},"
        f"{out.dist_end:.4f},{out.ang_end_deg:.2f},{out.done}"
    )


CSV_HEADER = (
    "t_s,trajectory,pose_x,pose_y,yaw_deg,path_idx,d,phi_deg,curv,"
    "v_cmd,omega_cmd,dist_end,ang_end_deg,done"
)
