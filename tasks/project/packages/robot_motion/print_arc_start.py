"""Print arc-start and exit pose from open-loop sim."""
from __future__ import annotations

import math
from pathlib import Path

from tasks.project.packages.robot_motion.config import (
    END_CONDITIONS,
    INTERSECTION_RESET_POSE,
    LANE_CONTROLLER,
    OPEN_LOOP_YAW_DAMPING,
)
from tasks.project.packages.robot_motion.intersection_drive import DriveTickInput, drive_tick
from tasks.project.packages.robot_motion.lane_controller import LaneController
from tasks.project.packages.robot_motion.trajectory import load_all_trajectories

DT = 0.05
TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"


def run_turn(tname: str) -> None:
    trajs = load_all_trajectories(TRAJ_DIR)
    lc = LaneController(**LANE_CONTROLLER)
    reset = INTERSECTION_RESET_POSE
    traj = trajs[tname]
    ec = END_CONDITIONS[tname]
    x, y, yaw = reset["x"], reset["y"], reset["yaw"]
    pi = 0

    print(f"=== {tname.upper()} reset_y={reset['y']} ===")
    for tick in range(500):
        out = drive_tick(
            DriveTickInput(
                x=x,
                y=y,
                yaw=yaw,
                path_index=pi,
                trajectory_name=tname,
                traj=traj,
                end_distance=ec["distance"],
                end_angle_deg=ec["angle_deg"],
                use_forward_search=False,
                dt=DT,
            ),
            lc,
            yaw_damping_factor=OPEN_LOOP_YAW_DAMPING,
        )
        if abs(out.curv) > 0.1:
            print(f"ARC_START tick={tick} t={tick * DT:.2f}s")
            print(
                f"  pose x={out.x:.4f} y={out.y:.4f} yaw={out.yaw:.4f} "
                f"({math.degrees(out.yaw):.2f} deg)"
            )
            print(
                f"  d={out.d:.4f} phi={out.phi:.4f} ({math.degrees(out.phi):.2f} deg) "
                f"curv={out.curv:.4f} path_idx={out.path_index}"
            )
            print(f"  v_cmd={out.v_cmd:.3f} omega_cmd={out.omega_cmd:.3f}")
            break
        x, y, yaw, pi = out.x, out.y, out.yaw, out.path_index
    else:
        print("  (no arc in 500 ticks)")

    lc.reset()
    x, y, yaw, pi = reset["x"], reset["y"], reset["yaw"], 0
    for tick in range(500):
        out = drive_tick(
            DriveTickInput(
                x=x,
                y=y,
                yaw=yaw,
                path_index=pi,
                trajectory_name=tname,
                traj=traj,
                end_distance=ec["distance"],
                end_angle_deg=ec["angle_deg"],
                use_forward_search=False,
                dt=DT,
            ),
            lc,
            yaw_damping_factor=OPEN_LOOP_YAW_DAMPING,
        )
        if out.done:
            print(f"EXIT tick={tick} t={tick * DT:.2f}s")
            print(
                f"  x={out.x:.4f} y={out.y:.4f} yaw={math.degrees(out.yaw):.2f} deg "
                f"d={out.d:.4f} phi={math.degrees(out.phi):.2f} deg"
            )
            print(
                f"  yaw_near_pi={2.64 <= out.yaw <= math.pi + 0.1} "
                f"x_negative={out.x < 0} d_near_0={abs(out.d) < 0.05}"
            )
            break
        x, y, yaw, pi = out.x, out.y, out.yaw, out.path_index
    print()


if __name__ == "__main__":
    for name in ("left", "right"):
        run_turn(name)
