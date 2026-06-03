"""Load proj-lfi virtual-lane trajectory YAML files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


def load_trajectory(yaml_path: str | Path, name: str) -> dict:
    """
    Load proj-lfi trajectory YAML.

    Args:
        yaml_path: Path to straight.yaml, left.yaml, or right1.yaml.
        name: Key prefix — ``straight``, ``left``, or ``right``
              (``right1.yaml`` uses ``right`` keys).
    """
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    track = np.vstack([
        np.array(data[f"xCoords_{name}"]),
        np.array(data[f"yCoords_{name}"]),
    ]).T
    return {
        "track": track,
        "tangent": np.array(data[f"tangentAngle_{name}"]),
        "curvature": np.array(data[f"curvature_{name}"]),
    }


def load_all_trajectories(trajectories_dir: str | Path) -> dict:
    """Load straight, left, and right trajectories from a directory."""
    base = Path(trajectories_dir)
    return {
        "straight": load_trajectory(base / "straight.yaml", "straight"),
        "left": load_trajectory(base / "left.yaml", "left"),
        "right": load_trajectory(base / "right1.yaml", "right"),
    }


def turn_start_index(traj: dict) -> int:
    """Index of the first curved point (straight segment ends here)."""
    for i, curv in enumerate(traj["curvature"]):
        if abs(float(curv)) > 1e-9:
            return i
    ys = traj["track"][:, 1]
    return int(np.argmin(np.abs(ys)))


def drive_start(traj: dict) -> tuple[dict, float, float, float]:
    """
    Trajectory slice and pose for path tracking after crossing the stop line.

    Returns ``(sliced_traj, x, y, yaw)`` at the intersection entry point.
    """
    idx = turn_start_index(traj)
    pt = traj["track"][idx]
    sliced = slice_trajectory_from(traj, idx)
    return sliced, float(pt[0]), float(pt[1]), float(traj["tangent"][idx])


def has_turn_section(traj: dict) -> bool:
    return any(abs(float(c)) > 1e-9 for c in traj["curvature"])


def straight_distance_from_reset(traj: dict, reset_y: float) -> float:
    """Meters of straight driving before the arc begins."""
    idx = turn_start_index(traj)
    turn_y = float(traj["track"][idx, 1])
    return max(0.0, turn_y - reset_y)


def slice_trajectory_from(traj: dict, start_idx: int) -> dict:
    """Return the turn arc (and exit straight) starting at ``start_idx``."""
    return {
        "track": traj["track"][start_idx:].copy(),
        "tangent": traj["tangent"][start_idx:].copy(),
        "curvature": traj["curvature"][start_idx:].copy(),
    }
