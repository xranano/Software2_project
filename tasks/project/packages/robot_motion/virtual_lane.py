"""Virtual lane planner — proj-lfi virtual_lane_node equivalent."""

from __future__ import annotations

import math

import numpy as np


def _wrap_phi(phi: float) -> float:
    """Wrap heading error to (-π, π] to avoid 2π spikes in the controller."""
    return (phi + math.pi) % (2 * math.pi) - math.pi


def global_path_index(x: float, y: float, traj: dict) -> int:
    """proj-lfi closest_track_point — global argmin over path."""
    car = np.array([x, y])
    diffs = traj["track"][:-1] - car
    return int(np.argmin(np.linalg.norm(diffs, axis=1)))


def end_pose_metrics(
    x: float, y: float, yaw: float, traj: dict, end_distance: float, end_angle_deg: float
) -> tuple[float, float, bool]:
    """Return (dist_end_m, ang_end_deg, done) for debug logging."""
    track = traj["track"]
    tangent = traj["tangent"]
    car = np.array([x, y])
    dist_end = float(np.linalg.norm(track[-1] - car))
    ang_end = abs(_wrap_phi(yaw - float(tangent[-1])))
    done = dist_end < end_distance and ang_end < np.deg2rad(end_angle_deg)
    return dist_end, math.degrees(ang_end), done


def compute_lane_pose(
    x: float,
    y: float,
    yaw: float,
    traj: dict,
    end_distance: float,
    end_angle_deg: float,
) -> tuple[float, float, float, bool]:
    """
    Closest-point path tracking (proj-lfi ``virtual_lane_node.relative_pose``).

    Returns:
        d: lateral offset (m). Positive = left of path.
        phi: heading error (rad).
        curvature: path curvature at closest point (1/m).
        done: True when near path end AND aligned with exit heading.
    """
    track = traj["track"]
    tangent = traj["tangent"]
    curvature = traj["curvature"]
    car = np.array([x, y])

    diffs = track[:-1] - car
    dists = np.linalg.norm(diffs, axis=1)
    idx = int(np.argmin(dists))
    min_dist = float(dists[idx])

    dist_end = float(np.linalg.norm(track[-1] - car))
    ang_end = abs(_wrap_phi(yaw - float(tangent[-1])))
    done = dist_end < end_distance and ang_end < np.deg2rad(end_angle_deg)

    p1, p2 = track[idx], track[idx + 1]
    cross = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
    side = -1.0 if cross > 0 else 1.0

    d = -min_dist * side
    phi = _wrap_phi(yaw - float(tangent[idx]))
    curv = float(curvature[idx])
    return d, phi, curv, done


def compute_lane_pose_forward(
    x: float,
    y: float,
    yaw: float,
    traj: dict,
    end_distance: float,
    end_angle_deg: float,
    path_index: int = 0,
    search_window: int = 45,
) -> tuple[float, float, float, bool, int]:
    """
    Forward-only variant — prevents shortcutting onto a later curve on the path.

    Returns ``(d, phi, curv, done, path_index)``.
    """
    track = traj["track"]
    tangent = traj["tangent"]
    curvature = traj["curvature"]
    car = np.array([x, y])
    n = len(track)

    hi = min(path_index + search_window, n - 1)
    lo = min(path_index, n - 2)
    segment = track[lo:hi + 1]
    diffs = segment[:-1] - car
    dists = np.linalg.norm(diffs, axis=1)
    local_idx = int(np.argmin(dists))
    idx = max(lo + local_idx, path_index)
    min_dist = float(dists[local_idx])

    dist_end = float(np.linalg.norm(track[-1] - car))
    ang_end = abs(_wrap_phi(yaw - float(tangent[-1])))
    done = dist_end < end_distance and ang_end < np.deg2rad(end_angle_deg)

    p1, p2 = track[idx], track[min(idx + 1, n - 1)]
    cross = (p2[0] - p1[0]) * (y - p1[1]) - (p2[1] - p1[1]) * (x - p1[0])
    side = -1.0 if cross > 0 else 1.0

    d = -min_dist * side
    phi = _wrap_phi(yaw - float(tangent[idx]))
    curv = float(curvature[idx])
    return d, phi, curv, done, idx
