"""Open-loop pose integration (proj-lfi velocity_to_pose fallback)."""

from __future__ import annotations

import numpy as np


def integrate_pose(
    x: float, y: float, yaw: float, v: float, omega: float, dt: float
) -> tuple[float, float, float]:
    """Integrate differential-drive motion over ``dt`` seconds."""
    x += v * np.cos(yaw) * dt
    y += v * np.sin(yaw) * dt
    yaw += omega * dt
    return x, y, yaw


def apply_yaw_damping(
    prev_yaw: float, new_yaw: float, omega_factor: float = 0.2
) -> float:
    """Reduce over-estimated yaw rate from open-loop integration (proj-lfi damping)."""
    prev = ((prev_yaw + np.pi / 2) % (2 * np.pi)) - np.pi / 2
    new = ((new_yaw + np.pi / 2) % (2 * np.pi)) - np.pi / 2
    dtheta = new - prev
    return prev + omega_factor * dtheta
