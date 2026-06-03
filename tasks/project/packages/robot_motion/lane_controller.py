"""Differential-drive lane tracker (proj-lfi / dt-core lane_controller_node)."""

from __future__ import annotations


class LaneController:
    """Differential-drive lane tracker. Equivalent to Duckietown lane_controller_node."""

    def __init__(
        self,
        v_bar: float = 0.23,
        k_d: float = -3.5,
        k_theta: float = -1.0,
        k_Id: float = 1.0,
        wheel_base: float = 0.103,
        omega_max: float = 4.0,
        use_feedforward: bool = True,
    ):
        self.v_bar = v_bar
        self.k_d = k_d
        self.k_theta = k_theta
        self.k_Id = k_Id
        self.wheel_base = wheel_base
        self.omega_max = omega_max
        self.use_feedforward = use_feedforward
        self.integral_d = 0.0

    def reset(self) -> None:
        """Clear lateral-error integral at intersection entry."""
        self.integral_d = 0.0

    def compute(
        self, d: float, phi: float, curvature: float, dt: float
    ) -> tuple[float, float, float, float]:
        """
        Compute wheel speeds from lane errors.

        Returns:
            (v_left, v_right, v, omega) in m/s and rad/s.
        """
        self.integral_d += d * dt
        omega = self.k_d * d + self.k_theta * phi + self.k_Id * self.integral_d
        if self.use_feedforward:
            omega += self.v_bar * curvature
        omega = max(-self.omega_max, min(self.omega_max, omega))
        v = self.v_bar
        half = self.wheel_base / 2.0
        v_left = v - omega * half
        v_right = v + omega * half
        return v_left, v_right, v, omega
