"""Tests for m/s → normalized PWM conversion."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from tasks.project.packages.robot_motion import hardware


@dataclass
class _FakeWheels:
    v_max: float = 1.0
    radius: float = 0.0318

    def set_wheels_speed(self, left: float, right: float) -> None:
        self.last_left = left
        self.last_right = right


class TestWheelSpeedConversion(unittest.TestCase):
    def setUp(self) -> None:
        hardware.init(wheels=_FakeWheels(), camera=None, stop_event=None)

    def test_v_bar_maps_to_reasonable_pwm(self) -> None:
        hardware.set_wheel_speeds(0.23, 0.23)
        wheels = hardware._ctx.wheels
        self.assertAlmostEqual(wheels.last_left, 0.23, places=3)
        self.assertAlmostEqual(wheels.last_right, 0.23, places=3)

    def test_differential_turn(self) -> None:
        hardware.set_wheel_speeds(0.10, 0.40)
        wheels = hardware._ctx.wheels
        self.assertAlmostEqual(wheels.last_left, 0.10, places=3)
        self.assertAlmostEqual(wheels.last_right, 0.40, places=3)


if __name__ == "__main__":
    unittest.main()
