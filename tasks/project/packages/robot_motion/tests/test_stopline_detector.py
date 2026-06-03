"""Tests for debounced stop-line detection."""

from __future__ import annotations

import unittest

import numpy as np

from tasks.project.packages.robot_motion.stopline_detector import StoplineDetector


class TestStoplineDetector(unittest.TestCase):
    def test_requires_consecutive_frames_to_latch(self) -> None:
        det = StoplineDetector(confirm_frames=3)
        red = np.zeros((480, 640, 3), dtype=np.uint8)
        red[440:, :, 0] = 255

        self.assertFalse(det.update(red))
        self.assertFalse(det.update(red))
        self.assertTrue(det.update(red))
        self.assertFalse(det.update(red))

    def test_disarm_ignores_until_reset(self) -> None:
        det = StoplineDetector(confirm_frames=1)
        red = np.zeros((480, 640, 3), dtype=np.uint8)
        red[440:, :, 0] = 255
        self.assertTrue(det.update(red))
        det.disarm_until_reset()
        det.reset()
        self.assertTrue(det.update(red))

    def test_reset_clears_state(self) -> None:
        det = StoplineDetector(confirm_frames=1)
        red = np.zeros((480, 640, 3), dtype=np.uint8)
        red[440:, :, 0] = 255
        self.assertTrue(det.update(red))
        det.reset()
        self.assertFalse(det.latched)


if __name__ == "__main__":
    unittest.main()
