"""Debounced red stop-line detection in the camera ROI."""

from __future__ import annotations

import cv2
import numpy as np

# Bottom strip only — latch when the stop line is under the wheels, not on the horizon.
ROI_Y_START = 0.90
RED_THRESHOLD = 0.12
CONFIRM_FRAMES = 8


def red_fraction(frame_rgb: np.ndarray) -> float:
    h = frame_rgb.shape[0]
    roi = frame_rgb[int(h * ROI_Y_START):, :]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 80, 80])
    upper2 = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
    return float(np.count_nonzero(mask)) / mask.size


def red_visible(frame_rgb: np.ndarray) -> bool:
    return red_fraction(frame_rgb) >= RED_THRESHOLD


class StoplineDetector:
    """Fire once when the robot reaches the stop line."""

    def __init__(self, confirm_frames: int = CONFIRM_FRAMES):
        self.confirm_frames = confirm_frames
        self._consecutive = 0
        self._latched = False
        self._armed = True

    def update(self, frame_rgb: np.ndarray) -> bool:
        """True exactly once when the stop line is reached."""
        if not self._armed or self._latched:
            return False
        if red_visible(frame_rgb):
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= self.confirm_frames:
            self._latched = True
            return True
        return False

    def reset(self) -> None:
        self._consecutive = 0
        self._latched = False
        self._armed = True

    def disarm_until_reset(self) -> None:
        """Ignore further stop-line hits until ``reset()`` (mid-intersection)."""
        self._armed = False

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def latched(self) -> bool:
        return self._latched

    @property
    def last_red_fraction(self) -> float:
        return getattr(self, "_last_fraction", 0.0)

    def sample(self, frame_rgb: np.ndarray) -> float:
        self._last_fraction = red_fraction(frame_rgb)
        return self._last_fraction
