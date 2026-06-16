"""Cheap Duckiebot detection via blue-pixel thresholding.

Instead of training a second model, we exploit the fact that a Duckiebot's
chassis is a strong, saturated blue. When enough blue pixels appear inside the
region of interest directly ahead of the camera, we treat it as a robot in the
path and request a stop.

Compatible with Python 3.6+ (no __future__ annotations, no lowercase generics).
"""
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'object_detection_config.yaml'
))


class BlueRobotDetector:
    """Detect a robot ahead by counting blue pixels in a region of interest."""

    def __init__(self, config_path=None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.enabled = bool(cfg.get('robot_detection_enabled', True))

        lower = cfg.get('robot_blue_hsv_lower', [90, 80, 60])
        upper = cfg.get('robot_blue_hsv_upper', [130, 255, 255])
        self.hsv_lower = np.array(lower, dtype=np.uint8)
        self.hsv_upper = np.array(upper, dtype=np.uint8)

        # Area (px) of the LARGEST connected blue blob required to call it a
        # robot. Using the biggest blob (not total pixels) ignores scattered
        # background blue and keys on one solid object.
        self.area_threshold = int(cfg.get('robot_blue_area_threshold',
                                          cfg.get('robot_blue_pixel_threshold', 1500)))

        # Temporal stability: consecutive hits needed to trigger, then keep
        # stopping for latch_seconds after the last confirmed detection.
        self.confirm_frames = int(cfg.get('robot_confirm_frames', 2))
        if 'robot_latch_seconds' in cfg:
            self.latch_seconds = float(cfg['robot_latch_seconds'])
        elif 'robot_latch_frames' in cfg:
            self.latch_seconds = max(0.1, float(cfg['robot_latch_frames']) / 30.0)
        else:
            self.latch_seconds = 2.0

        # Region of interest as frame fractions.
        self.roi_top    = float(cfg.get('robot_roi_top',    0.30))
        self.roi_bottom = float(cfg.get('robot_roi_bottom', 1.00))
        self.roi_left   = float(cfg.get('robot_roi_left',   0.05))
        self.roi_right  = float(cfg.get('robot_roi_right',  0.95))

        self._hit_counter   = 0
        self._latched_until = 0.0

        # Diagnostics for visualization / tuning.
        self.last_blue_pixels = 0       # largest blob area in px
        self.last_bbox = None           # type: Optional[tuple]
        self.last_mask = None           # type: Optional[np.ndarray]
        self.last_roi  = (0, 0, 0, 0)   # x0, y0, x1, y1 in pixels

    def _roi_bounds(self, h, w):
        y0 = max(0, min(h, int(h * self.roi_top)))
        y1 = max(0, min(h, int(h * self.roi_bottom)))
        x0 = max(0, min(w, int(w * self.roi_left)))
        x1 = max(0, min(w, int(w * self.roi_right)))
        return x0, y0, x1, y1

    def detect(self, frame_bgr):
        # type: (np.ndarray) -> Tuple[bool, int]
        """Return (robot_present, largest_blue_blob_area) for a BGR frame."""
        self.last_blue_pixels = 0
        self.last_bbox = None
        self.last_mask = None

        if not self.enabled or frame_bgr is None or frame_bgr.size == 0:
            self._hit_counter = 0
            self._latched_until = 0.0
            return False, 0

        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = self._roi_bounds(h, w)
        self.last_roi = (x0, y0, x1, y1)

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return False, 0

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # Open removes speckle; close fills holes so one robot reads as one blob.
        open_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

        full = np.zeros((h, w), dtype=np.uint8)
        full[y0:y1, x0:x1] = mask
        self.last_mask = full

        # Largest connected blue blob (robust to scattered background blue).
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        largest_area = 0
        largest_cnt  = None
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area > largest_area:
                largest_area = area
                largest_cnt  = cnt

        self.last_blue_pixels = largest_area
        if largest_cnt is not None:
            bx, by, bw, bh = cv2.boundingRect(largest_cnt)
            self.last_bbox = (x0 + bx, y0 + by, x0 + bx + bw, y0 + by + bh)

        # Temporal confirm + time latch: stay stopped for latch_seconds after
        # the last confirmed detection (avoids flicker release/resume).
        now = time.monotonic()
        if largest_area >= self.area_threshold:
            self._hit_counter += 1
        else:
            self._hit_counter = max(0, self._hit_counter - 1)

        if self._hit_counter >= self.confirm_frames:
            self._latched_until = now + self.latch_seconds

        return now < self._latched_until, largest_area
