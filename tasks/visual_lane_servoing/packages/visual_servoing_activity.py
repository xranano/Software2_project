from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')

_DEFAULT_YELLOW_LOWER = (20, 80, 100)
_DEFAULT_YELLOW_UPPER = (35, 255, 255)
_DEFAULT_WHITE_LOWER = (0, 0, 180)
_DEFAULT_WHITE_UPPER = (179, 60, 255)

try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}


def _bound(key: str, default: int, channel: int) -> int:
    value = _h.get(key, default)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _hsv_array(prefix: str, default_lower: tuple[int, int, int], default_upper: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([
        _bound(f"{prefix}_lower_h", default_lower[0], 0),
        _bound(f"{prefix}_lower_s", default_lower[1], 1),
        _bound(f"{prefix}_lower_v", default_lower[2], 2),
    ])
    upper = np.array([
        _bound(f"{prefix}_upper_h", default_upper[0], 0),
        _bound(f"{prefix}_upper_s", default_upper[1], 1),
        _bound(f"{prefix}_upper_v", default_upper[2], 2),
    ])
    return lower, upper


_yellow_lower, _yellow_upper = _hsv_array("yellow", _DEFAULT_YELLOW_LOWER, _DEFAULT_YELLOW_UPPER)
_white_lower, _white_upper = _hsv_array("white", _DEFAULT_WHITE_LOWER, _DEFAULT_WHITE_UPPER)


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Detect yellow (left/dashed) and white (right) lane markings in a BGR image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_yellow = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    mask_white = cv2.inRange(hsv, _white_lower, _white_upper)

    kernel = np.ones((3, 3), np.uint8)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    mask_white = cv2.morphologyEx(mask_white, cv2.MORPH_OPEN, kernel)

    return mask_yellow.astype(np.float32) / 255.0, mask_white.astype(np.float32) / 255.0


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower = np.array(white_lower)
    _white_upper = np.array(white_upper)


def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]),    'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]),    'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]),    'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]), 'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]), 'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]), 'white_upper_v':  int(_white_upper[2]),
    }
