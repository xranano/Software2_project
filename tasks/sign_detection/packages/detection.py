"""
detection.py
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional

from tasks.sign_detection.packages.sign_behavior_config import State
from tasks.sign_detection.packages.duck_detector import DuckDetector

_duck_detector = DuckDetector()

IMG_WIDTH = 640
IMG_HEIGHT = 480

CLASS_NAMES = {
    0: "duckie",
    1: "truck",
    2: "sign",
}

CLASS_COLORS = {
    0: (0, 215, 255),
    1: (180, 100, 220),
    2: (50, 205, 50),
}

Detection = Tuple[Tuple[int, int, int, int], float, int]

DUCK_STOP_Y2_RATIO = 0.70
DUCK_CONFIRM_FRAMES = 2
TRUCK_CONFIRM_FRAMES = 3
STOP_LATCH_FRAMES = 2

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0


def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)
    return cleaned


def _mask_to_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h, area))
    return boxes


def _duck_close_enough(bbox, score, state, white_x=None):
    if score < 0.45:
        return False
    x1, y1, x2, y2 = bbox
    area = (x2 - x1) * (y2 - y1)
    y2_ratio = y2 / IMG_HEIGHT
    if state != State.MOVING:
        return area > 10000
    return y2_ratio > DUCK_STOP_Y2_RATIO


def detect_obstacles(frame_rgb: np.ndarray) -> List[Detection]:
    global IMG_WIDTH, IMG_HEIGHT

    frame_h, frame_w = frame_rgb.shape[:2]
    IMG_WIDTH = frame_w
    IMG_HEIGHT = frame_h

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)
    detections: List[Detection] = []

    for bbox, score, cls_id in _duck_detector.detect(frame_rgb):
        detections.append((bbox, score, 0))

    blue_lower = np.array([95, 90, 60], dtype=np.uint8)
    blue_upper = np.array([125, 255, 220], dtype=np.uint8)

    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_mask[:int(frame_h * 0.22), :] = 0
    blue_mask = _clean_mask(blue_mask, open_size=5, close_size=9)

    for x1, y1, x2, y2, contour_area in _mask_to_bboxes(blue_mask):
        box_w = x2 - x1
        box_h = y2 - y1
        bbox_area = box_w * box_h

        if bbox_area < 4500:
            continue
        aspect = box_w / float(box_h + 1e-6)
        if aspect < 0.6 or aspect > 3.2:
            continue

        cy = (y1 + y2) / 2.0
        if cy < frame_h * 0.35:
            continue

        fill_ratio = contour_area / float(bbox_area + 1e-6)
        if fill_ratio < 0.35:
            continue

        score = min(1.0, bbox_area / 18000.0)
        detections.append(((x1, y1, x2, y2), score, 1))

    return detections


vehicle_min_bbox_area = 800


def vehicle_detected(detections):
    if detections is None:
        detections = []

    for (x1, y1, x2, y2), score, cls_id in detections:
        if cls_id != 1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < vehicle_min_bbox_area:
            continue
        centre_x = (x1 + x2) / 2.0
        offset = abs(centre_x - IMG_WIDTH / 2) / IMG_WIDTH
        print(f"[SignBehavior] vehicle detected (area={area:.0f}, offset={offset:.3f})")
        return True, offset

    return False, 0.0


def should_stop(
    detections: List[Detection],
    state: Optional[State] = None,
    white_x=[]
) -> Tuple[bool, str]:
    global _duck_counter, _truck_counter, _stop_latch

    if detections is None:
        detections = []

    if state == State.CHECKPATH or state == "CHECKPATH":
        return False, ""

    duck_threat = False
    truck_threat = False
    reason = ""

    for bbox, score, cls_id in detections:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        height = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if cls_id == 0:
            if _duck_close_enough(bbox, score, state, white_x):
                duck_threat = True
                reason = f"duckie close enough score={score:.2f} bbox={bbox}"
                print(f"[ObstacleStop] DUCKIE candidate close (area={area:.0f})")

        if cls_id == 1:
            if score < 0.45:
                continue
            if area < 18000:
                continue
            if height < IMG_HEIGHT * 0.12:
                continue
            if cy < IMG_HEIGHT * 0.42:
                continue
            if cx < 0.18 * IMG_WIDTH or cx > 0.82 * IMG_WIDTH:
                continue

            truck_threat = True
            reason = f"truck close area={area:.0f} h={height:.0f} score={score:.2f}"
            print(f"[ObstacleStop] TRUCK STOP confirmed candidate area={area:.0f}, h={height:.0f}")

    if duck_threat:
        _duck_counter += 1
    else:
        _duck_counter = max(0, _duck_counter - 1)

    if truck_threat:
        _truck_counter += 1
    else:
        _truck_counter = max(0, _truck_counter - 1)

    confirmed_duck = _duck_counter >= DUCK_CONFIRM_FRAMES
    confirmed_truck = _truck_counter >= TRUCK_CONFIRM_FRAMES

    if confirmed_duck or confirmed_truck:
        _stop_latch = STOP_LATCH_FRAMES
    else:
        _stop_latch = max(0, _stop_latch - 1)

    if _stop_latch > 0:
        return True, reason or "stop latch active"

    return False, ""


def reset_detection_state():
    global _duck_counter, _truck_counter, _stop_latch

    _duck_counter = 0
    _truck_counter = 0
    _stop_latch = 0
    print("[Detection] state reset")
