"""HSV red stop-line detection in the bottom camera strip."""
from typing import Optional, Tuple

import cv2
import numpy as np

from tasks.visual_lane_servoing.packages.sign_behavior_config import SignBehaviorConfig


def detect_red_line(
    fsm_context,
    frame_rgb: np.ndarray,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Return (red_line_detected, full-frame mask for debug overlay).
    """
    cfg = getattr(fsm_context, "config", SignBehaviorConfig())
    ignore_remaining = getattr(fsm_context, "_red_ignore_frames", 0)
    if ignore_remaining > 0:
        fsm_context._red_ignore_frames = ignore_remaining - 1
        return False, None

    if getattr(fsm_context, "_red_line_locked", False):
        return False, None

    if frame_rgb is None or frame_rgb.size == 0:
        return False, None

    h, w = frame_rgb.shape[:2]
    y0 = int(h * (1.0 - cfg.red_strip_frac))
    x0 = int(w * cfg.red_roi_left)
    x1 = int(w * cfg.red_roi_right)
    if y0 >= h or x0 >= x1:
        return False, None

    roi = frame_rgb[y0:h, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 70], dtype=np.uint8),
                    np.array([12, 255, 255], dtype=np.uint8)),
        cv2.inRange(hsv, np.array([168, 100, 70], dtype=np.uint8),
                    np.array([179, 255, 255], dtype=np.uint8)),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)))

    strip_h, strip_w = mask.shape[:2]
    red_px = int(np.count_nonzero(mask))
    red_ratio = red_px / float(max(strip_h * strip_w, 1))

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[y0:h, x0:x1] = mask

    if red_ratio < cfg.red_pixel_frac:
        return False, full_mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_width = strip_w * cfg.red_min_width_frac
    close_y = strip_h * cfg.red_line_close_y2_ratio

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < cfg.red_min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < min_width or bh <= 0:
            continue
        aspect = float(bw) / float(bh)
        if aspect < 1.6:
            continue
        if (y + bh) < close_y:
            continue
        return True, full_mask

    # Approaching a remembered sign: enough red in the strip is enough to trigger.
    if getattr(fsm_context, "_saved_tag", None) is not None:
        return True, full_mask

    return False, full_mask
