"""Lane-tracking debug overlay for the project sim web UI."""

from __future__ import annotations

import cv2
import numpy as np


def _draw_lane_overlay(cam: np.ndarray, debug_info: dict, scale_x: float, scale_y: float) -> None:
    """Draw slice lines, detected lane points, lane center, and robot target on the camera panel."""
    h_disp, w_disp = cam.shape[:2]

    for sy in debug_info.get("slice_ys", []):
        dy = int(sy * scale_y)
        cv2.line(cam, (0, dy), (w_disp, dy), (0, 255, 255), 1)

    yellow_xs = debug_info.get("yellow_xs", [])
    white_xs = debug_info.get("white_xs", [])
    slice_ys = debug_info.get("slice_ys", [])

    for i, x in enumerate(yellow_xs):
        if i < len(slice_ys):
            cv2.circle(cam, (int(x * scale_x), int(slice_ys[i] * scale_y)), 6, (0, 255, 255), -1)

    for i, x in enumerate(white_xs):
        if i < len(slice_ys):
            cv2.circle(cam, (int(x * scale_x), int(slice_ys[i] * scale_y)), 6, (255, 255, 255), -1)

    if yellow_xs and white_xs:
        y_mean = float(np.mean(yellow_xs))
        w_mean = float(np.mean(white_xs))
        center_x = (y_mean + w_mean) / 2.0
        if slice_ys:
            cy = int(float(np.mean(slice_ys)) * scale_y)
            cx = int(center_x * scale_x)
            cv2.line(cam, (cx, cy - 18), (cx, cy + 18), (0, 255, 0), 2)
            cv2.putText(cam, "lane center", (cx + 6, cy - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    target_x = w_disp // 2
    cv2.line(cam, (target_x, 0), (target_x, h_disp), (0, 0, 255), 1)
    cv2.putText(cam, "target", (target_x + 4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)


def create_lane_visualization(
    image_bgr: np.ndarray,
    debug_info: dict,
    pwm_left: float,
    pwm_right: float,
    motion: dict | None = None,
) -> np.ndarray:
    """Four-panel lane debug view + status strip."""
    display_w = 320
    h, w = image_bgr.shape[:2]
    display_h = int(h * display_w / w)
    scale_x = display_w / w
    scale_y = display_h / h

    cam = cv2.resize(image_bgr, (display_w, display_h))
    _draw_lane_overlay(cam, debug_info, scale_x, scale_y)

    lane_vis = cv2.resize(
        cv2.applyColorMap(debug_info.get("lane_mask", np.zeros((h, w), np.uint8)), cv2.COLORMAP_HOT),
        (display_w, display_h),
    )
    white_vis = cv2.resize(
        cv2.applyColorMap(debug_info.get("white_mask", np.zeros((h, w), np.uint8)), cv2.COLORMAP_BONE),
        (display_w, display_h),
    )
    ym = debug_info.get("yellow_mask", np.zeros((h, w), np.uint8))
    yellow_bgr = np.zeros((ym.shape[0], ym.shape[1], 3), dtype=np.uint8)
    yellow_bgr[:, :, 1] = ym
    yellow_bgr[:, :, 2] = ym
    yellow_vis = cv2.resize(yellow_bgr, (display_w, display_h))

    grid = np.vstack([
        np.hstack([cam, lane_vis]),
        np.hstack([white_vis, yellow_vis]),
    ])

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)
    cv2.putText(grid, "Camera + detections", (10, 20), font, 0.45, green, 1)
    cv2.putText(grid, "Both lines (mask)", (display_w + 10, 20), font, 0.45, green, 1)
    cv2.putText(grid, "White (right edge)", (10, display_h + 20), font, 0.45, green, 1)
    cv2.putText(grid, "Yellow (left edge)", (display_w + 10, display_h + 20), font, 0.45, green, 1)

    info = _info_strip(display_w * 2, debug_info, pwm_left, pwm_right, motion)
    return np.vstack([grid, info])


def create_virtual_lane_overlay(image_bgr: np.ndarray, motion: dict) -> np.ndarray:
    """Overlay for intersection virtual-lane tracking (reference path, not camera lines)."""
    display_w = 640
    h, w = image_bgr.shape[:2]
    display_h = int(h * display_w / w)
    cam = cv2.resize(image_bgr, (display_w, display_h))

    lines = [
        "PATH TRACKING (d, phi, curvature → lane PID)",
        f"trajectory: {motion.get('trajectory', '?')}",
        f"lateral error d: {motion.get('d', 0.0):.3f} m",
        f"heading error phi: {motion.get('phi', 0.0):.3f} rad",
        f"curvature: {motion.get('curvature', 0.0):.3f}",
        f"pose: ({motion.get('pose_x', 0.0):.2f}, {motion.get('pose_y', 0.0):.2f})",
        "",
        "d ≈ 0 and phi ≈ 0  →  on the reference path",
    ]
    y = 28
    for line in lines:
        cv2.putText(cam, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        y += 24

    bar_w = 400
    bar_x = 10
    bar_y = display_h - 50
    d = float(motion.get("d", 0.0))
    cv2.putText(cam, "d (lateral)", (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.rectangle(cam, (bar_x, bar_y), (bar_x + bar_w, bar_y + 16), (40, 40, 40), -1)
    cx = bar_x + bar_w // 2
    cv2.line(cam, (cx, bar_y), (cx, bar_y + 16), (120, 120, 120), 1)
    dp = int(np.clip(cx + d * 200, bar_x, bar_x + bar_w))
    cv2.circle(cam, (dp, bar_y + 8), 7, (0, 255, 255), -1)

    return cam


def _info_strip(width: int, debug_info: dict, pwm_left: float, pwm_right: float, motion: dict | None):
    h = 130
    canvas = np.zeros((h, width, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    bar_x, bar_w, bar_h = 110, 280, 18

    err = float(debug_info.get("lateral_error", 0.0))
    cv2.putText(canvas, "Lateral error:", (10, 22), font, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, "(0 = centered in lane)", (10, 38), font, 0.35, (180, 180, 180), 1)
    cv2.rectangle(canvas, (bar_x, 8), (bar_x + bar_w, 28), (50, 50, 50), -1)
    cx = bar_x + bar_w // 2
    cv2.line(canvas, (cx, 8), (cx, 28), (100, 100, 100), 1)
    ep = int(np.clip(cx + err * bar_w / 2, bar_x, bar_x + bar_w))
    ecol = (0, 255, 0) if abs(err) < 0.1 else (0, 255, 255) if abs(err) < 0.3 else (0, 0, 255)
    cv2.circle(canvas, (ep, 18), 8, ecol, -1)
    cv2.putText(canvas, f"{err:+.2f}", (bar_x + bar_w + 8, 22), font, 0.4, (200, 200, 200), 1)

    _draw_bar(canvas, "Left PWM:", bar_x, 48, bar_w, bar_h, pwm_left, font)
    _draw_bar(canvas, "Right PWM:", bar_x, 78, bar_w, bar_h, pwm_right, font)

    detected = debug_info.get("lane_detected", False)
    cv2.putText(
        canvas,
        "IN LANE" if detected else "NO LANE DETECTED",
        (10, 110),
        font,
        0.5,
        (0, 255, 0) if detected else (0, 0, 255),
        1,
    )
    cv2.putText(
        canvas,
        f"lane px: {debug_info.get('total_lane_pixels', 0)}",
        (200, 110),
        font,
        0.4,
        (200, 200, 200),
        1,
    )

    if motion:
        cv2.putText(
            canvas,
            f"FSM: {motion.get('state', '?')} | {motion.get('traffic_state', '?')}",
            (400, 110),
            font,
            0.4,
            (0, 255, 0),
            1,
        )

    return canvas


def _draw_bar(canvas, label, x0, y, bar_w, bar_h, value, font):
    cv2.putText(canvas, label, (10, y + 14), font, 0.45, (255, 255, 255), 1)
    cv2.rectangle(canvas, (x0, y), (x0 + bar_w, y + bar_h), (50, 50, 50), -1)
    fill = int(bar_w * np.clip(abs(value), 0, 1))
    color = (100, 100, 255) if value >= 0 else (255, 100, 100)
    cv2.rectangle(canvas, (x0, y), (x0 + fill, y + bar_h), color, -1)
    cv2.putText(canvas, f"{value:.2f}", (x0 + bar_w + 8, y + 14), font, 0.4, (200, 200, 200), 1)
