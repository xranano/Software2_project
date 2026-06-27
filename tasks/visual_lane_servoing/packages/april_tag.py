"""AprilTag 36h11 detection and multi-frame confirmation (OpenCV only)."""
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Lower 36 bits of known tag36h11 codes → tag ID.
_CODES_36H11 = {
    0xD97F18B49: 1,
    0xEBCBCA822: 4,
    0x265AD0472: 9,
    0x34FE91B86: 10,
    0x3FF962CD5: 11,
    0x81DA494AF: 20,
    0xA2CABC89C: 24,
    0xADC58D9EB: 25,
    0xB16E7DFB0: 26,
    0x9F53856B5: 39,
}

_MIN_QUAD_AREA = 300


def _swap_tag_10_11(tag_id: int, enabled: bool) -> int:
    if not enabled:
        return tag_id
    if tag_id == 10:
        return 11
    if tag_id == 11:
        return 10
    return tag_id


def _order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def _decode_bits_from_warp(warp: np.ndarray) -> Optional[int]:
    _, binary = cv2.threshold(warp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bits = (binary > 127).astype(np.uint8)
    cell = 80 // 8
    grid = np.zeros((8, 8), dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            grid[r, c] = 1 if bits[y0:y1, x0:x1].mean() > 0.5 else 0

    border = np.concatenate([
        grid[0, :], grid[-1, :], grid[1:-1, 0], grid[1:-1, -1],
    ])
    if border.mean() > 0.2:
        return None

    inner = grid[1:7, 1:7]
    code = 0
    for bit in inner.flatten():
        code = (code << 1) | int(bit)
    return code


def _decode_quad(gray: np.ndarray, quad: np.ndarray) -> Optional[int]:
    ordered = _order_corners(quad)
    dst = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(ordered, dst)
    warp = cv2.warpPerspective(gray, M, (80, 80))

    for rot in range(4):
        code = _decode_bits_from_warp(warp)
        if code is not None and code in _CODES_36H11:
            return _CODES_36H11[code]
        warp = cv2.rotate(warp, cv2.ROTATE_90_CLOCKWISE)
    return None


def _detect_opencv_aruco(gray: np.ndarray) -> List[dict]:
    if not hasattr(cv2, "aruco"):
        return []

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    if ids is None or len(corners) == 0:
        return []

    tags = []
    for marker_corners, tag_id in zip(corners, ids.flatten()):
        pts = marker_corners.reshape(4, 2).astype(np.float32)
        tags.append({"tag_id": int(tag_id), "corners": pts})
    return tags


def _detect_fallback(gray: np.ndarray) -> List[dict]:
    h, w = gray.shape[:2]
    frame_area = float(h * w)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5,
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    tags = []
    seen_ids = set()
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_QUAD_AREA or area > frame_area * 0.5:
            continue
        approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        tag_id = _decode_quad(gray, quad)
        if tag_id is None or tag_id in seen_ids:
            continue
        seen_ids.add(tag_id)
        tags.append({"tag_id": tag_id, "corners": _order_corners(quad)})
    return tags


def detect_tags(fsm_context, frame_rgb: np.ndarray) -> List[dict]:
    """Detect AprilTag 36h11 markers in an RGB frame."""
    if frame_rgb is None or frame_rgb.size == 0:
        return []

    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    tags = _detect_opencv_aruco(gray)
    if not tags:
        tags = _detect_fallback(gray)

    swap = getattr(getattr(fsm_context, "config", None), "tag_10_11_swap", False)
    min_tag_area = getattr(getattr(fsm_context, "config", None), "tag_min_area", 2000.0)
    
    filtered_tags = []
    for tag in tags:
        tag["tag_id"] = _swap_tag_10_11(int(tag["tag_id"]), swap)
        
        # Calculate bounding box area
        corners = tag["corners"]
        area = float(abs(cv2.contourArea(corners)))
        tag["area"] = area
        
        # Filter by minimum area (only detect close/large tags)
        if area >= min_tag_area:
            filtered_tags.append(tag)
        else:
            print(f"[AprilTag] Ignored tag {tag['tag_id']} (too small: {area:.0f}px² < {min_tag_area:.0f}px²)")
    
    return filtered_tags


def confirm_tags(fsm_context, raw_tags: List[dict]) -> List[int]:
    """Return tag IDs seen in >= tag_confirm_frames consecutive frames."""
    if not hasattr(fsm_context, "_tag_buffer"):
        fsm_context._tag_buffer = {}  # type: Dict[int, int]

    confirm_n = getattr(
        getattr(fsm_context, "config", None), "tag_confirm_frames", 2,
    )
    seen = {int(t["tag_id"]) for t in raw_tags}

    for tag_id in list(fsm_context._tag_buffer.keys()):
        if tag_id in seen:
            fsm_context._tag_buffer[tag_id] += 1
        else:
            del fsm_context._tag_buffer[tag_id]

    for tag_id in seen:
        if tag_id not in fsm_context._tag_buffer:
            fsm_context._tag_buffer[tag_id] = 1

    return [tid for tid, count in fsm_context._tag_buffer.items() if count >= confirm_n]
