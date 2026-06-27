"""
april_tag.py
"""

import cv2
import numpy as np

tag_confirm_frames = 2

# Lower 36 bits of each AprilTag 36h11 code (from tag36h11.c in the apriltag library).
# Only the IDs used by this project are listed.
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


def _order_corners(pts):
    """Sort 4 points into TL, TR, BR, BL order."""
    pts = pts.astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ])


def _decode_warped(bw80):
    row_centres = np.arange(8) * 10 + 5
    col_centres = np.arange(8) * 10 + 5
    bits = bw80[np.ix_(row_centres, col_centres)]

    border = np.concatenate([
        bits[0, :], bits[7, :],
        bits[1:7, 0], bits[1:7, 7]
    ])
    if np.any(border != 0):
        return None

    inner = bits[1:7, 1:7]

    for k in range(4):
        rotated = np.rot90(inner, k=k)
        code = int(rotated.ravel().dot(1 << np.arange(35, -1, -1, dtype=np.int64)))
        if code in _CODES_36H11:
            return _CODES_36H11[code]

    return None


_DST_PTS = np.array([[0, 0], [79, 0], [79, 79], [0, 79]], dtype=np.float32)


def _raw_detect(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        21, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    max_area = h * w * 0.5

    tags = []
    seen = set()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300 or area > max_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue

        src = _order_corners(approx.reshape(4, 2))
        M = cv2.getPerspectiveTransform(src, _DST_PTS)
        warped = cv2.warpPerspective(gray, M, (80, 80))

        _, bw = cv2.threshold(warped, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        tag_id = _decode_warped(bw)
        if tag_id is not None and tag_id not in seen:
            seen.add(tag_id)
            if tag_id == 10:
                tag_id = 11
            elif tag_id == 11:
                tag_id = 10

            tags.append({"tag_id": tag_id, "corners": src})

    return tags


def detect_tags(signBehavior, frame_rgb):
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

    try:
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            return []

        tags = []
        for c, tid in zip(corners, ids.flatten()):
            tid = int(tid)
            if tid == 10:
                tid = 11
            elif tid == 11:
                tid = 10
            tags.append({"tag_id": int(tid), "corners": c.reshape(4, 2)})
        return tags

    except AttributeError:
        pass

    return _raw_detect(gray)


def confirm_tags(signBehavior, raw_tags):
    seen_ids = {t["tag_id"] for t in raw_tags}

    for k in [k for k in signBehavior._tag_buffer if k not in seen_ids]:
        del signBehavior._tag_buffer[k]

    for tid in seen_ids:
        signBehavior._tag_buffer[tid] = (
            signBehavior._tag_buffer.get(tid, 0) + 1
        )

    required = int(getattr(getattr(signBehavior, "config", None), "tag_confirm_frames", tag_confirm_frames))
    confirmed = [
        tid
        for tid, cnt in signBehavior._tag_buffer.items()
        if cnt >= required
    ]

    return confirmed
