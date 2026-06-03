"""AprilTag 36h11 detection using OpenCV aruco."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    center: tuple[float, float]
    area: float


class AprilTagDetector:
    """Detect Duckietown traffic-sign AprilTags in a BGR camera frame."""

    def __init__(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

    def detect(self, frame_bgr: np.ndarray) -> list[TagDetection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return []

        h, w = gray.shape[:2]
        detections: list[TagDetection] = []
        for corner, tag_id in zip(corners, ids.flatten()):
            pts = corner.reshape(-1, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))
            area = float(cv2.contourArea(pts.astype(np.float32)))
            if area < 50.0:
                continue
            if cy > h * 0.92:
                continue
            detections.append(TagDetection(tag_id=int(tag_id), center=(cx, cy), area=area))

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
