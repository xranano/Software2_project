"""AprilTag 36h11 detection for Duckietown traffic signs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AprilTagDetection:
    tag_id: int
    corners: tuple[tuple[float, float], ...]
    center: tuple[float, float]
    area: float

    def as_dict(self) -> dict:
        return {
            "id": self.tag_id,
            "corners": [list(point) for point in self.corners],
            "center": list(self.center),
            "area": self.area,
        }


class AprilTagDetector:
    """Detect all tag36h11 tags and return the largest detections first."""

    def __init__(self, min_area: float = 50.0):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV AprilTag support is unavailable. Install opencv-contrib-python."
            )

        self.min_area = float(min_area)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        parameters = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._detect_markers = detector.detectMarkers
        else:
            self._detect_markers = lambda gray: cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters,
            )

    def detect(self, frame_bgr: np.ndarray) -> list[AprilTagDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        if frame_bgr.ndim == 2:
            gray = frame_bgr
        else:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self._detect_markers(gray)
        if ids is None:
            return []

        detections = []
        for marker_corners, tag_id in zip(corners, ids.flatten()):
            points = marker_corners.reshape(4, 2).astype(np.float32)
            area = float(abs(cv2.contourArea(points)))
            if area < self.min_area:
                continue

            center = tuple(float(value) for value in np.mean(points, axis=0))
            detections.append(AprilTagDetection(
                tag_id=int(tag_id),
                corners=tuple((float(x), float(y)) for x, y in points),
                center=center,
                area=area,
            ))

        detections.sort(key=lambda detection: detection.area, reverse=True)
        return detections
