"""AprilTag 36h11 detection for Duckietown traffic signs."""
# apriltag_detector.py
# Compatible with Python 3.6+  (no __future__ annotations, no lowercase generics)
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class AprilTagDetection:
    tag_id: int
    corners: Tuple[Tuple[float, float], ...]
    center:  Tuple[float, float]
    area:    float

    def as_dict(self) -> Dict:
        return {
            "id":      self.tag_id,
            "corners": [list(point) for point in self.corners],
            "center":  list(self.center),
            "area":    self.area,
        }


class AprilTagDetector:
    """Detect all tag36h11 tags and return the largest detections first."""

    def __init__(self, min_area: float = 50.0):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "OpenCV AprilTag support is unavailable. "
                "Install opencv-contrib-python."
            )

        self.min_area = float(min_area)

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        parameters = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            self._detect_markers = detector.detectMarkers
        else:
            def _detect(gray):
                return cv2.aruco.detectMarkers(
                    gray, dictionary, parameters=parameters
                )
            self._detect_markers = _detect

    def detect(self, frame_bgr: np.ndarray) -> List[AprilTagDetection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        gray = (frame_bgr if frame_bgr.ndim == 2
                else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))

        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            return []

        detections = []  # type: List[AprilTagDetection]
        for marker_corners, tag_id in zip(corners, ids.flatten()):
            points = marker_corners.reshape(4, 2).astype(np.float32)
            area   = float(abs(cv2.contourArea(points)))
            if area < self.min_area:
                continue
            center = tuple(float(v) for v in np.mean(points, axis=0))
            detections.append(AprilTagDetection(
                tag_id  = int(tag_id),
                corners = tuple((float(x), float(y)) for x, y in points),
                center  = center,
                area    = area,
            ))

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections