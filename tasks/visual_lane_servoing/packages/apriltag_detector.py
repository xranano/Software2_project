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
        self.min_area = float(min_area)
        self.backend = None  # type: str
        self._tag_detector = None

        if self._try_init_opencv():
            self.backend = "opencv"
        elif self._try_init_dt_apriltags():
            self.backend = "dt_apriltags"
        elif self._try_init_pupil():
            self.backend = "pupil"
        else:
            raise RuntimeError(
                "AprilTag detection unavailable. On the robot, install dt-apriltags "
                "(pip install dt-apriltags). On sim/laptop use opencv-contrib-python."
            )

        print("[AprilTag] Detector ready (backend={})".format(self.backend))

    def _try_init_opencv(self):
        if not hasattr(cv2, "aruco"):
            return False
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            )
        except (AttributeError, cv2.error):
            return False

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
        return True

    def _try_pip_install(self, package):
        import subprocess
        import sys
        try:
            print("[AprilTag] Trying: pip install {} --user".format(package))
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--user", "-q"],
                timeout=180,
            )
            return result.returncode == 0
        except Exception as exc:
            print("[AprilTag] pip install failed: {}".format(exc))
            return False

    def _try_init_dt_apriltags(self):
        """Duckietown ships dt-apriltags on many DB21 bots."""
        try:
            from dt_apriltags import Detector
        except ImportError:
            if not self._try_pip_install("dt-apriltags"):
                return False
            try:
                from dt_apriltags import Detector
            except ImportError:
                return False

        try:
            self._tag_detector = Detector(
                families="tag36h11",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
        except TypeError:
            self._tag_detector = Detector(
                families="tag36h11",
                nthreads=1,
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=True,
                decode_sharpening=0.25,
                debug=0,
            )
        return True

    def _try_init_pupil(self):
        try:
            from pupil_apriltags import Detector
        except ImportError:
            return False

        self._tag_detector = Detector(
            families="tag36h11",
            nthreads=1,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=True,
            decode_sharpening=0.25,
            debug=0,
        )
        return True

    def _to_gray(self, frame_bgr):
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        if frame_bgr.ndim == 2:
            return frame_bgr
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _from_corners(self, points, tag_id):
        # type: (np.ndarray, int) -> AprilTagDetection
        area = float(abs(cv2.contourArea(points)))
        if area < self.min_area:
            return None
        center = tuple(float(v) for v in np.mean(points, axis=0))
        return AprilTagDetection(
            tag_id=int(tag_id),
            corners=tuple((float(x), float(y)) for x, y in points),
            center=center,
            area=area,
        )

    def _detect_opencv(self, gray):
        corners, ids, _ = self._detect_markers(gray)
        if ids is None or len(corners) == 0:
            return []

        detections = []  # type: List[AprilTagDetection]
        for marker_corners, tag_id in zip(corners, ids.flatten()):
            points = marker_corners.reshape(4, 2).astype(np.float32)
            det = self._from_corners(points, tag_id)
            if det is not None:
                detections.append(det)
        return detections

    def _detect_native(self, gray):
        detections = []  # type: List[AprilTagDetection]
        for tag in self._tag_detector.detect(gray):
            points = np.asarray(tag.corners, dtype=np.float32).reshape(4, 2)
            det = self._from_corners(points, tag.tag_id)
            if det is not None:
                detections.append(det)
        return detections

    def detect(self, frame_bgr: np.ndarray) -> List[AprilTagDetection]:
        gray = self._to_gray(frame_bgr)
        if gray is None:
            return []

        if self.backend == "opencv":
            detections = self._detect_opencv(gray)
        else:
            detections = self._detect_native(gray)

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
