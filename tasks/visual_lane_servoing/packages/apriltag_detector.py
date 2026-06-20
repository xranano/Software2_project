"""AprilTag 36h11 detection wrapper (OpenCV aruco + raw fallback)."""
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tasks.visual_lane_servoing.packages import april_tag
from tasks.visual_lane_servoing.packages.sign_behavior_config import SignBehaviorConfig


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


class _DetectContext:
    def __init__(self, min_area: float, swap_10_11: bool = True):
        cfg = SignBehaviorConfig()
        cfg.tag_10_11_swap = swap_10_11
        self.config = cfg
        self.min_area = float(min_area)


class AprilTagDetector:
    """Detect tag36h11 markers; largest detections first."""

    def __init__(self, min_area: float = 50.0, swap_10_11: bool = True):
        self.min_area = float(min_area)
        self._ctx = _DetectContext(min_area, swap_10_11)
        backend = "opencv-aruco" if hasattr(cv2, "aruco") else "raw-fallback"
        print("[AprilTag] Detector ready (backend={})".format(backend))

    def _to_rgb(self, frame):
        if frame is None or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def detect(self, frame_bgr: np.ndarray) -> List[AprilTagDetection]:
        rgb = self._to_rgb(frame_bgr)
        if rgb is None:
            return []

        raw = april_tag.detect_tags(self._ctx, rgb)
        detections = []
        for tag in raw:
            pts = np.asarray(tag["corners"], dtype=np.float32).reshape(4, 2)
            area = float(abs(cv2.contourArea(pts)))
            if area < self.min_area:
                continue
            center = tuple(float(v) for v in np.mean(pts, axis=0))
            detections.append(AprilTagDetection(
                tag_id=int(tag["tag_id"]),
                corners=tuple((float(x), float(y)) for x, y in pts),
                center=center,
                area=area,
            ))
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections
