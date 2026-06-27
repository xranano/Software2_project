"""
duck_detector.py
"""

from typing import List, Tuple

import numpy as np

Detection = Tuple[Tuple[int, int, int, int], float, int]


class DuckDetector:
    """
    Placeholder detector interface used by detection.py.
    Returns empty detections unless replaced with a model-backed implementation.
    """

    def detect(self, frame_rgb: np.ndarray) -> List[Detection]:
        return []
