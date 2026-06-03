"""Thread-safe motion status for the web UI and virtual server."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass

from tasks.project.packages.sign_constants import RegulatorySign, TrafficState


@dataclass
class MotionStatus:
    running: bool = True
    traffic_state: str = TrafficState.DRIVING.value
    state: str = "LANE_FOLLOW"
    trajectory: str = "straight"
    last_tag_id: int | None = None
    regulatory_sign: str = RegulatorySign.NONE.value
    allowed_turns: tuple[str, ...] = ("straight",)
    pose_x: float = 0.0
    pose_y: float = 0.0
    pose_yaw: float = 0.0
    d: float = 0.0
    phi: float = 0.0
    curvature: float = 0.0
    intersection_done_count: int = 0
    game_over: bool = False


_lock = threading.Lock()
_status = MotionStatus()


def update(**kwargs) -> None:
    with _lock:
        for key, value in kwargs.items():
            if hasattr(_status, key):
                setattr(_status, key, value)


def snapshot() -> dict:
    with _lock:
        return asdict(_status)
