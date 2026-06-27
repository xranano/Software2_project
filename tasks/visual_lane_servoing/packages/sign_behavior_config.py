"""Sign behavior configuration and AprilTag ID → sign type mapping."""
from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39


# Raw AprilTag IDs → resolved sign type (STOP covers multiple physical tags).
_STOP_RAW_IDS = {1, 20, 24, 25, 26}
_YIELD_RAW_ID = 39
_INTERSECTION_RAW_IDS = {9, 10, 11}

TAG_TURNS: Dict[TagID, List[str]] = {
    TagID.TURN_RIGHT_FWD: ["forward", "right"],      # Tag 9: forward or right
    TagID.TURN_LEFT_FWD: ["forward", "left"],        # Tag 10: forward or left
    TagID.TURN_LEFT_RIGHT: ["right", "left"],        # Tag 11: right or left
}

TAG_NAMES = {
    TagID.STOP: "STOP",
    TagID.YIELD: "YIELD",
    TagID.TURN_RIGHT_FWD: "TURN FORWARD OR RIGHT",
    TagID.TURN_LEFT_FWD: "TURN FORWARD OR LEFT",
    TagID.TURN_LEFT_RIGHT: "TURN RIGHT OR LEFT",
}

# Intersection signs outrank stop/yield when multiple tags are confirmed.
_SIGN_PRIORITY = {
    TagID.TURN_RIGHT_FWD: 3,
    TagID.TURN_LEFT_FWD: 3,
    TagID.TURN_LEFT_RIGHT: 3,
    TagID.STOP: 2,
    TagID.YIELD: 1,
}


def resolve_tag(raw_id: int) -> Optional[TagID]:
    if raw_id in _INTERSECTION_RAW_IDS:
        return TagID(raw_id)
    if raw_id in _STOP_RAW_IDS:
        return TagID.STOP
    if raw_id == _YIELD_RAW_ID:
        return TagID.YIELD
    return None


def sign_priority(tag: TagID) -> int:
    return _SIGN_PRIORITY.get(tag, 0)


def is_intersection(tag: TagID) -> bool:
    return tag in TAG_TURNS


def is_stop_or_yield(tag: TagID) -> bool:
    return tag in (TagID.STOP, TagID.YIELD)


@dataclass
class SignBehaviorConfig:
    tag_confirm_frames: int = 2
    saved_sign_timeout_sec: float = 8.0
    tag_10_11_swap: bool = False

    red_strip_frac: float = 0.32
    red_roi_left: float = 0.12
    red_roi_right: float = 0.9
    red_line_close_y2_ratio: float = 0.3
    red_pixel_frac: float = 0.012
    red_min_area: int = 60
    red_min_width_frac: float = 0.12
    red_ignore_after_frames: int = 100
    default_forward_on_red_without_tag: bool = False

    stop_hold_frames: int = 18
    slow_ramp_factor: float = 0.8
    stopped_speed_threshold: float = 0.055

    approach_duration: float = 3.0
    approach_speed: float = 0.1

    check_left_frames: int = 8
    check_right_frames: int = 6
    check_turn_speed: float = 0.22
    check_vehicle_min_score: float = 0.4
    check_vehicle_class: int = 1  # truck in object-detection model

    post_stop_frames: int = 12
    post_stop_speed: float = 0.28

    yield_speed: float = 0.13
    yield_duration: float = 5.5

    preturn_left_frames: int = 3
    preturn_right_frames: int = 6
    preturn_speed: float = 0.12

    intersect_forward_frames: int = 40
    intersect_left_frames: int = 55
    intersect_right_frames: int = 40
    turn_left_inner: float = -0.05
    turn_left_outer: float = 0.30
    turn_right_inner: float = 0.30
    turn_right_outer: float = -0.05
    forward_turn_speed: float = 0.23

    exit_speed: float = 0.40
    exit_timeout_frames: int = 3

    @classmethod
    def from_dict(cls, cfg: dict) -> "SignBehaviorConfig":
        kwargs = {}
        for f in fields(cls):
            if f.name in cfg:
                kwargs[f.name] = cfg[f.name]
        return cls(**kwargs)


def clamp_pwm(left: float, right: float) -> Tuple[float, float]:
    left = float(max(-1.0, min(1.0, left)))
    right = float(max(-1.0, min(1.0, right)))
    peak = max(abs(left), abs(right), 1e-6)
    if peak > 1.0:
        left /= peak
        right /= peak
    return left, right
