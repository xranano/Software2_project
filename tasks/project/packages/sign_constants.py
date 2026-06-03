"""Duckietown AprilTag sign IDs and intersection turn mappings."""

from __future__ import annotations

from enum import Enum

# Regulatory signs (apriltagsDB.yaml — dt-core)
TAG_STOP = 1
TAG_YIELD = 2
TAG_NO_RIGHT = 3
TAG_NO_LEFT = 4
TAG_ONEWAY_RIGHT = 6
TAG_ONEWAY_LEFT = 7

# Intersection topology examples used in course demos / maps
TAG_FOUR_WAY = 8
TAG_RIGHT_T = 9
TAG_LEFT_T = 10
TAG_T_INTERSECT = 11

# Course-style intersection tags with explicit allowed turns
TAGS_TO_TURNS: dict[int, tuple[str, ...]] = {
    63: ("left", "straight"),
    59: ("straight", "right"),
    67: ("left", "right"),
    8: ("left", "straight", "right"),
    13: ("left", "straight", "right"),
    9: ("straight", "right"),
    10: ("left", "straight"),
    11: ("left", "straight"),
    6: ("right",),
    7: ("left",),
}

TOPOLOGY_TURNS: dict[str, tuple[str, ...]] = {
    "stop": ("straight",),
    "yield": ("straight",),
    "4-way-intersect": ("left", "straight", "right"),
    "right-T-intersect": ("straight", "right"),
    "left-T-intersect": ("left", "straight"),
    "T-intersection": ("left", "straight"),
    "oneway-right": ("right",),
    "oneway-left": ("left",),
    "no-right-turn": ("left", "straight"),
    "no-left-turn": ("straight", "right"),
}

TAG_TRAFFIC_SIGN_TYPE: dict[int, str] = {
    TAG_STOP: "stop",
    TAG_YIELD: "yield",
    TAG_NO_RIGHT: "no-right-turn",
    TAG_NO_LEFT: "no-left-turn",
    TAG_ONEWAY_RIGHT: "oneway-right",
    TAG_ONEWAY_LEFT: "oneway-left",
    TAG_FOUR_WAY: "4-way-intersect",
    TAG_RIGHT_T: "right-T-intersect",
    TAG_LEFT_T: "left-T-intersect",
    TAG_T_INTERSECT: "T-intersection",
}

VEHICLE_TAG_ID_MIN = 200


class RegulatorySign(str, Enum):
    NONE = "none"
    STOP = "stop"
    YIELD = "yield"


class TrafficState(str, Enum):
    DRIVING = "DRIVING"
    SIGN_DETECTED = "SIGN_DETECTED"
    DECISION = "DECISION"
    MANEUVERING = "MANEUVERING"
    WAITING = "WAITING"
