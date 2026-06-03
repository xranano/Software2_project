"""Duckietown 4-way intersection geometry (proj-lfi intersection_model)."""

from __future__ import annotations

import math

# proj-lfi / Duckietown physical constants
TILE_SIZE_M = 0.61
LANE_WIDTH_M = 0.205
STOPLINE_THICKNESS_M = 0.048

# Intersection frame: origin = centre of approach stopline, +y into intersection, +x right.
APPROACH_STOPLINE_Y = 0.0
INTERSECTION_CENTER_Y = TILE_SIZE_M / 2.0  # ~0.305 m from approach stopline centre

# Nominal stopline centres for a 4-way (used for vision hints).
STOPLINE_POSITIONS = {
    "approach": (0.0, APPROACH_STOPLINE_Y),
    "opposite": (0.0, TILE_SIZE_M),
    "left_arm": (-TILE_SIZE_M / 2.0, INTERSECTION_CENTER_Y),
    "right_arm": (TILE_SIZE_M / 2.0, INTERSECTION_CENTER_Y),
}


def wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi
