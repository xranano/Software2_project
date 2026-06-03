"""Lane detection shim — delegates to the shared visual servoing implementation."""

from __future__ import annotations

from tasks.visual_lane_servoing.packages.visual_servoing_activity import detect_lane_markings


def patch_lane_servoing_detector() -> None:
    """No-op: visual_servoing_activity now ships a working detector."""
    return None
