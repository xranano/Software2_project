"""Tests for sign tracking and intersection planning."""

from __future__ import annotations

import unittest

from tasks.project.packages.apriltag_detector import TagDetection
from tasks.project.packages.sign_constants import RegulatorySign, TAG_STOP
from tasks.project.packages.sign_tracker import (
    IntersectionPlan,
    SignTracker,
    crossing_traffic_visible,
)


class TestSignTracker(unittest.TestCase):
    def test_stop_and_four_way_tags(self) -> None:
        tracker = SignTracker()
        tracker.observe([
            TagDetection(tag_id=TAG_STOP, center=(320.0, 200.0), area=900.0),
            TagDetection(tag_id=8, center=(300.0, 180.0), area=800.0),
        ])
        self.assertEqual(tracker.regulatory, RegulatorySign.STOP)
        plan = tracker.freeze_plan()
        self.assertIn("left", plan.allowed_turns)
        self.assertIn("right", plan.allowed_turns)

    def test_topology_persists_without_redetection(self) -> None:
        tracker = SignTracker()
        tracker.observe([
            TagDetection(tag_id=8, center=(300.0, 180.0), area=800.0),
        ])
        tracker.observe([])
        self.assertIn("left", tracker.allowed_turns)

    def test_vehicle_tag_in_crossing_zone(self) -> None:
        detections = [TagDetection(tag_id=201, center=(320.0, 200.0), area=500.0)]
        self.assertTrue(crossing_traffic_visible(detections))

    def test_random_turn_respects_options(self) -> None:
        plan = IntersectionPlan(
            allowed_turns=("left", "right"),
            regulatory=RegulatorySign.NONE,
            topology_tag_id=8,
            regulatory_tag_id=None,
        )
        for _ in range(20):
            self.assertIn(plan.choose_turn(), ("left", "right"))


import unittest

from tasks.project.packages.sign_tracker import SignTracker


class TestSignTrackerRandomFallback(unittest.TestCase):
    def test_uses_fallback_turn_without_topology(self) -> None:
        tracker = SignTracker(default_turn="left")
        plan = tracker.freeze_plan(fallback_turn="left")
        self.assertEqual(plan.allowed_turns, ("left",))


if __name__ == "__main__":
    unittest.main()

