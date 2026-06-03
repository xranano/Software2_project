"""Persistent AprilTag observations while approaching an intersection."""

from __future__ import annotations

import random
from dataclasses import dataclass

from tasks.project.packages.apriltag_detector import TagDetection
from tasks.project.packages.sign_constants import (
    RegulatorySign,
    TAGS_TO_TURNS,
    TAG_TRAFFIC_SIGN_TYPE,
    TOPOLOGY_TURNS,
    VEHICLE_TAG_ID_MIN,
)

REGULATORY_TAGS = {1: RegulatorySign.STOP, 2: RegulatorySign.YIELD}


@dataclass(frozen=True)
class IntersectionPlan:
    """Sign readings frozen at the stop line."""

    allowed_turns: tuple[str, ...]
    regulatory: RegulatorySign
    topology_tag_id: int | None
    regulatory_tag_id: int | None

    def choose_turn(self) -> str:
        return random.choice(self.allowed_turns)


@dataclass
class _BestReading:
    area: float = 0.0
    tag_id: int | None = None


class SignTracker:
    """
    Accumulate the best regulatory and topology tag readings during lane follow.

    Readings persist across frames so a brief loss of line-of-sight does not erase
    the intersection plan.
    """

    def __init__(self, default_turn: str = "straight"):
        self._default_turn = default_turn
        self._topology_turns: tuple[str, ...] = (default_turn,)
        self._topology = _BestReading()
        self._regulatory = RegulatorySign.NONE
        self._regulatory_reading = _BestReading()

    def observe(self, detections: list[TagDetection]) -> None:
        for det in detections:
            if det.tag_id >= VEHICLE_TAG_ID_MIN:
                continue
            if det.tag_id in REGULATORY_TAGS and det.area >= self._regulatory_reading.area:
                self._regulatory_reading = _BestReading(area=det.area, tag_id=det.tag_id)
                self._regulatory = REGULATORY_TAGS[det.tag_id]
                continue
            turns = self._turns_for_tag(det.tag_id)
            if turns and det.area >= self._topology.area:
                self._topology = _BestReading(area=det.area, tag_id=det.tag_id)
                self._topology_turns = turns

    def freeze_plan(self, fallback_turn: str | None = None) -> IntersectionPlan:
        """Snapshot current readings for execution at the stop line."""
        hint = fallback_turn or self._default_turn
        if self._topology.tag_id is None:
            if hint in ("left", "right", "straight"):
                turns = (hint,)
            else:
                turns = random.choice([("left",), ("right",)])
        else:
            turns = self._topology_turns or (hint,)
        return IntersectionPlan(
            allowed_turns=turns,
            regulatory=self._regulatory,
            topology_tag_id=self._topology.tag_id,
            regulatory_tag_id=self._regulatory_reading.tag_id,
        )

    def clear(self) -> None:
        """Reset after completing an intersection maneuver."""
        self._topology_turns = (self._default_turn,)
        self._topology = _BestReading()
        self._regulatory = RegulatorySign.NONE
        self._regulatory_reading = _BestReading()

    @property
    def regulatory(self) -> RegulatorySign:
        return self._regulatory

    @property
    def allowed_turns(self) -> tuple[str, ...]:
        return self._topology_turns

    @property
    def last_topology_tag_id(self) -> int | None:
        return self._topology.tag_id

    @property
    def last_regulatory_tag_id(self) -> int | None:
        return self._regulatory_reading.tag_id

    @staticmethod
    def _turns_for_tag(tag_id: int) -> tuple[str, ...] | None:
        if tag_id in TAGS_TO_TURNS:
            return TAGS_TO_TURNS[tag_id]
        sign_type = TAG_TRAFFIC_SIGN_TYPE.get(tag_id)
        if sign_type and sign_type in TOPOLOGY_TURNS:
            return TOPOLOGY_TURNS[sign_type]
        return None


def crossing_traffic_visible(detections: list[TagDetection]) -> bool:
    """True when a vehicle tag appears in the forward crossing zone."""
    for det in detections:
        if det.tag_id >= VEHICLE_TAG_ID_MIN and det.center[1] < 360:
            return True
    return False
