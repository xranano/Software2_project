"""
sign_behavior.py
"""

import random
import time
from typing import Dict, List, Optional, Tuple

from tasks.visual_lane_servoing.packages import april_tag, red_line_detection
from tasks.visual_lane_servoing.packages.sign_behavior_config import (
    TagID,
    _TAG_TURNS,
    SignBehaviorConfig,
    State,
    resolve_tag,
)


class SignBehaviorFSM:
    FPS = 24.0

    @classmethod
    def from_dict(cls, cfg: dict):
        return cls(SignBehaviorConfig(**(cfg or {})))

    def __init__(self, config=None):
        self.cfg = config or SignBehaviorConfig()
        self.config = self.cfg
        self.state = State.MOVING

        self._tag_buffer: Dict[int, int] = {}
        self._saved_tag: Optional[TagID] = None
        self._saved_tag_time: Optional[float] = None
        self._saved_tag_timeout = float(
            getattr(self.cfg, "sign_ttl", getattr(self.cfg, "saved_sign_timeout_sec", 4.0))
        )

        self._chosen_turn: Optional[str] = None
        self._queued_turn: Optional[str] = None

        self._state_started_at = time.monotonic()
        self._last_step_time = None
        self._dt = 1.0 / self.FPS
        self._turn_counter = 0.0
        self._hold_counter = 0.0
        self._check_counter = 0.0
        self._check_wait_frames = 0.0

        self._red_line_locked = False
        self._ignore_red_counter = 0.0
        self._turn_prep_start = None

        self._slow_factor = 1.0
        self._last_raw_tags: List[dict] = []
        self._last_confirmed: List[int] = []
        self._last_red_line = False
        self.debug = {}

    @property
    def state_name(self):
        return self.state.name

    def reset(self):
        self.state = State.MOVING
        self._tag_buffer.clear()
        self._saved_tag = None
        self._saved_tag_time = None
        self._chosen_turn = None
        self._queued_turn = None
        self._state_started_at = time.monotonic()
        self._last_step_time = None
        self._dt = 1.0 / self.FPS
        self._turn_counter = 0.0
        self._hold_counter = 0.0
        self._check_counter = 0.0
        self._check_wait_frames = 0.0
        self._red_line_locked = False
        self._ignore_red_counter = 0.0
        self._turn_prep_start = None
        self._slow_factor = 1.0
        self._last_raw_tags = []
        self._last_confirmed = []
        self._last_red_line = False
        self.debug = {}

    def _set_state(self, new_state):
        self.state = new_state
        self._state_started_at = time.monotonic()
        self._turn_counter = 0.0
        self._hold_counter = 0.0
        self._check_counter = 0.0
        if new_state != State.SLOWING:
            self._slow_factor = 1.0

    def _elapsed_seconds(self):
        return max(0.0, time.monotonic() - self._state_started_at)

    def _elapsed_frames(self):
        return self._elapsed_seconds() * self.FPS

    def _update_dt(self):
        now = time.monotonic()
        if self._last_step_time is None:
            self._dt = 1.0 / self.FPS
        else:
            self._dt = max(0.0, now - self._last_step_time)
        self._last_step_time = now

    def _tick_timers(self):
        if self._ignore_red_counter > 0.0:
            self._ignore_red_counter = max(0.0, self._ignore_red_counter - self._dt)
        if self._check_wait_frames > 0.0:
            self._check_wait_frames = max(0.0, self._check_wait_frames - self._dt * self.FPS)

    def _parse_red_line(self, frame_rgb):
        result = red_line_detection.detect_red_line(self, frame_rgb)
        if isinstance(result, tuple):
            return bool(result[0])
        return bool(result)

    def _pick_last_seen_tag(self, confirmed_tags: List[int]) -> Optional[TagID]:
        """
        Last-seen wins: pick the most recent resolvable confirmed tag from raw order.
        """
        if not confirmed_tags:
            return None
        confirmed_set = set(int(t) for t in confirmed_tags)

        for tag_info in reversed(self._last_raw_tags):
            raw_id = int(tag_info.get("tag_id", -1))
            if raw_id not in confirmed_set:
                continue
            tag = resolve_tag(raw_id)
            if tag is not None:
                return tag

        for raw_id in reversed([int(t) for t in confirmed_tags]):
            tag = resolve_tag(raw_id)
            if tag is not None:
                return tag
        return None

    def _save_confirmed_sign(self, confirmed_tags):
        tag = self._pick_last_seen_tag(confirmed_tags)
        if tag is None:
            return
        if self._saved_tag != tag:
            print(f"[SignBehavior] sign saved: {tag.name}")
        self._saved_tag = tag
        self._saved_tag_time = time.monotonic()

    def _expire_saved_sign(self):
        if self._saved_tag is None or self._saved_tag_time is None:
            return
        if time.monotonic() - self._saved_tag_time > self._saved_tag_timeout:
            self._saved_tag = None
            self._saved_tag_time = None

    def _vehicle_on_right(self, detections: Optional[List], frame_w: int) -> bool:
        if not detections:
            return False
        mid = frame_w * 0.5
        for item in detections:
            if len(item) < 3:
                continue
            bbox, score, class_id = item[0], float(item[1]), int(item[2])
            # truck class and reasonable confidence
            if class_id != 1 or score < 0.4:
                continue
            x1, _, x2, _ = bbox
            cx = (float(x1) + float(x2)) * 0.5
            if cx >= mid:
                return True
        return False

    def _pick_turn(self):
        if self._chosen_turn is not None:
            return self._chosen_turn
        if self._queued_turn is not None:
            return self._queued_turn
        return "forward"

    def _finish_behavior(self):
        self._red_line_locked = False
        self._ignore_red_counter = float(getattr(self.cfg, "red_ignore_after_frames", 24)) / self.FPS
        self._chosen_turn = None
        self._queued_turn = None
        self._saved_tag = None
        self._saved_tag_time = None
        self._turn_prep_start = None
        self._set_state(State.MOVING)

    def step(self, frame_rgb, base_left, base_right, detections):
        self._update_dt()
        self._tick_timers()
        detections = detections or []

        self._last_raw_tags = april_tag.detect_tags(self, frame_rgb)
        self._last_confirmed = april_tag.confirm_tags(self, self._last_raw_tags)

        if self._ignore_red_counter > 0.0 or self._red_line_locked:
            red_line = False
        else:
            red_line = self._parse_red_line(frame_rgb)
        self._last_red_line = red_line

        self._save_confirmed_sign(self._last_confirmed)
        self._expire_saved_sign()

        left, right = base_left, base_right

        if self.state == State.MOVING:
            if red_line and self._saved_tag is not None:
                self._red_line_locked = True
                tag = self._saved_tag
                self._saved_tag = None
                self._saved_tag_time = None

                if tag == TagID.STOP:
                    self._queued_turn = "stop"
                    self._set_state(State.APPROACHING)
                elif tag == TagID.YIELD:
                    self._set_state(State.YIELDING)
                elif tag in _TAG_TURNS:
                    self._queued_turn = random.choice(_TAG_TURNS[tag])
                    self._set_state(State.APPROACHING)
                else:
                    self._queued_turn = random.choice(["left", "right", "forward"])
                    self._set_state(State.APPROACHING)

        elif self.state == State.SLOWING:
            self._slow_factor *= float(getattr(self.cfg, "slow_ramp_factor", 0.8))
            left = base_left * self._slow_factor
            right = base_right * self._slow_factor
            if max(abs(left), abs(right)) <= float(getattr(self.cfg, "stopped_speed_threshold", 0.055)):
                self._set_state(State.STOPPED)
                left, right = 0.0, 0.0

        elif self.state == State.STOPPED:
            left, right = 0.0, 0.0
            self._hold_counter = self._elapsed_frames()
            if self._hold_counter >= float(getattr(self.cfg, "stop_hold_frames", 18)):
                self._set_state(State.CHECKPATH)

        elif self.state == State.CHECKPATH:
            left, right = 0.0, 0.0
            w = frame_rgb.shape[1] if frame_rgb is not None and frame_rgb.size else 640
            check_left = float(getattr(self.cfg, "check_left_frames", 8))
            check_right = float(getattr(self.cfg, "check_right_frames", 6))
            c = self._elapsed_frames()
            if c < check_left:
                s = float(getattr(self.cfg, "check_turn_speed", 0.22))
                left, right = -s, s
            elif c < check_left + check_right:
                if self._vehicle_on_right(detections, w):
                    self._set_state(State.CHECKPATH)
                else:
                    s = float(getattr(self.cfg, "check_turn_speed", 0.22))
                    left, right = s, -s
            else:
                self._set_state(State.POST_STOP)

        elif self.state == State.YIELDING:
            peak = max(abs(base_left), abs(base_right), 1e-6)
            scale = min(1.0, float(getattr(self.cfg, "yield_speed", 0.13)) / peak)
            left, right = base_left * scale, base_right * scale
            if self._elapsed_seconds() >= float(getattr(self.cfg, "yield_duration", 5.5)):
                self._finish_behavior()

        elif self.state == State.POST_STOP:
            s = float(getattr(self.cfg, "post_stop_speed", 0.28))
            left = right = s
            if self._elapsed_frames() >= float(getattr(self.cfg, "post_stop_frames", 12)):
                self._finish_behavior()

        elif self.state == State.APPROACHING:
            # no slowdown in approach: keep lane-follow output
            left, right = base_left, base_right
            if self._queued_turn == "stop":
                if self._elapsed_seconds() >= 1.8:
                    self._set_state(State.SLOWING)
            elif self._elapsed_seconds() >= float(getattr(self.cfg, "approach_duration", 3.0)):
                self._set_state(State.INTERSECT)

        elif self.state == State.INTERSECT:
            self._chosen_turn = self._pick_turn()
            if self._chosen_turn in ("left", "right"):
                self._set_state(State.PRE_TURN)
            else:
                self._turn_prep_start = time.monotonic()
                self._set_state(State.TURNING)

        elif self.state == State.PRE_TURN:
            # Drive forward first, then slow prep
            if self._elapsed_seconds() < 0.8:
                left = right = float(getattr(self.cfg, "approach_speed", 0.23))
            else:
                s = float(getattr(self.cfg, "preturn_speed", 0.12))
                left = right = s
                budget = float(
                    getattr(self.cfg, "preturn_left_frames", 3)
                    if self._pick_turn() == "left"
                    else getattr(self.cfg, "preturn_right_frames", 6)
                )
                if self._elapsed_frames() >= budget + 24:
                    self._turn_prep_start = time.monotonic()
                    self._set_state(State.TURNING)

        elif self.state == State.TURNING:
            # Drive forward first if it's a forward "turn", or prepare for actual turn
            if self._chosen_turn == "forward" and self._elapsed_seconds() < 0.8:
                left = right = float(getattr(self.cfg, "approach_speed", 0.23))
            elif self._turn_prep_start is not None and (time.monotonic() - self._turn_prep_start) < float(getattr(self.cfg, "turn_prep_duration", 0.35)):
                left, right = base_left, base_right
            else:
                turn = self._pick_turn()
                if turn == "forward":
                    left = right = float(getattr(self.cfg, "forward_turn_speed", 0.23))
                    budget = float(getattr(self.cfg, "intersect_forward_frames", 40))
                elif turn == "left":
                    left = float(getattr(self.cfg, "turn_left_inner", -0.05))
                    right = float(getattr(self.cfg, "turn_left_outer", 0.30))
                    budget = float(getattr(self.cfg, "intersect_left_frames", 55))
                else:
                    left = float(getattr(self.cfg, "turn_right_inner", 0.30))
                    right = float(getattr(self.cfg, "turn_right_outer", -0.05))
                    budget = float(getattr(self.cfg, "intersect_right_frames", 40))

                if self._elapsed_frames() >= budget:
                    self._set_state(State.EXITING)

        elif self.state == State.EXITING:
            # Use lane-following commands for smooth transition
            left, right = base_left, base_right
            if self._elapsed_frames() >= float(getattr(self.cfg, "exit_timeout_frames", 12)):
                self._finish_behavior()

        self.debug = {
            "state": self.state.name,
            "confirmed_tags": [int(t) for t in self._last_confirmed],
            "saved_tag": self._saved_tag.name if self._saved_tag is not None else None,
            "red_line": bool(red_line),
            "chosen_turn": self._pick_turn() if self._queued_turn or self._chosen_turn else None,
            "state_elapsed_frames_24fps": int(round(self._elapsed_frames())),
            "dt_seconds": round(float(self._dt), 4),
        }

        return left, right