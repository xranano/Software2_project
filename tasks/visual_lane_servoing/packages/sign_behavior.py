"""Finite-state machine for stop/yield/intersection behavior on red-line trigger."""
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tasks.visual_lane_servoing.packages import april_tag
from tasks.visual_lane_servoing.packages import red_line_detection
from tasks.visual_lane_servoing.packages.sign_behavior_config import (
    SignBehaviorConfig,
    TagID,
    TAG_NAMES,
    TAG_TURNS,
    clamp_pwm,
    is_intersection,
    resolve_tag,
    sign_priority,
)


class SignBehaviorFSM:
    """AprilTag identity + red-line trigger orchestration layered on lane PWM."""

    STATES = (
        "MOVING", "APPROACHING", "SLOWING", "STOPPED", "CHECKPATH",
        "POST_STOP", "YIELDING", "INTERSECT", "PRE_TURN", "TURNING", "EXITING",
    )

    def __init__(self, config: Optional[SignBehaviorConfig] = None):
        self.config = config or SignBehaviorConfig()
        self.state = "MOVING"
        self._tag_buffer: Dict[int, int] = {}
        self._saved_tag: Optional[TagID] = None
        self._saved_tag_time: float = 0.0
        self._chosen_turn: Optional[str] = None
        self._state_frame: int = 0
        self._state_start: float = time.monotonic()
        self._slow_factor: float = 1.0
        self._check_phase: str = "left"
        self._check_sub_frame: int = 0
        self._check_wait_frames: int = 0
        self._red_ignore_frames: int = 0
        self._red_line_locked: bool = False
        self._last_raw_tags: List[dict] = []
        self._last_confirmed: List[int] = []
        self._last_red_line: bool = False
        self._last_red_mask: Optional[np.ndarray] = None
        self.debug: Dict[str, Any] = {}

    @classmethod
    def from_dict(cls, cfg: dict) -> "SignBehaviorFSM":
        return cls(SignBehaviorConfig.from_dict(cfg))

    @property
    def state_name(self) -> str:
        return self.state

    def reset(self) -> None:
        self.state = "MOVING"
        self._tag_buffer.clear()
        self._saved_tag = None
        self._saved_tag_time = 0.0
        self._chosen_turn = None
        self._state_frame = 0
        self._state_start = time.monotonic()
        self._slow_factor = 1.0
        self._check_phase = "left"
        self._check_sub_frame = 0
        self._check_wait_frames = 0
        self._red_ignore_frames = 0
        self._red_line_locked = False
        self._last_raw_tags = []
        self._last_confirmed = []
        self._last_red_line = False
        self._last_red_mask = None
        self.debug = {}

    def _enter_state(self, name: str, now: float) -> None:
        self.state = name
        self._state_frame = 0
        self._state_start = now
        if name != "SLOWING":
            self._slow_factor = 1.0
        if name == "CHECKPATH":
            self._check_phase = "left"
            self._check_sub_frame = 0
            self._check_wait_frames = 0

    def _vehicle_on_right(self, detections: Optional[List], frame_w: int) -> bool:
        if not detections:
            return False
        mid = frame_w * 0.5
        for item in detections:
            if len(item) < 3:
                continue
            bbox, score, class_id = item[0], item[1], item[2]
            if int(class_id) != self.config.check_vehicle_class:
                continue
            if float(score) < self.config.check_vehicle_min_score:
                continue
            xmin, _, xmax, _ = bbox
            cx = (float(xmin) + float(xmax)) * 0.5
            if cx >= mid:
                return True
        return False

    def _pick_saved_tag(self, confirmed: List[int]) -> Optional[TagID]:
        best: Optional[TagID] = None
        best_pri = -1
        for raw_id in confirmed:
            tag = resolve_tag(raw_id)
            if tag is None:
                continue
            pri = sign_priority(tag)
            if pri > best_pri:
                best_pri = pri
                best = tag
        return best

    def _save_confirmed_sign(self, confirmed: List[int], now: float) -> None:
        tag = self._pick_saved_tag(confirmed)
        if tag is None:
            return
        if self._saved_tag is None or sign_priority(tag) >= sign_priority(self._saved_tag):
            if self._saved_tag != tag:
                print("[Sign] Saved {} ({}) — keep driving until red line.".format(
                    TAG_NAMES.get(tag, tag.name), tag.name))
            self._saved_tag = tag
            self._saved_tag_time = now
        elif tag == self._saved_tag:
            self._saved_tag_time = now

    def _expire_saved_sign(self, now: float) -> None:
        if self._saved_tag is None:
            return
        if now - self._saved_tag_time > self.config.saved_sign_timeout_sec:
            print("[Sign] Saved sign expired ({:.1f}s).".format(
                self.config.saved_sign_timeout_sec))
            self._saved_tag = None

    def _begin_maneuver(self, now: float) -> None:
        self._red_line_locked = True
        saved = self._saved_tag
        self._saved_tag = None

        if saved is None:
            return

        if is_intersection(saved):
            options = TAG_TURNS[saved]
            self._chosen_turn = random.choice(options)
            
            # ENFORCE: Tag 10 cannot turn right
            if saved == TagID.TURN_LEFT_FWD and self._chosen_turn == "right":
                self._chosen_turn = "forward"
                print("[Sign] OVERRIDE: Tag 10 cannot go right, forcing FORWARD")
            
            # ENFORCE: Tag 11 cannot go forward
            if saved == TagID.TURN_LEFT_RIGHT and self._chosen_turn == "forward":
                self._chosen_turn = "left"
                print("[Sign] OVERRIDE: Tag 11 cannot go forward, forcing LEFT")
            
            print("[Sign] Intersection {} — chose {}.".format(
                TAG_NAMES.get(saved, saved.name), self._chosen_turn.upper()))
            self._enter_state("APPROACHING", now)
            return

        if saved == TagID.YIELD:
            print("[Sign] YIELD at red line — slowing to {:.2f} for {:.1f}s.".format(
                self.config.yield_speed, self.config.yield_duration))
            self._enter_state("YIELDING", now)
            return

        if saved == TagID.STOP:
            print("[Sign] STOP at red line — stopping.")
            self._enter_state("SLOWING", now)

    def _turn_pwm(self, direction: str) -> Tuple[float, float]:
        cfg = self.config
        if direction == "forward":
            s = cfg.forward_turn_speed
            return s, s
        if direction == "left":
            return cfg.turn_left_inner, cfg.turn_left_outer
        if direction == "right":
            return cfg.turn_right_inner, cfg.turn_right_outer
        return 0.0, 0.0

    def _turn_frame_budget(self, direction: str) -> int:
        cfg = self.config
        if direction == "forward":
            return cfg.intersect_forward_frames
        if direction == "left":
            return cfg.intersect_left_frames
        return cfg.intersect_right_frames

    def _finish_exit(self, now: float) -> None:
        cfg = self.config
        self._chosen_turn = None
        self._red_line_locked = False
        self._red_ignore_frames = cfg.red_ignore_after_frames
        self._enter_state("MOVING", now)
        print("[Sign] Maneuver complete — resuming lane follow.")

    def _update_debug(self) -> None:
        self.debug = {
            "state": self.state,
            "confirmed_tags": list(self._last_confirmed),
            "raw_tags": [
                {"id": t["tag_id"], "corners": t["corners"].tolist()
                 if hasattr(t["corners"], "tolist") else t["corners"]}
                for t in self._last_raw_tags
            ],
            "saved_tag": self._saved_tag.name if self._saved_tag else None,
            "red_line": self._last_red_line,
            "red_mask": self._last_red_mask,
            "chosen_turn": self._chosen_turn,
            "slow_factor": self._slow_factor,
            "state_frame": self._state_frame,
            "red_ignore_frames": self._red_ignore_frames,
            "red_line_locked": self._red_line_locked,
        }

    def step(
        self,
        frame_rgb: np.ndarray,
        base_left: float,
        base_right: float,
        detections: Optional[List] = None,
    ) -> Tuple[float, float, str]:
        now = time.monotonic()
        cfg = self.config

        self._last_raw_tags = april_tag.detect_tags(self, frame_rgb)
        self._last_confirmed = april_tag.confirm_tags(self, self._last_raw_tags)
        red_line, red_mask = red_line_detection.detect_red_line(self, frame_rgb)
        self._last_red_line = red_line
        self._last_red_mask = red_mask

        left, right = base_left, base_right

        if self.state == "MOVING":
            self._save_confirmed_sign(self._last_confirmed, now)
            self._expire_saved_sign(now)

            if red_line and self._saved_tag is not None:
                self._begin_maneuver(now)
            elif red_line and self._saved_tag is None and cfg.default_forward_on_red_without_tag:
                self._chosen_turn = "forward"
                self._enter_state("APPROACHING", now)
                self._red_line_locked = True

        if self.state == "SLOWING":
            self._slow_factor *= cfg.slow_ramp_factor
            left = base_left * self._slow_factor
            right = base_right * self._slow_factor
            peak = max(abs(left), abs(right))
            if peak <= cfg.stopped_speed_threshold or self._state_frame >= 60:
                left, right = 0.0, 0.0
                self._enter_state("STOPPED", now)

        if self.state == "STOPPED":
            left, right = 0.0, 0.0
            if self._state_frame >= cfg.stop_hold_frames:
                self._enter_state("CHECKPATH", now)

        if self.state == "CHECKPATH":
            left, right = 0.0, 0.0
            w = frame_rgb.shape[1] if frame_rgb is not None and frame_rgb.size else 640
            s = cfg.check_turn_speed

            if self._check_wait_frames > 0:
                self._check_wait_frames -= 1
            elif self._check_phase == "left":
                if self._check_sub_frame < cfg.check_left_frames:
                    left, right = -s, s
                    self._check_sub_frame += 1
                else:
                    self._check_phase = "right"
                    self._check_sub_frame = 0
            elif self._check_phase == "right":
                if self._vehicle_on_right(detections, w):
                    print("[Sign] Vehicle on right — waiting.")
                    self._check_wait_frames = max(cfg.check_right_frames, 5)
                    self._check_phase = "left"
                    self._check_sub_frame = 0
                elif self._check_sub_frame < cfg.check_right_frames:
                    left, right = s, -s
                    self._check_sub_frame += 1
                else:
                    self._enter_state("POST_STOP", now)

        if self.state == "POST_STOP":
            left = right = cfg.post_stop_speed
            if self._state_frame >= cfg.post_stop_frames:
                self._finish_exit(now)

        if self.state == "YIELDING":
            peak = max(abs(base_left), abs(base_right), 1e-6)
            scale = min(1.0, cfg.yield_speed / peak)
            left = base_left * scale
            right = base_right * scale
            if now - self._state_start >= cfg.yield_duration:
                print("[Sign] Yield complete — resuming normal speed.")
                self._finish_exit(now)

        if self.state == "APPROACHING":
            left = right = cfg.approach_speed
            # Wait until red line disappears before turning
            if not red_line:
                print(f"[Sign] Red line cleared — entering intersection")
                self._enter_state("INTERSECT", now)
            elif now - self._state_start >= cfg.approach_duration + 5.0:
                # Safety timeout in case red line never disappears
                print(f"[Sign] Approach timeout — forcing intersection entry")
                self._enter_state("INTERSECT", now)

        if self.state == "INTERSECT":
            turn = self._chosen_turn or "forward"
            print(f"[Sign] INTERSECT — chosen turn: {turn}")
            if turn in ("left", "right"):
                self._enter_state("PRE_TURN", now)
            else:
                self._enter_state("TURNING", now)

        if self.state == "PRE_TURN":
            left = right = cfg.preturn_speed
            turn = self._chosen_turn or "forward"
            budget = (cfg.preturn_left_frames if turn == "left"
                      else cfg.preturn_right_frames)
            if self._state_frame >= budget:
                print(f"[Sign] PRE_TURN complete — starting {turn.upper()} turn")
                self._enter_state("TURNING", now)

        if self.state == "TURNING":
            turn = self._chosen_turn or "forward"
            left, right = self._turn_pwm(turn)
            if self._state_frame == 0:
                print(f"[Sign] TURNING {turn.upper()}: L={left:.2f}, R={right:.2f}")
            if self._state_frame >= self._turn_frame_budget(turn):
                print(f"[Sign] Turn complete after {self._state_frame} frames")
                self._enter_state("EXITING", now)

        if self.state == "EXITING":
            left = right = cfg.exit_speed
            if self._state_frame >= cfg.exit_timeout_frames:
                self._finish_exit(now)

        self._state_frame += 1
        self._update_debug()
        return clamp_pwm(left, right)
