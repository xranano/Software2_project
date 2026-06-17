# agent.py
import os
import time
import random
import yaml
import numpy as np
import cv2
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.apriltag_detector import AprilTagDetector
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5

# ── AprilTag ID → sign name ──────────────────────────────────────────────────
TAG_INSTRUCTIONS = {
    0:  "STOP",
    1:  "YIELD",
    2:  "TURN LEFT",
    3:  "TURN RIGHT",
    4:  "STRAIGHT ONLY",
    5:  "PARKING",
    6:  "PEDESTRIAN CROSSING",
    7:  "NO ENTRY",
    8:  "TURN LEFT OR RIGHT",
    9:  "TURN RIGHT OR FORWARD",
    10: "TURN LEFT OR FORWARD",
    11: "TURN LEFT OR RIGHT",
    20: "STOP",
    24: "STOP",
    39: "YIELD",
}

# ── DETECTION-TEST MODE ──────────────────────────────────────────────────
# Every tag triggers the same fixed timed stop, regardless of its meaning.
# This is for validating AprilTag/object-detection quality, not real sign
# behaviour. To restore normal sign logic later, set this back to False and
# the original per-tag sets below (_PERMANENT_STOP_TAGS, _TIMED_STOP_TAGS,
# etc.) take over again.
_STOP_ON_ANY_SIGN     = True
_ANY_SIGN_STOP_DURATION = 2.0

_PERMANENT_STOP_TAGS  = {5, 7}          # PARKING, NO ENTRY
_TIMED_STOP_TAGS      = {20, 24}        # STOP only (0/6 no longer trigger a stop)
_TIMED_STOP_DURATIONS = {20: 2.0, 24: 2.0}
_YIELD_TAGS           = {39}            # YIELD only (tag 1 no longer triggers yield)
_LEFT_TURN_TAGS       = {10}            # TURN LEFT OR FORWARD
_RIGHT_TURN_TAGS      = {9}             # TURN RIGHT OR FORWARD
_RANDOM_TURN_TAGS     = {11}            # TURN LEFT OR RIGHT — choose randomly at the intersection
_STRAIGHT_TAGS        = set()           # nothing suppresses an in-progress turn anymore


def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white:  np.ndarray,
    h: int,
) -> Tuple[list, list]:
    slice_height = int(h * 0.35 / _NUM_SLICES)
    start_y      = int(h * _ROI_START)
    yellow_xs, white_xs = [], []

    for i in range(_NUM_SLICES):
        y = start_y + i * slice_height + slice_height // 2

        strip_y = mask_yellow[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_y > 0)[1]
        if len(idx) > 0:
            yellow_xs.append(int(np.mean(idx)))

        strip_w = mask_white[y - _SLICE_TOL: y + _SLICE_TOL, :]
        idx = np.where(strip_w > 0)[1]
        if len(idx) > 0:
            white_xs.append(int(np.mean(idx)))

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path=None):
        path = config_path or _CONFIG_FILE
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        # ── PD / speed ────────────────────────────────────────────────────
        self.p_gain              = cfg.get('p_gain',              0.5)
        self.d_gain              = cfg.get('d_gain',              0.58)
        self.max_steer           = cfg.get('max_steer',           0.2)
        self.base_speed          = cfg.get('base_speed',          0.23)
        self.min_cruise_speed    = cfg.get('min_cruise_speed',    0.23)
        self.min_wheel_speed     = cfg.get('min_wheel_speed',     0.08)
        self.turn_speed_ratio    = cfg.get('turn_speed_ratio',    0.0)
        self.curve_feedforward   = cfg.get('curve_feedforward',   0.12)
        self.curve_threshold     = cfg.get('curve_threshold',     350)
        self.detection_threshold = cfg.get('detection_threshold', 500)
        self.smooth_alpha        = cfg.get('smooth_alpha',        0.6)
        self.steer_smooth        = cfg.get('steer_smooth',        0.6)

        # ── AprilTag ──────────────────────────────────────────────────────
        self.apriltag_enabled  = cfg.get('apriltag_enabled',   True)
        self.apriltag_interval = max(1, int(cfg.get('apriltag_interval', 3)))
        self.apriltag_min_area = float(cfg.get('apriltag_min_area', 50.0))
        # Area threshold before a sign behaviour fires (tune per environment)
        self.apriltag_act_area = float(cfg.get('apriltag_act_area', 800.0))
        # Cooldown so the same tag doesn't re-trigger while rolling past
        self.apriltag_cooldown = float(cfg.get('apriltag_cooldown', 5.0))

        # ── Yield ─────────────────────────────────────────────────────────
        self.yield_speed    = float(cfg.get('yield_speed',    0.13))
        self.yield_duration = float(cfg.get('yield_duration', 2.0))

        # ── Left-turn FSM params ──────────────────────────────────────────
        self._left_straight_duration = cfg.get('left_straight_duration', 1.1)
        self._left_turn_max_duration = cfg.get('left_turn_max_duration',  2.5)
        self._left_straight_speed    = cfg.get('left_straight_speed',     0.23)
        self._left_turn_wheel_inner  = cfg.get('left_turn_wheel_inner',   0.07)
        self._left_turn_wheel_outer  = cfg.get('left_turn_wheel_outer',   0.26)

        # ── Right-turn FSM params ─────────────────────────────────────────
        self._right_straight_duration = cfg.get('right_straight_duration', 0.6)
        self._right_turn_max_duration = cfg.get('right_turn_max_duration',  2.0)
        self._right_straight_speed    = cfg.get('right_straight_speed',     0.23)
        self._right_turn_wheel_inner  = cfg.get('right_turn_wheel_inner',   0.26)
        self._right_turn_wheel_outer  = cfg.get('right_turn_wheel_outer',   0.07)

        # ── Runtime state ─────────────────────────────────────────────────
        self.frame_count         = 0
        self._prev_error         = 0.0
        self._filtered_error     = 0.0
        self._filtered_steering  = 0.0
        self._smooth_left        = None
        self._smooth_right       = None
        self._lane_half_width    = float(_LINE_OFFSET)

        # Left-turn FSM
        self._yellow_visible_frames  = 0
        self._left_turn_state        = 'none'
        self._left_turn_start        = 0.0
        self._left_turn_cooldown_end = 0.0

        # Right-turn FSM
        self._right_turn_state        = 'none'
        self._right_turn_start        = 0.0
        self._right_turn_cooldown_end = 0.0

        # Pending sign-decided turn ('none' | 'left' | 'right').
        # Set the moment a turn sign (9/10/11) is seen — remembers which way
        # to go at the upcoming intersection. The bot drives straight until
        # the lane lines vanish (same signal as the line-loss auto-turn),
        # then carries out this remembered direction. A minimum straight
        # dwell still applies so we don't react to a momentary dropout right
        # as the sign is first spotted.
        self._pending_sign_turn        = 'none'
        self._pending_sign_turn_start  = 0.0

        # Sign FSM  ('none' | 'stopped' | 'yielding' | 'parked')
        self._sign_state          = 'none'
        self._sign_state_start    = 0.0
        self._sign_state_duration = 0.0
        self._sign_cooldowns      = {}  # type: Dict[int, float]

        # AprilTag bookkeeping
        self.apriltag_detections = []
        self.apriltag_error      = None
        self._last_apriltag_ids  = ()
        self._apriltag_detector  = None
        if self.apriltag_enabled:
            try:
                self._apriltag_detector = AprilTagDetector(self.apriltag_min_area)
            except Exception as exc:
                self.apriltag_error = str(exc)
                print("[AprilTag] Disabled: {}".format(exc))

        self.last_debug_info = self._empty_debug_info(480, 640)

    # ── Sign helpers ─────────────────────────────────────────────────────────

    def _arm_sign_turn(self, direction, now):
        """Remember a turn decision from a sign, to be carried out once the
        lane lines vanish at the upcoming intersection (see compute_commands).
        """
        already_pending = self._pending_sign_turn != 'none'
        turn_in_progress = (self._left_turn_state != 'none'
                             or self._right_turn_state != 'none')
        on_cooldown = (now < self._left_turn_cooldown_end
                       or now < self._right_turn_cooldown_end)
        if already_pending or turn_in_progress or on_cooldown:
            return
        self._pending_sign_turn       = direction
        self._pending_sign_turn_start = now
        self._yellow_visible_frames   = 0
        print("[Sign] Remembering decision: turn {} at the next intersection.".format(
            direction.upper()))

    def _on_sign_detected(self, tag_id, area, now):
        name = TAG_INSTRUCTIONS.get(tag_id, "TAG_{}".format(tag_id))
        print("\n" + "=" * 60)
        print("[Sign] Acting on: {} (tag {}, area {:.0f} px^2)".format(name, tag_id, area))
        print("=" * 60 + "\n")

        if _STOP_ON_ANY_SIGN:
            self._sign_state          = 'stopped'
            self._sign_state_start    = now
            self._sign_state_duration = _ANY_SIGN_STOP_DURATION
            print("[Sign] DETECTION-TEST MODE: stopping for {:.1f}s on tag {} ({}).".format(
                _ANY_SIGN_STOP_DURATION, tag_id, name))
            return

        if tag_id in _PERMANENT_STOP_TAGS:
            self._sign_state = 'parked'
            print("[Sign] Permanently stopped ({}).".format(name))
            return

        if tag_id in _TIMED_STOP_TAGS:
            duration = _TIMED_STOP_DURATIONS[tag_id]
            self._sign_state          = 'stopped'
            self._sign_state_start    = now
            self._sign_state_duration = duration
            print("[Sign] Stopping for {:.1f}s.".format(duration))
            return

        if tag_id in _YIELD_TAGS:
            self._sign_state          = 'yielding'
            self._sign_state_start    = now
            self._sign_state_duration = self.yield_duration
            print("[Sign] Yielding for {:.1f}s.".format(self.yield_duration))
            return

        if tag_id in _RANDOM_TURN_TAGS:
            choice = random.choice(['left', 'right'])
            print("[Sign] Tag {} allows either turn — randomly chose: {}".format(
                tag_id, choice.upper()))
            self._arm_sign_turn(choice, now)
            return

        if tag_id in _LEFT_TURN_TAGS:
            self._arm_sign_turn('left', now)
            return

        if tag_id in _RIGHT_TURN_TAGS:
            self._arm_sign_turn('right', now)
            return

        if tag_id in _STRAIGHT_TAGS:
            self._left_turn_state  = 'none'
            self._right_turn_state = 'none'
            self._pending_sign_turn = 'none'
            print("[Sign] Straight-only: turn FSMs suppressed.")

    def _check_sign_behavior(self, now):
        for tag in self.apriltag_detections:
            tag_id = tag.get('id', tag.get('tag_id', -1))
            area   = tag.get('area', 0.0)
            if area < self.apriltag_act_area:
                continue
            if now < self._sign_cooldowns.get(tag_id, 0.0):
                continue
            self._sign_cooldowns[tag_id] = now + self.apriltag_cooldown
            self._on_sign_detected(tag_id, area, now)

    # ── PD helpers ───────────────────────────────────────────────────────────

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.mean(yellow_xs))
            w_mean = float(np.mean(white_xs))
            if w_mean <= y_mean:
                error = w / 2.0 - (y_mean + self._lane_half_width)
                return float(np.clip(error / (w / 2.0), -1.0, 1.0))
            measured = (w_mean - y_mean) / 2.0
            if measured > 20:
                self._lane_half_width = 0.9 * self._lane_half_width + 0.1 * measured
            error = w / 2.0 - (y_mean + w_mean) / 2.0
        elif left_det and yellow_xs:
            error = w / 2.0 - (float(np.mean(yellow_xs)) + self._lane_half_width)
        elif right_det and white_xs:
            error = w / 2.0 - (float(np.mean(white_xs)) - self._lane_half_width)
        else:
            error = self._prev_error
        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error):
        error_diff       = error - self._prev_error
        self._prev_error = error
        steering = self.p_gain * error + self.d_gain * error_diff
        return float(np.clip(steering, -self.max_steer, self.max_steer))

    def _cruise_speed(self, steering, is_curve, both_visible):
        return self.base_speed

    def _apply_wheel_floor(self, left, right):
        min_val = min(left, right)
        if min_val < self.min_wheel_speed:
            shift  = self.min_wheel_speed - min_val
            left  += shift
            right += shift
        peak = max(left, right)
        if peak > 1.0:
            left  /= peak
            right /= peak
        return float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0))

    def _motor_commands(self, steering, recovery, both_visible, is_curve):
        speed = self._cruise_speed(steering, is_curve, both_visible)
        diff  = float(np.clip(steering, -self.max_steer, self.max_steer))
        return self._apply_wheel_floor(speed - diff, speed + diff)

    def _smooth(self, left, right):
        alpha = self.smooth_alpha
        if self._smooth_left is None:
            self._smooth_left  = left
            self._smooth_right = right
        else:
            self._smooth_left  = alpha * left  + (1.0 - alpha) * self._smooth_left
            self._smooth_right = alpha * right + (1.0 - alpha) * self._smooth_right
        return self._smooth_left, self._smooth_right

    # ── Main loop ────────────────────────────────────────────────────────────

    def compute_commands(self, image):
        self.frame_count += 1
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        now = time.monotonic()

        # ── AprilTag detection ────────────────────────────────────────────
        if (self._apriltag_detector is not None
                and self.frame_count % self.apriltag_interval == 0):
            try:
                detections = self._apriltag_detector.detect(bgr)
                self.apriltag_detections = [d.as_dict() for d in detections]
                self.apriltag_error = None
                detected_ids = tuple(d.tag_id for d in detections)
                if detected_ids != self._last_apriltag_ids:
                    if detected_ids:
                        ids_str = ", ".join(str(i) for i in detected_ids)
                        print("\n" + "=" * 60)
                        print("[AprilTag] DETECTED TAG ID(s): {}".format(ids_str))
                        for d in detections:
                            instr = TAG_INSTRUCTIONS.get(d.tag_id, "UNKNOWN")
                            print("[AprilTag] Tag {}: {} (area: {:.1f} px^2)".format(
                                d.tag_id, instr, d.area))
                        print("=" * 60 + "\n")
                    elif self._last_apriltag_ids:
                        print("[AprilTag] Tags no longer visible")
                    self._last_apriltag_ids = detected_ids
            except Exception as exc:
                self.apriltag_detections = []
                self.apriltag_error = str(exc)
                print("[AprilTag] Detection error: {}".format(exc))

        self.last_debug_info.update({
            'frame_count':    self.frame_count,
            'apriltags':      list(self.apriltag_detections),
            'apriltag_error': self.apriltag_error,
        })

        # ── Evaluate sign behaviours ──────────────────────────────────────
        self._check_sign_behavior(now)

        # ── Sign FSM: permanent stop ──────────────────────────────────────
        if self._sign_state == 'parked':
            self._smooth_left = self._smooth_right = 0.0
            self._filtered_steering = 0.0
            return 0.0, 0.0

        # ── Sign FSM: timed stop ──────────────────────────────────────────
        if self._sign_state == 'stopped':
            if now - self._sign_state_start < self._sign_state_duration:
                self._smooth_left = self._smooth_right = 0.0
                self._filtered_steering = 0.0
                return 0.0, 0.0
            print("[Sign] Stop complete — resuming.")
            self._sign_state = 'none'

        # ── Lane detection ────────────────────────────────────────────────
        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print("[Agent] detect_lane_markings error: {}".format(e))
            return 0.0, 0.0

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        yellow_slice_count  = len(yellow_xs)
        white_slice_count   = len(white_xs)

        combined     = np.clip(mask_left + mask_right, 0, 1)
        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)
        self.last_debug_info = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'red_mask':          np.zeros((h, w), dtype=np.uint8),
            'red_px':            0,
            'red_line':          False,
            'total_lane_pixels': total_pixels,
            'lateral_error':     float(np.clip(self._prev_error, -1.0, 1.0)),
            'lane_detected':     total_pixels >= self.detection_threshold,
            'frame_count':       self.frame_count,
            'yellow_xs':         yellow_xs,
            'white_xs':          white_xs,
            'slice_ys':          [start_y + i * slice_height + slice_height // 2
                                   for i in range(_NUM_SLICES)],
            'is_curve':          False,
            'curve_dir':         0,
            'apriltags':         list(self.apriltag_detections),
            'apriltag_error':    self.apriltag_error,
            'sign_state':        self._sign_state,
            'pending_sign_turn': self._pending_sign_turn,
        }

        # ── Sign FSM: yield ───────────────────────────────────────────────
        yielding = False
        if self._sign_state == 'yielding':
            if now - self._sign_state_start < self._sign_state_duration:
                yielding = True
            else:
                print("[Sign] Yield complete — resuming normal speed.")
                self._sign_state = 'none'

        # ── Yellow-end tracker → arms turns at intersections ───────────────
        _YELLOW_MIN_FRAMES   = 8
        _PENDING_MIN_DWELL_S = 0.3  # ignore line dropout for this long after sign seen

        if yellow_slice_count > 0:
            self._yellow_visible_frames = min(self._yellow_visible_frames + 1, 999)
        else:
            lines_gone_long_enough = self._yellow_visible_frames >= _YELLOW_MIN_FRAMES

            if self._pending_sign_turn != 'none':
                # A sign (9/10/11) already told us which way to go. Wait for
                # the lines to actually vanish — i.e. we've reached the
                # intersection — before committing to that remembered turn.
                dwell_ok = (now - self._pending_sign_turn_start) >= _PENDING_MIN_DWELL_S
                if (lines_gone_long_enough and dwell_ok
                        and self._left_turn_state == 'none'
                        and self._right_turn_state == 'none'):
                    direction = self._pending_sign_turn
                    self._pending_sign_turn = 'none'
                    if direction == 'left':
                        self._left_turn_state = 'straight'
                        self._left_turn_start = now
                        print("[Agent] Lines gone — carrying out remembered LEFT turn")
                    else:
                        self._right_turn_state = 'straight'
                        self._right_turn_start = now
                        print("[Agent] Lines gone — carrying out remembered RIGHT turn")
            else:
                # No sign decision pending — fall back to the unrelated
                # default behaviour (always turn left when lines vanish).
                if (lines_gone_long_enough
                        and self._left_turn_state == 'none'
                        and self._right_turn_state == 'none'
                        and now >= self._left_turn_cooldown_end):
                    self._left_turn_state = 'straight'
                    self._left_turn_start = now
                    print("[Agent] Yellow gone — left turn: driving straight")

            self._yellow_visible_frames = 0

        # ── Left-turn FSM ─────────────────────────────────────────────────
        if self._left_turn_state == 'straight':
            elapsed = now - self._left_turn_start
            if elapsed < self._left_straight_duration:
                s = self._left_straight_speed
                return s, s
            self._left_turn_state = 'turning'
            self._left_turn_start = now
            print("[Agent] Left turn: now turning")

        if self._left_turn_state == 'turning':
            elapsed          = now - self._left_turn_start
            white_reappeared = white_slice_count >= 2
            timed_out        = elapsed >= self._left_turn_max_duration
            if white_reappeared or timed_out:
                self._left_turn_state        = 'none'
                self._left_turn_cooldown_end = now + 3.0
                self._yellow_visible_frames  = 0
                reason = "white reappeared" if white_reappeared else "timeout"
                print("[Agent] Left turn done ({}) — resuming".format(reason))
            else:
                return self._left_turn_wheel_inner, self._left_turn_wheel_outer

        # ── Right-turn FSM ────────────────────────────────────────────────
        if self._right_turn_state == 'straight':
            elapsed = now - self._right_turn_start
            if elapsed < self._right_straight_duration:
                s = self._right_straight_speed
                return s, s
            self._right_turn_state = 'turning'
            self._right_turn_start = now
            print("[Agent] Right turn: now turning")

        if self._right_turn_state == 'turning':
            elapsed           = now - self._right_turn_start
            yellow_reappeared = yellow_slice_count >= 2
            timed_out         = elapsed >= self._right_turn_max_duration
            if yellow_reappeared or timed_out:
                self._right_turn_state        = 'none'
                self._right_turn_cooldown_end = now + 3.0
                self._yellow_visible_frames   = 0
                reason = "yellow reappeared" if yellow_reappeared else "timeout"
                print("[Agent] Right turn done ({}) — resuming".format(reason))
            else:
                return self._right_turn_wheel_inner, self._right_turn_wheel_outer

        # ── Normal lane-following PD ──────────────────────────────────────
        recovery = total_pixels < self.detection_threshold

        white_on_wrong_side = (
            left_det and right_det
            and yellow_xs and white_xs
            and float(np.mean(white_xs)) <= float(np.mean(yellow_xs))
        )
        effective_right_det = right_det and not white_on_wrong_side
        both_visible        = left_det and effective_right_det and not recovery

        is_curve, curve_dir = detect_curve(
            yellow_xs,
            white_xs if not white_on_wrong_side else [],
            self.curve_threshold,
        )

        raw_error            = self._calculate_error(yellow_xs, white_xs,
                                                     left_det, right_det, w)
        self._filtered_error = 0.7 * self._filtered_error + 0.3 * raw_error
        steering             = self._calculate_steering(self._filtered_error)
        if is_curve:
            steering -= curve_dir * self.curve_feedforward
        steering = float(np.clip(steering, -self.max_steer, self.max_steer))
        self._filtered_steering = (
            self.steer_smooth * steering
            + (1.0 - self.steer_smooth) * self._filtered_steering
        )

        left, right = self._motor_commands(
            self._filtered_steering, recovery, both_visible, is_curve,
        )

        # Yield: scale wheel speeds down
        if yielding:
            peak = max(left, right, 1e-6)
            if peak > self.yield_speed:
                scale = self.yield_speed / peak
                left  *= scale
                right *= scale

        left, right = self._smooth(left, right)

        self.last_debug_info.update({
            'is_curve':      is_curve,
            'curve_dir':     curve_dir,
            'lateral_error': self._filtered_error,
            'lane_detected': not recovery,
        })

        return left, right

    def step(self, image, wheels_driver):
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def reset(self):
        self.frame_count             = 0
        self._prev_error             = 0.0
        self._filtered_error         = 0.0
        self._filtered_steering      = 0.0
        self._smooth_left            = None
        self._smooth_right           = None
        self._lane_half_width        = float(_LINE_OFFSET)
        self._yellow_visible_frames  = 0
        self._left_turn_state        = 'none'
        self._left_turn_start        = 0.0
        self._left_turn_cooldown_end = 0.0
        self._right_turn_state        = 'none'
        self._right_turn_start        = 0.0
        self._right_turn_cooldown_end = 0.0
        self._pending_sign_turn        = 'none'
        self._pending_sign_turn_start  = 0.0
        self._sign_state             = 'none'
        self._sign_state_start       = 0.0
        self._sign_state_duration    = 0.0
        self._sign_cooldowns         = {}
        self.apriltag_detections     = []
        self.apriltag_error          = None
        self._last_apriltag_ids      = ()
        print("[Agent] State reset")

    def get_debug_info(self, image):
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            'roi':               np.zeros((h, w, 3), dtype=np.uint8),
            'lane_mask':         np.zeros((h, w),    dtype=np.uint8),
            'white_mask':        np.zeros((h, w),    dtype=np.uint8),
            'yellow_mask':       np.zeros((h, w),    dtype=np.uint8),
            'red_mask':          np.zeros((h, w),    dtype=np.uint8),
            'red_px':            0,
            'red_line':          False,
            'total_lane_pixels': 0,
            'lateral_error':     0.0,
            'lane_detected':     False,
            'frame_count':       0,
            'apriltags':         [],
            'apriltag_error':    None,
            'sign_state':        'none',
            'pending_sign_turn': 'none',
        }