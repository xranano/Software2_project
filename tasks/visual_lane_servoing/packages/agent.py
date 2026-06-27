# agent.py
import os
import time
import yaml
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve
from tasks.sign_detection.packages.sign_behavior import SignBehaviorFSM

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_config.yaml'
))

_LINE_OFFSET = 160
_ROI_START   = 0.47
_NUM_SLICES  = 3
_SLICE_TOL   = 5


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

        self.frame_count         = 0
        self._prev_error         = 0.0
        self._filtered_error     = 0.0
        self._filtered_steering  = 0.0
        self._smooth_left        = None
        self._smooth_right       = None
        self._lane_half_width    = float(_LINE_OFFSET)

        self._sign_fsm = SignBehaviorFSM.from_dict(cfg)
        self.apriltag_error = None

        self.last_debug_info = self._empty_debug_info(480, 640)

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

    def _motor_commands(self, steering, recovery, both_visible, is_curve):
        speed = self._cruise_speed(steering, is_curve, both_visible)
        diff  = float(np.clip(steering, -self.max_steer, self.max_steer))
        left  = speed - diff
        right = speed + diff
        if left >= 0 and right >= 0:
            min_val = min(left, right)
            if min_val < self.min_wheel_speed:
                shift  = self.min_wheel_speed - min_val
                left  += shift
                right += shift
        peak = max(abs(left), abs(right))
        if peak > 1.0:
            left  /= peak
            right /= peak
        return float(np.clip(left, -1.0, 1.0)), float(np.clip(right, -1.0, 1.0))

    def _smooth(self, left, right):
        alpha = self.smooth_alpha
        if self._smooth_left is None:
            self._smooth_left  = left
            self._smooth_right = right
        else:
            self._smooth_left  = alpha * left  + (1.0 - alpha) * self._smooth_left
            self._smooth_right = alpha * right + (1.0 - alpha) * self._smooth_right
        return self._smooth_left, self._smooth_right

    def _compute_lane_pwm(self, bgr, image):
        """Lane-following PD only — no sign behavior."""
        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print("[Agent] detect_lane_markings error: {}".format(e))
            return 0.0, 0.0, None

        mask_y = (mask_left  * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels  = int(np.count_nonzero(mask_w))
        total_pixels  = yellow_pixels + white_pixels

        h, w      = mask_y.shape
        left_det  = yellow_pixels > 0
        right_det = white_pixels  > 0

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        combined     = np.clip(mask_left + mask_right, 0, 1)
        slice_height = int(h * 0.35 / _NUM_SLICES)
        start_y      = int(h * _ROI_START)

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
        left, right = self._smooth(left, right)

        lane_debug = {
            'roi':               image,
            'lane_mask':         (combined * 255).astype(np.uint8),
            'white_mask':        mask_w,
            'yellow_mask':       mask_y,
            'total_lane_pixels': total_pixels,
            'lateral_error':     self._filtered_error,
            'lane_detected':     not recovery,
            'yellow_xs':         yellow_xs,
            'white_xs':          white_xs,
            'slice_ys':          [start_y + i * slice_height + slice_height // 2
                                   for i in range(_NUM_SLICES)],
            'is_curve':          is_curve,
            'curve_dir':         curve_dir,
        }
        return left, right, lane_debug

    def compute_commands(self, image, detections=None):
        self.frame_count += 1

        base_left, base_right, lane_debug = self._compute_lane_pwm(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR), image,
        )
        if lane_debug is None:
            return 0.0, 0.0

        sign_step = self._sign_fsm.step(
            image, base_left, base_right, detections,
        )
        if isinstance(sign_step, tuple) and len(sign_step) >= 2:
            final_left, final_right = sign_step[0], sign_step[1]
        else:
            final_left, final_right = base_left, base_right

        sign_debug = self._sign_fsm.debug
        red_mask = sign_debug.get('red_mask')
        if red_mask is None:
            h = image.shape[0]
            w = image.shape[1]
            red_mask = np.zeros((h, w), dtype=np.uint8)

        apriltags = sign_debug.get('raw_tags', [])
        for tag in apriltags:
            tag['id'] = tag.get('id', tag.get('tag_id'))

        self.last_debug_info = {
            **lane_debug,
            'frame_count':         self.frame_count,
            'red_mask':            red_mask,
            'red_px':              int(np.count_nonzero(red_mask)),
            'red_line':            sign_debug.get('red_line', False),
            'apriltags':           apriltags,
            'apriltag_error':      self.apriltag_error,
            'sign_state':          sign_debug.get('state', 'MOVING'),
            'pending_sign_action': sign_debug.get('saved_tag'),
            'pending_sign_name':   sign_debug.get('saved_tag'),
            'confirmed_tags':      sign_debug.get('confirmed_tags', []),
            'saved_tag':           sign_debug.get('saved_tag'),
            'chosen_turn':         sign_debug.get('chosen_turn'),
            'sign_debug':          sign_debug,
        }

        return final_left, final_right

    @property
    def sign_state(self):
        return self._sign_fsm.state_name

    @property
    def sign_debug(self):
        return self._sign_fsm.debug

    @property
    def apriltag_detections(self):
        return self._sign_fsm.debug.get('raw_tags', [])

    def step(self, image, wheels_driver, detections=None):
        left, right = self.compute_commands(image, detections)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def reset(self):
        self.frame_count        = 0
        self._prev_error        = 0.0
        self._filtered_error    = 0.0
        self._filtered_steering = 0.0
        self._smooth_left       = None
        self._smooth_right      = None
        self._lane_half_width   = float(_LINE_OFFSET)
        self._sign_fsm.reset()
        self.apriltag_error     = None
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
            'sign_state':        'MOVING',
            'pending_sign_action': None,
            'confirmed_tags':    [],
            'saved_tag':         None,
            'chosen_turn':       None,
            'sign_debug':        {},
        }
