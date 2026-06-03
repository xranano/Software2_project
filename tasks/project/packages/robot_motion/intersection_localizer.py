"""
proj-lfi localization_node port — pose in intersection frame during INTERSECTION_DRIVE.

Predict: wheel odometry from executed (v, omega) + yaw damping (proj-lfi velocity_to_pose).
Correct (sim): Godot robot position relative to begin_drive anchor.
Correct (vision): red stopline row in camera → weak forward (y) nudge.
Fuse: weighted blend of available estimates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from tasks.project.packages.robot_motion.intersection_model import wrap_angle
from tasks.project.packages.robot_motion.odometry import apply_yaw_damping, integrate_pose
from tasks.project.packages.robot_motion.stopline_detector import red_fraction

logger = logging.getLogger(__name__)


@dataclass
class PoseEstimate:
    x: float
    y: float
    yaw: float
    source: str  # fused | odom | godot | stopline


@dataclass
class _Anchor:
    x: float
    y: float
    yaw: float
    godot_x: float | None = None
    godot_z: float | None = None
    godot_yaw: float | None = None
    odom_distance: float = 0.0


class IntersectionLocalizer:
    """Continuous pose estimate for virtual_lane_node (proj-lfi Layer 1)."""

    def __init__(
        self,
        yaw_damping_factor: float = 0.2,
        godot_weight: float = 0.65,
        stopline_weight: float = 0.25,
        odom_weight: float = 0.10,
    ):
        self.yaw_damping_factor = yaw_damping_factor
        self.godot_weight = godot_weight
        self.stopline_weight = stopline_weight
        self.odom_weight = odom_weight
        self._pose = PoseEstimate(0.0, 0.0, math.pi / 2, "odom")
        self._anchor: _Anchor | None = None
        self._last_source = "reset"

    def reset(
        self,
        x: float,
        y: float,
        yaw: float,
        *,
        godot_x: float | None = None,
        godot_z: float | None = None,
        godot_yaw: float | None = None,
        odom_distance: float = 0.0,
    ) -> None:
        self._pose = PoseEstimate(x, y, yaw, "reset")
        self._anchor = _Anchor(
            x=x,
            y=y,
            yaw=yaw,
            godot_x=godot_x,
            godot_z=godot_z,
            godot_yaw=godot_yaw,
            odom_distance=odom_distance,
        )
        self._last_source = "reset"
        logger.info(
            "Localizer reset pose=(%.2f, %.2f, %.2f) godot=(%s, %s, %s)",
            x,
            y,
            yaw,
            godot_x,
            godot_z,
            godot_yaw,
        )

    @property
    def pose(self) -> PoseEstimate:
        return self._pose

    @property
    def last_source(self) -> str:
        return self._last_source

    def update(
        self,
        dt: float,
        v_act: float,
        omega_act: float,
        frame_rgb: np.ndarray | None = None,
        *,
        godot_x: float | None = None,
        godot_z: float | None = None,
        godot_yaw: float | None = None,
        odom_distance: float | None = None,
    ) -> PoseEstimate:
        """Fuse odom prediction with Godot / stopline corrections."""
        if self._anchor is None:
            return self._pose

        px, py, pyaw = self._pose.x, self._pose.y, self._pose.yaw
        prev_yaw = pyaw
        ox, oy, oyaw = integrate_pose(px, py, pyaw, v_act, omega_act, dt)
        oyaw = apply_yaw_damping(prev_yaw, oyaw, omega_factor=self.yaw_damping_factor)
        odom = PoseEstimate(ox, oy, oyaw, "odom")

        estimates: list[tuple[PoseEstimate, float]] = [(odom, self.odom_weight)]
        sources: list[str] = ["odom"]

        godot = self._godot_pose(godot_x, godot_z, godot_yaw)
        if godot is not None:
            estimates.append((godot, self.godot_weight))
            sources.append("godot")

        if odom_distance is not None and self._anchor.odom_distance is not None:
            dy = odom_distance - self._anchor.odom_distance
            dist = PoseEstimate(
                self._anchor.x,
                self._anchor.y + dy,
                odom.yaw,
                "distance",
            )
            estimates.append((dist, 0.35 if godot is None else 0.15))
            sources.append("distance")

        stopline = self._stopline_pose(frame_rgb, odom.yaw)
        if stopline is not None:
            estimates.append((stopline, self.stopline_weight))
            sources.append("stopline")

        wx = sum(w * e.x for e, w in estimates)
        wy = sum(w * e.y for e, w in estimates)
        wyaw = sum(w * e.yaw for e, w in estimates)
        wsum = sum(w for _, w in estimates)
        fx = wx / wsum
        fy = wy / wsum
        fyaw = wrap_angle(wyaw / wsum)

        self._pose = PoseEstimate(fx, fy, fyaw, "+".join(sources))
        self._last_source = self._pose.source
        return self._pose

    def _godot_pose(
        self,
        godot_x: float | None,
        godot_z: float | None,
        godot_yaw: float | None,
    ) -> PoseEstimate | None:
        a = self._anchor
        if (
            a is None
            or a.godot_x is None
            or a.godot_z is None
            or a.godot_yaw is None
            or godot_x is None
            or godot_z is None
            or godot_yaw is None
        ):
            return None
        # Godot project map: robot drives along -Z; +X is right.
        dx = godot_x - a.godot_x
        dy = a.godot_z - godot_z
        dyaw = wrap_angle(godot_yaw - a.godot_yaw)
        return PoseEstimate(
            a.x + dx,
            a.y + dy,
            wrap_angle(a.yaw + dyaw),
            "godot",
        )

    def _stopline_pose(
        self, frame_rgb: np.ndarray | None, yaw: float
    ) -> PoseEstimate | None:
        """Weak y correction from red stopline band in camera (proj-lfi vision hint)."""
        if frame_rgb is None:
            return None
        frac = red_fraction(frame_rgb)
        if frac < 0.08:
            return None
        h = frame_rgb.shape[0]
        roi = frame_rgb[int(h * 0.75) :, :]
        if roi.size == 0:
            return None
        import cv2

        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255])) | cv2.inRange(
            hsv, np.array([160, 80, 80]), np.array([180, 255, 255])
        )
        ys, _xs = np.where(mask > 0)
        if len(ys) == 0:
            return None
        row_frac = 0.75 + float(np.mean(ys)) / h
        # Lower red band in image → stopline closer under axle (y → 0).
        y_hint = -0.25 + (row_frac - 0.82) * 1.2
        a = self._anchor
        if a is None:
            return None
        return PoseEstimate(a.x, y_hint, yaw, "stopline")
