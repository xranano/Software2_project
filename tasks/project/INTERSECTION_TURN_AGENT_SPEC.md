# Agent Task: Replicate proj-lfi Intersection Turn Control

## What you are implementing

proj-lfi intersection turns work by **path tracking**, not "rotate 90°".

```
pose (x,y,yaw)  →  virtual_lane  →  (d, phi, curvature)  →  lane_controller  →  wheel speeds
```

Your job: replicate `virtual_lane_node` + FSM + a `lane_controller` equivalent. Wire pose from encoders or camera. Copy trajectory YAMLs from proj-lfi.

## Implementation in this repo

| Spec component | File |
|----------------|------|
| Trajectory YAMLs | `packages/robot_motion/trajectories/{straight,left,right1}.yaml` |
| `compute_lane_pose` | `packages/robot_motion/virtual_lane.py` |
| `LaneController` | `packages/robot_motion/lane_controller.py` |
| FSM + pose reset | `packages/robot_motion/intersection_controller.py` |
| Main 20 Hz loop | `packages/robot_motion/main_loop.py` |
| Constants | `packages/robot_motion/config.py` |
| Hardware hooks | `packages/robot_motion/hardware.py` |
| Odometry | `packages/robot_motion/odometry.py` |

## Constants (from proj-lfi configs)

```python
END_CONDITIONS = {
    "straight": {"distance": 0.2, "angle_deg": 30},
    "left":     {"distance": 0.5, "angle_deg": 30},
    "right":    {"distance": 0.4, "angle_deg": 45},
}

RESET_POSE = {"x": 0.0, "y": -0.20, "yaw": 1.5707963267948966}
STOP_TIME_S = 2.0
CONTROL_DT = 0.05

LANE_CONTROLLER = {
    "v_bar": 0.23, "k_d": -3.5, "k_theta": -1.0, "k_Id": 1.0,
    "wheel_base": 0.103, "omega_max": 4.0, "use_feedforward": True,
}
```

## FSM

```
LANE_FOLLOW
  → at_stopline: stop motors, WAIT 2s, reset pose
  → INTERSECTION_DRIVE

INTERSECTION_DRIVE (loop at 20Hz):
  pose ← integrate_pose(v, omega)   # open-loop; vision optional
  (d,phi,curv,done) ← compute_lane_pose(pose, active_trajectory)
  (vl,vr) ← lane_controller(d, phi, curv)
  set_wheels(vl, vr)
  if done: stop, return to LANE_FOLLOW
```

## Acceptance

- Left turn follows arc, exits with yaw ≈ π
- Right turn follows arc, exits with yaw ≈ 0
- Straight stays on x≈0, curv=0
- Smooth arc motion, not pivot turns
