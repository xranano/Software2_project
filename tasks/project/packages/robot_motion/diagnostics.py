"""
Intersection troubleshooting — run without Godot:

    python -m tasks.project.packages.robot_motion.diagnostics

With sim log comparison after a run:

    python -m tasks.project.packages.robot_motion.diagnostics --compare drive_log.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from tasks.project.packages.robot_motion.config import (
    END_CONDITIONS,
    INTERSECTION_RESET_POSE,
    LANE_CONTROLLER,
    OPEN_LOOP_YAW_DAMPING,
    RESET_Y_BY_TRAJECTORY,
    YAW_DAMPING_FACTOR,
)
from tasks.project.packages.robot_motion.intersection_drive import (
    CSV_HEADER,
    format_drive_csv_row,
    simulate_maneuver,
)
from tasks.project.packages.robot_motion.trajectory import (
    has_turn_section,
    load_all_trajectories,
    straight_distance_from_reset,
    turn_start_index,
)
from tasks.project.packages.robot_motion.virtual_lane import _wrap_phi

_TRAJ_DIR = Path(__file__).resolve().parent / "trajectories"


class Check:
    def __init__(self, name: str):
        self.name = name
        self.ok = True
        self.detail: list[str] = []

    def fail(self, msg: str) -> None:
        self.ok = False
        self.detail.append(f"FAIL: {msg}")

    def note(self, msg: str) -> None:
        self.detail.append(msg)


def _section(title: str) -> None:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def check_config() -> Check:
    c = Check("config")
    c.note(f"INTERSECTION_RESET_POSE = {INTERSECTION_RESET_POSE}")
    c.note(f"RESET_Y_BY_TRAJECTORY = {RESET_Y_BY_TRAJECTORY}")
    c.note(f"END_CONDITIONS = {END_CONDITIONS}")
    c.note(f"YAW_DAMPING_FACTOR = {YAW_DAMPING_FACTOR}")
    c.note(f"OPEN_LOOP_YAW_DAMPING = {OPEN_LOOP_YAW_DAMPING}")
    c.note(f"LANE_CONTROLLER = {LANE_CONTROLLER}")
    if LANE_CONTROLLER["wheel_base"] != 0.103:
        c.note(f"wheel_base={LANE_CONTROLLER['wheel_base']} (Godot modcon_config uses 0.1 — synced at runtime)")
    return c


def check_yaml_geometry(trajs: dict) -> Check:
    c = Check("yaml_geometry")
    for name, traj in trajs.items():
        idx = turn_start_index(traj)
        reset_y = RESET_Y_BY_TRAJECTORY.get(name, INTERSECTION_RESET_POSE["y"])
        straight_m = straight_distance_from_reset(traj, reset_y)
        track = traj["track"]
        c.note(
            f"{name}: points={len(track)} turn_idx={idx} "
            f"turn_y={track[idx, 1]:.3f} straight_from_reset={straight_m:.3f}m "
            f"has_turn={has_turn_section(traj)} exit_yaw={math.degrees(traj['tangent'][-1]):.1f}°"
        )
        if name == "left" and straight_m < 0.15:
            c.fail(f"left straight segment too short ({straight_m:.3f}m)")
        if name == "right" and straight_m < 0.05:
            c.note(
                f"WARN: right straight only {straight_m:.3f}m with reset_y={reset_y} "
                f"(arc at y={track[idx, 1]:.3f}) — tight by design; use ROBOT_SKIP_TO_DRIVE to test"
            )
    return c


def check_open_loop_sim(trajs: dict, name: str) -> Check:
    c = Check(f"open_loop_sim_{name}")
    ec = END_CONDITIONS[name]
    reset_y = RESET_Y_BY_TRAJECTORY.get(name, INTERSECTION_RESET_POSE["y"])
    hist = simulate_maneuver(
        name,
        trajs[name],
        ec["distance"],
        ec["angle_deg"],
        reset_y=reset_y,
        yaw_damping_factor=OPEN_LOOP_YAW_DAMPING,
    )
    if not hist:
        c.fail("no ticks produced")
        return c

    last = hist[-1]
    c.note(f"ticks={len(hist)} duration={len(hist) * 0.05:.2f}s done={last.done}")
    c.note(
        f"final pose=({last.x:.3f},{last.y:.3f},{math.degrees(last.yaw):.1f}°) "
        f"dist_end={last.dist_end:.3f} ang_end={last.ang_end_deg:.1f}°"
    )

    # Milestone checks
    milestones = {0.0: None, 0.85: None, 0.90: None}
    for i, out in enumerate(hist):
        t = i * 0.05
        for key in list(milestones):
            if milestones[key] is None and t >= key:
                milestones[key] = (t, out)

    for t_key, data in milestones.items():
        if data is None:
            continue
        t, out = data
        c.note(
            f"  t={t:.2f}s y={out.y:.3f} idx={out.path_index} "
            f"curv={out.curv:.2f} omega={out.omega_cmd:.3f}"
        )

    if name == "left":
        if not last.done:
            c.fail("left maneuver did not complete")
        m90 = milestones.get(0.90)
        if m90 and abs(m90[1].curv) < 0.5:
            c.fail(f"curv still ~0 at t=0.90 (y={m90[1].y:.3f}) — path indexing or reset wrong")
        m85 = milestones.get(0.85)
        if m85 and abs(m85[1].curv) > 0.5:
            c.fail(f"curv nonzero at t=0.85 (y={m85[1].y:.3f}) — turning before stop line")

    if name == "right":
        if not last.done:
            c.fail("right maneuver did not complete")
        m10 = next(((i * 0.05, h) for i, h in enumerate(hist) if i * 0.05 >= 0.10), None)
        if m10 and abs(m10[1].curv) < 0.5:
            c.note(
                f"right: curv still 0 at t=0.10 (y={m10[1].y:.3f}) — "
                f"expected with reset_y={reset_y} (arc at y~-0.10)"
            )

    max_phi = max(abs(math.degrees(h.phi)) for h in hist)
    if max_phi > 45:
        c.fail(f"max |phi|={max_phi:.1f}° — phi wrap or yaw damping issue")
    else:
        c.note(f"max |phi|={max_phi:.1f}° OK")

    return c


def check_phi_wrap() -> Check:
    c = Check("phi_wrap")
    tests = [(math.pi / 2, math.pi, -math.pi / 2), (0.1, -3.0, _wrap_phi(0.1 - (-3.0)))]
    for yaw, tangent, expected in tests:
        got = _wrap_phi(yaw - tangent)
        if abs(got - expected) > 1e-6 and abs(got - expected - 2 * math.pi) > 1e-6:
            c.fail(f"wrap({yaw}-{tangent})={got}, expected {expected}")
    if c.ok:
        c.note("phi wrap OK")
    return c


def export_reference_csv(trajs: dict, out_path: Path) -> None:
    lines = [CSV_HEADER]
    for name in ("left", "right", "straight"):
        ec = END_CONDITIONS[name]
        reset_y = RESET_Y_BY_TRAJECTORY.get(name, INTERSECTION_RESET_POSE["y"])
        hist = simulate_maneuver(
            name,
            trajs[name],
            ec["distance"],
            ec["angle_deg"],
            reset_y=reset_y,
            yaw_damping_factor=OPEN_LOOP_YAW_DAMPING,
        )
        for i, out in enumerate(hist):
            lines.append(format_drive_csv_row(i * 0.05, name, out))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote reference CSV: {out_path}")


def compare_csv(path: Path) -> Check:
    c = Check(f"compare_csv({path.name})")
    if not path.exists():
        c.fail(f"file not found: {path}")
        return c
    rows = path.read_text(encoding="utf-8").strip().splitlines()
    if len(rows) < 2:
        c.fail("CSV empty")
        return c
    c.note(f"rows={len(rows) - 1}")
    c.note("First 3 data rows:")
    for line in rows[1:4]:
        c.note(f"  {line}")
    c.note("...")
    c.note("Last row:")
    c.note(f"  {rows[-1]}")
    return c


def print_troubleshooting_guide() -> None:
    print(
        """
TROUBLESHOOTING DECISION TREE
---------------------------
Run layers top → bottom. Stop at first FAIL that matches your symptom.

LAYER 0 — Pure Python (no Godot)
  python -m tasks.project.packages.robot_motion.diagnostics
  All checks must PASS. If FAIL here, bug is in YAML/config/math — not sim.

LAYER 1 — Motor sanity (Godot required)
  $env:ROBOT_MOTION_DEBUG="1"
  $env:ROBOT_DEBUG_OPEN_LOOP="1"
  $env:ROBOT_TRAJECTORY="left"
  python launch.py --sim --task project
  After stop+wait: robot must turn hard left 3s.
    Hard left     → motors OK, go to layer 2
    Barely moves  → set_twist / Godot wheel port / v_max
    Wrong way     → omega sign flip or frame mismatch

LAYER 2 — Skip lane follow (Godot required)
  $env:ROBOT_SKIP_TO_DRIVE="1"
  $env:ROBOT_MOTION_DEBUG="1"
  $env:ROBOT_TRAJECTORY="left"
  python launch.py --sim --task project
  Jumps straight to INTERSECTION_DRIVE. No stop line needed.
    Works → stop line / wait / begin_drive is the bug
    Fails   → path tracking vs Godot (layer 3)

LAYER 3 — Full pipeline log
  $env:ROBOT_MOTION_DEBUG="1"
  python launch.py --sim --task project
  Compare DRIVE lines to reference CSV:
    python -m tasks.project.packages.robot_motion.diagnostics --export reference_drive.csv
  Log file (auto): tasks/project/packages/robot_motion/drive_log.csv

LAYER 4 — Log vs video mismatch
  If DRIVE log looks healthy but Godot video wrong:
    Check v_cmd vs v_act and GODOT_NOT_MOVING warnings
    Check game distance: GET /status → distance_traveled
    Virtual pose uses executed PWM odom during INTERSECTION_DRIVE (Godot poll optional)

ENV VARS (all optional)
  ROBOT_MOTION_DEBUG=1       — DRIVE log every 50ms + drive_log.csv
  ROBOT_DEBUG_OPEN_LOOP=1    — 3s raw set_twist(0.23, 1.5) after wait
  ROBOT_SKIP_TO_DRIVE=1        — skip lane follow, begin_drive immediately
  ROBOT_TRAJECTORY=left|right|straight
  ROBOT_USE_FORWARD_SEARCH=1   — forward path window (default: global/proj-lfi)

UI CHECKLIST
  [ ] Click Start (or Reset — auto-starts)
  [ ] Set trajectory to left/right (not straight)
  [ ] Video shows INTERSECTION_WAIT then INTERSECTION_DRIVE overlay
  [ ] Godot sim connected (wheels port 5002, camera 5001)

FILES TO INSPECT
  config.py              — END_CONDITIONS, RESET_Y, gains
  trajectories/*.yaml      — path geometry (proj-lfi)
  main_loop.py             — FSM + drive loop
  intersection_drive.py    — compute_drive_commands + open-loop sim
  hardware.py              — set_twist, stopline, PWM readback
  stopline_detector.py     — ROI_Y_START, RED_THRESHOLD, CONFIRM_FRAMES
  intersection_controller.py — FSM phases
  virtual_lane.py          — d, phi, curv, done
  lane_controller.py       — PID + feedforward
  godot_wheels_driver.py   — baseline=0.1, set_velocity scaling
"""
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Intersection diagnostics")
    parser.add_argument("--export", type=Path, help="Write reference open-loop CSV")
    parser.add_argument("--compare", type=Path, help="Show contents of a drive log CSV")
    parser.add_argument("--guide", action="store_true", help="Print full troubleshooting guide")
    args = parser.parse_args(argv)

    if args.guide:
        print_troubleshooting_guide()
        return 0

    trajs = load_all_trajectories(_TRAJ_DIR)
    checks: list[Check] = [
        check_config(),
        check_yaml_geometry(trajs),
        check_phi_wrap(),
        check_open_loop_sim(trajs, "left"),
        check_open_loop_sim(trajs, "right"),
        check_open_loop_sim(trajs, "straight"),
    ]

    if args.export:
        export_reference_csv(trajs, args.export)
    if args.compare:
        checks.append(compare_csv(args.compare))

    _section("INTERSECTION DIAGNOSTICS")
    failed = 0
    for chk in checks:
        status = "PASS" if chk.ok else "FAIL"
        print(f"\n[{status}] {chk.name}")
        for line in chk.detail:
            print(f"  {line}")
        if not chk.ok:
            failed += 1

    print(f"\n{'=' * 60}")
    if failed:
        print(f"RESULT: {failed} check(s) FAILED — fix these before tuning Godot")
        print("Run with --guide for full troubleshooting tree")
        return 1
    print("RESULT: all checks PASSED — open-loop stack is OK")
    print("If sim still fails: run --guide layer 1-4 (Godot / stopline / pose sync)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
