import sys

import os

import threading



script_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.join(script_dir, '..', '..')

sys.path.insert(0, project_root)



from flask import Flask, Response, jsonify, request

import cv2

import numpy as np



from duckiebot.camera_driver.godot_camera_driver import GodotCameraDriver, GodotCameraConfig

from duckiebot.wheel_driver.godot_wheels_driver import GodotWheelsDriver

from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration

from launcher.ports import find_available_port

from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

from servers.project.lane_visualization import create_lane_visualization, create_virtual_lane_overlay

from servers.templates.project import get_template



import tasks.project.packages.agent as agent

from tasks.project.packages.robot_motion import hardware

from tasks.project.packages.robot_motion.config import DEFAULT_TRAJECTORY



app = Flask(__name__)



camera = None

wheels = None

stop_event = threading.Event()

agent_thread = None





def _uses_camera_lane_follow(motion: dict) -> bool:
    return motion.get("state") in ("LANE_FOLLOW", "INTERSECTION_WAIT")





def visualize(frame_rgb: np.ndarray) -> np.ndarray:

    """Build the debug view shown in the web UI video stream."""

    motion = hardware.get_ui_data()

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)



    if _uses_camera_lane_follow(motion):

        debug_info, pwm_left, pwm_right = hardware.get_lane_debug()

        if not debug_info:

            debug_info, pwm_left, pwm_right = hardware.analyze_lane_frame(frame_rgb)

        vis = create_lane_visualization(bgr, debug_info, pwm_left, pwm_right, motion)
        if motion.get("state") == "INTERSECTION_WAIT":
            rem = motion.get("wait_remaining", 0.0)
            cv2.putText(vis, f"STOPPED ({rem:.1f}s)", (10, vis.shape[0] - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return vis

    if motion.get("state") == "INTERSECTION_DRIVE":
        return create_virtual_lane_overlay(bgr, motion)



    # Waiting / paused — still show lane detection so the user can see the lines.

    debug_info, pwm_left, pwm_right = hardware.analyze_lane_frame(frame_rgb)

    vis = create_lane_visualization(bgr, debug_info, pwm_left, pwm_right, motion)

    if not motion.get("running", True):

        cv2.putText(vis, "PAUSED", (10, vis.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return vis





generate_frames = make_frame_generator(lambda: camera, visualize, quality=50)





@app.route('/')

def index():

    return get_template(title='Project', subtitle='Simulation — lane debug overlay')





@app.route('/video')

def video():

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





@app.route('/status')

def status():

    data = hardware.get_ui_data()

    debug, pwm_l, pwm_r = hardware.get_lane_debug()

    data["lane_detected"] = debug.get("lane_detected", False)

    data["lateral_error"] = debug.get("lateral_error", 0.0)

    data["pwm_left"] = pwm_l

    data["pwm_right"] = pwm_r

    data["diagnostics"] = hardware.get_diagnostics_snapshot()

    if wheels is not None:

        data['game_over'] = wheels.is_game_over()

    return jsonify(data)





@app.route('/start', methods=['POST'])

def start():

    hardware.set_running(True)

    return jsonify({'status': 'ok', 'running': True})





@app.route('/stop', methods=['POST'])

def stop():

    hardware.set_running(False)

    return jsonify({'status': 'ok', 'running': False})





@app.route('/reset', methods=['POST'])

def reset():

    trajectory = DEFAULT_TRAJECTORY

    if request.json:

        trajectory = request.json.get('trajectory', trajectory)

    if wheels is not None:

        wheels.reset_game()

    hardware.reset_motion(trajectory=trajectory)

    hardware.set_running(True)

    return jsonify({'status': 'ok', 'trajectory': trajectory})





@app.route('/command', methods=['POST'])

def command():

    data = request.json or {}

    key = str(data.get('key', '')).strip().lower()

    value = str(data.get('value', '')).strip().lower()



    if key == 'trajectory' and value in ('straight', 'left', 'right'):

        hardware.set_trajectory(value)

        return jsonify({'status': 'ok', 'trajectory': value})



    return jsonify({'status': 'error', 'message': f'Unknown command: {key}={value}'}), 400





@app.route('/shutdown')

def shutdown():

    shutdown_cleanup(wheels, camera, stop_event)

    return jsonify({'status': 'ok'})





def main():

    global camera, wheels, agent_thread



    import argparse

    ap = argparse.ArgumentParser(description='Project Virtual Server')

    ap.add_argument('--port', type=int, default=5000)

    ap.add_argument('--frame-port', type=int, default=5001)

    ap.add_argument('--wheel-port', type=int, default=5002)

    ap.add_argument('--godot-host', type=str, default='localhost')

    args = ap.parse_args()



    suppress_http_logs()

    print('=' * 60)

    print('PROJECT VIRTUAL SERVER — SIMULATION')

    print('=' * 60)



    print('\n[1/4] Initializing wheels driver...')

    wheels = GodotWheelsDriver(

        WheelPWMConfiguration(pwm_min=0),

        WheelPWMConfiguration(pwm_min=0),

        godot_host=args.godot_host,

        godot_port=args.wheel_port,

    )

    wheels.trim = 0

    print(f'  Wheels: {args.godot_host}:{args.wheel_port}')



    print('\n[2/4] Initializing camera driver...')

    print(f'  Waiting for Godot on port {args.frame_port}...')

    camera = GodotCameraDriver(

        godot_config=GodotCameraConfig(host='0.0.0.0', port=args.frame_port)

    )

    camera.start()

    print('  Camera: connected')



    print('\n[3/4] Starting agent thread...')

    stop_event.clear()

    agent_thread = threading.Thread(

        target=agent.main,

        args=(camera, wheels, None, stop_event),

        daemon=True,

        name='ProjectAgentThread',

    )

    agent_thread.start()

    hardware.set_running(True)

    print('  Agent running (auto-started)')



    web_port = find_available_port(args.port)

    if web_port != args.port:

        print(f'  Port {args.port} busy, using {web_port}')



    print('\n[4/4] Web server ready')

    print('=' * 60)

    print(f'Web Interface: http://localhost:{web_port}')

    print('Video stream shows yellow/white lane detection + lateral error.')

    print('=' * 60 + '\n')



    try:

        app.run(host='127.0.0.1', port=web_port, debug=False, threaded=True)

    except KeyboardInterrupt:

        print('\nShutting down...')

    finally:

        shutdown_cleanup(wheels, camera, stop_event)





if __name__ == '__main__':

    sys.exit(main())


