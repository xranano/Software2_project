import sys
import os
import signal
import threading
import time
import queue
import socket

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..', '..')
sys.path.insert(0, project_root)

import cv2
from flask import Flask, Response, render_template_string, jsonify, request

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.object_detection.packages.agent import ObjectDetectionAgent, CLASS_NAMES
from tasks.object_detection.packages.stop_activity import should_stop as student_should_stop
from tasks.object_detection.packages.robot_detector import BlueRobotDetector
from servers.object_detection.visualization import draw_detections, draw_status_overlay
from servers.templates.object_detection import OBJECT_DETECTION_TEMPLATE as HTML_TEMPLATE

from duckiebot.camera_driver import CameraDriver
from duckiebot.wheel_driver import DaguWheelsDriver
from duckiebot.wheel_driver.wheels_driver_abs import WheelPWMConfiguration
from launcher.ports import find_available_port
from servers.common import make_frame_generator, shutdown_cleanup, suppress_http_logs

SIGN_ACTIVE_STATES = frozenset({
    "SLOWING", "STOPPED", "CHECKPATH", "POST_STOP",
    "APPROACHING", "INTERSECT", "PRE_TURN", "TURNING", "EXITING",
})


app        = Flask(__name__)
lane_agent = None
det_agent  = None
robot_detector = None
camera     = None
wheels     = None
running    = False
manual_mode = False
stop_event = threading.Event()

_frame_queue     = queue.Queue(maxsize=1)
_last_detections = []
_detection_lock  = threading.Lock()
_stopped_by_det  = False
_stop_reason     = ''
_robot_blue_px   = 0

keys_pressed      = {'up': False, 'down': False, 'left': False, 'right': False}
_keys_lock        = threading.Lock()
_keys_last_update = time.time()


def manual_control_loop():
    global _keys_last_update
    while not stop_event.is_set():
        if not manual_mode or not wheels:
            time.sleep(0.05)
            continue

        if time.time() - _keys_last_update > 0.5:
            with _keys_lock:
                for k in keys_pressed:
                    keys_pressed[k] = False

        with _keys_lock:
            kc = keys_pressed.copy()

        left = right = 0.0
        if kc['up']:
            left, right = 0.5, 0.5
        if kc['down']:
            left, right = -0.5, -0.5
        if kc['up'] and kc['left']:
            left, right = 0.2, 0.5
        elif kc['up'] and kc['right']:
            left, right = 0.5, 0.2
        elif kc['left']:
            left, right = -0.3, 0.3
        elif kc['right']:
            left, right = 0.3, -0.3

        wheels.set_wheels_speed(left, right)
        time.sleep(0.05)


def detection_loop():
    global _last_detections
    while not stop_event.is_set():
        if det_agent is None or not det_agent.model_loaded:
            time.sleep(0.1)
            continue
        try:
            frame_rgb = _frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        result = det_agent.detect(frame_rgb)
        if result is not None:
            with _detection_lock:
                _last_detections = result


def _should_stop(detections, frame_h: int):
    return student_should_stop(detections, frame_h)


def visualize(frame_bgr):
    global _stopped_by_det, _stop_reason, _robot_blue_px

    if wheels is None:
        return draw_status_overlay(frame_bgr, 'Initializing...')

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if det_agent is not None and det_agent.model_loaded:
        try:
            small = cv2.resize(frame_rgb, (det_agent.img_size, det_agent.img_size))
            _frame_queue.put_nowait(small)
        except queue.Full:
            pass

    with _detection_lock:
        detections = list(_last_detections)

    pwm_left = pwm_right = 0.0
    if manual_mode:
        _stopped_by_det = False
        _stop_reason    = ''
    elif lane_agent is not None:
        pwm_left, pwm_right = lane_agent.compute_commands(frame_rgb, detections=detections)

        sign_state = getattr(lane_agent, "sign_state", "MOVING")
        sign_active = sign_state in SIGN_ACTIVE_STATES
        frame_h = det_agent.img_size if det_agent else frame_bgr.shape[0]
        if sign_active:
            should_stop, reason = False, ""
        else:
            should_stop, reason = _should_stop(detections, frame_h)

            if robot_detector is not None:
                robot_ahead, _robot_blue_px = robot_detector.detect(frame_bgr)
                if robot_ahead and not should_stop:
                    should_stop = True
                    reason = "robot ahead (blue px={})".format(_robot_blue_px)
                if lane_agent.frame_count % 30 == 0:
                    print("[Robot] enabled={} blue_area={} threshold={} -> {}".format(
                        robot_detector.enabled, _robot_blue_px,
                        robot_detector.area_threshold,
                        "STOP" if robot_ahead else "clear"))

        _stopped_by_det = should_stop
        _stop_reason    = reason

        if running and not should_stop:
            wheels.set_wheels_speed(pwm_left, pwm_right)
        else:
            wheels.set_wheels_speed(0.0, 0.0)

    if det_agent is not None and det_agent.model_loaded and detections:
        oh, ow = frame_bgr.shape[:2]
        sx = ow / det_agent.img_size
        sy = oh / det_agent.img_size
        scaled = [((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)), s, c)
                  for (x1, y1, x2, y2), s, c in detections]
        draw_detections(frame_bgr, scaled)

    _draw_robot_overlay(frame_bgr)

    return frame_bgr


def _draw_robot_overlay(frame_bgr):
    """Draw the blue-detection ROI and the single largest blue blob for tuning."""
    if robot_detector is None:
        return
    x0, y0, x1, y1 = robot_detector.last_roi
    cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (160, 160, 160), 1)
    triggered = _stopped_by_det and 'robot' in _stop_reason
    color = (0, 0, 255) if triggered else (255, 200, 0)
    if robot_detector.last_bbox is not None:
        bx1, by1, bx2, by2 = robot_detector.last_bbox
        cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), color, 2)
    cv2.putText(
        frame_bgr,
        "robot blue area: {} / {}".format(robot_detector.last_blue_pixels,
                                          robot_detector.area_threshold),
        (10, frame_bgr.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA,
    )


generate_frames = make_frame_generator(lambda: camera, visualize, quality=50, rgb=False)


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, config=det_agent, hostname=socket.gethostname(), virtual=False)

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start():
    global running
    running = True
    return jsonify({'status': 'running'})

@app.route('/stop', methods=['POST'])
def stop():
    global running
    running = False
    if wheels:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'status': 'stopped'})

@app.route('/set_mode', methods=['POST'])
def set_mode():
    global manual_mode
    mode = request.json.get('mode', 'auto') if request.json else 'auto'
    manual_mode = (mode == 'manual')
    if wheels and not manual_mode:
        wheels.set_wheels_speed(0.0, 0.0)
    return jsonify({'mode': 'manual' if manual_mode else 'auto'})

@app.route('/keys', methods=['POST'])
def update_keys():
    global _keys_last_update
    data = request.json or {}
    with _keys_lock:
        for k in keys_pressed:
            keys_pressed[k] = bool(data.get(k, False))
    _keys_last_update = time.time()
    return jsonify({'status': 'ok'})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    value = request.json.get('value') if request.json else None
    if det_agent and value is not None:
        det_agent.conf_threshold = float(value)
    return jsonify({'conf_threshold': det_agent.conf_threshold if det_agent else None})

@app.route('/set_robot_threshold', methods=['POST'])
def set_robot_threshold():
    value = request.json.get('value') if request.json else None
    if robot_detector and value is not None:
        robot_detector.area_threshold = int(value)
    return jsonify({
        'robot_blue_area_threshold': robot_detector.area_threshold if robot_detector else None
    })

@app.route('/status')
def status():
    with _detection_lock:
        dets = list(_last_detections)
    return jsonify({
        'running':              running,
        'manual_mode':          manual_mode,
        'model_loaded':         det_agent.model_loaded if det_agent else False,
        'load_error':           det_agent.load_error if det_agent else None,
        'trt_building':         getattr(det_agent, 'trt_building', False) if det_agent else False,
        'stopped_by_detection': _stopped_by_det,
        'stop_reason':          _stop_reason,
        'robot_blue_pixels':    _robot_blue_px,
        'conf_threshold': det_agent.conf_threshold if det_agent else 0.5,
        'detections': [
            {'class': CLASS_NAMES.get(c, str(c)), 'score': round(s, 3), 'bbox': list(b)}
            for b, s, c in dets
        ],
    })


def main():
    global lane_agent, det_agent, camera, wheels

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=5000)
    args = ap.parse_args()

    suppress_http_logs()
    print('=' * 60)
    print('OBJECT DETECTION — LANE FOLLOW + STOP ON DETECTION')
    print('=' * 60)

    def _init_wheels():
        global wheels
        wheels = DaguWheelsDriver(WheelPWMConfiguration(), WheelPWMConfiguration())
        print('[Init] Wheels ready')

    def _init_camera():
        global camera
        try:
            cam = CameraDriver()
            cam.start()
            camera = cam
            print('[Init] Camera ready')
        except Exception as exc:
            print('[Init] Camera FAILED: {}'.format(exc))
            print('[Init] Stop any other task, wait 10s, then redeploy. '
                  'If it persists, power-cycle the bot.')

    def _init_agents():
        global lane_agent, det_agent, robot_detector
        try:
            lane_agent = LaneServoingAgent()
            print('[Init] Lane agent ready (speed={})'.format(lane_agent.base_speed))
            robot_detector = BlueRobotDetector()
            print('[Init] Robot (blue) detector ready '
                  '(enabled={}, area_threshold={})'.format(
                      robot_detector.enabled, robot_detector.area_threshold))
            det_agent = ObjectDetectionAgent()
            if det_agent.model_loaded:
                print('[Init] Detection model ready ({}px)'.format(det_agent.img_size))
            else:
                print('[Init] Detection model: {}'.format(det_agent.load_error))
        except Exception as exc:
            print('[Init] Agent setup FAILED: {}'.format(exc))
            import traceback
            traceback.print_exc()

    threading.Thread(target=_init_wheels,      daemon=True).start()
    threading.Thread(target=_init_camera,      daemon=True).start()
    threading.Thread(target=_init_agents,      daemon=True).start()
    threading.Thread(target=detection_loop,    daemon=True).start()
    threading.Thread(target=manual_control_loop, daemon=True).start()

    def _shutdown(signum, frame):
        shutdown_cleanup(wheels, camera, stop_event)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    web_port = find_available_port(args.port)
    print(f'\nWeb Interface: http://{socket.gethostname()}.local:{web_port}')
    print('=' * 60 + '\n')

    try:
        app.run(host='0.0.0.0', port=web_port, debug=False, threaded=True)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        shutdown_cleanup(wheels, camera, stop_event)


if __name__ == '__main__':
    sys.exit(main())
