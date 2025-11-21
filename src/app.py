# src/app.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Silence TF / Mediapipe info/warnings

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
from drowsiness_ear import init_detector, process_frame, release_detector

app = Flask(__name__,template_folder='C:/Users/nishc/OneDrive/Desktop/DRIVER DROWSINESS DETECTION/templates')
camera_running = False
cap = None
lock = threading.Lock()

def generate_frames():
    global cap, camera_running
    while camera_running:
        if cap is None:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Send the frame to detection function which annotates & may play alarm
        annotated = process_frame(frame)

        # encode
        ret2, buffer = cv2.imencode('.jpg', annotated)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    # When loop ends, ensure resources cleaned here as a safeguard
    try:
        if cap is not None and cap.isOpened():
            cap.release()
    except:
        pass
    release_detector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_running, cap
    with lock:
        if camera_running:
            return jsonify({"status": "already_running"})
        # initialize detector (mediapipe, alarm, etc.)
        init_detector()
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = None
            return jsonify({"status": "camera_error"}), 500
        camera_running = True
    return jsonify({"status": "started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global camera_running, cap
    with lock:
        if not camera_running:
            return jsonify({"status": "already_stopped"})
        camera_running = False
        # generator will release cap; release here too for immediate effect
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except:
            pass
        cap = None
        release_detector()
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)