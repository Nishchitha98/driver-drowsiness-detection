import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, Response, jsonify
import cv2
import time
import threading

from drowsiness_ear import init_detector, process_frame, release_detector

app = Flask(__name__)

camera = None
running = False
lock = threading.Lock()


# ---------------- VIDEO STREAM ----------------
def generate_frames():
    global camera, running

    while running:
        if camera is None:
            time.sleep(0.1)
            continue

        success, frame = camera.read()
        if not success:
            print("❌ Frame grab failed")
            break

        frame = process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        time.sleep(0.03)  # ~30 FPS

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    print("🛑 Stream stopped")


# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global camera, running

    with lock:
        if running:
            return jsonify({"status": "already_running"})

        print("▶ Initializing detector")
        init_detector()

        print("▶ Opening webcam")
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not camera.isOpened():
            camera = None
            print("❌ Webcam open failed")
            return jsonify({"status": "camera_error"}), 500

        running = True
        print("✅ Webcam started")

    return jsonify({"status": "started"})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global camera, running

    with lock:
        running = False

        if camera:
            camera.release()
            camera = None

        release_detector()
        print("⏹ Webcam stopped")

    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    print("📡 Client connected to video feed")
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚀 Flask server started")
    app.run(debug=True, threaded=True)
