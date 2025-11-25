import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time

from drowsiness_ear import init_detector, process_frame, release_detector

app = Flask(__name__, template_folder='../templates')

# Global variables
camera = None
stream_thread = None
running = False
lock = threading.Lock()


# ------------------- FRAME GENERATOR --------------------
def frame_generator():
    global camera, running

    while running:
        if camera is None:
            time.sleep(0.05)
            continue

        ok, frame = camera.read()
        if not ok:
            continue

        # process the frame
        annotated = process_frame(frame)

        ok2, buf = cv2.imencode('.jpg', annotated)
        if not ok2:
            continue

        frame_bytes = buf.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')

    # cleanup
    try:
        if camera:
            camera.release()
    except:
        pass

    release_detector()
    camera = None


# ------------------- ROUTES -----------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_camera", methods=["POST"])
def start_camera():
    global running, camera

    with lock:
        if running:
            return jsonify({"status": "already_running"})

        # Start detection model
        init_detector()

        # (Re)open webcam
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        time.sleep(0.3)  # give time to warm up

        if not camera.isOpened():
            camera = None
            return jsonify({"status": "camera_error"}), 500

        running = True

    return jsonify({"status": "started"})


@app.route("/stop_camera", methods=["POST"])
def stop_camera():
    global running, camera

    with lock:
        if not running:
            return jsonify({"status": "already_stopped"})

        running = False

        try:
            if camera is not None and camera.isOpened():
                camera.release()
        except:
            pass

        camera = None
        release_detector()

    return jsonify({"status": "stopped"})


@app.route("/video_feed")
def video_feed():
    return Response(frame_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------- MAIN -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
