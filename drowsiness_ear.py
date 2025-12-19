import cv2
import numpy as np
import mediapipe as mp
import pygame
import warnings
warnings.filterwarnings("ignore")

pygame.mixer.init()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = None
alarm_sound = None

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14, 87, 317]
NOSE_TIP = 1

# Thresholds
EAR_THRESHOLD = 0.21
CLOSED_FRAMES_REQUIRED = 20
HEAD_BEND_OFFSET = 40   # pixels down from face center

blink_counter = 0


def calculate_EAR(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return 0 if C == 0 else (A + B) / (2 * C)


def init_detector():
    global face_mesh, alarm_sound

    try:
        alarm_sound = pygame.mixer.Sound("assets/alarm.wav")
    except:
        alarm_sound = None
        print("⚠ alarm.wav missing")

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def play_alarm():
    if alarm_sound and not pygame.mixer.get_busy():
        alarm_sound.play()


def release_detector():
    global face_mesh
    if face_mesh:
        face_mesh.close()
    face_mesh = None
    pygame.mixer.stop()


def process_frame(frame):
    global blink_counter

    if face_mesh is None:
        return frame

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    if not res.multi_face_landmarks:
        blink_counter = 0
        cv2.putText(frame, "NO FACE DETECTED", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame

    face = res.multi_face_landmarks[0]

    def pt(i):
        lm = face.landmark[i]
        return np.array([lm.x * w, lm.y * h])

    # --- EAR ---
    left_eye = np.array([pt(i) for i in LEFT_EYE])
    right_eye = np.array([pt(i) for i in RIGHT_EYE])
    ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2

    # --- NOSE POSITION ---
    nose_y = pt(NOSE_TIP)[1]
    face_center_y = h / 2

    # ---------------- LOGIC ----------------
    reasons = []
    status = "AWAKE"
    color = (0, 255, 0)

    # Eyes closed
    if ear < EAR_THRESHOLD:
        blink_counter += 1
    else:
        blink_counter = 0

    if blink_counter > CLOSED_FRAMES_REQUIRED:
        reasons.append("Eyes closed for long")

    # Head bending down
    if nose_y > face_center_y + HEAD_BEND_OFFSET:
        reasons.append("Head bent down")

    # Alert
    if reasons:
        status = "DROWSINESS ALERT"
        color = (0, 0, 255)
        play_alarm()

    # ---------------- DRAW ----------------
    cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(frame, status, (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    y = 130
    for r in reasons:
        cv2.putText(frame, f"- {r}", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35

    return frame
