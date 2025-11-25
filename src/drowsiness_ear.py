import cv2
import mediapipe as mp
import numpy as np
import pygame

pygame.mixer.init()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = None
alarm_sound = None
blink_counter = 0
last_status = "Awake"
last_reason = ""
alarm_enabled = True

def set_alarm_enabled(value: bool):
    global alarm_enabled
    alarm_enabled = value

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14, 87, 317]

EAR_THRESHOLD = 0.23
MAR_THRESHOLD = 0.55
CLOSED_FRAMES_REQUIRED = 14
HEAD_DOWN_THRESHOLD = 18


def calculate_EAR(pts):
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    if C == 0:
        return 0.0
    return (A + B) / (2.0 * C)


def calculate_MAR(pts):
    A = np.linalg.norm(pts[2] - pts[3])
    C = np.linalg.norm(pts[0] - pts[1])
    if C == 0:
        return 0.0
    return A / C


def init_detector(alarm_path="assets/alarm.wav"):
    global face_mesh, alarm_sound

    pygame.mixer.init()
    try:
        alarm_sound = pygame.mixer.Sound(alarm_path)
    except:
        alarm_sound = None
        print("⚠ Alarm disabled (file missing)")

    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.6
    )


def play_alarm():
    if alarm_sound is not None and not pygame.mixer.get_busy():
        try:
            alarm_sound.play()
        except:
            pass


def release_detector():
    global face_mesh
    try:
        if face_mesh is not None:
            face_mesh.close()
    except:
        pass

    face_mesh = None
    pygame.mixer.stop()


def process_frame(frame):
    global face_mesh, blink_counter

    if face_mesh is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    status_text = "NO FACE DETECTED"
    reason_text = ""
    color = (0, 255, 255)

    if not res.multi_face_landmarks:
        cv2.putText(annotated, status_text, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
        blink_counter = 0
        return annotated

    face = res.multi_face_landmarks[0]

    def pt(id):
        lm = face.landmark[id]
        return np.array([lm.x * w, lm.y * h])

    left_eye = np.array([pt(i) for i in LEFT_EYE])
    right_eye = np.array([pt(i) for i in RIGHT_EYE])
    mouth = np.array([pt(i) for i in MOUTH])

    ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2.0
    mar = calculate_MAR(mouth)
    nose_y = face.landmark[1].y * h
    head_down = nose_y > (h / 2 + HEAD_DOWN_THRESHOLD)

    cv2.putText(annotated, f"EAR: {ear:.2f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"MAR: {mar:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    status_text = "AWAKE"
    color = (0, 255, 0)

    if mar > MAR_THRESHOLD:
        status_text = "YAWNING"
        reason_text = "(Mouth wide open)"
        color = (0, 255, 255)

    drowsy_reason = []

    if ear < EAR_THRESHOLD:
        drowsy_reason.append("Eyes Closed")

    if head_down:
        drowsy_reason.append("Head Down")

    if drowsy_reason:
        blink_counter += 1
    else:
        blink_counter = 0

    if blink_counter >= CLOSED_FRAMES_REQUIRED:
        status_text = "DROWSINESS ALERT!"
        reason_text = "(" + ", ".join(drowsy_reason) + ")"
        color = (0, 0, 255)
        play_alarm()

    cv2.putText(annotated, status_text, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    if reason_text:
        cv2.putText(annotated, reason_text, (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    return annotated
