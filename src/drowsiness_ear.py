import cv2
import mediapipe as mp
import numpy as np
import pygame
import warnings
warnings.filterwarnings("ignore")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = None
alarm_sound = None
blink_counter = 0

# Landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 308, 13, 14, 87, 317]

# Thresholds
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


def init_detector():
    """Initialize Mediapipe + Alarm."""
    global face_mesh, alarm_sound
    pygame.mixer.init()

    # FIXED alarm path — no errors
    try:
        alarm_sound = pygame.mixer.Sound("assets/alarm.wav")
    except:
        alarm_sound = None
        print("⚠ Alarm file not found (assets/alarm.wav)")

    # Stable face mesh settings
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )


def play_alarm():
    if alarm_sound and not pygame.mixer.get_busy():
        try:
            alarm_sound.play()
        except:
            pass


def release_detector():
    global face_mesh
    try:
        if face_mesh:
            face_mesh.close()
    except:
        pass

    face_mesh = None
    try:
        pygame.mixer.stop()
    except:
        pass


def process_frame(frame):
    global face_mesh, blink_counter

    if face_mesh is None:
        return frame

    annotated = frame.copy()
    h, w = annotated.shape[:2]
    rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    # -----------------------------------------------------
    # NO FACE DETECTED
    # -----------------------------------------------------
    if not result.multi_face_landmarks:
        cv2.putText(annotated, "NO FACE DETECTED", (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
        blink_counter = 0
        return annotated

    # Get landmarks
    face = result.multi_face_landmarks[0]

    def pt(id):
        lm = face.landmark[id]
        return np.array([lm.x * w, lm.y * h])

    left_eye = np.array([pt(i) for i in LEFT_EYE])
    right_eye = np.array([pt(i) for i in RIGHT_EYE])
    mouth = np.array([pt(i) for i in MOUTH])

    ear = (calculate_EAR(left_eye) + calculate_EAR(right_eye)) / 2
    mar = calculate_MAR(mouth)

    # Show EAR / MAR
    cv2.putText(annotated, f"EAR: {ear:.2f}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(annotated, f"MAR: {mar:.2f}", (30, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Detect head down
    nose_y = face.landmark[1].y * h
    head_down = nose_y > (h / 2 + HEAD_DOWN_THRESHOLD)

    status = "AWAKE"
    color = (0, 255, 0)

    # Yawn detection
    if mar > MAR_THRESHOLD:
        status = "YAWNING"
        color = (0, 255, 255)

    # Eye or head down
    if ear < EAR_THRESHOLD or head_down:
        blink_counter += 1
    else:
        blink_counter = 0

    # Drowsiness
    if blink_counter >= CLOSED_FRAMES_REQUIRED:
        status = "DROWSINESS ALERT!"
        color = (0, 0, 255)
        play_alarm()

    # Draw status
    cv2.putText(annotated, status, (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

    return annotated
