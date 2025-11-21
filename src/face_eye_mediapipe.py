# src/face_eye_mediapipe.py
import warnings
warnings.filterwarnings("ignore")
import cv2
import mediapipe as mp
import numpy as np
import os
from datetime import datetime
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Landmark index groups for eye regions (common indices)
LEFT_EYE_IDX = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144]
RIGHT_EYE_IDX = [263, 466, 388, 387, 386, 385, 384, 398, 362, 380, 381, 382, 374, 373]

def landmarks_to_bbox(landmarks, img_w, img_h, indices, padding=5):
    """Return bounding box (x1,y1,x2,y2) from a list of landmark indices."""
    xs = [int(landmarks[i].x * img_w) for i in indices]
    ys = [int(landmarks[i].y * img_h) for i in indices]
    x1, x2 = max(min(xs) - padding, 0), min(max(xs) + padding, img_w - 1)
    y1, y2 = max(min(ys) - padding, 0), min(max(ys) + padding, img_h - 1)
    return x1, y1, x2, y2

def draw_bbox_with_label(frame, bbox, label, color=(0,255,0)):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main(save_eyes=False):
    global save_flag
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    os.makedirs("data/eyes/open", exist_ok=True)
    os.makedirs("data/eyes/closed", exist_ok=True)
    save_count = 0
    mode = "open"  # press 'm' to toggle between 'open' and 'closed'
    print("Press 'q' to quit, 's' to save eye crops, 'm' to toggle save mode, 'p' to take screenshot")

    left_crop, right_crop = None, None  # placeholders for eye crops

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Face bounding box
            all_x = [int(lm.x * img_w) for lm in face_landmarks]
            all_y = [int(lm.y * img_h) for lm in face_landmarks]
            fx1, fy1 = max(min(all_x) - 20, 0), max(min(all_y) - 20, 0)
            fx2, fy2 = min(max(all_x) + 20, img_w-1), min(max(all_y) + 20, img_h-1)
            draw_bbox_with_label(frame, (fx1, fy1, fx2, fy2), "Face")

            # Left and right eyes
            left_bbox = landmarks_to_bbox(face_landmarks, img_w, img_h, LEFT_EYE_IDX, padding=5)
            right_bbox = landmarks_to_bbox(face_landmarks, img_w, img_h, RIGHT_EYE_IDX, padding=5)
            draw_bbox_with_label(frame, left_bbox, "Left Eye", color=(255,0,0))
            draw_bbox_with_label(frame, right_bbox, "Right Eye", color=(0,165,255))

            # Crop the eyes each frame
            lx1, ly1, lx2, ly2 = left_bbox
            rx1, ry1, rx2, ry2 = right_bbox
            left_crop = frame[ly1:ly2, lx1:lx2].copy()
            right_crop = frame[ry1:ry2, rx1:rx2].copy()

            # Auto save mode (optional)
            if save_eyes and save_flag:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"data/eyes/{mode}/left_{timestamp}.jpg", left_crop)
                cv2.imwrite(f"data/eyes/{mode}/right_{timestamp}.jpg", right_crop)
                save_count += 1
                print(f"✅ Saved #{save_count} to data/eyes/{mode}/")

        # Display mode
        cv2.putText(frame, f"Mode: {mode}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow("Face & Eye Detection - Press q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            mode = "closed" if mode == "open" else "open"
            print("Switched mode to:", mode)
        elif key == ord('s'):
            # Save current eye crops
            if left_crop is not None and right_crop is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(f"data/eyes/{mode}/left_{timestamp}.jpg", left_crop)
                cv2.imwrite(f"data/eyes/{mode}/right_{timestamp}.jpg", right_crop)
                print(f"✅ Saved eyes to data/eyes/{mode}/")
            else:
                print("⚠️ No eye detected to save.")
        elif key == ord('p'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg", frame)
            print("📸 Screenshot saved.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    save_flag = False
    main(save_eyes=False)
