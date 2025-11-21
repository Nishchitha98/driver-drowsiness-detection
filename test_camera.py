import cv2

# Open the default webcam (0 means built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not access the webcam.")
else:
    print("✅ Webcam opened successfully! Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Show the video feed
    cv2.imshow("Camera Test - Press 'q' to Quit", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
