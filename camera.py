import cv2
import time

# Initialize camera
camera_id = 0
width, height = 1920, 1080

cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    raise ValueError(f"Cannot open camera with ID {camera_id}")

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Capture image
ret, photo = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Failed to capture image")
# Release camera
cap.release()

# Show image
cv2.imshow('Photo', photo)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save image
filename = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
cv2.imwrite(filename, photo)