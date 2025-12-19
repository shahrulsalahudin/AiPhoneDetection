import cv2
from ultralytics import YOLO
import time

# Load your trained YOLO model (replace with the correct path to your best.pt file)
model = YOLO('yolov8n.pt')  # Use the smallest model for better FPS

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam index, change if needed

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Set a smaller resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Set target FPS (optional, but may help for smoother performance)
cap.set(cv2.CAP_PROP_FPS, 30)

frame_counter = 0  # Frame counter to skip frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame_counter += 1

    # Skip frames to reduce processing load (process every 2nd frame)
    if frame_counter % 2 != 0:
        continue

    # Resize frame to speed up detection
    resized_frame = cv2.resize(frame, (416, 416))  # Change this to a smaller size like 320x320 or 256x256

    # Perform inference on the captured frame
    start_time = time.time()
    results = model(resized_frame, conf=0.4)  # Lower confidence threshold to speed up detection
    print(f"Inference time: {time.time() - start_time:.3f}s")

    # Draw results on the frame
    if isinstance(results, list):
        for result in results:
            frame = result.plot()  # Draw results on the frame
    else:
        frame = results.plot()

    # Display the frame with the detections
    cv2.imshow('Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
