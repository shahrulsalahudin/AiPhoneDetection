import cv2
import time
from telegram import Bot
from ultralytics import YOLO
import asyncio

# Telegram bot setup
API_TOKEN = '8528483748:AAGHXX-by5kjBo9ambAb3XHDD-m06L3DUrw'  # Your bot token
CHAT_ID = 7517574115  # Your chat ID (int, no quotes)
bot = Bot(token=API_TOKEN)

# Load YOLO model
model = YOLO('C:/Users/Shahrul Hakim/Desktop/FYP2/Train/AiPhoneDetection/runs/detect/train/weights/best.pt')  # Path to your trained model

# Camera setup
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

start_time = None
phone_detected = False
detection_time = 0

# Asynchronous function to send the image to Telegram
async def send_telegram_message(image_path, message="Phone usage detected in Restrict Area"):
    try:
        with open(image_path, 'rb') as photo:
            await bot.send_photo(chat_id=CHAT_ID, photo=photo, caption=message)
        print(f"Sent image to Telegram: {image_path}")
    except Exception as e:
        print(f"Error sending message to Telegram: {e}")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform inference on the captured frame
    results = model(frame)

    phone_detected_in_frame = False

    # Check for "phone" or "on phone" in the results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            confidence = box.conf[0]  # Get the confidence score

            # Only process detections with confidence greater than 0.5
            if confidence > 0.5:  # You can adjust this threshold as needed
                class_id = int(box.cls[0])  # Extract class ID

                # Set different colors for "phone" and "on phone"
                if result.names[class_id] == "on phone":
                    color = (0, 255, 0)  # Green for "on phone"
                elif result.names[class_id] == "phone":
                    color = (0, 0, 255)  # Red for "phone"
                else:
                    color = (255, 0, 0)  # Blue for other detections

                x1, y1, x2, y2 = box.xyxy[0]  # Get the bounding box coordinates
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = result.names[class_id]
                label_with_confidence = f"{label} {confidence:.2f}"  # Display label and confidence score
                cv2.putText(frame, label_with_confidence, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                phone_detected_in_frame = True

    # If phone is detected, start timing
    if phone_detected_in_frame:
        if start_time is None:
            start_time = time.time()  # Start the timer
        detection_time = time.time() - start_time  # Calculate elapsed time

        print(f"Detection time: {detection_time:.2f} seconds")

        if detection_time >= 10:  # If phone is detected for 10 seconds
            timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())  # Get timestamp for the image
            image_path = f'detected_{timestamp}.png'
            cv2.imwrite(image_path, frame)  # Save the current frame as an image

            # Send the image to Telegram asynchronously
            print(f"Image saved as: {image_path}")
            asyncio.run(send_telegram_message(image_path))  # Run the async function

            # Reset the timer
            start_time = None
            detection_time = 0
    else:
        # Reset the timer if no phone is detected
        start_time = None
        detection_time = 0

    # Show the frame with bounding boxes and labels
    cv2.imshow('Webcam Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close OpenCV window
