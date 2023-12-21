import os
from ultralytics import YOLO
import cv2

# Use 0 as the argument for VideoCapture to access the default webcam
cap = cv2.VideoCapture(0)

# Load the YOLO model
model_path = os.path.join('.', 'models', 'best.pt')
model = YOLO(model_path)

threshold = 0.5

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    H, W, _ = frame.shape

    # Object detection using YOLO
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Draw bounding box and display class name
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 2)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Webcam Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
