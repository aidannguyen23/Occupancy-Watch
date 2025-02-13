import cv2
import torch

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# capture the front webcam
cap = cv2.VideoCapture(0)

# while camera open, read a frame in
while cap.isOpened():
    ret, frame = cap.read()
    # stop loop if camera not working
    if not ret:
        break

    # Perform object detection
    results = model(frame)

    # Extract detections
    detections = results.xyxy[0].numpy()  # numpy array with x1, y1, x2, y2, conf, class
    # x1, y1, x2, y2: Coordinates of the bounding box.
    # conf: Confidence score of the detection.
    # class: Class ID (e.g., 0 for "person")

    people_count = sum(1 for d in detections if int(d[5]) == 0)  # Class 0 is "person"

    for d in detections:
        x1, y1, x2, y2, conf, cls = map(int, d[:6])  # Extract coordinates and class
        if cls == 0:  # Class 0 is "person"
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            # Add label
            cv2.putText(
                frame,
                f"Person {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    # Display the count
    cv2.putText(frame, f'People Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('People Counter', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()