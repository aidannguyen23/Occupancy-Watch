import cv2
import torch
from sort import Sort  # SORT or DeepSORT for tracking
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize SORT tracker
tracker = Sort()

# Variables for user-defined door region
drawing = False  # True if mouse is pressed
door_x1, door_y1, door_x2, door_y2 = -1, -1, -1, -1
door_defined = False

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global door_x1, door_y1, door_x2, door_y2, drawing, door_defined

    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        door_x1, door_y1 = x, y

    elif event == cv2.EVENT_MOUSEMOVE:  # Update rectangle
        if drawing:
            door_x2, door_y2 = x, y

    elif event == cv2.EVENT_LBUTTONUP:  # Finish drawing
        drawing = False
        door_x2, door_y2 = x, y
        door_defined = True  # Door region is now defined

# Open camera feed
cap = cv2.VideoCapture(0)
cv2.namedWindow('Select Door Region')
cv2.setMouseCallback('Select Door Region', draw_rectangle)

# Wait for user to define the door region
while not door_defined:
    ret, frame = cap.read()
    if not ret:
        break

    # Show instructions
    cv2.putText(frame, "Drag to select the door region, then press 'q' to confirm.", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw the rectangle while dragging
    if drawing:
        cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (0, 0, 255), 2)

    cv2.imshow('Select Door Region', frame)

    # Exit when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q') and door_defined:
        break

cv2.destroyWindow('Select Door Region')

# Track person movements
person_ids = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = model(frame)
    detections = results.xyxy[0].numpy()

    # Filter bounding boxes (only people, class=0) and retain necessary fields
    bboxes = [d[:5] for d in detections if int(d[5]) == 0]  # x1, y1, x2, y2, conf
    bboxes = np.array(bboxes)  # Convert to NumPy array

    # Update tracker only if there are detections
    if bboxes.shape[0] > 0:
        tracks = tracker.update(bboxes)
    else:
        tracks = []

    # Process tracks
    for track in tracks:
        x1, y1, x2, y2, track_id = map(int, track)
        centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Update person position
        prev_pos = person_ids.get(track_id)
        person_ids[track_id] = (centroid_x, centroid_y)

        if prev_pos is not None:
            prev_x, prev_y = prev_pos

            # Check if the person enters or exits the door region
            if door_x1 < centroid_x < door_x2 and door_y1 < centroid_y < door_y2:
                if not (door_x1 < prev_x < door_x2 and door_y1 < prev_y < door_y2):
                    print(f"Person {track_id} ENTERED the door.")
            elif door_x1 < prev_x < door_x2 and door_y1 < prev_y < door_y2:
                print(f"Person {track_id} EXITED the door.")

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw door region
    cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (0, 0, 255), 2)
    cv2.putText(frame, 'Door Region', (door_x1, door_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Door Entry/Exit Tracker', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
