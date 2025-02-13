import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, nn_budget=30, nms_max_overlap=1.0)

# Variables for user-defined door region
drawing = False
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

    cv2.putText(frame, "Drag to select the door region, then press 'q' to confirm.", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if drawing:
        cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (0, 0, 255), 2)

    cv2.imshow('Select Door Region', frame)

    if cv2.waitKey(1) & 0xFF == ord('q') and door_defined:
        break

cv2.destroyWindow('Select Door Region')

# Tracking variables
room_count = 0
# Initialize trackers
person_states = {}          # Tracks if a person is inside or outside
person_last_position = {}   # Tracks the last known position of each person's centroid
vanished_at = {}            # Tracks when and where a person last vanished
transition_state = {}       # Tracks if a person is mid-transition (e.g., near the door)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results.xyxy[0].numpy()

    # Filter for people (class=0)
    bboxes = [d[:4] for d in detections if int(d[5]) == 0]
    confidences = [d[4] for d in detections if int(d[5]) == 0]
    bboxes = [bbox.tolist() for bbox in bboxes]

    # Convert bboxes to [x, y, w, h]
    def convert_bbox_format(bbox):
        x, y = bbox[0], bbox[1]
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        return [x, y, w, h]

    raw_detections = [
        [convert_bbox_format(bbox), conf, "person"]
        for bbox, conf in zip(bboxes, confidences)
    ]

    # Update tracks
    if raw_detections:
        tracks = tracker.update_tracks(raw_detections, frame=frame)
    else:
        tracks = []

    # Track active IDs
    active_ids = {track.track_id for track in tracks}

    # Handle vanishing logic
    for track_id, last_position in list(person_last_position.items()):
        if track_id not in active_ids:
            last_x, last_y = last_position
            vanished_at[track_id] = (last_x, last_y)  # Record vanish location
            if door_x1 < last_x < door_x2 and door_y1 < last_y < door_y2:
                # If vanished inside the door, count as exiting
                if person_states.get(track_id) == "inside":
                    room_count -= 1
                    person_states[track_id] = "outside"
                    print(f"Person {track_id} EXITED. Room count: {room_count}")

    # Process active tracks
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_tlbr())  # Bounding box
        centroid_x, centroid_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Check if appearing at the door
        if track_id not in person_last_position and door_x1 < centroid_x < door_x2 and door_y1 < centroid_y < door_y2:
            if person_states.get(track_id, "outside") == "outside":
                room_count += 1
                person_states[track_id] = "inside"
                print(f"Person {track_id} ENTERED. Room count: {room_count}")

        # Update last position
        person_last_position[track_id] = (centroid_x, centroid_y)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Remove vanished IDs from active tracking
    for track_id in list(vanished_at.keys()):
        if track_id in active_ids:
            del vanished_at[track_id]

    # Draw door region
    cv2.rectangle(frame, (door_x1, door_y1), (door_x2, door_y2), (0, 0, 255), 2)
    cv2.putText(frame, 'Door Region', (door_x1, door_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Door Entry/Exit Tracker', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
