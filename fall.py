import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import time
import datetime

# please write a function where i can resize the frame to 640x640
def resize_frame(frame):
    return cv2.resize(frame, (640, 640))

# Load the YOLOv8 model
model = YOLO('Fall_NModelv1.pt')

# Open the video
cap = cv2.VideoCapture('fall.mp4')

# Prepare video writer
# current_time_vdo = datetime.datetime.now()
# timevdo = current_time_vdo.strftime('%Y-%m-%d %H:%M:%S.%f')
# timestamp_for_filename_vdo = timevdo.replace(':', '_').replace('.', '_')
# output_video = f'{timestamp_for_filename_vdo}_rec_video.mp4'
# fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# rec = cv2.VideoWriter(output_video, fourcc, 10.0, (640, 640))

# Initialize data structures
class_times = {}
last_seen = {}
history = deque(maxlen=300)  # Store the last 10 seconds (assuming 30 FPS)

# Set up variables for logic
warning_issued = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get current timestamp
    current_time = time.time()

    # Perform inference
    results = model(frame)

    # Process detections
    for result in results:
        if result.boxes is None:
            continue
        boxes = result.boxes.xyxy.cpu().numpy()
        class_names = result.boxes.cls.tolist()

        for box, class_idx in zip(boxes, class_names):
            class_name = model.names[int(class_idx)]
            if class_name not in last_seen:
                last_seen[class_name] = current_time
                class_times[class_name] = 0
            else:
                class_times[class_name] += current_time - last_seen[class_name]
                last_seen[class_name] = current_time

            x1, y1, x2, y2 = box
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            color = (0, 255, 0) if class_name != 'Fallen' else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, f"- {int(class_times[class_name])}s", (int(x1) + 80, int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Append detection to history
            history.append((current_time, class_name))

    # Check for 'fallen' state
    if 'Fallen' in class_times and class_times['Fallen'] > 5:
        if not warning_issued:
            warning_issued = True
            cv2.putText(frame, "Level 2 Warning: Fallen detected for more than 5 seconds.", (int(100), int(350) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            print("Level 2 Warning: Fallen detected for more than 5 seconds.")
        
        # Analyze last 10 seconds of data
        recent_history = [cls for t, cls in history if current_time - t <= 10]
        print(f"Recent History: {recent_history}")

        if 'Falling' in recent_history:
            print("Fall detected from Falling to Fallen.")
            cv2.putText(frame,"Fall detected from Falling to Fallen.", (int(100), int(300) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 200, 0), 2)
        elif 'Sitting' in recent_history:
            print("Transition from Sitting to Fallen detected.")
            cv2.putText(frame, "Level 2 Warning: Fallen detected for more than 5 seconds.", (int(100), int(300) - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 250, 0), 2)

    # Display total time for each class on the video frame
    y_offset = 20
    for class_name, total_time in class_times.items():
        cv2.putText(frame, f"{class_name}: {int(total_time)}s", (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        y_offset += 15

    # Display the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 640))
    cv2.imshow('Frame', frame)

    # Write to output video
   # rec.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Release everything if the job is finished
cap.release()
# rec.release()
cv2.destroyAllWindows()
