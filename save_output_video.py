import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Input and output video paths
input_path = r"D:\AI_Journey\VechileParkingSystem\inputs\entrance_video2.mp4"
output_path = r"D:\AI_Journey\VechileParkingSystem\outputs\entrance_video_output2.mp4"

# Open input video
cap = cv2.VideoCapture(input_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

COUNT_LINE_Y = 100
FOUR_WHEELERS = ['car']
TWO_WHEELERS = ['motorcycle']

vehicle_history = {}
two_wheeler_count = 0
four_wheeler_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else None
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            label = model.names[cls_id]
            if label not in FOUR_WHEELERS + TWO_WHEELERS or track_id is None:
                continue
            if track_id not in vehicle_history:
                vehicle_history[track_id] = {"label": label, "counted": False, "last_cy": cy}
            last_cy = vehicle_history[track_id]["last_cy"]
            counted = vehicle_history[track_id]["counted"]
            if not counted and last_cy < COUNT_LINE_Y and cy >= COUNT_LINE_Y:
                vehicle_history[track_id]["counted"] = True
                if label in FOUR_WHEELERS:
                    four_wheeler_count += 1
                elif label in TWO_WHEELERS:
                    two_wheeler_count += 1
            vehicle_history[track_id]["last_cy"] = cy
            color = (0, 255, 0) if label in FOUR_WHEELERS else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, f"Two Wheelers: {two_wheeler_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Four Wheelers: {four_wheeler_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    out.write(frame)

cap.release()
out.release()
print(f"✅ Output video saved as {output_path}") 