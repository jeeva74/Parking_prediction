import csv
import os
import time
from datetime import datetime

import cv2
from ultralytics import YOLO


# -----------------------------
# Step 3: Parking Capacity Definition
# -----------------------------
TOTAL_PARKING_SLOTS = 10
LOG_INTERVAL_SECONDS = 300  # 5 minutes for historical storage (step 5)
HISTORY_CSV_PATH = os.path.join("outputs", "parking_history.csv")


COUNT_LINE_Y = None  # will be set dynamically based on frame height

# Real-time counters (vehicles that have entered = occupied slots)
two_wheeler_count = 0
four_wheeler_count = 0

# Dictionary to hold tracking history per vehicle ID.
# For each track_id we store:
# - label: class name (car / motorcycle)
# - last_cy: last vertical center position
# - last_side: "above" or "below" the counting line to detect crossings
vehicle_history = {}

# Classes to detect - simplified to just car and motorcycle
FOUR_WHEELERS = ["car"]
TWO_WHEELERS = ["motorcycle"]


def compute_parking_status(occupied: int):
    """
    Step 3 & 4:
    - Calculate available slots
    - Derive qualitative status and color code.
    """
    occupied = max(0, min(occupied, TOTAL_PARKING_SLOTS))
    available = TOTAL_PARKING_SLOTS - occupied

    if TOTAL_PARKING_SLOTS == 0:
        ratio = 1.0
    else:
        ratio = occupied / TOTAL_PARKING_SLOTS

    # Map to UI status for Step 9
    if ratio < 0.6:
        status = "Available"
        color = (0, 255, 0)  # Green
    elif ratio < 0.9:
        status = "Likely available"
        color = (0, 255, 255)  # Yellow
    else:
        status = "Full"
        color = (0, 0, 255)  # Red

    return occupied, available, status, color


def predict_future_availability(
    occupied: int, available: int, now: datetime | None = None
):
    """
    Step 7: Simple rule-based parking availability prediction.
    
    Very simple logic based on available slots:
    - If available > 20: High availability
    - If available > 0: Likely available
    - If available = 0: Low availability (Full)
    """
    if now is None:
        now = datetime.now()

    if available > 20:
        level = "High availability"
        indicator = "🟢"
    elif available > 0:
        level = "Likely available"
        indicator = "🟡"
    else:
        level = "Low availability"
        indicator = "🔴"

    return {
        "time": now.strftime("%H:%M"),
        "level": level,
        "indicator": indicator,
    }


def recommend_parking_slot(prediction: dict, available: int):
    """
    Step 8: Decision making based on predicted level and current availability.
    """
    level = prediction.get("level", "")

    if available <= 0 or "Low" in level:
        return "Recommendation: Use alternate parking – current lot is or will be full."
    if "High" in level:
        return "Recommendation: Use main parking – high chance of free slots."
    return "Recommendation: Parking may be partially available – plan a short wait."


def log_parking_snapshot(occupied: int, available: int):
    """
    Step 5 & 6: Historical data storage + basic preprocessing.

    We:
    - Store time, occupied, available
    - Add hour and weekday as basic features for later analysis.
    """
    os.makedirs(os.path.dirname(HISTORY_CSV_PATH), exist_ok=True)

    file_exists = os.path.isfile(HISTORY_CSV_PATH)
    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    hour = now.hour
    weekday = now.strftime("%A")

    with open(HISTORY_CSV_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Hour", "Weekday", "Occupied", "Available"])
        writer.writerow([timestamp, hour, weekday, occupied, available])


def draw_parking_dashboard(
    frame,
    two_w_count: int,
    four_w_count: int,
    occupied: int,
    available: int,
    status: str,
    status_color,
    prediction: dict,
    recommendation: str,
):
    """
    Step 4, 7, 8, 9: Dashboard matching the exact reference UI layout.
    Single panel with all information organized as per reference design.
    """
    h, w, _ = frame.shape
    
    # Single panel dimensions - INCREASED HEIGHT for better spacing
    panel_w = 550
    panel_h = 240
    panel_x0 = 15
    panel_y0 = 15
    panel_x1 = panel_x0 + panel_w
    panel_y1 = panel_y0 + panel_h
    
    # Draw semi-transparent panel background (light gray as in reference)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (200, 200, 200), thickness=-1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Clean border
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y1), (100, 100, 100), 2)

    # Vertical partition line between left and right sections
    divider_x = panel_x0 + panel_w // 2 + 10
    cv2.line(
        frame,
        (divider_x, panel_y0 + 8),
        (divider_x, panel_y1 - 8),
        (120, 120, 120),
        1,
    )
    
    # Font settings - SMALLER for compact display
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale_title = 0.55
    font_scale_label = 0.45
    font_scale_value = 0.50
    font_scale_small = 0.40
    font_thick_title = 1
    font_thick_normal = 1
    font_thick_value = 1
    
    padding_x = 15
    padding_y = 18
    line_height = 26  # Increased for better spacing
    section_gap = 10  # Gap between major sections
    y = panel_y0 + padding_y
    
    # ========== TITLE ==========
    cv2.putText(
        frame, "PARKING STATUS:",
        (panel_x0 + padding_x, y),
        font, font_scale_title, (0, 0, 0), font_thick_title
    )
    y += line_height + 8  # More space after title
    
    # LEFT SIDE - Vehicle counts and occupancy
    left_x = panel_x0 + padding_x
    right_x = panel_x0 + panel_w // 2 + 20  # Start right side from middle
    
    # ========== LEFT SIDE: TWO-WHEEL and FOUR-WHEEL ==========
    label_x1 = left_x
    label_x2 = label_x1 + 150
    
    cv2.putText(
        frame, "TWO-WHEEL:",
        (label_x1, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    cv2.putText(
        frame, "FOUR-WHEEL:",
        (label_x2, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    y += line_height
    
    cv2.putText(
        frame, f"{two_w_count}",
        (label_x1, y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    cv2.putText(
        frame, f"{four_w_count}",
        (label_x2, y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    y += line_height + section_gap
    
    # ========== LINE 3: OCCUPIED and AVAILABLE labels ==========
    cv2.putText(
        frame, "OCCUPIED:",
        (label_x1, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    cv2.putText(
        frame, "AVAILABLE:",
        (label_x2, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    y += line_height
    
    # ========== LINE 4: Values for occupied and available ==========
    cv2.putText(
        frame, f"{occupied}",
        (label_x1, y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    cv2.putText(
        frame, f"{available}",
        (label_x2, y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    y += line_height + section_gap  # More space before prediction section
    
    # ========== LEFT SIDE BOTTOM: TOTAL ==========
    total_y = panel_y1 - padding_y - line_height
    cv2.putText(
        frame, "TOTAL:",
        (label_x1, total_y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    cv2.putText(
        frame, f"{TOTAL_PARKING_SLOTS}",
        (label_x1 + 80, total_y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    
    # ========== RIGHT SIDE: PREDICTION SECTION ==========
    pred_level = prediction.get("level", "")
    pred_time = prediction.get("time", "--:--")
    color_circle_x = panel_x0 + panel_w - 25
    
    # Reset y for right side
    y = panel_y0 + padding_y
    
    # PREDICT: label
    cv2.putText(
        frame, "PREDICT:",
        (right_x, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    y += line_height + 3
    
    # Prediction value (LIKELY/HIGH/LOW)
    if "High availability" in pred_level:
        pred_display = "HIGH"
    elif "Likely available" in pred_level:
        pred_display = "LIKELY"
    elif "Low availability" in pred_level:
        pred_display = "LOW"
    else:
        pred_display = pred_level[:10] if len(pred_level) > 10 else pred_level
    
    cv2.putText(
        frame, pred_display,
        (right_x, y),
        font, font_scale_value, (0, 0, 0), font_thick_value
    )
    # Green circle indicator on right
    cv2.circle(frame, (color_circle_x, y - 6), 7, status_color, thickness=-1)
    cv2.circle(frame, (color_circle_x, y - 6), 7, (0, 0, 0), 1)
    y += line_height + 5
    
    # PLAN SHORT WAIT
    rec_text = recommendation.replace("Recommendation:", "").strip()
    if "alternate" in rec_text.lower() or "full" in rec_text.lower():
        rec_display = "USE ALTERNATE"
    elif "main parking" in rec_text.lower() or "high chance" in rec_text.lower():
        rec_display = "USE MAIN"
    else:
        rec_display = "PLAN SHORT WAIT"
    
    cv2.putText(
        frame, rec_display,
        (right_x, y),
        font, font_scale_label, (0, 0, 0), font_thick_normal
    )
    y += line_height + 5
    
    # TIME
    cv2.putText(
        frame, f"TIME: {pred_time}",
        (right_x, y),
        font, font_scale_small, (0, 0, 0), font_thick_normal
    )
    y += line_height + 5
    
    # MAY BE AVAILABLE (bottom right)
    if "High" in pred_level:
        avail_msg = "AVAILABLE NOW"
    elif "Likely" in pred_level:
        avail_msg = "MAY BE AVAILABLE"
    else:
        avail_msg = "FULL - WAIT"
    
    cv2.putText(
        frame, avail_msg,
        (right_x, y),
        font, font_scale_small, (0, 0, 0), font_thick_normal
    )


def send_to_cisco_smart_city(payload: dict):
    """
    Step 10: Cisco Smart City Integration (conceptual).

    This is a placeholder to show how the system could push data to a Cisco IoT /
    smart city platform over MQTT, HTTP, or other protocols.

    For now, we don't perform actual network operations here; this function can
    be extended with Cisco-specific SDKs or APIs.
    """
    # Example (conceptual only):
    # cisco_iot_client.publish("smart-city/parking", json.dumps(payload))
    _ = payload  # avoid unused variable warning


def main():
    global two_wheeler_count, four_wheeler_count
    model = YOLO("yolov8n.pt")

    # Use a relative path to the input video inside the project.
    # Change the filename here if you want to use a different input video.
    video_path = os.path.join("inputs", "entrance_video.mp4")
    print(f"🎥 Using video source: {os.path.abspath(video_path)}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Could not open video source. Please check the path in main.py.")
        return

    print("✅ Video source opened successfully.")
    print("▶ Smart Parking Dashboard is starting...")
    print(f"   Total parking slots configured: {TOTAL_PARKING_SLOTS}")
    print("   A GUI window will appear. Press 'q' in the window to stop.\n")

    last_log_time = 0.0
    first_frame_printed = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("✅ Video complete.")
            break

        # Set counting line position dynamically on first frame (around 60% height)
        global COUNT_LINE_Y
        if COUNT_LINE_Y is None:
            frame_height = frame.shape[0]
            COUNT_LINE_Y = int(frame_height * 0.6)
            print(f"📏 COUNT_LINE_Y set to {COUNT_LINE_Y} (60% of frame height).")

        if not first_frame_printed:
            print("✅ First frame processed. Real-time detection and parking status are running.")
            first_frame_printed = True

        results = model.track(
            source=frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )[0]

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                track_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cy = int((y1 + y2) / 2)

                label = model.names[cls_id]

                if label not in FOUR_WHEELERS + TWO_WHEELERS or track_id is None:
                    continue

                # Initialize if new track_id
                if track_id not in vehicle_history:
                    initial_side = "above" if cy < COUNT_LINE_Y else "below"
                    vehicle_history[track_id] = {
                        "label": label,
                        "last_cy": cy,
                        "last_side": initial_side,
                    }

                # Check if it crossed the line in either direction
                last_cy = vehicle_history[track_id]["last_cy"]
                last_side = vehicle_history[track_id]["last_side"]
                current_side = "above" if cy < COUNT_LINE_Y else "below"

                # Entry: moved from above -> below the line
                if last_side == "above" and current_side == "below":
                    if label in FOUR_WHEELERS:
                        four_wheeler_count += 1
                    elif label in TWO_WHEELERS:
                        two_wheeler_count += 1

                # Exit: moved from below -> above the line
                elif last_side == "below" and current_side == "above":
                    if label in FOUR_WHEELERS and four_wheeler_count > 0:
                        four_wheeler_count -= 1
                    elif label in TWO_WHEELERS and two_wheeler_count > 0:
                        two_wheeler_count -= 1

                # Update last y position and side
                vehicle_history[track_id]["last_cy"] = cy
                vehicle_history[track_id]["last_side"] = current_side

                # Draw box and label
                color = (0, 255, 0) if label in FOUR_WHEELERS else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"{label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        # Compute real-time parking status
        occupied = two_wheeler_count + four_wheeler_count
        occupied, available, status, status_color = compute_parking_status(occupied)

        # Simple rule-based prediction (now depends on current occupancy)
        prediction = predict_future_availability(occupied, available)
        recommendation = recommend_parking_slot(prediction, available)

        # Step 5: Log historical data at fixed intervals
        now_ts = time.time()
        if now_ts - last_log_time >= LOG_INTERVAL_SECONDS:
            log_parking_snapshot(occupied, available)
            last_log_time = now_ts

            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{log_time}] 📊 Logged snapshot to '{HISTORY_CSV_PATH}': "
                f"occupied={occupied}, available={available}, status={status}"
            )

            # Conceptual Cisco integration: send a summarized payload
            payload = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "total_slots": TOTAL_PARKING_SLOTS,
                "occupied": occupied,
                "available": available,
                "status": status,
                "prediction": prediction,
            }
            send_to_cisco_smart_city(payload)

        # Draw the counting line so you can see where vehicles are counted
        cv2.line(
            frame,
            (0, COUNT_LINE_Y),
            (frame.shape[1], COUNT_LINE_Y),
            (0, 255, 255),
            2,
        )

        # Step 9: UI overlay – smart parking dashboard
        draw_parking_dashboard(
            frame,
            two_wheeler_count,
            four_wheeler_count,
            occupied,
            available,
            status,
            status_color,
            prediction,
            recommendation,
        )

        cv2.imshow("Smart Parking Dashboard", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
