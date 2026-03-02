import cv2
import csv
import os
from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VIDEO_PATH  = '../videos/traffic_sample.mp4'
OUTPUT_CSV  = 'yolo_detection/counts_output.csv'
SKIP_MS     = 5000          # skip first 5 seconds (black intro)
SAMPLE_EVERY = 15           # process every 15th frame (saves time on CPU)
MIN_CONTOUR_AREA = 300      # minimum blob size to count as a vehicle
YOLO_CONF   = 0.20          # YOLO confidence threshold

# ─────────────────────────────────────────────
# LANE ROI COORDINATES  (x1, y1, x2, y2)
# Defined from roi_final.jpg analysis
# ─────────────────────────────────────────────

LANE_ROIS = {
    'NORTH': (220,  0,  420, 110),
    'SOUTH': (240, 255, 400, 360),
    'WEST':  (0,   110, 230, 255),
    'EAST':  (370,  80, 640, 270),
}

LANE_COLORS = {
    'NORTH': (0, 255, 0),     # green
    'SOUTH': (0, 0, 255),     # red
    'WEST':  (255, 0, 0),     # blue
    'EAST':  (0, 255, 255),   # yellow
}

# ─────────────────────────────────────────────
# VEHICLE CLASSES IN YOLO (COCO dataset)
# 2=car, 3=motorcycle, 5=bus, 7=truck
# ─────────────────────────────────────────────
VEHICLE_CLASSES = [2, 3, 5, 7]


def count_vehicles_in_roi(frame, roi, bg_subtractor, yolo_model):
    """
    Hybrid detection:
    1. MOG2 background subtraction finds moving blobs
    2. YOLO confirms detections within the ROI
    Returns the higher of the two counts for robustness.
    """
    x1, y1, x2, y2 = roi
    roi_frame = frame[y1:y2, x1:x2]

    # --- Method 1: Background Subtraction (MOG2) ---
    fg_mask = bg_subtractor.apply(roi_frame)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mog2_count  = sum(1 for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA)

    # --- Method 2: YOLO Detection ---
    results     = yolo_model(roi_frame, device='cpu', conf=YOLO_CONF, verbose=False,
                             classes=VEHICLE_CLASSES)
    yolo_count  = len(results[0].boxes)

    # Take the max of both methods
    return max(mog2_count, yolo_count)


def draw_overlay(frame, lane_counts):
    """Draw ROI boxes and vehicle counts on the frame for visualization."""
    overlay = frame.copy()
    for lane, (x1, y1, x2, y2) in LANE_ROIS.items():
        color = LANE_COLORS[lane]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"{lane}: {lane_counts[lane]}"
        cv2.putText(overlay, label, (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return overlay


def run_detection():
    print("=" * 50)
    print("  Vehicle Detection Pipeline Starting")
    print("=" * 50)

    # Load YOLO model
    print("\n[1/4] Loading YOLOv8 model...")
    model = YOLO('yolov8m.pt')
    print("      YOLOv8m loaded.")

    # Open video
    print(f"\n[2/4] Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("ERROR: Could not open video. Check the path.")
        return

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"      FPS: {fps:.1f} | Total Frames: {total_frames}")

    # Skip intro black frames
    cap.set(cv2.CAP_PROP_POS_MSEC, SKIP_MS)

    # One background subtractor per lane
    bg_subtractors = {
        lane: cv2.createBackgroundSubtractorMOG2(
            history=100, varThreshold=40, detectShadows=False
        )
        for lane in LANE_ROIS
    }

    # Warm up background subtractors (30 frames)
    print("\n[3/4] Warming up background subtractor (30 frames)...")
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        for lane, roi in LANE_ROIS.items():
            x1, y1, x2, y2 = roi
            bg_subtractors[lane].apply(frame[y1:y2, x1:x2])

    # ─────────────────────────────────────────
    # MAIN DETECTION LOOP
    # ─────────────────────────────────────────
    print("\n[4/4] Running detection on video frames...")
    print("      (Processing every 15th frame to save time)\n")

    os.makedirs('yolo_detection', exist_ok=True)
    csv_file   = open(OUTPUT_CSV, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['frame', 'timestamp_sec', 'NORTH', 'SOUTH', 'WEST', 'EAST', 'total'])

    frame_idx      = 0
    saved_count    = 0
    preview_saved  = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Only process every Nth frame
        if frame_idx % SAMPLE_EVERY != 0:
            continue

        timestamp = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2)

        # Count vehicles in each lane
        lane_counts = {}
        for lane, roi in LANE_ROIS.items():
            lane_counts[lane] = count_vehicles_in_roi(
                frame, roi, bg_subtractors[lane], model
            )

        total = sum(lane_counts.values())

        # Write to CSV
        csv_writer.writerow([
            frame_idx,
            timestamp,
            lane_counts['NORTH'],
            lane_counts['SOUTH'],
            lane_counts['WEST'],
            lane_counts['EAST'],
            total
        ])
        saved_count += 1

        # Save one preview image with overlays
        if not preview_saved and total > 5:
            preview = draw_overlay(frame, lane_counts)
            cv2.imwrite('yolo_detection/detection_preview.jpg', preview)
            preview_saved = True

        # Print progress every 10 saved rows
        if saved_count % 10 == 0:
            print(f"  Frame {frame_idx:4d} | {timestamp:6.1f}s | "
                  f"N:{lane_counts['NORTH']} "
                  f"S:{lane_counts['SOUTH']} "
                  f"W:{lane_counts['WEST']} "
                  f"E:{lane_counts['EAST']} | "
                  f"Total:{total}")

    cap.release()
    csv_file.close()

    print(f"\n{'='*50}")
    print(f"  Detection Complete!")
    print(f"  Rows saved to CSV : {saved_count}")
    print(f"  Output file       : {OUTPUT_CSV}")
    print(f"  Preview image     : yolo_detection/detection_preview.jpg")
    print(f"{'='*50}")


if __name__ == '__main__':
    run_detection()