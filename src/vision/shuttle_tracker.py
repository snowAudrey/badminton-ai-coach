import cv2
import json
import os
import numpy as np
from ultralytics import YOLO
 
# ============================================================
# MODEL SETUP
# ============================================================
# Load the YOLOv8 nano model as a starting point
# Use shuttlecock-specific weights file, best.pt, trained using 8,000+ labeled images
# ============================================================
 
#yolo_model = YOLO("yolov8n.pt")
yolo_model = YOLO("models/best.pt") 
 
# ============================================================
# TUNING PARAMETERS
# Adjust these values if detection is too noisy or too sparse
# ============================================================
 
# Minimum confidence score to accept a shuttle detection (0.0 - 1.0)
# Lower = more detections but more false positives
# Higher = fewer detections but more reliable
CONFIDENCE_THRESHOLD = 0.3

# A rally is considered "active" when the shuttle moves at least this many pixels
# between consecutive frames. Below this = shuttle is still / between points.
MIN_SHUTTLE_MOVEMENT_PX = 5
 
# How many consecutive frames without movement before we call a rally "ended"
# At 30fps, 90 frames = 3 seconds of stillness
RALLY_END_FRAMES = 90
 
# class 0 = 'Shuttlecock' in the custom model
SHUTTLE_CLASS_ID = 0

# If a detected position recurs at nearly the same pixel across many frames,
# it's almost certainly a static background object (e.g. equipment) being
# misclassified as the shuttlecock — a real shuttle in play is essentially
# always moving. Positions within STATIC_OBJECT_RADIUS_PX of each other that
# occur in at least STATIC_OBJECT_MIN_FRAMES frames across the whole video
# are treated as background clutter and excluded from detection.
STATIC_OBJECT_RADIUS_PX = 10
STATIC_OBJECT_MIN_FRAMES = 100


def detect_shuttle_candidates(frame):
    """
    Runs YOLOv8 on a single frame to find the shuttlecock.
    Returns ALL detections above the confidence threshold (not just the
    best one), as a list of (x_center, y_center, confidence) tuples — so
    that static false positives can be filtered out later by looking at
    the whole video, instead of guessing from a single frame.
    """

    # Run YOLOv8 on the frame
    # verbose=False suppresses YOLOv8's default console output
    results = yolo_model(frame, verbose=False)

    candidates = []

    # Loop through all detected objects in the frame
    for result in results:
        for box in result.boxes:

            # Get the confidence score for this detection
            confidence = float(box.conf)

            # Only keep detections above our confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # Only accept the shuttlecock class
            if int(box.cls) != SHUTTLE_CLASS_ID:
                continue

            # xyxy gives us (x1, y1, x2, y2) bounding box corners
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Calculate the center point of the bounding box
            # The shuttle is small so the center is a good single-point representation
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            candidates.append((x_center, y_center, confidence))

    return candidates


def detect_shuttle(frame):
    """
    Runs YOLOv8 on a single frame and returns only the highest-confidence
    shuttlecock detection, or None if nothing was found.

    Kept for quick single-frame checks. Full-video processing uses
    detect_shuttle_candidates() plus static-object filtering instead,
    since a single frame has no way to know which detection is a static
    background false positive.
    """

    candidates = detect_shuttle_candidates(frame)

    if not candidates:
        return None

    # Return the highest-confidence candidate
    return max(candidates, key=lambda c: c[2])


def find_static_false_positives(frame_candidates: list) -> list:
    """
    Scans every candidate detection across the whole video and flags any
    position that recurs at nearly the same pixel in an implausibly large
    number of frames. A real shuttle in play is essentially always moving —
    a position that stays fixed for a large fraction of the video is almost
    certainly a static background object that resembles a shuttlecock, not
    the shuttle itself.

    Args:
        frame_candidates: List of dicts, each with a "candidates" key
                           holding that frame's list of (x, y, confidence)
                           tuples from detect_shuttle_candidates()

    Returns:
        List of (x, y) positions to exclude.
    """

    # Round each candidate's position to group near-identical detections
    # together, and count how many frames each rounded position appears in
    position_counts = {}
    for entry in frame_candidates:
        for x, y, _ in entry["candidates"]:
            bucket = (round(x / STATIC_OBJECT_RADIUS_PX), round(y / STATIC_OBJECT_RADIUS_PX))
            position_counts[bucket] = position_counts.get(bucket, 0) + 1

    static_buckets = [
        bucket for bucket, count in position_counts.items()
        if count >= STATIC_OBJECT_MIN_FRAMES
    ]

    # Convert bucket coordinates back to approximate pixel coordinates
    return [
        (bucket_x * STATIC_OBJECT_RADIUS_PX, bucket_y * STATIC_OBJECT_RADIUS_PX)
        for bucket_x, bucket_y in static_buckets
    ]


def _is_near_static_position(candidate, static_positions: list) -> bool:
    """
    Returns True if this candidate's position is close to one of the
    flagged static background positions.
    """

    x, y, _ = candidate
    tolerance = STATIC_OBJECT_RADIUS_PX * 2

    return any(
        np.sqrt((x - sx) ** 2 + (y - sy) ** 2) <= tolerance
        for sx, sy in static_positions
    )


def get_shuttle_positions(video_path: str, output_json_path: str):
    """
    Loops through every frame of a video and records the shuttle position at each frame.
    Saves results as a JSON file with frame index, timestamp, and shuttle coordinates.
 
    Each entry in the output looks like:
    {
        "frame": 142,
        "timestamp": 4.733,
        "shuttle_x": 320.5,    <- pixel x coordinate, or null if not detected
        "shuttle_y": 180.2,    <- pixel y coordinate, or null if not detected
        "confidence": 0.87,    <- detection confidence, or null if not detected
        "detected": true       <- false means the shuttle wasn't found this frame
    }
    """
 
    # Open the video file
    cap = cv2.VideoCapture(video_path)
 
    # Get frames per second — needed to calculate timestamps
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    # Get total frame count — used for progress printing
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
 
    # List to store ALL candidate detections for every frame — filtering
    # down to one detection per frame happens after this first pass, once
    # we know which positions are static background clutter
    frame_candidates = []

    # Counter to track which frame we're on
    frame_index = 0

    print(f"Processing {total_frames} frames for shuttle detection...")

    # Loop through every frame in the video
    while cap.isOpened():

        # Read the next frame
        ret, frame = cap.read()

        # Stop if the video has ended
        if not ret:
            break

        # Collect every detection above the confidence threshold this frame
        candidates = detect_shuttle_candidates(frame)

        frame_candidates.append({
            "frame": frame_index,
            "timestamp": round(frame_index / fps, 3),
            "candidates": candidates
        })

        # Print progress every 100 frames so you know it's running
        if frame_index % 100 == 0:
            print(f"  Frame {frame_index} / {total_frames}")

        frame_index += 1

    # Release the video file
    cap.release()

    # Find any position that recurs suspiciously often across the whole
    # video — almost certainly a static background object, not the shuttle
    static_positions = find_static_false_positives(frame_candidates)
    if static_positions:
        print(f"Filtering out {len(static_positions)} static false-positive position(s): {static_positions}")

    # Second pass: for each frame, pick the highest-confidence candidate
    # that isn't a flagged static position
    all_frames = []
    for entry in frame_candidates:

        remaining = [
            c for c in entry["candidates"]
            if not _is_near_static_position(c, static_positions)
        ]

        if remaining:

            # Shuttle was found — take the highest-confidence remaining candidate
            x, y, conf = max(remaining, key=lambda c: c[2])

            frame_data = {
                "frame": entry["frame"],
                "timestamp": entry["timestamp"],
                "shuttle_x": round(x, 2),
                "shuttle_y": round(y, 2),
                "confidence": round(conf, 3),
                "detected": True
            }

        else:

            # Shuttle was not found this frame — store nulls so the frame is still recorded
            frame_data = {
                "frame": entry["frame"],
                "timestamp": entry["timestamp"],
                "shuttle_x": None,
                "shuttle_y": None,
                "confidence": None,
                "detected": False
            }

        all_frames.append(frame_data)

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
 
    # Save all frame data to a JSON file
    # indent=2 makes it human-readable when you open it in VS Code
    with open(output_json_path, "w") as f:
        json.dump(all_frames, f, indent=2)
 
    # Count how many frames actually had a detection for a quick accuracy check
    detected_count = sum(1 for f in all_frames if f["detected"])
    print(f"Done! {detected_count} / {frame_index} frames had shuttle detections → {output_json_path}")
 
    return all_frames


def get_rally_segments(shuttle_positions: list, fps: float):
    """
    Takes the list of shuttle positions (from get_shuttle_positions) and groups
    frames into rally segments based on shuttle movement.

    A rally is "active" when the shuttle is detected and moving.
    A rally "ends" when the shuttle stops moving or disappears for RALLY_END_FRAMES consecutive frames.

    Returns a list of rally segments:
    [
        {
            "rally_id": 1,
            "start_frame": 45,
            "end_frame": 210,
            "start_time": 1.5,
            "end_time": 7.0,
            "frame_count": 165
        },
        ...
    ]
    """

    # List to store completed rally segments
    rallies = []

    # Rally ID counter — increments each time a new rally starts
    rally_id = 0

    # Track whether we are currently inside an active rally
    in_rally = False

    # Store the frame index where the current rally started
    rally_start_frame = None
    rally_start_time = None

    # Track the last known shuttle position — used to measure movement
    last_x = None
    last_y = None

    # Count consecutive frames where the shuttle hasn't moved
    # When this hits RALLY_END_FRAMES, we declare the rally over
    still_frame_count = 0

    # Loop through every frame's data
    for frame_data in shuttle_positions:

        frame_idx = frame_data["frame"]
        timestamp = frame_data["timestamp"]
        detected = frame_data["detected"]

        if not detected:

            # Shuttle not found this frame
            if in_rally:
                # We're inside a rally — count this as a "still" frame
                still_frame_count += 1

                # If the shuttle has been gone long enough, end the rally
                if still_frame_count >= RALLY_END_FRAMES:
                    rallies.append({
                        "rally_id": rally_id,
                        "start_frame": rally_start_frame,
                        # End the rally RALLY_END_FRAMES ago (last time we saw movement)
                        "end_frame": frame_idx - RALLY_END_FRAMES,
                        "start_time": rally_start_time,
                        "end_time": round(timestamp - (RALLY_END_FRAMES / fps), 3),
                        "frame_count": (frame_idx - RALLY_END_FRAMES) - rally_start_frame
                    })
                    # Reset rally tracking state
                    in_rally = False
                    last_x = None
                    last_y = None
                    still_frame_count = 0

            # Nothing more to do if shuttle isn't detected
            continue

        # Shuttle WAS detected this frame — get its position
        current_x = frame_data["shuttle_x"]
        current_y = frame_data["shuttle_y"]

        if last_x is None:
            # First detection — can't measure movement yet, just store the position
            last_x = current_x
            last_y = current_y

            # If we weren't in a rally, start one
            if not in_rally:
                in_rally = True
                rally_id += 1
                rally_start_frame = frame_idx
                rally_start_time = timestamp
                still_frame_count = 0

            continue

        # Calculate how far the shuttle moved since the last detected frame
        # This is straight-line (Euclidean) distance in pixels
        movement = np.sqrt((current_x - last_x) ** 2 + (current_y - last_y) ** 2)

        if movement >= MIN_SHUTTLE_MOVEMENT_PX:

            # Shuttle is moving — this is an active rally frame
            still_frame_count = 0  # Reset the stillness counter

            if not in_rally:
                # Movement detected but we weren't tracking a rally — start one
                in_rally = True
                rally_id += 1
                rally_start_frame = frame_idx
                rally_start_time = timestamp

        else:

            # Shuttle barely moved — count as a still frame
            if in_rally:
                still_frame_count += 1

                if still_frame_count >= RALLY_END_FRAMES:
                    # End the current rally
                    rallies.append({
                        "rally_id": rally_id,
                        "start_frame": rally_start_frame,
                        "end_frame": frame_idx - RALLY_END_FRAMES,
                        "start_time": rally_start_time,
                        "end_time": round(timestamp - (RALLY_END_FRAMES / fps), 3),
                        "frame_count": (frame_idx - RALLY_END_FRAMES) - rally_start_frame
                    })
                    in_rally = False
                    last_x = None
                    last_y = None
                    still_frame_count = 0
                    continue

        # Update last known position
        last_x = current_x
        last_y = current_y

    # If we reached the end of the video while still inside a rally, close it out
    if in_rally and shuttle_positions:
        last_frame = shuttle_positions[-1]
        rallies.append({
            "rally_id": rally_id,
            "start_frame": rally_start_frame,
            "end_frame": last_frame["frame"],
            "start_time": rally_start_time,
            "end_time": last_frame["timestamp"],
            "frame_count": last_frame["frame"] - rally_start_frame
        })

    print(f"Found {len(rallies)} rally segments")
    for r in rallies:
        print(f"  Rally {r['rally_id']}: frames {r['start_frame']}–{r['end_frame']} "
              f"({r['start_time']}s – {r['end_time']}s)")

    return rallies


def visualize_shuttle(video_path: str, output_video_path: str, shuttle_positions: list):
    """
    Saves an annotated copy of the video with the shuttle position marked on each frame.
    Use this to visually verify detection is working — open the output video and check.

    Detected frames show a yellow circle at the shuttle position.
    Undetected frames are left unchanged.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties so output video matches input exactly
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Set up video writer
    # VideoWriter_fourcc(*"mp4v") sets the video compression format to MP4
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Convert the shuttle_positions list into a dict keyed by frame index
    # This makes it O(1) to look up the shuttle position for any frame
    positions_by_frame = {entry["frame"]: entry for entry in shuttle_positions}

    frame_index = 0

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Check if we have shuttle data for this frame
        frame_data = positions_by_frame.get(frame_index)

        if frame_data and frame_data["detected"]:

            # Draw a yellow circle at the shuttle's detected position
            # (0, 255, 255) is yellow in BGR color format
            x = int(frame_data["shuttle_x"])
            y = int(frame_data["shuttle_y"])

            # Outer circle — filled yellow dot
            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

            # Inner circle — white center so it's visible on light backgrounds
            cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)

            # Label the confidence score above the circle
            conf_text = f"{frame_data['confidence']:.2f}"
            cv2.putText(frame, conf_text, (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Write the annotated frame to the output video
        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"Saved shuttle visualization → {output_video_path}")


# ============================================================
# This block only runs when you execute this file directly.
# It won't run if this file is imported by another file.
# ============================================================

if __name__ == "__main__":

    # Path to the test video 
    VIDEO_PATH = "data/test_clip.MOV"
    #VIDEO_PATH = "data/test_clip_pro.MOV"

    # Step 1: extract shuttle positions for every frame and save to JSON
    shuttle_positions = get_shuttle_positions(
        video_path=VIDEO_PATH,
        output_json_path="outputs/shuttle_data.json"
    )

    # Step 2: group frames into rally segments based on shuttle movement
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    rallies = get_rally_segments(shuttle_positions, fps)

    # Step 3: save an annotated video so you can visually check detection quality
    visualize_shuttle(
        video_path=VIDEO_PATH,
        output_video_path="outputs/shuttle_overlay.mp4",
        shuttle_positions=shuttle_positions
    )
