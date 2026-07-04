# Import the OpenCV library for reading video files and frames
import cv2

# Import all four functions from shuttle_tracker
# "from src.vision.shuttle_tracker" tells Python where to find the file
# src/ → vision/ → shuttle_tracker.py
from src.vision.shuttle_tracker import detect_shuttle, get_shuttle_positions, get_rally_segments, visualize_shuttle

# ============================================================
# IMPORTANT: Update this to match your actual video filename
# Run "ls data/" in the terminal to see your exact filename
# ============================================================
#VIDEO_PATH = "data/test_clip_pro.MOV"
VIDEO_PATH = "data/test_clip.MOV"

# ============================================================
# STEP 1: Test detect_shuttle()
# Goal: confirm the function runs on a single frame without crashing
# ============================================================

# Open the video file so we can read a single frame
cap = cv2.VideoCapture(VIDEO_PATH)

# Read the very first frame from the video
# ret = True if the frame was read successfully, False if something went wrong
# frame = the actual image data for that frame
ret, frame = cap.read()

# Close the video file — we only needed one frame for this test
cap.release()

# Check if the frame loaded correctly
if not ret:
    # If ret is False, the video couldn't be opened — likely wrong path or format
    print("ERROR: Could not read video. Check your file path and format.")
else:
    # Run shuttle detection on the single frame
    # Returns (x, y, confidence) if found, or None if not found
    detection = detect_shuttle(frame)

    if detection is not None:
        # Unpack the result — x and y are pixel coordinates, conf is 0.0–1.0
        x, y, conf = detection
        print(f"Shuttle detected at pixel ({x:.1f}, {y:.1f}) with confidence {conf:.2f}")
    else:
        # None means no detection — expected with the general model
        print("No shuttle detected in the first frame (expected with general model — not a bug)")


# ============================================================
# STEP 2: Test get_shuttle_positions()
# Goal: confirm the function processes the full video and saves a JSON
#
# This will take a minute depending on video length — watch the progress prints.
#
# What to check when it's done:
#   1. No crash — it ran to completion
#   2. Open outputs/shuttle_data.json in VS Code
#   3. Confirm every frame has an entry with frame, timestamp, detected fields
#   4. Check the final print — it tells you how many frames had detections
#      With the general model, expect a low detection count — that's fine for now
# ============================================================

# Run shuttle detection across the entire video
# Saves results to outputs/shuttle_data.json automatically
shuttle_positions = get_shuttle_positions(
    video_path=VIDEO_PATH,
    output_json_path="outputs/shuttle_data.json"
)

# Print how many total frames were processed
print(f"Total frames processed: {len(shuttle_positions)}")

# Print the first 3 entries so you can see the data structure
print("First 3 frame entries:")
for entry in shuttle_positions[:3]:
    print(f"  {entry}")


# ============================================================
# STEP 3: Test get_rally_segments()
# Goal: confirm the function groups frames into rally segments
#
# What to check:
#   - It prints a list of rallies with start/end times
#   - With low shuttle detection (from the general model), may get 0 rallies
#     or very short/noisy ones — that is expected and will improve with better weights
# ============================================================

# Pass the shuttle positions into the rally segmentation function
# This doesn't need the video — it works purely from the position data
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()
rallies = get_rally_segments(shuttle_positions, fps)

# Print a summary of what was found
print(f"Total rallies found: {len(rallies)}")

# Print each rally's details
for rally in rallies:
    print(f"  Rally {rally['rally_id']}: "
          f"{rally['start_time']}s → {rally['end_time']}s  "
          f"({rally['frame_count']} frames)")


# ============================================================
# STEP 4: Test visualize_shuttle()
# Goal: save an annotated video for visually checking detection quality
#
# What to check:
#   - Open outputs/shuttle_overlay.mp4 after this runs
#   - Frames where the shuttle was detected should show a yellow dot
# ============================================================

# Save an annotated copy of the video with yellow dots where the shuttle was detected
visualize_shuttle(
    video_path=VIDEO_PATH,
    output_video_path="outputs/shuttle_overlay.mp4",
    shuttle_positions=shuttle_positions
)

# Remind the user where to find the output
print("Done! Open outputs/shuttle_overlay.mp4 to visually check detection quality.")
print("Yellow dots = shuttle detected. No dots = model couldn't find the shuttle.")
