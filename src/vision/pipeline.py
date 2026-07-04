# cv2 is the OpenCV library — used here only to check video FPS
import cv2
import json
import os

# Import the pose extraction function from the vision module
from src.vision.pose_estimator import extract_pose_data, select_player

# Import the shuttle tracking functions from the vision module
from src.vision.shuttle_tracker import get_shuttle_positions, get_rally_segments


def run_pipeline(video_path: str, output_dir: str = "outputs", player_box: tuple = None):
    """
    Runs the full CV pipeline on a badminton video:
      1. Extracts pose keypoints for the selected player (pose_estimator.py)
      2. Extracts shuttlecock positions for every frame (shuttle_tracker.py)
      3. Detects rally segments from shuttle movement (shuttle_tracker.py)
      4. Merges everything into one unified JSON per frame

    Each entry in the output looks like:
    {
        "frame": 142,
        "timestamp": 4.733,
        "rally_id": 2,          <- null if frame is between rallies
        "player_keypoints": [   <- list of 33 landmarks, or [] if not detected
            { "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99 },
            ...
        ],
        "shuttle_x": 320.5,     <- null if not detected
        "shuttle_y": 180.2,     <- null if not detected
        "shuttle_detected": true
    }

    Args:
        video_path:   Path to the input badminton video file.
        output_dir:   Folder where all output files will be saved.
        player_box:   (x1, y1, x2, y2) bounding box of the player to track.
                      If None, a selection window will open automatically.

    Returns:
        List of merged frame dicts (same data as saved to unified_data.json).
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # STEP 1: Player selection
    # If no player_box was passed in, open the click-to-select window
    # ============================================================

    if player_box is None:
        print("Step 1/4 — Select player...")
        print("  A window will open. Click on the player you want to analyze, then press any key.")
        player_box = select_player(video_path)
    else:
        print(f"Step 1/4 — Using provided player box: {player_box}")

    # ============================================================
    # STEP 2: Extract pose keypoints for selected player
    # Saves intermediate output to outputs/pose_data.json
    # ============================================================

    print("\nStep 2/4 — Extracting pose keypoints...")
    pose_json_path = os.path.join(output_dir, "pose_data.json")

    pose_frames = extract_pose_data(
        video_path=video_path,
        output_json_path=pose_json_path,
        player_box=player_box
    )

    # ============================================================
    # STEP 3: Extract shuttle positions for every frame
    # Saves intermediate output to outputs/shuttle_data.json
    # ============================================================

    print("\nStep 3/4 — Tracking shuttlecock...")
    shuttle_json_path = os.path.join(output_dir, "shuttle_data.json")

    shuttle_frames = get_shuttle_positions(
        video_path=video_path,
        output_json_path=shuttle_json_path
    )

    # ============================================================
    # STEP 3b: Detect rally segments from shuttle movement
    # No file output — we embed rally_id directly in each frame below
    # ============================================================

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    rallies = get_rally_segments(shuttle_frames, fps)

    # Build a frame → rally_id lookup dict so we can tag each frame efficiently
    # Any frame not covered by a rally will get rally_id = None
    frame_to_rally = {}
    for rally in rallies:
        for frame_idx in range(rally["start_frame"], rally["end_frame"] + 1):
            frame_to_rally[frame_idx] = rally["rally_id"]

    # ============================================================
    # STEP 4: Merge pose + shuttle + rally into one unified structure
    # Joined on frame index — both pipelines process the same video
    # so frame numbers are guaranteed to align
    # ============================================================

    print("\nStep 4/4 — Merging pose, shuttle, and rally data...")

    # Convert shuttle list to a dict keyed by frame index for O(1) lookups
    shuttle_by_frame = {entry["frame"]: entry for entry in shuttle_frames}

    unified_frames = []

    for pose_entry in pose_frames:

        frame_idx = pose_entry["frame"]
        timestamp = pose_entry["timestamp"]

        # Look up shuttle data for this frame (may be None if video lengths differ)
        shuttle_entry = shuttle_by_frame.get(frame_idx, {})

        # Build the unified entry
        unified_entry = {
            "frame": frame_idx,
            "timestamp": timestamp,

            # rally_id is None when the shuttle is between points / not moving
            "rally_id": frame_to_rally.get(frame_idx, None),

            # Pose keypoints — list of 33 landmarks, or empty list if player not detected
            "player_keypoints": pose_entry.get("keypoints", []),

            # Shuttle position — None values when shuttle wasn't detected this frame
            "shuttle_x": shuttle_entry.get("shuttle_x", None),
            "shuttle_y": shuttle_entry.get("shuttle_y", None),
            "shuttle_detected": shuttle_entry.get("detected", False),
        }

        unified_frames.append(unified_entry)

    # Save the merged output
    unified_json_path = os.path.join(output_dir, "unified_data.json")
    with open(unified_json_path, "w") as f:
        json.dump(unified_frames, f, indent=2)

    # ============================================================
    # Summary
    # ============================================================

    total_frames = len(unified_frames)
    pose_detected = sum(1 for f in unified_frames if f["player_keypoints"])
    shuttle_detected = sum(1 for f in unified_frames if f["shuttle_detected"])
    rally_frames = sum(1 for f in unified_frames if f["rally_id"] is not None)

    print(f"\nPipeline complete — {total_frames} frames processed")
    print(f"  Pose detected:    {pose_detected} / {total_frames} frames")
    print(f"  Shuttle detected: {shuttle_detected} / {total_frames} frames")
    print(f"  Rally frames:     {rally_frames} / {total_frames} frames ({len(rallies)} rallies)")
    print(f"  Output saved → {unified_json_path}")

    return unified_frames


# ============================================================
# This block only runs when you execute this file directly.
# It won't run if this file is imported by another agent module.
# ============================================================

if __name__ == "__main__":

    # Path to the test video 
    VIDEO_PATH = "data/test_clip.MOV"

    unified = run_pipeline(
        video_path=VIDEO_PATH,
        output_dir="outputs"
        # player_box not passed — will open click-to-select window
    )

    # Print first 3 entries so you can verify the structure looks right
    print("\nFirst 3 unified entries:")
    for entry in unified[:3]:
        print(f"  frame={entry['frame']}  t={entry['timestamp']}s  "
              f"rally={entry['rally_id']}  "
              f"keypoints={len(entry['player_keypoints'])}  "
              f"shuttle=({entry['shuttle_x']}, {entry['shuttle_y']})")