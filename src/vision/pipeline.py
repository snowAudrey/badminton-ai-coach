import json
import os

# Import the pose extraction function from the vision module
from src.vision.pose_estimator import extract_pose_data, select_player

# Import the shuttle tracking functions from the vision module
from src.vision.shuttle_tracker import get_shuttle_positions


def run_pipeline(video_path: str, output_dir: str = "outputs", player_box: tuple = None):
    """
    Runs the full CV pipeline on a badminton video:
      1. Extracts pose keypoints for the selected player (pose_estimator.py)
      2. Extracts shuttlecock positions for every frame (shuttle_tracker.py)
      3. Merges everything into one unified JSON per frame

    Each entry in the output looks like:
    {
        "frame": 142,
        "timestamp": 4.733,
        "player_keypoints": [   <- list of 33 landmarks, or [] if not detected
            { "x": 0.5, "y": 0.3, "z": -0.1, "visibility": 0.99,
              "px": 612.0, "py": 744.5 },   <- px/py are full-frame pixels
            ...
        ],
        "world_keypoints": [    <- same 33 landmarks in 3D metres (hip-centred)
            { "x": 0.12, "y": -0.41, "z": 0.05, "visibility": 0.99 },
            ...
        ],
        "crop_box": [x1, y1, x2, y2],   <- full-frame pixel box the player was cropped from
        "shuttle_x": 320.5,     <- null if not detected
        "shuttle_y": 180.2,     <- null if not detected
        "shuttle_confidence": 0.63,   <- null if not detected
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
        print("Step 1/3 — Select player...")
        print("  A window will open. Click on the player you want to analyze, then press any key.")
        player_box = select_player(video_path)
    else:
        print(f"Step 1/3 — Using provided player box: {player_box}")

    # ============================================================
    # STEP 2: Extract pose keypoints for selected player
    # Saves intermediate output to outputs/pose_data.json
    # ============================================================

    print("\nStep 2/3 — Extracting pose keypoints...")
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

    print("\nStep 3/3 — Tracking shuttlecock...")
    shuttle_json_path = os.path.join(output_dir, "shuttle_data.json")

    shuttle_frames = get_shuttle_positions(
        video_path=video_path,
        output_json_path=shuttle_json_path
    )

    # ============================================================
    # MERGE: pose + shuttle into one unified structure
    # Joined on frame index — both pipelines process the same video
    # so frame numbers are guaranteed to align
    # ============================================================

    print("\nMerging pose and shuttle data...")

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

            # Pose keypoints — list of 33 landmarks, or empty list if player not detected
            "player_keypoints": pose_entry.get("keypoints", []),

            # The same landmarks in 3D metres (used for joint angles), and the
            # full-frame pixel box the player was cropped from
            "world_keypoints": pose_entry.get("world_keypoints", []),
            "crop_box": pose_entry.get("crop_box", None),

            # Shuttle position — None values when shuttle wasn't detected this frame
            "shuttle_x": shuttle_entry.get("shuttle_x", None),
            "shuttle_y": shuttle_entry.get("shuttle_y", None),
            "shuttle_confidence": shuttle_entry.get("confidence", None),
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

    print(f"\nPipeline complete — {total_frames} frames processed")
    print(f"  Pose detected:    {pose_detected} / {total_frames} frames")
    print(f"  Shuttle detected: {shuttle_detected} / {total_frames} frames")
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
              f"keypoints={len(entry['player_keypoints'])}  "
              f"shuttle=({entry['shuttle_x']}, {entry['shuttle_y']})")