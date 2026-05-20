import cv2 
from src.vision.pose_estimator import detect_players, select_player, crop_to_player, extract_pose_data, visualize_pose
import json

VIDEO_PATH = "data/test_clip.MOV"


# ============================================================
# STEP 1: Test detect_players()
# Goal: confirm YOLOv8 can detect players in the first frame
# ============================================================
# Open the video file to read frames from it
cap = cv2.VideoCapture(VIDEO_PATH)

# Read the first frame from the video
# ret = True if the frame was read successfully, False otherwise
# frame = the actual image data of the frame
ret, frame = cap.read()

# Close the video file 
cap.release()

if not ret:
    print("ERROR: Could not read frame from video.")
else:
    boxes = detect_players(frame)
    print(f"Number of players detected: {len(boxes)}")

    # Loop through each detected player and print their bounding box coordinates
    for i, box in enumerate(boxes):
        print(f"    Player {i + 1}: {box}")


# ============================================================
# STEP 2: Test select_player()
# Goal: confirm a window opens and user can click on a player
# A window will pop up — click on your player then press any key
# ============================================================
 
# Open a window showing the first frame with all detected players highlighted
# Returns the bounding box of whichever player the user clicks on
box = select_player(VIDEO_PATH)
 
# Print the bounding box of the player the user selected
print(f"Selected box: {box}")


# ============================================================
# STEP 3: Test crop_to_player()
# Goal: confirm the frame is cropped to just the selected player
# Check outputs/cropped_player.jpg to visually verify
# ============================================================
 
# Open the video file again to get a fresh frame
cap = cv2.VideoCapture(VIDEO_PATH)
 
# Read the first frame
ret, frame = cap.read()
 
# Close the video file
cap.release()
 
# Only run if the frame was read successfully
if ret:
    # Crop the frame to just the selected player's bounding box
    # cropped = the cropped image data
    # adjusted_box = the bounding box coordinates after padding was added
    cropped, adjusted_box = crop_to_player(frame, box)
 
    # Save the cropped frame as an image file 
    # Open outputs/cropped_player.jpg and confirm it shows just one player
    cv2.imwrite("outputs/cropped_player.jpg", cropped)
    print(f"Cropped frame saved — check outputs/cropped_player.jpg")
    print(f"Adjusted box with padding: {adjusted_box}")
 

 # ============================================================
# STEP 4: Test extract_pose_data()
# Goal: confirm MediaPipe extracts keypoints for the selected
# player and saves them correctly to a JSON file
# ============================================================
 
# Run pose extraction on the full video for the selected player
# box = the bounding box from select_player() in Step 2
all_frames = extract_pose_data(
    video_path=VIDEO_PATH,
    output_json_path="outputs/pose_data.json",
    player_box=box
)
 
# ── Basic checks ──────────────────────────────────────────
 
# Print how many frames were processed
# Should match the length of your video × frames per second
# For a 16 second video at 30fps you'd expect ~480 frames
print(f"Total frames processed: {len(all_frames)}")
 
# Print the first frame's data so we can inspect the structure
# Should show frame index 0, timestamp 0.0, and 33 keypoints
print(f"\nFirst frame data:")
print(f"  Frame index: {all_frames[0]['frame']}")
print(f"  Timestamp: {all_frames[0]['timestamp']}")
print(f"  Number of keypoints: {len(all_frames[0]['keypoints'])}")
 
# Print the last frame's data to confirm timestamps are correct
# For a 16 second video the last timestamp should be close to 16.0
print(f"\nLast frame data:")
print(f"  Frame index: {all_frames[-1]['frame']}")
print(f"  Timestamp: {all_frames[-1]['timestamp']}")
print(f"  Number of keypoints: {len(all_frames[-1]['keypoints'])}")
 
# ── Keypoint checks ───────────────────────────────────────
 
# Count how many frames successfully detected a person
# If this is very low, MediaPipe is struggling to detect the player
frames_with_keypoints = [f for f in all_frames if len(f['keypoints']) > 0]
print(f"\nFrames with detected keypoints: {len(frames_with_keypoints)} / {len(all_frames)}")
 
# Print the first keypoint of the first frame that has detections
# Each keypoint should have x, y, z, and visibility values between 0 and 1
first_detected_frame = frames_with_keypoints[0]
print(f"\nSample keypoint from frame {first_detected_frame['frame']}:")
print(f"  {first_detected_frame['keypoints'][0]}")
 
# ── JSON file checks ──────────────────────────────────────
 
# Open the saved JSON file and confirm it was written correctly
with open("outputs/pose_data.json", "r") as f:
    # Load the JSON data back into Python
    saved_data = json.load(f)
 
# Confirm the number of frames in the file matches what we processed
print(f"\nFrames saved in JSON file: {len(saved_data)}")
 
# Confirm the JSON file structure looks correct
# Should print the same values as all_frames[0] above
print(f"First frame in JSON file:")
print(f"  Frame index: {saved_data[0]['frame']}")
print(f"  Timestamp: {saved_data[0]['timestamp']}")
print(f"  Number of keypoints: {len(saved_data[0]['keypoints'])}")


# ============================================================
# STEP 5: Test visualize_pose()
# Goal: confirm an annotated video is saved with skeleton
# drawn on the selected player only
# Check outputs/pose_overlay.mp4 to visually verify
# ============================================================
 
# Run visualize_pose on the full video for the selected player
# box = the bounding box from select_player() in Step 2
# This will take a bit longer than Step 4 since it writes a video file
visualize_pose(
    video_path=VIDEO_PATH,
    output_video_path="outputs/pose_overlay.mp4",
    player_box=box
)
 
# ── File check ────────────────────────────────────────────
 
# Import os so we can check if the output file was created
import os
 
# Check if the output video file was actually created
if os.path.exists("outputs/pose_overlay.mp4"):
 
    # Get the file size in megabytes so we know it's not empty
    file_size_mb = os.path.getsize("outputs/pose_overlay.mp4") / (1024 * 1024)
 
    # Print confirmation that the file exists and its size
    print(f"pose_overlay.mp4 saved successfully")
    print(f"File size: {file_size_mb:.2f} MB")
 
else:
    # If the file doesn't exist, something went wrong
    print("ERROR: pose_overlay.mp4 was not created — check for errors above")
 
# ── Visual check ──────────────────────────────────────────
# Open outputs/pose_overlay.mp4 and confirm:
# 1. The full court is visible — not just the cropped player region
# 2. Skeleton is drawn on your selected player only
# 3. Other players have no skeleton drawn on them
# 4. The skeleton follows the player as they move around the court
print(f"\nOpen outputs/pose_overlay.mp4 and visually confirm:")
print(f"  1. Full court is visible")
print(f"  2. Skeleton appears on your selected player only")
print(f"  3. Skeleton follows the player as they move")
 