# cv2 is the OpenCV library — used for reading video files and processing frames
import cv2

# mediapipe is Google's library for detecting human body keypoints
import mediapipe as mp

# json is a built-in Python library for saving data as JSON files
import json

# os is a built-in Python library for file and folder operations
import os

# numpy is a math library — used here for distance calculation between two points
import numpy as np

# YOLO is the object detection model from the ultralytics library
from ultralytics import YOLO

# Load YOLOv8 model for person detection
# "yolov8n.pt" is the model weights file — downloads automatically on first run
yolo_model = YOLO("yolov8n.pt")


def detect_players(frame):
    '''
    Uses YOLOv8 to detect players in a frame.
    Returns a list of bounding boxes for detected players: [(x1, y1, x2, y2), ...]
    '''
    results = yolo_model(frame, verbose = False)
    boxes = []

    for result in results: 
        #Loop through each individual detected object in the frame 
        for box in result.boxes:
            # Class 0 means "person" in YOLOv8
            if int(box.cls) == 0 and float(box.conf) > 0.5:
                # Converts the coordinates from floats to integers
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
    return boxes 


def select_player(video_path: str):
    '''
    Shows the first frame of the video with all detected players highlighted.
    User clicks in the player they want to track.
    Returns the bounding box of the selected player: (x1, y1, x2, y2)
    '''
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Could not read video: {video_path}")
    
    boxes = detect_players(first_frame)
    if not boxes:
        raise ValueError("No players detected in the first frame.")
    
    # Make a copy of the first frame to draw on, so we don't modify the original frame data
    display_frame = first_frame.copy()

    # Loop through each detected player and draw a green box around them
    for i, (x1,y1,x2,y2) in enumerate(boxes):
        # Draw a green rectangle around the player
        # 2 is the thickness of the rectangle border in pixels
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add a label above each box so user knows which player is which
        # (x1, y1 - 10) places the text just above the top-left corner of the box
        cv2.putText(display_frame, f"Player {i + 1}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Empty list to store where the user clicks 
    click_coords = []
    
    # This function runs automatically whenever the user clicks on the window
    def on_click(event, x, y, flags, param):
        # EVENT_LBUTTONDOWN means the left mouse button was clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            click_coords.append((x,y))

    # Show the frame, listen for clicks, wait for keypress, then clean up
    cv2.imshow("Click on the player you want to analyze (then press any key)", display_frame)
    cv2.setMouseCallback("Click on the player you want to analyze (then press any key)", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if not click_coords:
        raise ValueError("No player was selected. You need to click on a player.")
    click_x, click_y = click_coords[0]
    selected_box = None
    min_distance = float("inf")

    # Loop through every detected player box to find which one is closest to the click
    for (x1, y1, x2, y2) in boxes:
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance = np.sqrt((click_x - center_x) ** 2 + (click_y - center_y) ** 2)
        if distance < min_distance:
            min_distance = distance
            selected_box = (x1, y1, x2, y2)
    print(f"Player selected at bounding box: {selected_box}")
    return selected_box


def crop_to_player(frame, box, padding=40):
    """
    Crops a frame to just the selected player's bounding box.
    Adds a small padding around the box so we don't cut off limbs.
    """
 
    x1, y1, x2, y2 = box
 
    # Get the full height and width of the frame
    # shape returns (height, width, channels) 
    h, w = frame.shape[:2]
 
    # Expand the box slightly in all directions so we don't clip the player's limbs
    # max(0, ...) prevents going above or left of the frame edge
    # min(w/h, ...) prevents going below or right of the frame edge
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
 
    # Crop the frame to just the player region
    # frame[y1:y2, x1:x2] is NumPy array slicing — rows from y1 to y2, columns from x1 to x2
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def extract_pose_data(video_path: str, output_json_path: str, player_box: tuple):
    """
    Extracts pose keypoints for every frame, analyzing only the selected player.
    Re-detects the player position every frame so the crop follows the player
    as they move around the court.
    Saves results as a JSON file with frame index, timestamp, and 33 body landmarks.
    """
 
    # Access MediaPipe's pose detection module
    mp_pose = mp.solutions.pose
    
    # Opens the video file and gets frames per second to calculate timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
 
    all_frames = []
    frame_index = 0
    # Start with the user's selected bounding box as the last known position
    last_known_box = player_box
 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            # Run YOLO on the current frame to find all visible players
            boxes = detect_players(frame)
 
            # Find which detected player is our selected player 
            if boxes:

                # Calculate the center of the last known box where our player was in the previous frame
                last_center_x = (last_known_box[0] + last_known_box[2]) / 2
                last_center_y = (last_known_box[1] + last_known_box[3]) / 2
                min_distance = float("inf")
                closest_box = None
 
                for (x1, y1, x2, y2) in boxes:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
 
                    # Calculate straight-line distance between this detection
                    # and where our player was last seen
                    distance = np.sqrt(
                        (center_x - last_center_x) ** 2 +
                        (center_y - last_center_y) ** 2
                    )
 
                    if distance < min_distance:
                        min_distance = distance
                        closest_box = (x1, y1, x2, y2)
 
                # Only update last known box to the closest detection if it is within a reasonable distance
                # This becomes the starting point for the next frame's search 
                if min_distance < 300:
                    last_known_box = closest_box


            # Crop the frame to just the selected player's current position
            cropped_frame, adjusted_box = crop_to_player(frame, last_known_box)
 
            # MediaPipe requires RGB, OpenCV loads BGR
            rgb_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
 
            # Run pose detection on the cropped player region
            results = pose.process(rgb_frame)
 
            # Empty list to store keypoints for this frame
            keypoints = []
 
            # Only extract keypoints if a person was detected
            if results.pose_landmarks:
 
                # Loop through all 33 body landmarks
                for landmark in results.pose_landmarks.landmark:
 
                    # x, y, z are fractions of the cropped region (0.0-1.0), not pixel coordinates
                    keypoints.append({
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                        "visibility": landmark.visibility
                    })
 
            all_frames.append({
                "frame": frame_index,
                # Convert frame number to seconds using fps
                "timestamp": round(frame_index / fps, 3),
                "keypoints": keypoints
            })
            frame_index += 1
 
    cap.release()
 
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
 
    # Save all frame data to a JSON file
    # indent=2 makes the JSON human-readable
    with open(output_json_path, "w") as f:
        json.dump(all_frames, f, indent=2)
 
    print(f"Done! Processed {frame_index} frames → {output_json_path}")
    return all_frames


def visualize_pose(video_path: str, output_video_path: str, player_box: tuple):
    """
    Saves a copy of the full video with skeleton overlay drawn on the selected player only.
    Re-detects the player position every frame so the skeleton follows the player
    as they move around the court.
    Use this to visually verify pose detection is working correctly.
    """

    # Access MediaPipe's pose detection and drawing modules
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
 
    # Open the video file and get video properties
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
 
    # Set up video writer to save the annotated output video
    # VideoWriter_fourcc(*"mp4v") sets the video compression format to MP4
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
 
    last_known_box = player_box
 
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
 
            boxes = detect_players(frame)
            if boxes:
                last_center_x = (last_known_box[0] + last_known_box[2]) / 2
                last_center_y = (last_known_box[1] + last_known_box[3]) / 2
                min_distance = float("inf")
                closest_box = None

                for (x1, y1, x2, y2) in boxes:
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    distance = np.sqrt(
                        (center_x - last_center_x) ** 2 +
                        (center_y - last_center_y) ** 2
                    )

                    if distance < min_distance:
                        min_distance = distance
                        closest_box = (x1, y1, x2, y2)
 
                if min_distance < 300:
                    last_known_box = closest_box
 
            cropped_frame, (cx1, cy1, cx2, cy2) = crop_to_player(frame, last_known_box)
            rgb_cropped = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_cropped)
 
            # If a person was detected, draw the skeleton
            if results.pose_landmarks:
 
                # Draw the skeleton lines and joint dots on the cropped frame
                mp_drawing.draw_landmarks(
                    # Frame to draw on
                    cropped_frame,
                    # The 33 detected landmark positions
                    results.pose_landmarks,
                    # Lines connecting the landmarks (e.g. shoulder → elbow → wrist)
                    mp_pose.POSE_CONNECTIONS
                )
 
                # Paste the annotated cropped region back into the full frame
                # This keeps the rest of the video intact while only annotating our player
                frame[cy1:cy2, cx1:cx2] = cropped_frame
 
            # Write the full annotated frame to the output video
            out.write(frame)
 
    cap.release()
    out.release()
    print(f"Saved annotated video → {output_video_path}")


# ============================================================
# This block only runs when you execute this file directly.
# It won't run if this file is imported by another file.
# ============================================================

if __name__ == "__main__":

    #VIDEO_PATH = "data/test_clip.MOV"
    VIDEO_PATH = "data/test_clip_pro.MOV"

    # Step 1: click to select player
    player_box = select_player(VIDEO_PATH)

    # Step 2: extract keypoints → saves pose_data.json
    extract_pose_data(
        video_path=VIDEO_PATH,
        output_json_path="outputs/pose_data.json",
        player_box=player_box
    )

    # Step 3: generate annotated video → saves pose_overlay.mp4
    visualize_pose(
        video_path=VIDEO_PATH,
        output_video_path="outputs/pose_overlay.mp4",
        player_box=player_box
    )   


