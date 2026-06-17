import json
import os
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Sample every Nth frame from rally frames
# At 30fps, every 5th frame = ~6 samples per second
# Increase this number for longer videos to keep data manageable
SAMPLE_EVERY_N_FRAMES = 5

# Minimum visibility score to trust a keypoint
# MediaPipe assigns 0.0-1.0 — below 0.5 means the landmark is
# partially hidden or MediaPipe is uncertain about its position
MIN_VISIBILITY = 0.5


# ============================================================
# MEDIAPIPE LANDMARK INDICES
# These are the keypoint indices most relevant to badminton
# technique — racket arm, non-racket arm, hips, and shoulders
# ============================================================

LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28
    #"left_foot_index": 31,    
    #"right_foot_index": 32   
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_keypoint(keypoints: list, index: int) -> dict | None:
    """
    Returns a single keypoint by index if its visibility is above threshold.
    Returns None if the keypoint is not visible enough to trust.

    Args:
        keypoints: List of 33 keypoint dicts from unified_data.json
        index: The MediaPipe landmark index (0-32)

    Returns:
        Keypoint dict with x, y, z, visibility — or None if not reliable
    """

    if index >= len(keypoints):
        return None

    kp = keypoints[index]

    # Only return the keypoint if MediaPipe is confident it's visible
    if kp.get("visibility", 0) < MIN_VISIBILITY:
        return None

    return kp


def calculate_angle(a: dict, b: dict, c: dict) -> float | None:
    """
    Calculates the angle at point B formed by points A-B-C.
    Used to measure joint angles like elbow bend or knee flex.

    For example:
      A = shoulder, B = elbow, C = wrist → gives elbow angle
      A = hip, B = knee, C = ankle → gives knee angle

    Returns angle in degrees, or None if any keypoint is missing.
    """

    if a is None or b is None or c is None:
        return None

    # Convert keypoint dicts to numpy arrays (using x, y only — ignoring z)
    a_vec = np.array([a["x"], a["y"]])
    b_vec = np.array([b["x"], b["y"]])
    c_vec = np.array([c["x"], c["y"]])

    # Vectors from B to A and from B to C
    ba = a_vec - b_vec
    bc = c_vec - b_vec

    # Cosine of the angle between the two vectors
    # Clamp to [-1, 1] to avoid floating point errors in arccos
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)

    # Convert from radians to degrees
    angle = np.degrees(np.arccos(cosine))

    return round(float(angle), 1)


def extract_pose_observations(frame: dict, side: str = "right") -> dict:
    """
    Converts raw keypoint data for one frame into human-readable
    pose observations the Orchestrator can reason about.

    Args:
        frame: One entry from unified_data.json with player_keypoints
        side:  "right" or "left" — which is the player's racket arm.
               Defaults to "right". This affects which elbow/wrist
               is treated as the racket arm in observations.

    Returns:
        Dict of observations for this frame, e.g.:
        {
            "timestamp": 4.033,
            "frame": 121,
            "rally_id": 1,
            "shuttle_detected": True,
            "elbow_angle": 95.2,
            "knee_angle_racket_side": 142.3,
            "observations": ["elbow angle is 95°"]
        }
    """

    keypoints = frame.get("player_keypoints", [])
    timestamp = frame.get("timestamp", 0)
    frame_idx = frame.get("frame", 0)
    rally_id = frame.get("rally_id")
    shuttle_detected = frame.get("shuttle_detected", False)

    # Determine which side is the racket arm
    if side == "right":
        shoulder_idx = LANDMARKS["right_shoulder"]
        elbow_idx = LANDMARKS["right_elbow"]
        wrist_idx = LANDMARKS["right_wrist"]
        hip_idx = LANDMARKS["right_hip"]
        knee_idx = LANDMARKS["right_knee"]
        ankle_idx = LANDMARKS["right_ankle"]
        opp_shoulder_idx = LANDMARKS["left_shoulder"]
        opp_hip_idx = LANDMARKS["left_hip"]
    else:
        shoulder_idx = LANDMARKS["left_shoulder"]
        elbow_idx = LANDMARKS["left_elbow"]
        wrist_idx = LANDMARKS["left_wrist"]
        hip_idx = LANDMARKS["left_hip"]
        knee_idx = LANDMARKS["left_knee"]
        ankle_idx = LANDMARKS["left_ankle"]
        opp_shoulder_idx = LANDMARKS["right_shoulder"]
        opp_hip_idx = LANDMARKS["right_hip"]

    # Get the keypoints we need
    shoulder = get_keypoint(keypoints, shoulder_idx)
    elbow = get_keypoint(keypoints, elbow_idx)
    wrist = get_keypoint(keypoints, wrist_idx)
    hip = get_keypoint(keypoints, hip_idx)
    knee = get_keypoint(keypoints, knee_idx)
    ankle = get_keypoint(keypoints, ankle_idx)

    # Calculate joint angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    knee_angle = calculate_angle(hip, knee, ankle)

    # Build human-readable observations — raw facts, not judgments.
    # Whether an angle is "good" or "bad" depends on shot type/phase,
    # which isn't known here; that interpretation happens downstream
    # with retrieved coaching context.
    observations = []

    if elbow_angle is not None:
        observations.append(f"elbow angle is {elbow_angle}°")

    if knee_angle is not None:
        observations.append(f"knee angle is {knee_angle}°")

    if shuttle_detected:
        observations.append("shuttle detected this frame")

    return {
        "timestamp": timestamp,
        "frame": frame_idx,
        "rally_id": rally_id,
        "shuttle_detected": shuttle_detected,
        "elbow_angle": elbow_angle,
        "knee_angle": knee_angle,
        "observations": observations
    }


# ============================================================
# MAIN VISION AGENT FUNCTION
# ============================================================

def analyze_video(unified_data_path: str, racket_arm: str = "right") -> list:
    """
    Main entry point for the Vision Agent.

    Loads the unified data from the CV pipeline, samples every Nth frame
    from rally segments, extracts pose observations from each sampled frame,
    and returns a list of observations the Orchestrator can reason about.

    Args:
        unified_data_path: Path to unified_data.json from run_pipeline()
        racket_arm:        "right" or "left" — player's racket arm side

    Returns:
        List of observation dicts, one per sampled frame, sorted by timestamp.
        Each dict contains timestamp, frame index, joint angles, and
        human-readable observation strings.
    """

    # Load the unified data
    if not os.path.exists(unified_data_path):
        raise FileNotFoundError(
            f"Unified data not found: {unified_data_path}\n"
            "Run pipeline.py first to generate it."
        )

    with open(unified_data_path, "r") as f:
        all_frames = json.load(f)

    print(f"Loaded {len(all_frames)} frames from {unified_data_path}")

    # Filter to only rally frames — non-rally frames are between points
    # and don't contain useful technique data
    rally_frames = [f for f in all_frames if f.get("rally_id") is not None]
    print(f"Rally frames: {len(rally_frames)} / {len(all_frames)}")

    # Sample every Nth frame to avoid redundant data
    # Consecutive frames are nearly identical at 30fps
    sampled_frames = rally_frames[::SAMPLE_EVERY_N_FRAMES]
    print(f"Sampled frames: {len(sampled_frames)} (every {SAMPLE_EVERY_N_FRAMES}th frame)")

    # Extract pose observations from each sampled frame
    observations = []

    for frame in sampled_frames:

        # Skip frames with no keypoint data
        # (player was not detected by MediaPipe in this frame)
        if not frame.get("player_keypoints"):
            continue

        obs = extract_pose_observations(frame, side=racket_arm)

        # Only include frames where we could extract at least one observation
        if obs["observations"]:
            observations.append(obs)

    print(f"Extracted observations from {len(observations)} frames")

    return observations


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":

    # Path to your unified data — update to match your actual output
    UNIFIED_DATA_PATH = "outputs/unified_data.json"

    # Change to "left" if the player is left-handed
    RACKET_ARM = "right"

    # Run the vision agent
    observations = analyze_video(UNIFIED_DATA_PATH, racket_arm=RACKET_ARM)

    # Print a summary
    print(f"\nTotal observations: {len(observations)}")
    print("\nFirst 3 sampled frames:")
    for obs in observations[:3]:
        print(f"\n  t={obs['timestamp']}s  frame={obs['frame']}  rally={obs['rally_id']}")
        print(f"  Elbow angle: {obs['elbow_angle']}°")
        print(f"  Knee angle:  {obs['knee_angle']}°")
        print(f"  Shuttle:     {obs['shuttle_detected']}")
        print(f"  Notes:")
        for note in obs["observations"]:
            print(f"    - {note}")