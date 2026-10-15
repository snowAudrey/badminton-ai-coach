import json
import os

import cv2
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

# Sample every Nth frame of the clip
# At 30fps, every 5th frame = ~6 samples per second
# Increase this number for longer videos to keep data manageable
SAMPLE_EVERY_N_FRAMES = 5

# Minimum visibility score to trust a keypoint. MediaPipe assigns 0.0-1.0.
# When the camera is behind the player, the racket arm spends much of each
# swing behind the body and only scores 0.3-0.5 even when its position is
# still well placed, so the cutoff is deliberately below the usual 0.5.
# Below ~0.2 the positions do become unreliable.
MIN_VISIBILITY = 0.3

# Warn when the racket arm could be measured in fewer than this share of
# the sampled frames — usually the wrong side was chosen, or the camera
# angle hides the racket arm
MIN_RACKET_ARM_COVERAGE = 0.4


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
}

# The joints that matter on each side, so the racket side can be swapped
# with one lookup
SIDES = {
    side: {
        joint: LANDMARKS[f"{side}_{joint}"]
        for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")
    }
    for side in ("right", "left")
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
        Keypoint dict — or None if not reliable
    """

    if index >= len(keypoints):
        return None

    kp = keypoints[index]

    # Only return the keypoint if MediaPipe is confident it's visible
    if kp.get("visibility", 0) < MIN_VISIBILITY:
        return None

    return kp


def _x(kp: dict) -> float:
    """Horizontal position in full-frame pixels (falls back to crop fraction for old data)."""
    return kp.get("px", kp["x"])


def _y(kp: dict) -> float:
    """Vertical position in full-frame pixels (falls back to crop fraction for old data)."""
    return kp.get("py", kp["y"])


def _as_vector(kp: dict, use_3d: bool) -> np.ndarray:
    """Turns a keypoint dict into the coordinates used for angle maths."""
    if use_3d:
        return np.array([kp["x"], kp["y"], kp["z"]])
    return np.array([_x(kp), _y(kp)])


def calculate_angle(a: dict, b: dict, c: dict, use_3d: bool = False) -> float | None:
    """
    Calculates the angle at point B formed by points A-B-C.
    Used to measure joint angles like elbow bend or knee flex.

    For example:
      A = shoulder, B = elbow, C = wrist → gives elbow angle
      A = hip, B = knee, C = ankle → gives knee angle

    With use_3d=True the points are MediaPipe world landmarks (metres),
    which gives a true 3D joint angle. Otherwise the points are 2D pixel
    positions, which is only an approximation of the real angle.

    Returns angle in degrees, or None if any keypoint is missing.
    """

    if a is None or b is None or c is None:
        return None

    a_vec = _as_vector(a, use_3d)
    b_vec = _as_vector(b, use_3d)
    c_vec = _as_vector(c, use_3d)

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
        side:  "right" or "left" — which MediaPipe side is the racket arm
               (confirm it with the check image, since MediaPipe can mirror
               sides when it sees a player from behind)

    Returns:
        Dict of observations for this frame, e.g.:
        {
            "timestamp": 4.033,
            "frame": 121,
            "shuttle_detected": True,
            "elbow_angle": 95.2,
            "knee_angle": 142.3,
            "wrist_height": "overhead",
            "observations": ["elbow angle is 95.2°", ...]
        }
    """

    keypoints = frame.get("player_keypoints", [])
    world = frame.get("world_keypoints", [])
    timestamp = frame.get("timestamp", 0)
    frame_idx = frame.get("frame", 0)
    shuttle_detected = frame.get("shuttle_detected", False)

    idx = SIDES[side]

    # Get the keypoints we need (image coordinates, with visibility check)
    nose = get_keypoint(keypoints, LANDMARKS["nose"])
    shoulder = get_keypoint(keypoints, idx["shoulder"])
    elbow = get_keypoint(keypoints, idx["elbow"])
    wrist = get_keypoint(keypoints, idx["wrist"])
    hip = get_keypoint(keypoints, idx["hip"])
    knee = get_keypoint(keypoints, idx["knee"])
    ankle = get_keypoint(keypoints, idx["ankle"])

    # Calculate joint angles. Prefer the 3D world landmarks: the player crop
    # is tall and narrow, so angles measured from crop fractions are
    # distorted. Fall back to pixel coordinates for data saved before world
    # landmarks were recorded.
    if len(world) == 33:
        elbow_angle = calculate_angle(get_keypoint(world, idx["shoulder"]),
                                      get_keypoint(world, idx["elbow"]),
                                      get_keypoint(world, idx["wrist"]), use_3d=True)
        knee_angle = calculate_angle(get_keypoint(world, idx["hip"]),
                                     get_keypoint(world, idx["knee"]),
                                     get_keypoint(world, idx["ankle"]), use_3d=True)
    else:
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)

    # Wrist height relative to head and shoulder. Clear, smash, and drop are
    # hit overhead (above head height); drive and net are hit just above
    # shoulder height; lift and defensive shots are hit below shoulder height.
    # Image y grows downward, so a smaller y means higher up.
    wrist_height = None
    if wrist is not None and nose is not None and shoulder is not None:
        if _y(wrist) < _y(nose):
            wrist_height = "overhead"
        elif _y(wrist) < _y(shoulder):
            wrist_height = "above shoulder"
        else:
            wrist_height = "below shoulder"

    # Build human-readable observations — raw facts, not judgments.
    # Whether an angle is "good" or "bad" depends on shot type/phase,
    # which isn't known here; that interpretation happens downstream
    # with retrieved coaching context.
    observations = []

    if elbow_angle is not None:
        observations.append(f"elbow angle is {elbow_angle}°")

    if knee_angle is not None:
        observations.append(f"knee angle is {knee_angle}°")

    if wrist_height is not None:
        observations.append(f"wrist is {wrist_height}")

    if shuttle_detected:
        observations.append("shuttle detected this frame")

    return {
        "timestamp": timestamp,
        "frame": frame_idx,
        "shuttle_detected": shuttle_detected,
        "elbow_angle": elbow_angle,
        "knee_angle": knee_angle,
        "wrist_height": wrist_height,
        "observations": observations
    }


# ============================================================
# LEFT/RIGHT CONSISTENCY
# ============================================================

# Pairs of (left, right) landmarks used to tell whether MediaPipe swapped
# sides between two frames: shoulders, elbows, wrists, hips
SIDE_PAIRS = [(11, 12), (13, 14), (15, 16), (23, 24)]


# Only treat a frame as mirrored when the evidence is strong: the swapped
# assignment must cost less than this fraction of keeping the labels. When
# the player is side-on and both arms overlap, the two costs are nearly
# equal and a wrong swap would stick for hundreds of frames.
SWAP_EVIDENCE_RATIO = 0.5


def make_sides_consistent(all_frames: list) -> list:
    """
    MediaPipe labels landmarks "left" and "right" from the player's point of
    view, but on footage shot from behind it can mirror them. If it mirrors
    the whole clip the same way, the check image will show the wrong arm
    highlighted and --racket-arm can be flipped to match; the harder case,
    handled here, is when the labels flip between frames.

    This walks through the frames and, whenever the "left" and "right"
    landmarks have clearly jumped to each other's positions since the
    previous frame, swaps them back. The result is a copy of the frames in
    which each label follows one physical arm for the whole clip. Which
    physical arm is called "left" is arbitrary — see the check image.

    The decision uses shoulders, elbows, wrists and hips together, weighted
    by how visible each pair is, so a genuine swap (every pair jumps) is
    easy to tell from normal movement or a half-hidden arm.
    """

    consistent = []
    previous = None  # positions and visibilities after any swap, from the previous frame

    for frame in all_frames:
        keypoints = frame.get("player_keypoints", [])
        if len(keypoints) < 33:
            consistent.append(frame)
            previous = None
            continue

        positions = {i: np.array([_x(keypoints[i]), _y(keypoints[i])])
                     for pair in SIDE_PAIRS for i in pair}
        visibility = {i: keypoints[i].get("visibility", 0) for pair in SIDE_PAIRS for i in pair}

        swap = False
        if previous is not None:
            keep_cost = 0.0
            swap_cost = 0.0
            for left, right in SIDE_PAIRS:
                # Trust a pair only as much as its least visible point
                weight = min(visibility[left], visibility[right],
                             previous["visibility"][left], previous["visibility"][right])
                keep_cost += weight * (np.linalg.norm(positions[left] - previous["positions"][left])
                                       + np.linalg.norm(positions[right] - previous["positions"][right]))
                swap_cost += weight * (np.linalg.norm(positions[right] - previous["positions"][left])
                                       + np.linalg.norm(positions[left] - previous["positions"][right]))
            swap = swap_cost < SWAP_EVIDENCE_RATIO * keep_cost

        if swap:
            frame = dict(frame)
            frame["player_keypoints"] = _swap_sides(keypoints)
            if len(frame.get("world_keypoints", [])) == 33:
                frame["world_keypoints"] = _swap_sides(frame["world_keypoints"])
            frame["sides_swapped"] = True
            positions = {i: positions[j] for pair in SIDE_PAIRS for i, j in (pair, pair[::-1])}
            visibility = {i: visibility[j] for pair in SIDE_PAIRS for i, j in (pair, pair[::-1])}

        previous = {"positions": positions, "visibility": visibility}
        consistent.append(frame)

    swapped = sum(1 for f in consistent if f.get("sides_swapped"))
    if swapped:
        print(f"Left/right labels swapped back in {swapped} of {len(all_frames)} frames to keep them consistent")

    return consistent


def _swap_sides(keypoints: list) -> list:
    """Returns a copy of 33 landmarks with every left/right pair exchanged."""
    swapped = list(keypoints)
    # MediaPipe pairs: eyes/ears/mouth (1-10), then limbs (11-32) alternate left/right
    for left, right in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10)] + [(i, i + 1) for i in range(11, 33, 2)]:
        swapped[left], swapped[right] = keypoints[right], keypoints[left]
    return swapped


# ============================================================
# RACKET ARM CHECK IMAGE
# ============================================================

def save_racket_arm_check_image(video_path: str, all_frames: list, side: str,
                                frame_indices: list, output_path: str) -> str | None:
    """
    Saves a strip of video frames with the chosen racket arm drawn on each,
    so the choice can be checked by eye. This matters because MediaPipe
    labels sides from the player's point of view and can mirror them on
    footage shot from behind. Returns the saved path, or None if no frame
    could be read.
    """

    frames_by_index = {f.get("frame"): f for f in all_frames}
    idx = SIDES[side]
    tiles = []

    cap = cv2.VideoCapture(video_path)
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, image = cap.read()
        if not ret:
            continue

        frame = frames_by_index.get(frame_index)
        keypoints = frame.get("player_keypoints", []) if frame else []

        # Draw shoulder → elbow → wrist in yellow, with a circle on the wrist.
        # Needs full-frame pixel positions (px/py), which older data lacks.
        if len(keypoints) == 33 and "px" in keypoints[idx["wrist"]]:
            points = [(int(keypoints[idx[j]]["px"]), int(keypoints[idx[j]]["py"]))
                      for j in ("shoulder", "elbow", "wrist")]
            cv2.line(image, points[0], points[1], (0, 255, 255), 6)
            cv2.line(image, points[1], points[2], (0, 255, 255), 6)
            cv2.circle(image, points[2], 18, (0, 255, 255), 4)

        # Zoom in on the player when the crop box is known
        box = frame.get("crop_box") if frame else None
        if box:
            margin = 200
            x1, y1 = max(0, box[0] - margin), max(0, box[1] - margin)
            x2, y2 = min(image.shape[1], box[2] + margin), min(image.shape[0], box[3] + margin)
            image = image[y1:y2, x1:x2]

        image = cv2.resize(image, (int(image.shape[1] * 600 / image.shape[0]), 600))
        cv2.putText(image, f"frame {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(image, f"frame {frame_index}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(image)
    cap.release()

    if not tiles:
        print(f"  Could not read frames from {video_path} for the racket-arm check image")
        return None

    strip = np.hstack(tiles)
    banner = np.zeros((70, strip.shape[1], 3), dtype=np.uint8)
    lines = [
        "The arm drawn in yellow is the one the coach treats as your RACKET arm.",
        "If it is the wrong arm, rerun with --racket-arm " + ("left" if side == "right" else "right"),
    ]
    for i, text in enumerate(lines):
        cv2.putText(banner, text, (10, 28 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, np.vstack([banner, strip]))
    return output_path


# ============================================================
# MAIN VISION AGENT FUNCTION
# ============================================================

def analyze_video(unified_data_path: str, racket_arm: str = "right",
                  video_path: str = None, output_dir: str = "outputs") -> list:
    """
    Main entry point for the Vision Agent.

    Loads the unified data from the CV pipeline, keeps MediaPipe's left/right
    labels consistent across frames, samples every Nth frame, extracts pose
    observations from each sampled frame, and returns a list of
    observations the Orchestrator can reason about.

    Args:
        unified_data_path: Path to unified_data.json from run_pipeline()
        racket_arm:        "right" or "left" — the player's racket arm.
        video_path:        If given, a check image is saved to
                           <output_dir>/racket_arm_check.jpg showing which
                           arm is being measured, so it can be confirmed by
                           eye (MediaPipe can mirror sides on footage shot
                           from behind).
        output_dir:        Folder for the check image.

    Returns:
        List of observation dicts, one per sampled frame, sorted by timestamp.
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

    # Make MediaPipe's left/right labels follow the same physical arms
    # throughout the clip
    all_frames = make_sides_consistent(all_frames)
    side = racket_arm
    if side not in SIDES:
        raise ValueError(f"racket_arm must be 'right' or 'left', got {racket_arm!r}")

    if video_path:
        # Show four frames spread across the clip where the racket wrist was
        # clearly seen, so the drawn arm can be judged
        wrist_index = SIDES[side]["wrist"]
        clear = [f["frame"] for f in all_frames
                 if len(f.get("player_keypoints", [])) == 33
                 and f["player_keypoints"][wrist_index].get("visibility", 0) >= 0.5]
        pool = clear if len(clear) >= 4 else [f["frame"] for f in all_frames]
        frame_indices = sorted({pool[int(i * (len(pool) - 1) / 3)] for i in range(4)}) if pool else []
        saved = save_racket_arm_check_image(
            video_path, all_frames, side, frame_indices,
            os.path.join(output_dir, "racket_arm_check.jpg"))
        if saved:
            print(f"Racket-arm check image saved → {saved}")

    # Sample every Nth frame to avoid redundant data
    # Consecutive frames are nearly identical at 30fps
    sampled_frames = all_frames[::SAMPLE_EVERY_N_FRAMES]
    print(f"Sampled frames: {len(sampled_frames)} (every {SAMPLE_EVERY_N_FRAMES}th frame)")

    # Extract pose observations from each sampled frame
    observations = []

    for frame in sampled_frames:

        # Skip frames with no keypoint data
        # (player was not detected by MediaPipe in this frame)
        if not frame.get("player_keypoints"):
            continue

        obs = extract_pose_observations(frame, side=side)

        # Only include frames where we could extract at least one observation
        if obs["observations"]:
            observations.append(obs)

    # How often the racket arm was actually measurable — low numbers mean
    # the wrong side was chosen or the camera angle hides the racket arm
    with_elbow = sum(1 for o in observations if o["elbow_angle"] is not None)
    with_wrist = sum(1 for o in observations if o["wrist_height"] is not None)
    print(f"Extracted observations from {len(observations)} frames "
          f"(elbow angle in {with_elbow}, wrist height in {with_wrist})")
    if observations and with_elbow / len(observations) < MIN_RACKET_ARM_COVERAGE:
        print(f"WARNING: the '{side}' arm was measurable in only "
              f"{with_elbow / len(observations):.0%} of sampled frames. Check "
              f"{os.path.join(output_dir, 'racket_arm_check.jpg')} — if the yellow arm is not your "
              f"racket arm, rerun with --racket-arm {'left' if side == 'right' else 'right'}; "
              f"if it is, the camera angle is hiding your racket arm for most of the clip.")

    return observations


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":

    # Path to your unified data — update to match your actual output
    UNIFIED_DATA_PATH = "outputs/unified_data.json"

    # The clip the data came from — used for the racket-arm check image
    VIDEO_PATH = "data/test_clip.MOV"

    # Change to "left" if the player is left-handed
    RACKET_ARM = "right"

    # Run the vision agent
    observations = analyze_video(UNIFIED_DATA_PATH, racket_arm=RACKET_ARM, video_path=VIDEO_PATH)

    # Print a summary
    print(f"\nTotal observations: {len(observations)}")
    print("\nFirst 3 sampled frames:")
    for obs in observations[:3]:
        print(f"\n  t={obs['timestamp']}s  frame={obs['frame']}")
        print(f"  Elbow angle: {obs['elbow_angle']}°")
        print(f"  Knee angle:  {obs['knee_angle']}°")
        print(f"  Wrist height: {obs['wrist_height']}")
        print(f"  Shuttle:     {obs['shuttle_detected']}")
        print(f"  Notes:")
        for note in obs["observations"]:
            print(f"    - {note}")
