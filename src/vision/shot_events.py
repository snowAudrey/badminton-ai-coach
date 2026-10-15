"""
Finds the moments the tracked player hits the shuttle ("shot events") and
records a small set of facts about each one for the shot classifier.

How a shot is found: the racket wrist's speed, measured relative to the
player's shoulders (so running around the court doesn't count) and scaled by
shoulder width (so distance from the camera doesn't matter), spikes during
a swing. Each spike that stands well above the clip's typical wrist speed,
and is at least MIN_SHOT_GAP_S from the previous one, is a shot. The speed
peak is taken as the moment of contact.

Run on its own:
    python -m src.vision.shot_events
"""

import json
import os

import numpy as np

from src.agents.vision_agent import (
    LANDMARKS, SIDES, MIN_VISIBILITY, get_keypoint, calculate_angle, make_sides_consistent
)


# ============================================================
# TUNING PARAMETERS
# ============================================================

# A swing's wrist speed must be at least this many times the clip's median
# wrist speed. Tuned on hand-labelled shots: 2.0 found 9 of 10 shots with
# 2 extra detections; 3.0 found only the hardest-hit 5.
SPEED_PEAK_MULTIPLIER = 2.0

# Two shots can't be closer together than this (seconds)
MIN_SHOT_GAP_S = 0.6

# Wrist positions are smoothed over this many frames before measuring speed
SMOOTHING_FRAMES = 5

# A hidden wrist is interpolated across gaps up to this many frames; longer
# gaps are left empty so no speed is measured there
MAX_INTERPOLATED_GAP = 5

# How far around the contact moment to look when describing a shot (seconds)
FEATURE_WINDOW_S = 0.2
PREPARATION_WINDOW_S = 0.5
SHUTTLE_WINDOW_S = 0.5

# The wrist's height this long before the speed peak tells whether the shot
# was hit from overhead or from below: a lift's follow-through swings up to
# overhead AFTER contact, but a quarter-second before the fastest movement
# its wrist is still below the shoulder
CONTACT_HEIGHT_LEAD_S = 0.25

# The shuttle's own flight after a hit: the first run of detections that
# starts within START_WITHIN_S of contact and has no gap longer than MAX_GAP_S.
# Stopping at the first gap keeps the opponent's reply out of it.
FLIGHT_START_WITHIN_S = 1.0
FLIGHT_MAX_GAP_S = 0.5


# ============================================================
# WRIST TRACK
# ============================================================

def racket_wrist_track(frames: list, side: str) -> dict:
    """
    Builds the racket wrist's path through the clip.

    Returns a dict with, per frame (NaN where the wrist wasn't usable):
      "position": wrist relative to the shoulder midpoint, in shoulder widths
      "speed":    smoothed speed of that relative position, in shoulder widths per second
      "fps":      frames per second, worked out from the timestamps
    """

    n = len(frames)
    idx = SIDES[side]
    left_sh, right_sh = LANDMARKS["left_shoulder"], LANDMARKS["right_shoulder"]

    timestamps = np.array([f["timestamp"] for f in frames])
    fps = float(1.0 / np.median(np.diff(timestamps))) if n > 1 else 30.0

    position = np.full((n, 2), np.nan)
    shoulder_width = np.full(n, np.nan)

    for i, frame in enumerate(frames):
        keypoints = frame.get("player_keypoints", [])
        if len(keypoints) < 33 or keypoints[idx["wrist"]].get("visibility", 0) < MIN_VISIBILITY:
            continue
        ls, rs, wrist = keypoints[left_sh], keypoints[right_sh], keypoints[idx["wrist"]]
        mid = np.array([(ls["px"] + rs["px"]) / 2, (ls["py"] + rs["py"]) / 2])
        shoulder_width[i] = np.hypot(ls["px"] - rs["px"], ls["py"] - rs["py"])
        position[i] = np.array([wrist["px"], wrist["py"]]) - mid

    # One scale for the whole clip — a per-frame width jumps around when the
    # player turns side-on
    scale = np.nanmedian(shoulder_width)
    if not np.isfinite(scale) or scale <= 0:
        return {"position": position, "speed": np.full(n, np.nan), "fps": fps, "shoulder_width": 0.0}
    position /= scale

    # Fill short gaps where the wrist was hidden; leave long gaps empty
    index = np.arange(n)
    for axis in range(2):
        series = position[:, axis]
        missing = np.isnan(series)
        if missing.all():
            continue
        filled = np.interp(index, index[~missing], series[~missing])
        run = 0
        long_gap = np.zeros(n, dtype=bool)
        for j in range(n):
            run = run + 1 if missing[j] else 0
            if run > MAX_INTERPOLATED_GAP:
                long_gap[j - run + 1:j + 1] = True
        filled[long_gap] = np.nan
        position[:, axis] = filled

    speed = np.full(n, np.nan)
    speed[1:] = np.hypot(*(position[1:] - position[:-1]).T) * fps

    half = SMOOTHING_FRAMES // 2
    smoothed = np.full(n, np.nan)
    for i in range(n):
        window = speed[max(0, i - half):i + half + 1]
        if np.any(~np.isnan(window)):
            smoothed[i] = np.nanmean(window)

    return {"position": position, "speed": smoothed, "fps": fps, "shoulder_width": float(scale)}


def find_speed_peaks(speed: np.ndarray, fps: float) -> list:
    """Frame indices where the wrist speed peaks above the shot threshold."""

    if np.all(np.isnan(speed)):
        return []
    threshold = SPEED_PEAK_MULTIPLIER * np.nanmedian(speed)
    half_gap = int(round(MIN_SHOT_GAP_S * fps / 2))

    peaks = []
    for i in range(1, len(speed) - 1):
        value = speed[i]
        if np.isnan(value) or value < threshold:
            continue
        window = speed[max(0, i - half_gap):i + half_gap + 1]
        if value >= np.nanmax(window):
            # Only one peak per window: skip if we already kept a close one
            if peaks and i - peaks[-1] < 2 * half_gap and speed[peaks[-1]] >= value:
                continue
            peaks.append(i)
    return peaks


# ============================================================
# SHOT FEATURES
# ============================================================

def _nearest_frame_with(frames: list, centre: int, radius: int, is_usable) -> dict | None:
    """The frame closest to `centre` (within `radius`) for which is_usable(frame) holds."""
    for offset in range(radius + 1):
        for j in (centre - offset, centre + offset):
            if 0 <= j < len(frames) and is_usable(frames[j]):
                return frames[j]
    return None


def _wrist_height(frame: dict, side: str) -> str | None:
    """Same overhead / above shoulder / below shoulder rule as vision_agent."""
    keypoints = frame.get("player_keypoints", [])
    idx = SIDES[side]
    wrist = get_keypoint(keypoints, idx["wrist"])
    nose = get_keypoint(keypoints, LANDMARKS["nose"])
    shoulder = get_keypoint(keypoints, idx["shoulder"])
    if wrist is None or nose is None or shoulder is None:
        return None
    if wrist["py"] < nose["py"]:
        return "overhead"
    if wrist["py"] < shoulder["py"]:
        return "above shoulder"
    return "below shoulder"


def _elbow_angle(frame: dict, side: str) -> float | None:
    world = frame.get("world_keypoints", [])
    if len(world) != 33:
        return None
    idx = SIDES[side]
    return calculate_angle(get_keypoint(world, idx["shoulder"]), get_keypoint(world, idx["elbow"]),
                           get_keypoint(world, idx["wrist"]), use_3d=True)


def _knee_angle(frame: dict, side: str) -> float | None:
    world = frame.get("world_keypoints", [])
    if len(world) != 33:
        return None
    idx = SIDES[side]
    return calculate_angle(get_keypoint(world, idx["hip"]), get_keypoint(world, idx["knee"]),
                           get_keypoint(world, idx["ankle"]), use_3d=True)


def _wrist_side_of_body(frame: dict, side: str) -> float | None:
    """
    How far the racket wrist sits toward the racket side of the body, in
    shoulder widths. Positive = out on the racket side (a forehand position),
    negative = across the body (a backhand position). Measured along the
    line between the shoulders, so it works whichever way the player faces.
    """
    keypoints = frame.get("player_keypoints", [])
    idx = SIDES[side]
    other = SIDES["left" if side == "right" else "right"]
    wrist = get_keypoint(keypoints, idx["wrist"])
    racket_sh = get_keypoint(keypoints, idx["shoulder"])
    other_sh = get_keypoint(keypoints, other["shoulder"])
    if wrist is None or racket_sh is None or other_sh is None:
        return None
    axis = np.array([racket_sh["px"] - other_sh["px"], racket_sh["py"] - other_sh["py"]])
    width = np.linalg.norm(axis)
    if width < 1e-6:
        return None
    mid = np.array([(racket_sh["px"] + other_sh["px"]) / 2, (racket_sh["py"] + other_sh["py"]) / 2])
    offset = np.array([wrist["px"], wrist["py"]]) - mid
    return round(float(np.dot(offset, axis / width) / width), 2)


def _shuttle_motion(frames: list, start: int, stop: int) -> dict | None:
    """Where the shuttle went between two frames, if it was detected there."""
    seen = [(f["frame"], f["shuttle_x"], f["shuttle_y"]) for f in frames[max(0, start):stop]
            if f.get("shuttle_detected")]
    if len(seen) < 2:
        return None
    (f0, x0, y0), (f1, x1, y1) = seen[0], seen[-1]
    if f1 == f0:
        return None
    return {
        "detections": len(seen),
        "vertical": "rising" if y1 < y0 else "falling",   # image y grows downward
        "speed_px_per_frame": round(float(np.hypot(x1 - x0, y1 - y0) / (f1 - f0)), 1),
    }


def _shuttle_rise_after(frames: list, contact: int, fps: float, head_y: float,
                        shoulder_width: float) -> dict | None:
    """
    How high the shuttle got on its own flight after the player's hit, in
    shoulder widths above the player's head. The camera is behind the
    player, so a shuttle that climbs far up the frame went high and deep
    (clear, lift); one that stays around head height stayed low (drop, net
    shot, drive). Only the first arc of detections is used — it ends at the
    first gap or when the shuttle starts rising again — so the opponent's
    reply doesn't get mixed in.
    """
    stop = contact + int(round((FLIGHT_START_WITHIN_S + 1.0) * fps)) + 1
    seen = [(f["frame"], f["shuttle_y"]) for f in frames[contact:stop] if f.get("shuttle_detected")]
    if not seen or (seen[0][0] - contact) / fps > FLIGHT_START_WITHIN_S or shoulder_width <= 0:
        return None
    # Follow the flight while it is one arc: it may go up (y decreasing) and
    # then come down (y increasing), but once it starts going up again that
    # is a new hit — the opponent's reply — so stop there
    run = [seen[0]]
    descending = False
    for previous, current in zip(seen, seen[1:]):
        if (current[0] - previous[0]) / fps > FLIGHT_MAX_GAP_S:
            break
        if current[1] > previous[1]:
            descending = True
        elif descending and current[1] < previous[1]:
            break
        run.append(current)
    if len(run) < 2:
        return None
    ys = [y for _, y in run]
    return {
        "detections": len(run),
        "starts_after_s": round((run[0][0] - contact) / fps, 2),
        "rise_shoulder_widths": round(float((head_y - min(ys)) / shoulder_width), 1),
        "end_shoulder_widths": round(float((head_y - ys[-1]) / shoulder_width), 1),
    }


def _shuttle_incoming_speed(frames: list, contact: int, fps: float, shoulder_width: float) -> float | None:
    """
    Speed of the shuttle in the 0.3 s before contact, in shoulder widths per
    second — so the number means the same thing at any video resolution or
    camera distance.
    """
    start = max(0, contact - int(round(0.3 * fps)))
    seen = [(f["frame"], f["shuttle_x"], f["shuttle_y"]) for f in frames[start:contact + 1]
            if f.get("shuttle_detected")]
    if len(seen) < 2 or seen[-1][0] == seen[0][0] or shoulder_width <= 0:
        return None
    (f0, x0, y0), (f1, x1, y1) = seen[0], seen[-1]
    return round(float(np.hypot(x1 - x0, y1 - y0) / (f1 - f0) * fps / shoulder_width), 1)


def describe_shot(frames: list, contact: int, side: str, track: dict) -> dict:
    """Collects the facts about one shot around its contact frame."""

    fps = track["fps"]
    window = int(round(FEATURE_WINDOW_S * fps))
    prep = int(round(PREPARATION_WINDOW_S * fps))
    shuttle_window = int(round(SHUTTLE_WINDOW_S * fps))
    frame = frames[contact]

    def usable_wrist(f):
        return _wrist_height(f, side) is not None

    def usable_elbow(f):
        return _elbow_angle(f, side) is not None

    nearest_wrist = _nearest_frame_with(frames, contact, window, usable_wrist)
    nearest_elbow = _nearest_frame_with(frames, contact, window, usable_elbow)
    lead = _nearest_frame_with(frames, contact - int(round(CONTACT_HEIGHT_LEAD_S * fps)), 2, usable_wrist)

    # Vertical size of the swing: how far the wrist travelled up and down
    # around the peak, in shoulder widths (big for clears and lifts, small
    # for blocks, serves and net shots)
    pos = track["position"]
    lo_sw, hi_sw = max(0, contact - int(round(0.4 * fps))), min(len(frames) - 1, contact + int(round(0.15 * fps)))
    swing_ys = pos[lo_sw:hi_sw + 1, 1]
    swing = round(float(np.nanmax(swing_ys) - np.nanmin(swing_ys)), 1) if np.any(~np.isnan(swing_ys)) else None

    before = frames[max(0, contact - prep):contact + 1]
    after = frames[contact:min(len(frames), contact + window + 1)]
    elbows_before = [a for a in (_elbow_angle(f, side) for f in before) if a is not None]
    elbows_after = [a for a in (_elbow_angle(f, side) for f in after) if a is not None]
    knees_before = [a for a in (_knee_angle(f, side) for f in before) if a is not None]
    heights_before = [h for h in (_wrist_height(f, side) for f in before) if h is not None]

    # Wrist moving up or down through the contact
    lo, hi = max(0, contact - window // 2), min(len(frames) - 1, contact + window // 2)
    vertical = None
    if not np.isnan(pos[lo, 1]) and not np.isnan(pos[hi, 1]) and hi > lo:
        dy = (pos[hi, 1] - pos[lo, 1]) * fps / (hi - lo)   # shoulder widths per second, + = down
        vertical = "rising" if dy < -0.5 else "falling" if dy > 0.5 else "level"

    box = frame.get("crop_box")

    # Reference heights for the shuttle facts
    keypoints = frame.get("player_keypoints", [])
    head_y = keypoints[LANDMARKS["nose"]]["py"] if len(keypoints) == 33 else None
    shoulder_width = track["shoulder_width"]

    return {
        "time_s": round(frame["timestamp"], 3),
        "frame": frame["frame"],
        "peak_wrist_speed": round(float(track["speed"][contact]), 1),
        "wrist_height": _wrist_height(nearest_wrist, side) if nearest_wrist else None,
        "wrist_height_before_peak": _wrist_height(lead, side) if lead else None,
        "wrist_swing_shoulder_widths": swing,
        "highest_wrist_before": ("overhead" if "overhead" in heights_before
                                 else "above shoulder" if "above shoulder" in heights_before
                                 else "below shoulder" if heights_before else None),
        "elbow_angle": _elbow_angle(nearest_elbow, side) if nearest_elbow else None,
        "elbow_angle_min_before": min(elbows_before) if elbows_before else None,
        "elbow_angle_after": max(elbows_after) if elbows_after else None,
        "knee_angle_min_before": min(knees_before) if knees_before else None,
        "wrist_side_of_body": _wrist_side_of_body(nearest_wrist, side) if nearest_wrist else None,
        "wrist_vertical_motion": vertical,
        "player_box": box,
        "shuttle_before": _shuttle_motion(frames, contact - shuttle_window, contact),
        "shuttle_after": _shuttle_motion(frames, contact + 1, contact + shuttle_window + 1),
        "shuttle_rise_after": (_shuttle_rise_after(frames, contact, fps, head_y, shoulder_width)
                               if head_y is not None else None),
        "shuttle_incoming_speed": _shuttle_incoming_speed(frames, contact, fps, shoulder_width),
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def detect_shots(unified_data_path: str, racket_arm: str = "right",
                 output_path: str = "outputs/shot_events.json") -> list:
    """
    Finds every shot in the clip and saves them, in time order, to
    output_path. Each entry has the contact time and the facts the shot
    classifier needs (shot_type is added later by the classifier).
    """

    with open(unified_data_path) as f:
        frames = make_sides_consistent(json.load(f))

    track = racket_wrist_track(frames, racket_arm)
    usable = int(np.sum(~np.isnan(track["speed"])))
    print(f"Racket wrist usable in {usable} / {len(frames)} frames")

    peaks = find_speed_peaks(track["speed"], track["fps"])
    events = [describe_shot(frames, peak, racket_arm, track) for peak in peaks]

    # Clip-level context: how long since the previous shot (a serve follows a
    # long pause) and where the player stood compared with their usual spot
    # (lower in the frame = nearer the camera = further back in the court)
    bottoms = [f["crop_box"][3] for f in frames if f.get("crop_box")]
    usual_bottom = float(np.median(bottoms)) if bottoms else None
    previous = None
    for event in events:
        event["seconds_since_previous_shot"] = (round(event["time_s"] - previous, 2)
                                                if previous is not None else None)
        previous = event["time_s"]
        if usual_bottom and event.get("player_box") and track["shoulder_width"] > 0:
            event["court_depth_shoulder_widths"] = round(
                float((event["player_box"][3] - usual_bottom) / track["shoulder_width"]), 1)
        else:
            event["court_depth_shoulder_widths"] = None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(events, f, indent=2)

    print(f"Found {len(events)} shot(s) → {output_path}")
    for event in events:
        print(f"  t={event['time_s']:6.2f}s  wrist {event['wrist_height'] or '?':<15} "
              f"elbow {event['elbow_angle'] or '?':>6}  speed {event['peak_wrist_speed']}")
    return events


if __name__ == "__main__":

    UNIFIED_DATA_PATH = "outputs/unified_data.json"

    # Change to "left" if the player is left-handed
    RACKET_ARM = "right"

    detect_shots(UNIFIED_DATA_PATH, racket_arm=RACKET_ARM)
