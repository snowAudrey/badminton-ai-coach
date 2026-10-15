"""
Names each detected shot (smash, clear, drop, ...).

The main method is a set of rules on the measured facts from shot_events.py:
where the racket wrist was at contact (overhead or below the shoulder), how
high the shuttle climbed afterwards (a clear or lift goes high and deep, a
drop or net shot stays low), how fast the shuttle was coming in (a block
returns a fast shot), and whether the shot followed a long pause (a serve).

Claude can optionally be asked instead (--use-claude), given the same facts
plus video frames. In testing on amateur phone footage this was LESS
accurate than the rules: the shuttle is too small to see in the frames, so
the model guessed. It is kept for experiments only.

Run on its own (after shot_events.json exists):
    python -m src.agents.shot_classifier
"""

import base64
import json
import os
import time

import anthropic
import cv2
from dotenv import load_dotenv


# ============================================================
# SETUP
# ============================================================

load_dotenv()
client = anthropic.Anthropic()

# Haiku 4.5 first — it reads images and is cheap. Switch to "claude-sonnet-5"
# if accuracy on the labelled shots is too low.
MODEL = "claude-haiku-4-5"

# The fixed list of answers the classifier may give
SHOT_TYPES = ["smash", "clear", "drop", "drive", "net shot", "lift",
              "block/defence", "serve", "return of serve", "other", "no shot"]

# Close-up frames of the swing, as offsets from the detected contact (seconds)
SWING_FRAME_OFFSETS_S = (-0.2, 0.0)

# Wider court frames after contact that show where the shuttle went. A smash
# has landed by ~0.6 s, a drop is hanging near the net then, a clear is still
# high over the far court at ~1 s.
FLIGHT_FRAME_OFFSETS_S = (0.3, 0.6, 1.0)

# Images are scaled to these heights before sending (smaller = cheaper)
CROP_HEIGHT = 512
COURT_HEIGHT = 768

# Extra space around the player's box so the racket and shuttle stay in view,
# as a fraction of the box size
CROP_MARGIN = 0.6

API_DELAY_SECONDS = 0.5


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an experienced badminton coach. You identify which shot a player
just played from a few video frames and some pose measurements.

The clip is shot from behind ONE player's end of the court (the player we
are coaching), so the far end of the court and the opponent are toward the
top of the frame. You get two kinds of frames:
- Close-ups of that player just before and at the detected contact. A
  yellow circle marks the player's racket hand when it was visible.
- Wider court frames 0.3, 0.6 and 1.0 seconds after contact, showing where
  the shuttle went. The coached player is outlined in yellow. The shuttle is
  small and white — look for it. After this player's hit it travels away from
  the camera (upward in the frame). A smash is fast and has usually landed by
  0.6 s; a drop floats and is near the net around 0.6 s; a clear is still
  high over the far court at 1.0 s; a lift also goes high but starts from a
  low contact; a drive stays flat and fast; a net shot barely crosses the
  net. Ignore the opponent's own shots.

Shot types you may answer with, and how to tell them apart:
- smash: overhead contact, hit hard and steeply downward
- clear: overhead contact, hit high and deep to the back of the opponent's court
- drop: overhead contact, hit softly so it falls just over the net
- drive: fast, flat shot, contact around shoulder height
- net shot: played close to the net, gently over, contact around or below net height
- lift: contact below the shoulder (often near the net), hit high and deep
- block/defence: returning an attacking shot from below shoulder height with a short action
- serve: the stroke that starts a rally
- return of serve: the first shot after the opponent's serve
- other: a real hit that fits none of the above
- no shot: the player did not hit the shuttle in these frames (e.g. moving into position, swinging without contact)

Use the pose measurements as hints, but trust what you can see: the
contact height, the swing, and above all where the shuttle goes afterwards.
Overhead contact alone does not separate a clear, a smash and a drop —
the flight does. If the frames genuinely don't show enough, lower your
confidence rather than guessing.

Answer with ONLY a JSON object, no markdown:
{"shot_type": "<one of the types above>",
 "stroke_side": "forehand" | "backhand" | "unknown",
 "confidence": <0.0 to 1.0>,
 "candidates": ["<most likely type>", "<second>", "<third>"],
 "reasoning": "<one short sentence>"}
"""


def describe_features(event: dict, racket_arm: str) -> str:
    """Turns a shot event's measurements into plain text for the prompt."""

    lines = [f"The player is {racket_arm}-handed. Measurements around the detected contact at t={event['time_s']}s:"]

    def add(label, value, unit=""):
        if value is not None:
            lines.append(f"- {label}: {value}{unit}")

    add("racket wrist height at the speed peak", event.get("wrist_height"))
    add("highest wrist position in the 0.5s before", event.get("highest_wrist_before"))
    add("elbow angle near contact", event.get("elbow_angle"), "° (180° = straight)")
    add("most bent the elbow got in the 0.5s before", event.get("elbow_angle_min_before"), "°")
    add("elbow angle just after", event.get("elbow_angle_after"), "°")
    add("deepest knee bend in the 0.5s before", event.get("knee_angle_min_before"), "° (smaller = deeper)")

    # Where the wrist sits relative to the body only hints at forehand vs
    # backhand for shots hit below the shoulder; overhead contact points sit
    # above the head and the hint is meaningless there
    side = event.get("wrist_side_of_body")
    if side is not None and event.get("highest_wrist_before") != "overhead":
        where = "out on the racket side (forehand position)" if side > 0.15 else \
                "across the body (backhand position)" if side < -0.15 else "in front of the body"
        lines.append(f"- racket wrist relative to the body: {where}")

    add("racket wrist moving", event.get("wrist_vertical_motion"))
    add("peak wrist speed", event.get("peak_wrist_speed"), " shoulder-widths/s (soft shots are usually under 15)")

    return "\n".join(lines)


# ============================================================
# FRAMES
# ============================================================

def _read_frame(cap, index: int, total: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(max(0, index), total - 1))
    ret, image = cap.read()
    return image if ret else None


def _encode(image, height: int) -> bytes | None:
    scale = height / image.shape[0]
    image = cv2.resize(image, (max(1, int(image.shape[1] * scale)), height))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes() if ok else None


def extract_contact_frames(video_path: str, event: dict, racket_arm: str, frames_by_index: dict) -> list:
    """
    Returns labelled JPEG frames for one shot: close-ups of the player around
    the contact (racket hand marked), then wider court views after contact
    that show where the shuttle went (player outlined).
    """

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    from src.agents.vision_agent import SIDES, MIN_VISIBILITY
    wrist_index = SIDES[racket_arm]["wrist"]
    results = []

    for offset_s in SWING_FRAME_OFFSETS_S:
        index = event["frame"] + int(round(offset_s * fps))
        image = _read_frame(cap, index, total)
        if image is None:
            continue
        frame = frames_by_index.get(index) or {}
        box = frame.get("crop_box") or event.get("player_box")
        keypoints = frame.get("player_keypoints", [])
        if len(keypoints) == 33 and keypoints[wrist_index].get("visibility", 0) >= MIN_VISIBILITY:
            cv2.circle(image, (int(keypoints[wrist_index]["px"]), int(keypoints[wrist_index]["py"])),
                       14, (0, 255, 255), 3)
        if box:
            x1, y1, x2, y2 = box
            mx, my = int((x2 - x1) * CROP_MARGIN), int((y2 - y1) * CROP_MARGIN)
            image = image[max(0, y1 - my):min(image.shape[0], y2 + my),
                          max(0, x1 - mx):min(image.shape[1], x2 + mx)]
        jpeg = _encode(image, CROP_HEIGHT)
        if jpeg:
            label = "close-up, 0.2 s before contact" if offset_s < 0 else "close-up, at contact"
            results.append((label, jpeg))

    for offset_s in FLIGHT_FRAME_OFFSETS_S:
        index = event["frame"] + int(round(offset_s * fps))
        image = _read_frame(cap, index, total)
        if image is None:
            continue
        frame = frames_by_index.get(index) or {}
        box = frame.get("crop_box") or event.get("player_box")
        if box:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 255), 3)
            # Keep everything from the top of the frame down to just below the player
            image = image[:min(image.shape[0], box[3] + 120), :]
        jpeg = _encode(image, COURT_HEIGHT)
        if jpeg:
            results.append((f"court view, {offset_s:.1f} s after contact", jpeg))

    cap.release()
    return results


# ============================================================
# RULE-BASED CLASSIFICATION
# ============================================================

# A shuttle that climbs at least this many shoulder widths above the head
# after contact went high and deep (clear or lift); below it stayed low
SHUTTLE_HIGH_RISE = 3.5

# An incoming shuttle at least this fast (shoulder widths per second, so it
# is the same at any resolution or camera distance) is an attacking shot
# being returned. Returned with a short action (small wrist swing) that is a
# block/defence; with a full swing it is a lift
FAST_INCOMING_SPEED = 8
SHORT_ACTION_SWING = 1.5

# A shot this long after the previous one, with nothing coming in, is a serve
SERVE_PAUSE_S = 4.0

# Wrist speed (shoulder widths per second) above which an overhead shot with
# no shuttle data is called a clear rather than a drop
HARD_SWING_SPEED = 25


def classify_by_rules(event: dict) -> dict:
    """
    Names a shot from its measured facts alone. Returns the same fields as
    the Claude classifier: shot_type, stroke_side, confidence, candidates,
    reasoning. Confidence reflects how much evidence was actually available.

    Tuned on one hand-labelled clip (10 shots), so expect it to need
    adjusting as more clips are labelled.
    """

    lead = event.get("wrist_height_before_peak")      # a quarter-second before the fastest movement
    at_peak = event.get("wrist_height")
    motion = event.get("wrist_vertical_motion")
    swing = event.get("wrist_swing_shoulder_widths") or 0
    speed = event.get("peak_wrist_speed") or 0
    rise = (event.get("shuttle_rise_after") or {}).get("rise_shoulder_widths")
    incoming = event.get("shuttle_incoming_speed")
    pause = event.get("seconds_since_previous_shot")

    # Contact height. A lift's follow-through swings up to overhead after
    # contact, so "wrist overhead at the speed peak" is not enough on its
    # own: the wrist must not have been below the shoulder just before.
    overhead = (
        lead == "overhead"
        or (at_peak == "overhead" and lead != "below shoulder")
        or (lead == "above shoulder" and motion == "falling" and swing >= 2)
    )

    # Forehand vs backhand is not reported yet. The wrist's position relative
    # to the body at the speed peak looked like a usable cue, but on the
    # labelled clip it called three forehand lifts/net shots "backhand":
    # by the speed peak the follow-through has already carried the wrist
    # across the body. The raw offset is kept in the event for future work.
    side = "unknown"

    def answer(shot, confidence, candidates, reasoning):
        return {"shot_type": shot, "stroke_side": side, "confidence": confidence,
                "candidates": candidates, "reasoning": reasoning}

    if overhead:
        if rise is not None:
            if rise >= SHUTTLE_HIGH_RISE:
                return answer("clear", 0.75, ["clear", "smash", "drop"],
                              f"overhead contact; shuttle climbed {rise} shoulder widths above head height afterwards")
            return answer("drop", 0.65, ["drop", "smash", "clear"],
                          f"overhead contact; shuttle stayed low afterwards (rose {rise} shoulder widths)")
        if speed >= HARD_SWING_SPEED:
            return answer("clear", 0.45, ["clear", "smash", "drop"],
                          f"overhead contact with a hard swing ({speed} shoulder widths/s); shuttle not tracked afterwards")
        return answer("drop", 0.45, ["drop", "clear", "smash"],
                      f"overhead contact with a gentle swing ({speed} shoulder widths/s); shuttle not tracked afterwards")

    # Hit from below the shoulder from here on
    if (pause is None or pause >= SERVE_PAUSE_S) and incoming is None:
        return answer("serve", 0.6, ["serve", "lift", "net shot"],
                      "below-shoulder contact after a long pause with no shuttle coming in")
    if incoming is not None and incoming >= FAST_INCOMING_SPEED and swing < SHORT_ACTION_SWING:
        return answer("block/defence", 0.6, ["block/defence", "lift", "net shot"],
                      f"short action ({swing} shoulder widths of wrist travel) returning a fast incoming "
                      f"shuttle ({incoming} shoulder widths/s)")
    if rise is not None and rise >= SHUTTLE_HIGH_RISE:
        return answer("lift", 0.7, ["lift", "block/defence", "net shot"],
                      f"below-shoulder contact; shuttle climbed {rise} shoulder widths above head height afterwards")
    if incoming is not None and incoming >= FAST_INCOMING_SPEED:
        return answer("block/defence", 0.55, ["block/defence", "lift", "net shot"],
                      f"below-shoulder contact returning a fast incoming shuttle ({incoming} shoulder widths/s)")
    if rise is not None:
        return answer("net shot", 0.55, ["net shot", "lift", "block/defence"],
                      f"below-shoulder contact; shuttle stayed low afterwards (rose {rise} shoulder widths)")
    return answer("net shot", 0.4, ["net shot", "lift", "block/defence"],
                  "below-shoulder contact; shuttle not tracked, so this is the most common case")


# ============================================================
# CLAUDE CLASSIFICATION (optional)
# ============================================================

def classify_shot(event: dict, frames: list, racket_arm: str, model: str = MODEL) -> dict:
    """Asks Claude to name one shot. Returns the parsed answer (or an 'other' fallback)."""

    content = [{"type": "text", "text": describe_features(event, racket_arm)}]
    for label, jpeg in frames:
        content.append({"type": "text", "text": f"Frame: {label}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                    "data": base64.b64encode(jpeg).decode()}})
    content.append({"type": "text", "text": "Which shot was this? Answer with the JSON object only."})

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        # Some models return a thinking block before the answer text
        raw = next(block.text for block in response.content if getattr(block, "type", "") == "text").strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:-1]).strip()
        answer = json.loads(raw)
    except Exception as e:
        print(f"  ERROR classifying shot at t={event['time_s']}s: {e}")
        return {"shot_type": "other", "stroke_side": "unknown", "confidence": 0.0,
                "candidates": ["other"], "reasoning": f"classification failed: {e}"}

    shot_type = str(answer.get("shot_type", "other")).lower().strip()
    if shot_type not in SHOT_TYPES:
        shot_type = "other"
    candidates = [c for c in (str(x).lower().strip() for x in answer.get("candidates", [])) if c in SHOT_TYPES]
    if shot_type not in candidates:
        candidates = [shot_type] + candidates
    return {
        "shot_type": shot_type,
        "stroke_side": str(answer.get("stroke_side", "unknown")).lower().strip(),
        "confidence": float(answer.get("confidence", 0.0)),
        "candidates": candidates[:3],
        "reasoning": str(answer.get("reasoning", "")),
    }


def classify_shots(events_path: str, video_path: str, unified_data_path: str,
                   racket_arm: str = "right", use_claude: bool = False, model: str = MODEL) -> list:
    """
    Names every shot in events_path and writes the answers back into it.
    Each event gains shot_type, stroke_side, confidence, candidates,
    reasoning and classifier_model ("rules" unless Claude was used).
    """

    with open(events_path) as f:
        events = json.load(f)

    if use_claude:
        with open(unified_data_path) as f:
            frames_by_index = {fr["frame"]: fr for fr in json.load(f)}
        print(f"Classifying {len(events)} shot(s) with {model}...")
    else:
        print(f"Classifying {len(events)} shot(s) with rules...")

    for event in events:
        if use_claude:
            frames = extract_contact_frames(video_path, event, racket_arm, frames_by_index)
            answer = classify_shot(event, frames, racket_arm, model=model)
            time.sleep(API_DELAY_SECONDS)
        else:
            answer = classify_by_rules(event)
        event.update(answer)
        event["classifier_model"] = model if use_claude else "rules"
        print(f"  t={event['time_s']:6.2f}s  {event['shot_type']:<15} {event['stroke_side']:<9} "
              f"conf {event['confidence']:.2f}  — {event['reasoning']}")

    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Saved → {events_path}")
    return events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Name the shots in outputs/shot_events.json.")
    parser.add_argument("--video", default="data/test_clip.MOV")
    parser.add_argument("--racket-arm", choices=["right", "left"], default="right")
    parser.add_argument("--use-claude", action="store_true",
                        help="Ask Claude (with video frames) instead of using the rules")
    parser.add_argument("--model", default=MODEL, help="with --use-claude: e.g. claude-haiku-4-5 or claude-sonnet-5")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    classify_shots(os.path.join(args.output_dir, "shot_events.json"), args.video,
                   os.path.join(args.output_dir, "unified_data.json"),
                   racket_arm=args.racket_arm, use_claude=args.use_claude, model=args.model)
