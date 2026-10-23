"""
Turns the shots found in a clip into searches of the coaching knowledge
base, and collects the facts that come back, grouped by shot type.

For every shot type in the clip there is one base query (the shot's
preparation, contact point and follow-through) plus one query for each
measurement that stands out (elbow straight at contact, deep knee bend,
and so on — see shot_flags). The queries are fixed templates, so the same
clip always asks the same questions. The earlier version asked Claude to
write queries from raw angle lines, and which shots it covered changed from
run to run.

Run on its own (after shot_events.json exists):
    python -m src.agents.knowledge_agent --output-dir outputs
"""

import json
import os

# query_knowledge_base() searches the ChromaDB knowledge base built in vector_store.py
from src.knowledge.vector_store import query_knowledge_base
from src.knowledge.tags import SHOT_TYPES


# ============================================================
# SETTINGS
# ============================================================

# How many coaching facts to retrieve per query (weak matches are dropped
# by the knowledge base's distance cut-off, so a query can return fewer)
N_RESULTS_PER_QUERY = 4

# Roughly how much of a clip is handled in one feedback call, in seconds.
# A clip is cut into equal windows, as many as it takes to keep each one
# near this length (a 34 s clip is one window, a 50 s clip two of 25 s)
WINDOW_SECONDS = 30

# A shot named with at least this confidence is taken as that shot. Below
# it, the classifier's second candidate is searched as well, so the
# feedback can hedge between the two
CONFIDENT = 0.7

# Measurements that stand out. Angles in degrees (180 = straight joint),
# swing in shoulder widths of racket-wrist travel
ELBOW_STRAIGHT = 160
ELBOW_BENT = 120
KNEE_DEEP = 120
KNEE_STRAIGHT = 165
SHORT_SWING = 1.5
BIG_SWING = 3.0

# Which shots are hit from above the head, which need a full swing, and
# which are soft touches where a big swing is a fault
OVERHEAD_SHOTS = {"smash", "clear", "drop"}
FULL_SWING_SHOTS = {"clear", "smash", "lift"}
SOFT_SHOTS = {"net shot", "block/defence", "serve"}


# ============================================================
# STEP 1: GROUP SHOTS BY TIME WINDOW
# ============================================================

def group_events_by_window(events: list, window_seconds: float = WINDOW_SECONDS) -> list:
    """
    Splits the clip's shots into equal consecutive windows of roughly
    window_seconds. Returns [(label, events)], e.g. [("whole clip", [...])]
    or [("0-25s", [...]), ("25-50s", [...])].
    """
    if not events:
        return []
    end = max(e["time_s"] for e in events)
    count = max(1, int(round(end / window_seconds)))
    if count == 1:
        return [("whole clip", list(events))]

    length = end / count
    windows = []
    for i in range(count):
        start, stop = i * length, (i + 1) * length
        chunk = [e for e in events if start <= e["time_s"] < stop or (i == count - 1 and e["time_s"] >= start)]
        if chunk:
            windows.append((f"{int(start)}-{int(round(stop))}s", chunk))
    return windows


# ============================================================
# STEP 2: WHAT STANDS OUT IN A SHOT'S MEASUREMENTS
# ============================================================

def shot_flags(event: dict) -> list:
    """
    The measurements of one shot that deserve a search of their own, as
    (query phrase, plain-language observation) pairs.
    """
    shot = event.get("shot_type")
    elbow = event.get("elbow_angle")
    knee = event.get("knee_angle_min_before") if shot != "serve" else None   # a serve starts from standing
    swing = event.get("wrist_swing_shoulder_widths")
    highest = event.get("highest_wrist_before")      # highest the racket hand got around the swing
    flags = []

    if elbow is not None:
        if elbow >= ELBOW_STRAIGHT:
            flags.append(("elbow straight at contact", f"elbow {elbow:.0f}° at contact, nearly straight"))
        elif elbow <= ELBOW_BENT:
            flags.append(("elbow very bent at contact", f"elbow {elbow:.0f}° at contact, very bent"))

    if knee is not None:
        if knee <= KNEE_DEEP:
            flags.append(("deep knee bend before the shot", f"deepest knee bend before the shot {knee:.0f}°, deep"))
        elif knee >= KNEE_STRAIGHT:
            flags.append(("legs straight, no knee bend before the shot",
                          f"knees never below {knee:.0f}° before the shot, legs straight"))

    if shot in OVERHEAD_SHOTS and highest and highest != "overhead":
        flags.append(("contact point below head height", f"racket hand never above the head around the swing ({highest} at best)"))

    if swing is not None:
        if shot in FULL_SWING_SHOTS and swing < SHORT_SWING:
            flags.append(("short swing, little racket travel", f"swing only {swing} shoulder widths, short for a {shot}"))
        elif shot in SOFT_SHOTS and swing >= BIG_SWING:
            flags.append(("big swing on a soft shot", f"swing {swing} shoulder widths, big for a {shot}"))

    # A high serve and a low serve are different techniques; ask for the right one
    if shot == "serve" and event.get("serve_height") == "high":
        flags.append(("high serve, height and length to the back of the court", "the serve went high"))
    elif shot == "serve" and event.get("serve_height") == "low":
        flags.append(("low serve, tight over the net", "the serve stayed low"))

    return flags


# ============================================================
# STEP 3: BUILD THE QUERIES AND RETRIEVE FACTS
# ============================================================

def base_query(shot: str) -> str:
    return f"{shot} technique: preparation, contact point and follow-through"


def build_queries(events: list) -> dict:
    """
    The search queries for one window, grouped by the shot type they are
    filtered to: {"clear": ["clear technique: ...", "clear: elbow straight at contact"], ...}
    """
    queries = {}

    def add(shot, text):
        if shot in SHOT_TYPES and text not in queries.setdefault(shot, []):
            queries[shot].append(text)

    for event in events:
        shot = event.get("shot_type")
        if shot not in SHOT_TYPES:           # "other", "no shot"
            continue
        add(shot, base_query(shot))
        for phrase, _ in shot_flags(event):
            add(shot, f"{shot}: {phrase}")
        if (event.get("confidence") or 0) < CONFIDENT:
            for alternative in (event.get("candidates") or [])[1:2]:
                add(alternative, base_query(alternative))

    return queries


def retrieve_facts(queries: dict, discipline: str) -> dict:
    """
    Runs every query against the knowledge base, limited to its shot type
    and to the player's discipline. A fact that already came back for an
    earlier query is not repeated. Returns {shot: {query: [facts]}}.
    """
    results = {}
    seen_ids = set()
    for shot, texts in queries.items():
        results[shot] = {}
        for text in texts:
            facts = [fact for fact in query_knowledge_base(text, n_results=N_RESULTS_PER_QUERY,
                                                           shot_type=shot, discipline=discipline)
                     if fact.get("id") not in seen_ids]
            seen_ids.update(fact.get("id") for fact in facts)
            results[shot][text] = facts
    return results


# ============================================================
# STEP 4: TEXT FOR THE FEEDBACK PROMPT
# ============================================================

def describe_event(event: dict) -> str:
    """One shot as a line of plain language for the feedback prompt."""
    shot = event.get("shot_type", "unknown")
    confidence = event.get("confidence") or 0
    alternatives = [c for c in (event.get("candidates") or []) if c != shot][:2]
    name = shot
    if shot == "serve" and event.get("serve_height") in ("high", "low"):
        name = f"{event['serve_height']} serve"
    head = f"At {event['time_s']:.1f}s: {name} (confidence {confidence:.2f}"
    if alternatives:
        head += f"; alternatives: {', '.join(alternatives)}"
    head += ")."

    parts = []
    lead, at_peak = event.get("wrist_height_before_peak"), event.get("wrist_height")
    if lead and at_peak:
        parts.append(f"racket hand {lead} a quarter-second before the fastest moment and {at_peak} at it")
    elif at_peak:
        parts.append(f"racket hand {at_peak} at the fastest moment")
    if event.get("elbow_angle") is not None:
        parts.append(f"elbow {event['elbow_angle']:.0f}° at contact")
    if event.get("elbow_angle_min_before") is not None:
        parts.append(f"most bent the elbow got before the shot {event['elbow_angle_min_before']:.0f}°")
    if event.get("knee_angle_min_before") is not None:
        parts.append(f"deepest knee bend before the shot {event['knee_angle_min_before']:.0f}°")
    if event.get("wrist_swing_shoulder_widths") is not None:
        parts.append(f"swing {event['wrist_swing_shoulder_widths']} shoulder widths")
    rise = (event.get("shuttle_rise_after") or {}).get("rise_shoulder_widths")
    if rise is not None:
        parts.append(f"shuttle climbed {rise} shoulder widths above head height afterwards")
    elif not event.get("shuttle_after"):
        parts.append("shuttle not tracked after contact, so where it went is unknown")
    if event.get("shuttle_incoming_speed") is not None:
        parts.append(f"shuttle arrived at {event['shuttle_incoming_speed']} shoulder widths/s")
    if event.get("seconds_since_previous_shot") is not None:
        parts.append(f"{event['seconds_since_previous_shot']} s since the previous shot")

    line = head + (" Measurements: " + "; ".join(parts) + "." if parts else " No measurements.")
    stands_out = [observation for _, observation in shot_flags(event)]
    if stands_out:
        line += " Stands out: " + "; ".join(stands_out) + "."
    return line


def format_shots_for_prompt(events: list) -> str:
    return "\n".join(describe_event(event) for event in events)


def format_facts_for_prompt(results: dict) -> str:
    """The retrieved facts grouped by shot type, with every non-empty field."""
    lines = []
    for shot, by_query in results.items():
        facts = [fact for query_facts in by_query.values() for fact in query_facts]
        if not facts:
            continue
        lines.append(f"Facts about the {shot}:")
        for fact in facts:
            bits = []
            for label, key in (("preparation", "preparation"), ("execution", "execution"),
                               ("common mistake", "common_mistake"), ("cue", "cue")):
                if fact.get(key):
                    bits.append(f"{label}: {fact[key]}")
            lines.append(f"  - ({fact.get('technique', '')}) " + " | ".join(bits))
        lines.append("")
    return "\n".join(lines).strip() or "No facts were retrieved."


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Show the queries and facts the feedback step would use.")
    parser.add_argument("--output-dir", default="outputs", help="folder holding shot_events.json")
    parser.add_argument("--discipline", choices=["singles", "doubles"], default="singles")
    args = parser.parse_args()

    with open(os.path.join(args.output_dir, "shot_events.json")) as f:
        events = json.load(f)

    for label, window_events in group_events_by_window(events):
        print(f"\n{'=' * 60}\n{label.upper()} — {len(window_events)} shots\n{'=' * 60}")
        print(format_shots_for_prompt(window_events))
        queries = build_queries(window_events)
        results = retrieve_facts(queries, args.discipline)
        for shot, by_query in results.items():
            for query, facts in by_query.items():
                print(f"\nQUERY [{shot}]: {query}  → {len(facts)} fact(s)")
                for fact in facts:
                    print(f"   {fact['relevance_distance']}  {fact['cue'][:110]}")
