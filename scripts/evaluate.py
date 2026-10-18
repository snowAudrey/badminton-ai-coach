"""
Score the coach against hand labels.

Usage:
    python scripts/evaluate.py --labels data/labels/test_clip.csv
    python scripts/evaluate.py --labels data/labels/test_clip.csv --note "after phase 2"

Reports, using whatever is available in outputs/:
  - shuttle detection rate                 (outputs/shuttle_data.json)
  - elbow-angle and wrist-height coverage  (outputs/observations.json, Phase 1.4+)
  - shot detection recall / precision      (outputs/shot_events.json, Phase 2+)
  - shot-type top-1 / top-2 accuracy       (same)
Each run appends one row to outputs/eval_history.csv so runs can be compared.
"""

import argparse
import csv
import json
import os
from datetime import date


# A label and a detected shot count as the same shot if their times are
# within this many seconds of each other
MATCH_TOLERANCE_S = 0.5


def normalize_type(name) -> str:
    """Maps shot-type spellings onto the fixed label vocabulary."""
    if not name:
        return ""
    name = str(name).strip().lower()
    aliases = {
        "block": "block/defence",
        "defence": "block/defence",
        "defense": "block/defence",
        "defensive shot": "block/defence",
        "net": "net shot",
        "drop shot": "drop",
        "return": "return of serve",
    }
    return aliases.get(name, name)


def load_labels(path: str) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["time_s"] = float(row["time_s"])
        row["shot_type"] = normalize_type(row.get("shot_type"))
    return sorted(rows, key=lambda r: r["time_s"])


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def match_events(labels: list, events: list, tolerance: float) -> list:
    """
    Pairs each label with at most one detected shot, closest in time first.
    Returns a list of (label, event) pairs.
    """
    candidates = []
    for li, label in enumerate(labels):
        for ei, event in enumerate(events):
            dt = abs(label["time_s"] - float(event.get("time_s", -1e9)))
            if dt <= tolerance:
                candidates.append((dt, li, ei))
    candidates.sort()

    used_labels, used_events, pairs = set(), set(), []
    for _, li, ei in candidates:
        if li in used_labels or ei in used_events:
            continue
        used_labels.add(li)
        used_events.add(ei)
        pairs.append((labels[li], events[ei]))
    return pairs


def pct(numerator, denominator) -> str:
    return "n/a" if not denominator else f"{100 * numerator / denominator:.0f}% ({numerator}/{denominator})"


def main():
    parser = argparse.ArgumentParser(description="Score the coach against hand labels.")
    parser.add_argument("--labels", required=True, help="CSV from scripts/label_shots.py")
    parser.add_argument("--outputs", default="outputs", help="Folder holding the pipeline outputs")
    parser.add_argument("--note", default="", help="Short description saved with this run")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    shuttle = load_json(os.path.join(args.outputs, "shuttle_data.json"))
    observations = load_json(os.path.join(args.outputs, "observations.json"))
    events = load_json(os.path.join(args.outputs, "shot_events.json"))

    row = {"date": date.today().isoformat(), "note": args.note, "labels": len(labels)}
    print(f"\nLabels: {len(labels)} shots from {args.labels}")

    # --- shuttle detection rate ---
    if shuttle:
        detected = sum(1 for f in shuttle if f.get("detected"))
        row["shuttle_rate"] = round(detected / len(shuttle), 3)
        print(f"Shuttle detected:      {pct(detected, len(shuttle))} of frames")
    else:
        print("Shuttle detected:      n/a (no outputs/shuttle_data.json)")

    # --- how often the racket arm was measurable ---
    if observations:
        n = len(observations)
        with_elbow = sum(1 for o in observations if o.get("elbow_angle") is not None)
        with_wrist = sum(1 for o in observations if o.get("wrist_height") is not None)
        row["elbow_coverage"] = round(with_elbow / n, 3) if n else None
        row["wrist_coverage"] = round(with_wrist / n, 3) if n else None
        print(f"Elbow angle present:   {pct(with_elbow, n)} of observation lines")
        print(f"Wrist height present:  {pct(with_wrist, n)} of observation lines")
    else:
        print("Angle coverage:        n/a (no outputs/observations.json)")

    # --- shot detection and naming ---
    if events is None:
        print("Shot detection:        n/a (no outputs/shot_events.json yet — Phase 2)")
    else:
        # Detections the classifier judged to be no hit at all don't count as shots
        events = [e for e in events if normalize_type(e.get("shot_type")) != "no shot"]
        pairs = match_events(labels, events, MATCH_TOLERANCE_S)
        row["shot_recall"] = round(len(pairs) / len(labels), 3) if labels else None
        row["shot_precision"] = round(len(pairs) / len(events), 3) if events else None
        print(f"Shots found (±{MATCH_TOLERANCE_S}s): {pct(len(pairs), len(labels))} of labeled shots")
        print(f"False shots:           {pct(len(events) - len(pairs), len(events))} of detected shots")

        typed = [(l, e) for l, e in pairs if e.get("shot_type")]
        if typed:
            top1 = sum(1 for l, e in typed if normalize_type(e["shot_type"]) == l["shot_type"])
            top2 = 0
            for l, e in typed:
                cands = [normalize_type(c) for c in (e.get("candidates") or [e["shot_type"]])[:2]]
                if normalize_type(e["shot_type"]) not in cands:
                    cands = [normalize_type(e["shot_type"])] + cands[:1]
                top2 += l["shot_type"] in cands
            row["type_top1"] = round(top1 / len(typed), 3)
            row["type_top2"] = round(top2 / len(typed), 3)
            print(f"Shot type correct:     {pct(top1, len(typed))} first guess, {pct(top2, len(typed))} within top two")

            # Only score forehand/backhand where the coach actually committed to one
            sided = [(l, e) for l, e in typed
                     if str(e.get("stroke_side") or e.get("side") or "").lower() in ("forehand", "backhand")
                     and l.get("side") in ("forehand", "backhand")]
            if sided:
                ok = sum(1 for l, e in sided if str(e.get("stroke_side") or e.get("side")).lower() == l["side"])
                row["side_acc"] = round(ok / len(sided), 3)
                print(f"Forehand/backhand:     {pct(ok, len(sided))}")

    # --- keep a history so runs can be compared ---
    history = os.path.join(args.outputs, "eval_history.csv")
    columns = ["date", "note", "labels", "shuttle_rate", "elbow_coverage", "wrist_coverage",
               "shot_recall", "shot_precision", "type_top1", "type_top2", "side_acc"]
    new_file = not os.path.exists(history)
    os.makedirs(args.outputs, exist_ok=True)
    with open(history, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})
    print(f"\nSaved to {history}")


if __name__ == "__main__":
    main()
