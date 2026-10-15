"""
Label the shots in a clip by hand, so the coach can be scored against them.

Usage:
    python scripts/label_shots.py --video data/test_clip.MOV

Controls (with the video window focused):
    a / d        step back / forward 1 frame
    s / w        step back / forward 10 frames
    space        play / pause
    m            mark the current frame as the moment of contact, then keep
                 watching to see where the shuttle lands before choosing the
                 shot type — the label is saved at the marked time
    1-9, 0       log a shot (at the marked frame if one is set, else the
                 current frame):
                   1 smash   2 clear   3 drop    4 drive   5 net shot
                   6 lift    7 block/defence     8 serve   9 return of serve   0 other
                 then press f (forehand) or b (backhand); any other key = unknown
    u            undo the last label
    q            save and quit

Labels are saved to data/labels/<video name>.csv after every change, so you
can quit and resume at any time.
"""

import argparse
import csv
import os

import cv2


SHOT_KEYS = {
    ord("1"): "smash",
    ord("2"): "clear",
    ord("3"): "drop",
    ord("4"): "drive",
    ord("5"): "net shot",
    ord("6"): "lift",
    ord("7"): "block/defence",
    ord("8"): "serve",
    ord("9"): "return of serve",
    ord("0"): "other",
}

SIDE_KEYS = {ord("f"): "forehand", ord("b"): "backhand"}

LABEL_COLUMNS = ["time_s", "frame", "shot_type", "side"]

HELP_LINES = [
    "a/d: -1/+1 frame   s/w: -10/+10   space: play/pause   m: mark contact   u: undo   q: save & quit",
    "1 smash  2 clear  3 drop  4 drive  5 net  6 lift  7 block  8 serve  9 return  0 other  (then f/b)",
]

# Longest side of the preview window, in pixels
MAX_DISPLAY_HEIGHT = 900


def load_labels(path: str) -> list:
    """Loads existing labels so a labeling session can be resumed."""
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["time_s"] = float(row["time_s"])
        row["frame"] = int(row["frame"])
    return rows


def save_labels(path: str, labels: list):
    """Writes all labels, sorted by time, so the file is always complete."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        for row in sorted(labels, key=lambda r: r["time_s"]):
            writer.writerow({k: row[k] for k in LABEL_COLUMNS})


def draw_overlay(frame, lines: list):
    """Draws text lines on a dark strip at the top of the frame."""
    pad = 8
    line_h = 22
    cv2.rectangle(frame, (0, 0), (frame.shape[1], pad * 2 + line_h * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Hand-label badminton shots in a clip.")
    parser.add_argument("--video", default="data/test_clip.MOV", help="Path to the clip to label")
    parser.add_argument("--out", default=None,
                        help="Output CSV (default: data/labels/<video name>.csv)")
    args = parser.parse_args()

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_path = args.out or os.path.join("data", "labels", f"{stem}.csv")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    labels = load_labels(out_path)
    print(f"Labeling {args.video} ({total} frames, {fps:.2f} fps)")
    print(f"Saving to {out_path} — {len(labels)} label(s) loaded")

    window = "Label shots"
    cv2.namedWindow(window)

    idx = 0
    playing = False
    message = ""
    mark = None  # frame index of the contact moment, set with "m"

    while True:
        idx = max(0, min(idx, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break

        scale = min(1.0, MAX_DISPLAY_HEIGHT / frame.shape[0])
        display = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1.0 else frame.copy()

        last = sorted(labels, key=lambda r: r["time_s"])[-1] if labels else None
        status = f"frame {idx}/{total - 1}   t={idx / fps:.3f}s   labels: {len(labels)}"
        if mark is not None:
            status += f"   MARK @ {mark / fps:.2f}s"
        if last:
            status += f"   last: {last['shot_type']} ({last['side']}) @ {last['time_s']:.2f}s"
        lines = [status] + HELP_LINES + ([message] if message else [])
        cv2.imshow(window, draw_overlay(display, lines))

        key = cv2.waitKey(int(1000 / fps) if playing else 0) & 0xFF

        if playing and key == 255:
            idx += 1
            if idx >= total - 1:
                playing = False
            continue

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("d"):
            idx += 1
        elif key == ord("a"):
            idx -= 1
        elif key == ord("w"):
            idx += 10
        elif key == ord("s"):
            idx -= 10
        elif key == ord("m"):
            mark = idx
            message = f"marked contact at {idx / fps:.2f}s — watch where it lands, then press the shot number"
        elif key == ord("u"):
            if labels:
                removed = labels.pop()
                save_labels(out_path, labels)
                message = f"removed {removed['shot_type']} @ {removed['time_s']:.2f}s"
            else:
                message = "nothing to undo"
        elif key in SHOT_KEYS:
            playing = False
            shot = SHOT_KEYS[key]
            prompt = lines[:-1] if message else lines
            cv2.imshow(window, draw_overlay(display.copy(), prompt + [f"{shot}: press f (forehand) or b (backhand)"]))
            side_key = cv2.waitKey(0) & 0xFF
            side = SIDE_KEYS.get(side_key, "unknown")
            at = mark if mark is not None else idx
            labels.append({
                "time_s": round(at / fps, 3),
                "frame": at,
                "shot_type": shot,
                "side": side,
            })
            save_labels(out_path, labels)
            message = f"logged {shot} ({side}) @ {at / fps:.2f}s"
            mark = None

    save_labels(out_path, labels)
    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} label(s) → {out_path}")


if __name__ == "__main__":
    main()
