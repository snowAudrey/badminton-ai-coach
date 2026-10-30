"""
Optional history of analyzed clips, kept in a PostgreSQL database (for
example on Amazon RDS), so a player can see their progress across sessions.

Nothing here runs unless asked: main.py only saves when given
--save-history, and a failed save never stops the rest of the run.

The database address comes from DATABASE_URL in the .env file:
    DATABASE_URL=postgresql://USERNAME:PASSWORD@ENDPOINT:5432/badminton?sslmode=require

Run on its own:
    python -m src.storage.history                          # sessions saved for "player"
    python -m src.storage.history --shot clear             # every clear, oldest first
    python -m src.storage.history --save outputs/pro --video data/test_clip_pro.MOV
                                                           # save a folder from an earlier run
"""

import argparse
import datetime
import json
import os

import psycopg
from psycopg.types.json import Jsonb
from dotenv import load_dotenv


# ============================================================
# SETTINGS
# ============================================================

DEFAULT_PLAYER = "player"

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")

# Give up on connecting after this many seconds, so an unreachable database
# (e.g. a changed home IP address) fails quickly instead of hanging the run
CONNECT_TIMEOUT_S = 10


# ============================================================
# CONNECTION AND TABLES
# ============================================================

def connect() -> psycopg.Connection:
    """Opens a connection to the database named in DATABASE_URL."""
    load_dotenv()
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set in .env")
    return psycopg.connect(url, connect_timeout=CONNECT_TIMEOUT_S)


def create_tables(conn: psycopg.Connection) -> None:
    """Creates the clips, shots and feedback tables if they don't exist yet."""
    with open(SCHEMA_PATH) as f:
        conn.execute(f.read())


# ============================================================
# SAVE ONE RUN
# ============================================================

def _shot_row(clip_id: int, event: dict) -> tuple:
    """One shot from shot_events.json as a row of the shots table."""
    rise = (event.get("shuttle_rise_after") or {}).get("rise_shoulder_widths")
    alternatives = [c for c in (event.get("candidates") or []) if c != event.get("shot_type")]
    return (
        clip_id,
        event["time_s"],
        event.get("shot_type") or "unknown",
        event.get("confidence"),
        ", ".join(alternatives) or None,
        event.get("elbow_angle"),
        event.get("knee_angle_min_before"),
        event.get("wrist_height"),
        event.get("wrist_swing_shoulder_widths"),
        rise,
        event.get("shuttle_incoming_speed"),
        event.get("shuttle_outgoing_speed"),
        Jsonb(event),
    )


def save_run(conn: psycopg.Connection, events: list, feedback_by_window: dict | None,
             video_name: str, player: str = DEFAULT_PLAYER, played_on: datetime.date = None,
             discipline: str = "singles", racket_arm: str = "right") -> int:
    """
    Saves one analyzed clip with its shots and feedback, all in one step: if
    anything fails, nothing is saved. A clip saved earlier for the same
    player, video and day is replaced. Returns the new clip's id.
    """
    played_on = played_on or datetime.date.today()
    create_tables(conn)

    with conn.transaction():
        # Replace an earlier save of the same clip (its shots and feedback go with it)
        conn.execute("DELETE FROM clips WHERE player = %s AND video_name = %s AND played_on = %s",
                     (player, video_name, played_on))

        clip_id = conn.execute(
            """INSERT INTO clips (player, video_name, played_on, discipline, racket_arm, shot_count)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (player, video_name, played_on, discipline, racket_arm, len(events)),
        ).fetchone()[0]

        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO shots (clip_id, time_s, shot_type, confidence, alternatives,
                                      elbow_angle, knee_angle_min, wrist_height, swing,
                                      shuttle_rise, shuttle_incoming_speed, shuttle_outgoing_speed, details)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [_shot_row(clip_id, event) for event in events],
            )
            feedback_rows = [(clip_id, label, text) for label, text in (feedback_by_window or {}).items() if text]
            if feedback_rows:
                cur.executemany("INSERT INTO feedback (clip_id, window_label, text) VALUES (%s, %s, %s)",
                                feedback_rows)

    return clip_id


def save_run_safely(events: list, feedback_by_window: dict | None, video_path: str,
                    player: str = DEFAULT_PLAYER, played_on: datetime.date = None,
                    discipline: str = "singles", racket_arm: str = "right") -> int | None:
    """
    save_run for main.py: opens and closes its own connection, and turns any
    failure into a printed warning, so the local results are never affected.
    """
    try:
        with connect() as conn:
            clip_id = save_run(conn, events, feedback_by_window, os.path.basename(video_path),
                               player=player, played_on=played_on,
                               discipline=discipline, racket_arm=racket_arm)
        print(f"Saved to history: clip {clip_id}, {len(events)} shot(s), player '{player}'")
        return clip_id
    except Exception as e:
        print(f"WARNING: could not save to the history database ({type(e).__name__}: {e}). "
              f"The local results in outputs/ are unaffected. If the connection timed out, "
              f"check that your current IP is allowed in the database's security group.")
        return None


def load_output_folder(output_dir: str) -> tuple:
    """
    The shots and feedback saved in an output folder by an earlier run. The
    feedback is used only if it was written after the shots, so feedback
    left over from a different, older run is not saved with them.
    """
    events_path = os.path.join(output_dir, "shot_events.json")
    feedback_path = os.path.join(output_dir, "feedback.json")
    with open(events_path) as f:
        events = json.load(f)
    feedback = None
    if os.path.exists(feedback_path) and os.path.getmtime(feedback_path) >= os.path.getmtime(events_path):
        with open(feedback_path) as f:
            feedback = json.load(f)
    elif os.path.exists(feedback_path):
        print("Note: feedback.json is older than shot_events.json, so it belongs to an earlier run "
              "and is not saved.")
    return events, feedback


# ============================================================
# SHOW HISTORY
# ============================================================

def print_sessions(conn: psycopg.Connection, player: str) -> None:
    """One line per saved clip: when, which video, and how many of each shot."""
    rows = conn.execute(
        """SELECT c.played_on, c.video_name, c.shot_count,
                  string_agg(s.shot_type || ' ' || s.n, ', ' ORDER BY s.n DESC, s.shot_type) AS mix
           FROM clips c
           JOIN (SELECT clip_id, shot_type, count(*) AS n FROM shots GROUP BY clip_id, shot_type) s
             ON s.clip_id = c.id
           WHERE c.player = %s
           GROUP BY c.id
           ORDER BY c.played_on, c.saved_at""",
        (player,),
    ).fetchall()
    if not rows:
        print(f"No sessions saved for '{player}' yet.")
        return
    print(f"Sessions for '{player}':")
    for played_on, video, count, mix in rows:
        print(f"  {played_on}  {video:<28} {count:>3} shots: {mix}")


def print_shots(conn: psycopg.Connection, player: str, shot_type: str) -> None:
    """Every saved shot of one type, oldest first, with its main measurements."""
    rows = conn.execute(
        """SELECT c.played_on, c.video_name, s.time_s, s.confidence, s.elbow_angle,
                  s.knee_angle_min, s.wrist_height
           FROM shots s JOIN clips c ON c.id = s.clip_id
           WHERE c.player = %s AND s.shot_type = %s
           ORDER BY c.played_on, c.saved_at, s.time_s""",
        (player, shot_type),
    ).fetchall()
    if not rows:
        print(f"No {shot_type} shots saved for '{player}' yet.")
        return

    def fmt(value, unit=""):
        return "-" if value is None else f"{value:.0f}{unit}"

    print(f"Every {shot_type} saved for '{player}':")
    print(f"  {'date':<10}  {'video':<28} {'time':>6}  {'conf':>4}  {'elbow':>5}  {'knee':>5}  hand")
    for played_on, video, time_s, conf, elbow, knee, hand in rows:
        print(f"  {played_on}  {video:<28} {time_s:>5.1f}s  {conf if conf is not None else 0:>4.2f}  "
              f"{fmt(elbow, '°'):>5}  {fmt(knee, '°'):>5}  {hand or '-'}")


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Show or add to the history of analyzed clips.")
    parser.add_argument("--player", default=DEFAULT_PLAYER, help=f"player name (default: {DEFAULT_PLAYER})")
    parser.add_argument("--shot", help="list every saved shot of this type, e.g. clear")
    parser.add_argument("--save", metavar="OUTPUT_DIR",
                        help="save the results already in this folder (e.g. outputs/pro) instead of showing history")
    parser.add_argument("--video", help="with --save: the clip those results came from")
    parser.add_argument("--date", type=datetime.date.fromisoformat,
                        help="with --save: the day the clip was played, YYYY-MM-DD (default: today)")
    parser.add_argument("--discipline", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--racket-arm", choices=["right", "left"], default="right")
    args = parser.parse_args()

    if args.save:
        if not args.video:
            parser.error("--save needs --video, the clip the results came from")
        events, feedback = load_output_folder(args.save)
        save_run_safely(events, feedback, args.video, player=args.player, played_on=args.date,
                        discipline=args.discipline, racket_arm=args.racket_arm)
    else:
        with connect() as conn:
            if args.shot:
                print_shots(conn, args.player, args.shot)
            else:
                print_sessions(conn, args.player)
