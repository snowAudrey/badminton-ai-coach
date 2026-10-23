"""
Writes the coaching feedback: one Claude call per clip (or per 30-second
window of a longer clip) that receives the named shots in time order, their
measurements, and the coaching facts retrieved for them.

Run on its own (after unified_data.json and shot_events.json exist):
    python -m src.agents.orchestrator --video data/test_clip.MOV
"""

import json
import os
import textwrap
import time

import anthropic

# load_dotenv reads the .env file and makes ANTHROPIC_API_KEY available to os.getenv
from dotenv import load_dotenv

# analyze_video() turns CV pipeline output into pose observations (kept for
# the racket-arm check image and observations.json)
from src.agents.vision_agent import analyze_video

from src.agents.knowledge_agent import (
    group_events_by_window,
    build_queries,
    retrieve_facts,
    format_shots_for_prompt,
    format_facts_for_prompt,
)


# ============================================================
# SETUP
# ============================================================

# Load the ANTHROPIC_API_KEY from your .env file
load_dotenv()

# Create the Anthropic client — it automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Model that writes the feedback. One call per clip, so the stronger model
# is affordable. Sonnet 5 thinks before it answers, and that thinking counts
# against the reply limit — 1,500 tokens was used up entirely by thinking
# and no feedback text came back, so the limit is generous
MODEL = "claude-sonnet-5"
MAX_FEEDBACK_TOKENS = 8000

# Small delay between API calls in seconds — avoids hitting rate limits
API_DELAY_SECONDS = 0.5


# ============================================================
# FEEDBACK PROMPT
# ============================================================

SYNTHESIS_PROMPT = """You are a badminton coach reviewing a short clip of one player, who plays {discipline}.

You are given:
1. The shots the player played, in time order. Each has its detected name,
   how sure the detection is (0 to 1), the alternatives, and a few
   measurements taken from the video.
2. Coaching facts retrieved from a knowledge base for those shot types and
   those measurements. Each fact may have preparation, execution, a common
   mistake and a cue.

Write feedback for the player, shot by shot, in time order. Address the
player as "you". For each shot, one short paragraph that starts with the
time and the shot name, e.g. "At 5.5s, lift:", followed by 2-4 sentences.

Rules:
- Name the shot outright when its confidence is 0.7 or higher. Below that,
  say what it most likely was and name the alternative ("most likely a
  drop, possibly a clear"), then give advice that holds for both or say
  which applies to which.
- Use only the retrieved facts. Do not add advice from your own knowledge
  and do not invent measurements.
- A measurement is an observation, not proof of a mistake. Before applying
  a common mistake, check that the measurement actually shows it; skip the
  warning when the measurement points the other way or there is no
  measurement for it.
- Angles: 180° is a straight joint, smaller is more bent. Wrist height
  says where the racket hand was relative to the head and shoulder.
- If no fact applies to a shot, write one line, e.g. "At 17.9s, serve:
  nothing to flag", rather than padding.
- "Nothing to flag" means the available measurements look fine. If the
  shuttle was not tracked after a shot, say that where it went could not be
  checked, so the player knows the outcome was not judged.
- End with two sentences on the one or two things that came up most often
  in the clip.

Shots:
{shots_text}

Retrieved coaching facts:
{knowledge_text}
"""


# ============================================================
# STEP 1: WRITE THE FEEDBACK (ONE CLAUDE CALL)
# ============================================================

def synthesize_feedback(events: list, results: dict, discipline: str) -> str:
    """
    Sends one window's shots and retrieved facts to Claude and returns the
    feedback text. Empty string if the API call fails.
    """

    # Fill in the prompt template (.replace() rather than .format(), so
    # braces in the facts can't be mistaken for placeholders)
    prompt = (SYNTHESIS_PROMPT
              .replace("{discipline}", discipline)
              .replace("{shots_text}", format_shots_for_prompt(events))
              .replace("{knowledge_text}", format_facts_for_prompt(results)))

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_FEEDBACK_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        # The reply is plain text; take the text block (the model also
        # returns a thinking block, which is not the feedback)
        text = " ".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            print(f"  ERROR during synthesis: the reply had no text (stop reason: {response.stop_reason}). "
                  f"If the stop reason is max_tokens, raise MAX_FEEDBACK_TOKENS.")
        return text

    except Exception as e:
        print(f"  ERROR during synthesis: {type(e).__name__}: {e}")
        return ""


# ============================================================
# STEP 2: RUN THE FULL FEEDBACK STEP
# ============================================================

def print_ready(feedback: str, width: int = 88) -> str:
    """Wraps each paragraph of the feedback to the given width for the terminal."""
    if not feedback:
        return "(no feedback was produced — see the ERROR line above)"
    return "\n\n".join(textwrap.fill(paragraph.strip(), width) for paragraph in feedback.split("\n") if paragraph.strip())


def _save_json(path: str, data) -> None:
    """Writes data as readable JSON, creating the folder if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_pipeline(unified_data_path: str, events_path: str, racket_arm: str = "right",
                 discipline: str = "singles", video_path: str = None, output_dir: str = "outputs") -> dict:
    """
    Writes the feedback for a clip whose shots have already been found and named:
      1. vision_agent.analyze_video() — racket-arm check image + observations.json
      2. knowledge_agent — queries from the shots, facts from the knowledge base
      3. synthesize_feedback() — one Claude call per window

    Everything Claude was given is also saved, so the feedback can be checked:
      <output_dir>/observations.json       pose observations (reference)
      <output_dir>/queries_and_facts.json  the queries and the facts they found, per window
      <output_dir>/feedback.json           the feedback per window

    Returns:
        Dict mapping window label (e.g. "whole clip" or "0-30s") -> feedback text
    """

    observations = analyze_video(unified_data_path, racket_arm=racket_arm,
                                 video_path=video_path, output_dir=output_dir)
    _save_json(os.path.join(output_dir, "observations.json"), observations)

    with open(events_path) as f:
        events = json.load(f)

    feedback_by_window = {}
    queries_and_facts = {}

    windows = group_events_by_window(events)
    if not windows:
        feedback_by_window["whole clip"] = "No shots were found in this clip, so there is nothing to give feedback on."

    for label, window_events in windows:
        print(f"\nProcessing {label} ({len(window_events)} shots, {discipline})...")
        queries = build_queries(window_events)
        results = retrieve_facts(queries, discipline)
        queries_and_facts[label] = results
        feedback_by_window[label] = synthesize_feedback(window_events, results, discipline)
        time.sleep(API_DELAY_SECONDS)

    _save_json(os.path.join(output_dir, "queries_and_facts.json"), queries_and_facts)
    _save_json(os.path.join(output_dir, "feedback.json"), feedback_by_window)
    print(f"\nSaved observations, queries/facts and feedback → {output_dir}/")

    return feedback_by_window


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write coaching feedback from a clip's saved shots.")
    parser.add_argument("--video", default="data/test_clip.MOV", help="the clip (for the racket-arm check image)")
    parser.add_argument("--racket-arm", choices=["right", "left"], default="right")
    parser.add_argument("--discipline", choices=["singles", "doubles"], default="singles")
    parser.add_argument("--output-dir", default="outputs", help="folder holding unified_data.json and shot_events.json")
    args = parser.parse_args()

    feedback_by_window = run_pipeline(
        os.path.join(args.output_dir, "unified_data.json"),
        os.path.join(args.output_dir, "shot_events.json"),
        racket_arm=args.racket_arm, discipline=args.discipline,
        video_path=args.video, output_dir=args.output_dir,
    )

    for label, feedback in feedback_by_window.items():
        print(f"\n{'=' * 60}")
        print(f"FEEDBACK — {label}")
        print('=' * 60)
        print(print_ready(feedback))
