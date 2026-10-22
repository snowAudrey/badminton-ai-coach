import anthropic
import json
import os

# time is a built-in Python library — used here to add a small delay between API calls
# to avoid hitting rate limits
import time

# load_dotenv reads the .env file and makes ANTHROPIC_API_KEY available to os.getenv
from dotenv import load_dotenv

# Import the caption parser to load chunks directly from it
from src.knowledge.caption_parser import parse_all_vtt_files

# The fixed-choice tags, shared with the vector store and the agents
from src.knowledge.tags import SHOT_TYPES, STROKE_SIDES, DISCIPLINES, BODY_PARTS


# ============================================================
# SETUP
# ============================================================

# Load the ANTHROPIC_API_KEY from your .env file
load_dotenv()

# Create the Anthropic client — it automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Model to use for extraction
# Haiku 4.5 is fast and cheap — ideal for batch processing
MODEL = "claude-haiku-4-5"

# How many chunks to process in test mode
# Always test on a small batch before running all 416
TEST_BATCH_SIZE = 5

# Where the extracted facts go. The earlier, untagged extraction lives in
# outputs/coaching_facts_full.json; this run is kept separate so the two
# can be compared
TEST_OUTPUT_PATH = "outputs/coaching_facts_test.json"
FULL_OUTPUT_PATH = "outputs/coaching_facts_v2.json"

# Every fact gets the fixed-choice tags from src/knowledge/tags.py, spelled
# exactly, so the knowledge base can be filtered by them — "which facts are
# about a lift, for singles?" — instead of hoping a free-text label happens
# to match.

# Small delay between API calls in seconds
# Prevents hitting rate limits on large batches
API_DELAY_SECONDS = 0.5

# Longest reply allowed, in tokens. A 500-word chunk can hold a dozen
# facts, each with several sentences; 2000 cut one chunk's reply off in the
# middle of a fact, which made the whole reply unreadable
MAX_REPLY_TOKENS = 4000


# ============================================================
# EXTRACTION PROMPT
# ============================================================
# This is the most important part — it tells Claude exactly what to extract
# and how to format it. 

EXTRACTION_PROMPT = """You are extracting badminton coaching knowledge from a video transcript.

Given this transcript chunk from a badminton tutorial video, extract every concrete coaching fact you can find.
Extract one JSON object per distinct coaching point — do not bundle multiple aspects of a technique into one object.

For each coaching fact, output a JSON object with these fields:
- "technique": the stroke or skill being discussed, in a few words (e.g. "drop shot", "clear", "corner footwork")
- "shot_types": a list of one to three entries chosen from exactly this list: "smash", "clear", "drop", "drive", "net shot", "lift", "block/defence", "serve", "return of serve", "net kill", "footwork", "general". Name the shot(s) the advice is about. Use "footwork" for movement advice not tied to one shot, and "general" for tactics, mindset, grip changes or anything else. Copy the spelling exactly.
- "stroke_side": "forehand", "backhand", or "any" when the advice applies to both sides or the side is not mentioned
- "discipline": "doubles" if the advice depends on having a partner (doubles rotation, formation, "your partner"), "singles" if it depends on covering the whole court alone, otherwise "both"
- "body_parts": a list of zero to three entries chosen from exactly this list: "grip", "wrist", "elbow", "shoulder", "arm", "torso", "hips", "legs", "feet", "head". Name the body parts the advice is about.
- "preparation": the body position, stance, grip, or setup required before executing the shot — leave empty string "" if not mentioned
- "execution": the swing, contact point, follow-through, or movement during the shot — leave empty string "" if not mentioned
- "common_mistake": the specific error and what it causes (e.g. "dropping the elbow causes the racket face to open and the shuttle to land short") — leave empty string "" if not mentioned
- "cue": the one or two line actionable coaching instruction — use the original transcript language as closely as possible, only rephrase to remove filler words or fix unclear references like "like this"
- "context": when or where this applies (e.g. "forehand side, opponent has lifted to back court", "moving to front corner from base position") — leave empty string "" if not mentioned

Rules:
- Extract one fact per object — if a chunk covers three aspects of a drop shot, output three separate objects
- Only extract concrete, actionable coaching facts
- Skip channel introductions, filler phrases, and off-topic content
- Skip anything that references a visual demonstration without explaining it (e.g. "like this", "as you can see" with no further explanation)
- Each fact must make sense as standalone advice without needing the video
- Keep cue language close to the transcript — do not invent coaching advice not present in the text
- If no coaching facts are present in the chunk, return an empty array []
- Tag values must come from the lists above, spelled exactly as given

Return ONLY a valid JSON array. No explanation, no markdown, no preamble.

Example output format:
[
  {
    "technique": "drop shot",
    "shot_types": ["drop"],
    "stroke_side": "forehand",
    "discipline": "both",
    "body_parts": ["torso", "legs"],
    "preparation": "Turn your whole body so you're facing sideways, feet wider than shoulder-width, most of your weight on the back leg.",
    "execution": "",
    "common_mistake": "",
    "cue": "Turn your whole body so you're facing sideways before the shuttle arrives.",
    "context": "Forehand side, setting up body position before playing the drop shot."
  },
  {
    "technique": "overhead preparation",
    "shot_types": ["drop", "clear", "smash"],
    "stroke_side": "forehand",
    "discipline": "both",
    "body_parts": ["grip", "shoulder", "elbow"],
    "preparation": "Use a forehand grip with racket arm shoulder and elbow at 90 degrees.",
    "execution": "",
    "common_mistake": "Racket arm too straight or too bent reduces power and control.",
    "cue": "Have your racket arm shoulder and elbow at 90 degrees before swinging.",
    "context": "Racket arm setup for any overhead shot."
  },
  {
    "technique": "doubles rotation",
    "shot_types": ["general"],
    "stroke_side": "any",
    "discipline": "doubles",
    "body_parts": [],
    "preparation": "",
    "execution": "",
    "common_mistake": "Both players standing side by side when attacking leaves the front of the court open.",
    "cue": "When your partner smashes, move forward to cover the net.",
    "context": "Attacking in doubles after a lift from the opponents."
  }
]

Transcript chunk:
{chunk_text}"""


# ============================================================
# TAG CLEANING
# ============================================================

def _pick(values, allowed: list, limit: int) -> list:
    """Keeps the entries of `values` that are in `allowed` (case-insensitive), at most `limit`."""
    if isinstance(values, str):
        values = [values]
    kept = []
    for value in values or []:
        value = str(value).strip().lower()
        if value in allowed and value not in kept:
            kept.append(value)
    return kept[:limit]


def clean_tags(fact: dict) -> dict:
    """
    Forces every tag onto the fixed lists. Anything Claude spelled its own
    way is dropped, and an empty shot list becomes "general", so a later
    filter never meets a value it doesn't know.
    """
    fact["shot_types"] = _pick(fact.get("shot_types"), SHOT_TYPES, 3) or ["general"]
    fact["stroke_side"] = (_pick(fact.get("stroke_side"), STROKE_SIDES, 1) or ["any"])[0]
    fact["discipline"] = (_pick(fact.get("discipline"), DISCIPLINES, 1) or ["both"])[0]
    fact["body_parts"] = _pick(fact.get("body_parts"), BODY_PARTS, 3)
    for field in ("technique", "preparation", "execution", "common_mistake", "cue", "context"):
        fact[field] = str(fact.get(field) or "").strip()
    return fact


def _fact_key(fact: dict) -> tuple:
    """What makes two facts the same tip: the same cue, execution and mistake text."""
    def norm(text):
        return " ".join(str(text or "").lower().split())
    return norm(fact.get("cue")), norm(fact.get("execution")), norm(fact.get("common_mistake"))


# ============================================================
# CORE EXTRACTION FUNCTION
# ============================================================

def extract_from_chunk(chunk: dict) -> list:
    """
    Sends a single text chunk to Claude and returns a list of extracted coaching facts.

    Args:
        chunk: A dict with "text", "source", and "chunk_index" keys
               (output format from caption_parser.py)

    Returns:
        List of coaching fact dicts. Empty list if nothing useful was found
        or if the API call failed.
    """

    # Fill in the chunk text into the prompt template
    # (using .replace() instead of .format() — the template contains literal
    # JSON braces in its example output, which .format() would misinterpret
    # as additional placeholders)
    prompt = EXTRACTION_PROMPT.replace("{chunk_text}", chunk["text"])

    try:
        # Call the Claude API
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_REPLY_TOKENS,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the text response from the API result
        raw_response = response.content[0].text.strip()

        # Claude sometimes wraps its output in a markdown code fence
        # (```json ... ```) despite being told not to — strip it if present
        if raw_response.startswith("```"):
            lines = raw_response.split("\n")
            raw_response = "\n".join(lines[1:-1]).strip()

        # Parse the JSON response
        facts = json.loads(raw_response)

        # Clean the tags and add source metadata so we know which video each fact came from
        cleaned = []
        for fact in facts:
            if not isinstance(fact, dict):
                continue
            fact = clean_tags(fact)
            fact["source"] = chunk["source"]
            fact["video_title"] = chunk.get("video_title", "")
            fact["level"] = chunk.get("level", "")
            fact["chunk_index"] = chunk["chunk_index"]
            cleaned.append(fact)

        return cleaned

    except json.JSONDecodeError:
        # Claude returned something that wasn't valid JSON
        # This can happen if the chunk had no coaching content and Claude added explanation text
        print(f"  WARNING: JSON parse failed for {chunk['source']} chunk {chunk['chunk_index']}")
        print(f"  Raw response: {raw_response[:200]}")
        return []

    except Exception as e:
        # Any other API error (rate limit, network issue, etc.)
        print(f"  ERROR on {chunk['source']} chunk {chunk['chunk_index']}: {e}")
        return []


# ============================================================
# BATCH EXTRACTION FUNCTIONS
# ============================================================

def extract_batch(chunks: list, output_json_path: str) -> list:
    """
    Runs extraction on a list of chunks and saves results to a JSON file.
    Prints progress every 10 chunks so you can monitor a long run.

    Args:
        chunks: List of chunk dicts from caption_parser.py
        output_json_path: Where to save the extracted facts JSON

    Returns:
        Flat list of all extracted coaching facts across all chunks
    """

    all_facts = []
    seen_keys = set()
    repeats = 0
    total = len(chunks)

    print(f"Starting extraction on {total} chunks...")

    for i, chunk in enumerate(chunks):

        # Print progress every 10 chunks
        if i % 10 == 0:
            print(f"  Processing chunk {i + 1} / {total} — {len(all_facts)} facts extracted so far")

        # Extract coaching facts from this chunk, skipping a tip already
        # recorded word for word (chunks overlap by 50 words, so the same
        # sentence can be extracted twice)
        for fact in extract_from_chunk(chunk):
            key = _fact_key(fact)
            if key in seen_keys:
                repeats += 1
                continue
            seen_keys.add(key)
            all_facts.append(fact)

        # Small delay between calls to avoid rate limits
        time.sleep(API_DELAY_SECONDS)

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save all extracted facts to JSON
    with open(output_json_path, "w") as f:
        json.dump(all_facts, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"  Chunks processed: {total}")
    print(f"  Facts extracted:  {len(all_facts)} ({repeats} word-for-word repeats dropped)")
    print(f"  Saved → {output_json_path}")

    return all_facts


def run_test_extraction(transcripts_dir: str = "data/transcripts"):
    """
    Runs extraction on just the first TEST_BATCH_SIZE chunks.
    Always run this before the full batch to verify quality and estimate cost.
    """

    print(f"=== TEST MODE: processing first {TEST_BATCH_SIZE} chunks only ===\n")

    # Load all chunks from the VTT files
    all_chunks = parse_all_vtt_files(transcripts_dir)

    if not all_chunks:
        print("No chunks found — check your transcripts directory path.")
        return []

    # Take just the first few chunks for testing
    test_chunks = all_chunks[:TEST_BATCH_SIZE]

    # Run extraction on the small test batch
    facts = extract_batch(
        chunks=test_chunks,
        output_json_path=TEST_OUTPUT_PATH
    )

    # Print the extracted facts so you can evaluate quality
    print(f"\nExtracted facts from test batch:")
    for fact in facts:
        print(f"\n  Technique: {fact.get('technique', '?')}")
        print(f"  Tags: shots {fact.get('shot_types')} | side {fact.get('stroke_side')} | "
              f"{fact.get('discipline')} | body {fact.get('body_parts')} | level {fact.get('level')}")
        print(f"  Preparation: {fact.get('preparation', '?')}")
        print(f"  Execution: {fact.get('execution', '?')}")
        print(f"  Common mistake: {fact.get('common_mistake', '?')}")
        print(f"  Cue: {fact.get('cue', '?')}")
        print(f"  Context: {fact.get('context', '?')}")
        print(f"  Source: {fact.get('video_title', '?')}")

    return facts


def run_full_extraction(transcripts_dir: str = "data/transcripts"):
    """
    Runs extraction on ALL chunks. Only call this after verifying test results look good.
    This will make ~416 API calls — takes several minutes and costs well under a dollar.
    """

    print("=== FULL EXTRACTION MODE ===\n")
    print("This will process all chunks and may take several minutes.")
    print("Make sure you've run run_test_extraction() first and are happy with the quality.\n")

    # Load all chunks
    all_chunks = parse_all_vtt_files(transcripts_dir)

    if not all_chunks:
        print("No chunks found — check your transcripts directory path.")
        return []

    # Run extraction on everything
    facts = extract_batch(
        chunks=all_chunks,
        output_json_path=FULL_OUTPUT_PATH
    )

    return facts


def redo_videos(title_text: str, transcripts_dir: str = "data/transcripts",
                facts_json_path: str = FULL_OUTPUT_PATH) -> list:
    """
    Re-extracts only the videos whose title contains title_text and swaps
    their facts into the saved facts file, so one failed chunk doesn't need
    the whole run again.
    """

    wanted = [c for c in parse_all_vtt_files(transcripts_dir) if title_text.lower() in c["video_title"].lower()]
    if not wanted:
        print(f"No video title contains {title_text!r}")
        return []
    titles = sorted({c["video_title"] for c in wanted})
    print(f"Re-extracting {len(wanted)} chunk(s) from: {titles}")

    with open(facts_json_path) as f:
        facts = json.load(f)
    kept = [fact for fact in facts if fact.get("video_title") not in titles]
    redone = extract_batch(wanted, output_json_path=TEST_OUTPUT_PATH)

    with open(facts_json_path, "w") as f:
        json.dump(kept + redone, f, indent=2)
    print(f"Replaced {len(facts) - len(kept)} fact(s) with {len(redone)} → {facts_json_path} now holds {len(kept) + len(redone)} facts")
    return redone


# ============================================================
# This block only runs when you execute this file directly.
#   python -m src.knowledge.knowledge_extractor --test             first few chunks → outputs/coaching_facts_test.json
#   python -m src.knowledge.knowledge_extractor --full             every chunk      → outputs/coaching_facts_v2.json
#   python -m src.knowledge.knowledge_extractor --video "Serve"    redo the videos whose title contains "Serve"
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract tagged coaching facts from the tutorial transcripts.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true", help=f"process the first {TEST_BATCH_SIZE} chunks only")
    mode.add_argument("--full", action="store_true", help="process every chunk (run --test first)")
    mode.add_argument("--video", metavar="TITLE_TEXT",
                      help="re-extract only the videos whose title contains this text, and update the saved facts")
    args = parser.parse_args()

    if args.test:
        run_test_extraction()
    elif args.full:
        run_full_extraction()
    else:
        redo_videos(args.video)