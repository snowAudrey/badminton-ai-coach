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


# ============================================================
# SETUP
# ============================================================

# Load the ANTHROPIC_API_KEY from your .env file
load_dotenv()

# Create the Anthropic client — it automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Model to use for extraction
# claude-haiku-3-5 is the fastest and cheapest model — ideal for batch processing
MODEL = "claude-haiku-4-5"

# How many chunks to process in test mode
# Always test on a small batch before running all 618
TEST_BATCH_SIZE = 5

# Small delay between API calls in seconds
# Prevents hitting rate limits on large batches
API_DELAY_SECONDS = 0.5


# ============================================================
# EXTRACTION PROMPT
# ============================================================
# This is the most important part — it tells Claude exactly what to extract
# and how to format it. 

EXTRACTION_PROMPT = """You are extracting badminton coaching knowledge from a video transcript.

Given this transcript chunk from a badminton tutorial video, extract every concrete coaching fact you can find.
Extract one JSON object per distinct coaching point — do not bundle multiple aspects of a technique into one object.

For each coaching fact, output a JSON object with these fields:
- "technique": the stroke or skill being discussed (e.g. "drop shot", "clear", "corner footwork") 
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

Return ONLY a valid JSON array. No explanation, no markdown, no preamble.

Example output format:
[
  {
    "technique": "drop shot",
    "preparation": "Turn your whole body so you're facing sideways, feet wider than shoulder-width, most of your weight on the back leg.",
    "execution": "",
    "common_mistake": "",
    "cue": "Turn your whole body so you're facing sideways before the shuttle arrives.",
    "context": "Forehand side, setting up body position before playing the drop shot."
  },
  {
    "technique": "drop shot",
    "preparation": "Use a forehand grip with racket arm shoulder and elbow at 90 degrees.",
    "execution": "",
    "common_mistake": "Racket arm too straight or too bent reduces power and control.",
    "cue": "Have your racket arm shoulder and elbow at 90 degrees before swinging.",
    "context": "Racket arm setup for drop shot."
  },
  {
    "technique": "drop shot",
    "preparation": "",
    "execution": "Raise non-racket arm up almost pointing to the shuttle. Strike the shuttle just in front of your body and rotate hips through contact.",
    "common_mistake": "Not rotating hips and using only the arm creates awkward movement and loss of control.",
    "cue": "Have your non-racket arm up pointing to the shuttle and rotate your hips through the shot.",
    "context": "Execution phase of the drop shot."
  }
]

Transcript chunk:
{chunk_text}"""


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
            max_tokens=2000,
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

        # Add source metadata to each fact so we know which video it came from
        for fact in facts:
            fact["source"] = chunk["source"]
            fact["chunk_index"] = chunk["chunk_index"]

        return facts

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
    total = len(chunks)

    print(f"Starting extraction on {total} chunks...")

    for i, chunk in enumerate(chunks):

        # Print progress every 10 chunks
        if i % 10 == 0:
            print(f"  Processing chunk {i + 1} / {total} — {len(all_facts)} facts extracted so far")

        # Extract coaching facts from this chunk
        facts = extract_from_chunk(chunk)
        all_facts.extend(facts)

        # Small delay between calls to avoid rate limits
        time.sleep(API_DELAY_SECONDS)

    # Create output folder if it doesn't exist
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Save all extracted facts to JSON
    with open(output_json_path, "w") as f:
        json.dump(all_facts, f, indent=2)

    print(f"\nExtraction complete!")
    print(f"  Chunks processed: {total}")
    print(f"  Facts extracted:  {len(all_facts)}")
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
        output_json_path="outputs/coaching_facts_test.json"
    )

    # Print the extracted facts so you can evaluate quality
    print(f"\nExtracted facts from test batch:")
    for fact in facts:
        print(f"\n  Technique: {fact.get('technique', '?')}")
        print(f"  Preparation: {fact.get('preparation', '?')}")
        print(f"  Execution: {fact.get('execution', '?')}")
        print(f"  Common mistake: {fact.get('common_mistake', '?')}")
        print(f"  Cue: {fact.get('cue', '?')}")
        print(f"  Context: {fact.get('context', '?')}")
        print(f"  Source: {fact.get('source', '?')}")

    return facts


def run_full_extraction(transcripts_dir: str = "data/transcripts"):
    """
    Runs extraction on ALL chunks. Only call this after verifying test results look good.
    This will make ~618 API calls — takes several minutes and costs ~$1.50-2.00.
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
        output_json_path="outputs/coaching_facts_full.json"
    )

    return facts


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":

    # STEP 1: Always run the test first
    # Check outputs/coaching_facts_test.json and the printed facts
    # before proceeding to full extraction
    # test_facts = run_test_extraction()

    # STEP 2: Only uncomment this after you're happy with the test results
    full_facts = run_full_extraction()