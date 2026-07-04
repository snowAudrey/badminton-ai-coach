import json
import time
import anthropic

# load_dotenv reads the .env file and makes ANTHROPIC_API_KEY available to os.getenv
from dotenv import load_dotenv

# query_knowledge_base() searches the ChromaDB knowledge base built in vector_store.py
from src.knowledge.vector_store import query_knowledge_base

# analyze_video() turns CV pipeline output into pose observations — only needed
# for the __main__ test block below
from src.agents.vision_agent import analyze_video


# ============================================================
# SETUP
# ============================================================

# Load the ANTHROPIC_API_KEY from your .env file
load_dotenv()

# Create the Anthropic client — it automatically reads ANTHROPIC_API_KEY from environment
client = anthropic.Anthropic()

# Model to use for translating observations into search queries
MODEL = "claude-haiku-4-5"

# How many coaching facts to retrieve per generated search query
N_RESULTS_PER_QUERY = 6

# Small delay between API calls in seconds — avoids hitting rate limits
API_DELAY_SECONDS = 0.5


# ============================================================
# TRANSLATION PROMPT
# ============================================================
# This prompt is step 1 of the bridge: it asks the LLM to look at a
# sequence of pose observations from one rally and turn any notable
# patterns into search queries written in coaching language.
#
# The LLM is NOT looking at the coaching knowledge base here — it's
# using its own general badminton knowledge to guess what techniques
# might be involved. The actual coaching facts are retrieved afterwards
# in retrieve_knowledge_for_rally().

TRANSLATION_PROMPT = """You are reviewing pose data sampled from one rally of a badminton match.

Below is a sequence of frame-by-frame observations, ordered by time. Each line
describes joint angles, wrist height relative to the head and shoulder, and
shuttle detection for one sampled frame.

Note on angle convention: a smaller elbow or knee angle means the joint is
more bent (flexed); an angle closer to 180° means the joint is straighter
(extended).

Wrist height is a rough indicator of shot type: overhead usually means clear,
smash, or drop shot; above shoulder usually means drive or net shot; below
shoulder usually means lift or a defensive shot. When an observation suggests
one of these height categories, generate queries covering all of that
category's shot types (e.g. all three of clear, smash, and drop for an
overhead pattern) rather than picking just one or two — they can look nearly
identical in joint angles alone, and the synthesis step can only hedge
between shots that actually have retrieved facts to draw on.

Look for notable patterns across these frames — specifically, points in time
where a joint angle changes substantially over a short window, or the wrist
height transitions between overhead, above shoulder, and below shoulder —
either of which likely marks a swing or shot being played. Describe what
changed and roughly when, without judging whether the angle or wrist height
itself is correct or typical — you don't know what shot was being played or
what phase of the shot this is.

Based on these patterns, generate 3 to 8 search queries that could be used to
look up relevant badminton coaching advice. Each query must name a specific
badminton shot or technique (e.g. smash, clear, drop shot, net shot, lift,
drive, serve) together with the body part or movement involved — not a vague
or abstract description. For example, write "elbow extension during overhead
clear", "knee bend when preparing for a smash", or "standing position when
preparing to return a smash" — not "dynamic knee adjustment during rally
exchanges" or "elbow flexion recovery" with no named shot.

Observations:
{observations_text}

Return ONLY a valid JSON array of query strings. No explanation, no markdown.

Example output:
["elbow extension during overhead clear", "knee bend preparation for smash", "standing position when preparing to return a smash", "wrist height during net shot"]
"""


# ============================================================
# STEP 1: GROUP OBSERVATIONS BY RALLY
# ============================================================

def group_observations_by_rally(observations: list) -> dict:
    """
    Groups a flat list of observation dicts (from vision_agent.analyze_video)
    into a dict keyed by rally_id.

    This lets us send one batch of observations per rally to the LLM,
    instead of one frame at a time (which would have no context) or
    the whole video at once (which would mix unrelated rallies together).

    Args:
        observations: List of observation dicts, each with a "rally_id" key

    Returns:
        Dict mapping rally_id -> list of observation dicts, e.g.
        {1: [obs1, obs2, ...], 2: [obs1, obs2, ...]}
    """

    rallies = {}

    for obs in observations:
        rally_id = obs.get("rally_id")

        if rally_id not in rallies:
            rallies[rally_id] = []

        rallies[rally_id].append(obs)

    return rallies


# ============================================================
# STEP 2: FORMAT OBSERVATIONS FOR THE PROMPT
# ============================================================

def format_observations_for_prompt(observations: list) -> str:
    """
    Converts a list of observation dicts into a compact, line-per-frame
    text block that gets dropped into TRANSLATION_PROMPT.

    Each line looks like:
        t=4.3s: elbow angle is 110.8°, knee angle is 138.1°, shuttle detected this frame

    Args:
        observations: List of observation dicts for ONE rally

    Returns:
        Multi-line string, one line per observation
    """

    lines = []

    for obs in observations:
        timestamp = obs.get("timestamp", 0)
        notes = ", ".join(obs.get("observations", []))
        lines.append(f"t={timestamp}s: {notes}")

    return "\n".join(lines)


# ============================================================
# STEP 3: TRANSLATE OBSERVATIONS INTO SEARCH QUERIES (LLM CALL 1)
# ============================================================

def translate_observations_to_queries(observations: list) -> list:
    """
    Sends one rally's worth of observations to the LLM and asks it to
    generate search queries based on any notable patterns it sees.

    Args:
        observations: List of observation dicts for ONE rally

    Returns:
        List of search query strings (in coaching language). Empty list
        if the LLM call fails or returns something that isn't valid JSON.
    """

    # Build the text block describing this rally's observations
    observations_text = format_observations_for_prompt(observations)

    # Fill in the prompt template
    # (using .replace() instead of .format() — same reason as in
    # knowledge_extractor.py: the example output contains literal
    # JSON brackets that .format() would misinterpret)
    prompt = TRANSLATION_PROMPT.replace("{observations_text}", observations_text)

    try:
        # Call the Claude API
        response = client.messages.create(
            model=MODEL,
            max_tokens=500,
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

        # Parse the JSON response — should be a list of strings
        queries = json.loads(raw_response)

        return queries

    except json.JSONDecodeError:
        print(f"  WARNING: JSON parse failed for translation response")
        print(f"  Raw response: {raw_response[:200]}")
        return []

    except Exception as e:
        print(f"  ERROR during translation: {e}")
        return []


# ============================================================
# STEP 4: RETRIEVE COACHING FACTS FOR ONE RALLY
# ============================================================

def retrieve_knowledge_for_rally(observations: list) -> dict:
    """
    Full pipeline for one rally:
      1. Translate the rally's observations into search queries (LLM call)
      2. Run each query against the ChromaDB knowledge base
      3. Return a dict mapping each query to its retrieved coaching facts

    Args:
        observations: List of observation dicts for ONE rally

    Returns:
        Dict like:
        {
            "elbow extension during overhead clear": [fact1, fact2, ...],
            "knee bend preparation for smash": [fact1, fact2, ...]
        }
    """

    # Step 1: generate search queries from the pose patterns in this rally
    queries = translate_observations_to_queries(observations)

    # Step 2: retrieve coaching facts for each query
    results = {}

    for query in queries:
        facts = query_knowledge_base(query, n_results=N_RESULTS_PER_QUERY)
        results[query] = facts

        # Small delay between API-backed calls to avoid rate limits
        time.sleep(API_DELAY_SECONDS)

    return results


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":

    # Path to your unified data — same as vision_agent.py
    UNIFIED_DATA_PATH = "outputs/unified_data.json"

    # Change to "left" if the player is left-handed
    RACKET_ARM = "right"

    # Step 1: get pose observations from the Vision Agent
    observations = analyze_video(UNIFIED_DATA_PATH, racket_arm=RACKET_ARM)

    # Step 2: group observations by rally
    rallies = group_observations_by_rally(observations)
    print(f"\nFound {len(rallies)} rally(s)")

    # Step 3: for each rally, translate observations into queries and
    # retrieve matching coaching facts
    for rally_id, rally_observations in rallies.items():
        print(f"\n{'=' * 60}")
        print(f"RALLY {rally_id} — {len(rally_observations)} sampled frames")
        print('=' * 60)

        results = retrieve_knowledge_for_rally(rally_observations)

        for query, facts in results.items():
            print(f"\nQUERY: {query}")

            for i, fact in enumerate(facts):
                print(f"\n  Result {i + 1} (distance: {fact['relevance_distance']})")
                print(f"  Technique: {fact['technique']}")
                if fact["cue"]:
                    print(f"  Cue: {fact['cue']}")
                if fact["common_mistake"]:
                    print(f"  Common mistake: {fact['common_mistake']}")
