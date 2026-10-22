"""
The fixed-choice tags every coaching fact carries. Kept in one place so the
extractor (which asks Claude for them), the vector store (which stores and
filters by them) and the agents (which ask for them) all agree on the
spellings.
"""

# Shot types: the names the shot classifier produces, plus "net kill" (a
# shot the tutorials teach that the classifier doesn't predict), "footwork"
# and "general" (tactics, mindset, grip, anything else)
SHOT_TYPES = ["smash", "clear", "drop", "drive", "net shot", "lift", "block/defence",
              "serve", "return of serve", "net kill", "footwork", "general"]
STROKE_SIDES = ["forehand", "backhand", "any"]
DISCIPLINES = ["singles", "doubles", "both"]
BODY_PARTS = ["grip", "wrist", "elbow", "shoulder", "arm", "torso", "hips", "legs", "feet", "head"]


def shot_flag(shot_type: str) -> str:
    """
    The metadata field that says whether a fact is about a shot type:
    "net shot" -> "shot_net_shot", "block/defence" -> "shot_block_defence".
    One yes/no field per shot type, because ChromaDB can filter on a single
    value but not on "list contains".
    """
    return "shot_" + shot_type.replace("/", "_").replace(" ", "_")
