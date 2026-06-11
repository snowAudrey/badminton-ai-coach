# chromadb is the vector database — it stores text with embeddings and
# lets us search by semantic similarity instead of exact keywords
import chromadb
import json
import os


# ============================================================
# SETTINGS
# ============================================================

# Where ChromaDB saves its database files on disk
# PersistentClient means the data survives between runs —
# you build the database once and query it forever after
CHROMA_DB_PATH = "data/chromadb"

# Name of the collection (like a table in a regular database)
COLLECTION_NAME = "badminton_coaching"

# Path to the extracted coaching facts from knowledge_extractor.py
FACTS_JSON_PATH = "outputs/coaching_facts_full.json"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_useful(fact: dict) -> bool:
    """
    Returns True if the fact has at least one meaningful content field.

    A fact with only "technique" and "context" but nothing actionable
    (no preparation, execution, mistake, or cue) is useless to the agent,
    so we skip it when building the database.
    """

    return any([
        fact.get("preparation"),
        fact.get("execution"),
        fact.get("common_mistake"),
        fact.get("cue")
    ])


def fact_to_embedding_text(fact: dict) -> str:
    """
    Builds the text string that gets embedded for semantic search.

    We concatenate ALL non-empty fields so that a query can match a fact
    through any of its aspects:
      - by technique name      → matches "technique" field      (e.g. "drop shot execution")
      - by observed problem    → matches "common_mistake" field  (e.g. "racket face open at contact")
      - by situation           → matches "context" field         (e.g. "late to the shuttle")

    Empty fields are skipped so they don't add noise to the embedding.

    The structured fields are stored separately as metadata, so this
    concatenation is ONLY used for matching — the agent still gets
    the individual fields back after retrieval.
    """

    parts = []

    # Each field is added as "label: content" — the label gives the
    # embedding model context about what role the text plays
    if fact.get("technique"):
        parts.append(f"technique: {fact['technique']}")

    if fact.get("preparation"):
        parts.append(f"preparation: {fact['preparation']}")

    if fact.get("execution"):
        parts.append(f"execution: {fact['execution']}")

    if fact.get("common_mistake"):
        parts.append(f"common mistake: {fact['common_mistake']}")

    if fact.get("cue"):
        parts.append(f"cue: {fact['cue']}")

    if fact.get("context"):
        parts.append(f"context: {fact['context']}")

    # Join everything into one string separated by ". "
    return ". ".join(parts)


# ============================================================
# BUILD THE KNOWLEDGE BASE (run once)
# ============================================================

def build_knowledge_base(facts_json_path: str = FACTS_JSON_PATH):
    """
    Loads the extracted coaching facts and stores them in ChromaDB.

    Run this ONCE after extraction is complete. ChromaDB persists to disk,
    so there is no need to rebuild unless the facts file changes.

    Steps:
      1. Load facts from the JSON file
      2. Filter out facts with no meaningful content
      3. Build the embedding text for each fact
      4. Store everything in ChromaDB (text + metadata)
    """

    # ----- Step 1: Load the facts -----

    if not os.path.exists(facts_json_path):
        raise FileNotFoundError(
            f"Facts file not found: {facts_json_path}\n"
            "Run knowledge_extractor.py first to generate it."
        )

    with open(facts_json_path, "r") as f:
        facts = json.load(f)

    print(f"Loaded {len(facts)} facts from {facts_json_path}")

    # ----- Step 2: Filter out empty facts -----

    useful_facts = [fact for fact in facts if is_useful(fact)]
    skipped = len(facts) - len(useful_facts)

    print(f"Keeping {len(useful_facts)} useful facts ({skipped} skipped as empty)")

    # ----- Step 3: Connect to ChromaDB -----

    # PersistentClient saves the database to disk at CHROMA_DB_PATH
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete the collection if it already exists — this makes the build
    # repeatable: re-running always gives a fresh, consistent database
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for a fresh rebuild")

    # Create the collection — ChromaDB uses its default embedding model
    # (all-MiniLM-L6-v2) automatically, no API key or setup needed
    collection = client.create_collection(name=COLLECTION_NAME)

    # ----- Step 4: Prepare the data for insertion -----

    # ChromaDB takes three parallel lists:
    #   ids        — a unique string ID per record
    #   documents  — the text that gets embedded for search
    #   metadatas  — the structured fields, returned alongside search results

    ids = []
    documents = []
    metadatas = []

    for i, fact in enumerate(useful_facts):

        # Unique ID per fact — just the index as a string
        ids.append(f"fact_{i}")

        # The concatenated text used for semantic matching
        documents.append(fact_to_embedding_text(fact))

        # The structured fields, preserved individually
        # ChromaDB metadata values must be strings/numbers/bools — no None
        metadatas.append({
            "technique": fact.get("technique", ""),
            "preparation": fact.get("preparation", ""),
            "execution": fact.get("execution", ""),
            "common_mistake": fact.get("common_mistake", ""),
            "cue": fact.get("cue", ""),
            "context": fact.get("context", ""),
            "source": fact.get("source", ""),
            "chunk_index": fact.get("chunk_index", -1)
        })

    # ----- Step 5: Insert in batches -----

    # ChromaDB handles large inserts better in batches — 500 at a time
    # also gives us progress prints for us to know it's working
    BATCH_SIZE = 500

    for start in range(0, len(ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(ids))

        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end]
        )

        print(f"  Inserted {end} / {len(ids)} facts")

    print(f"\nKnowledge base built!")
    print(f"  Collection: '{COLLECTION_NAME}'")
    print(f"  Total facts stored: {collection.count()}")
    print(f"  Saved to: {CHROMA_DB_PATH}")

    return collection


# ============================================================
# QUERY THE KNOWLEDGE BASE (used by the Knowledge Agent)
# ============================================================

def query_knowledge_base(question: str, n_results: int = 6) -> list:
    """
    Searches the knowledge base for facts relevant to a question.

    This is the function the Knowledge Agent will call.

    Args:
        question:  Natural language query, e.g. "correct elbow position
                   for backhand clear" or "why does my serve go too high"
        n_results: How many facts to return (default 5)

    Returns:
        List of fact dicts, most relevant first. Each dict has the
        structured fields plus a "relevance_distance" score
        (lower = more relevant).
    """

    # Connect to the existing database on disk
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Get the collection — raises an error if build_knowledge_base()
    # hasn't been run yet
    collection = client.get_collection(name=COLLECTION_NAME)

    # Run the semantic search
    # ChromaDB embeds the question and finds the closest stored facts
    results = collection.query(
        query_texts=[question],
        n_results=n_results
    )

    # ChromaDB returns parallel lists nested one level deep
    # (because we can query multiple questions at once — we only sent one)
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Repackage into a clean list of fact dicts
    facts = []
    for metadata, distance in zip(metadatas, distances):
        fact = dict(metadata)
        fact["relevance_distance"] = round(distance, 4)
        facts.append(fact)

    return facts


# ============================================================
# This block only runs when you execute this file directly.
# ============================================================

if __name__ == "__main__":

    # STEP 1: Build the database from the extracted facts
    # Only needs to run once — comment it out after the first successful build
    build_knowledge_base()

    # STEP 2: Test retrieval with a few queries
    # Check that the returned facts are actually relevant to each question
    test_queries = [
        "correct grip for backhand serve",
        "why does my serve go too high over the net",
        "footwork to the forehand front corner",
        "elbow position on clear shots"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"QUERY: {query}")
        print('=' * 60)

        facts = query_knowledge_base(query, n_results=3)

        for i, fact in enumerate(facts):
            print(f"\n  Result {i + 1} (distance: {fact['relevance_distance']})")
            print(f"  Technique: {fact['technique']}")
            if fact["preparation"]:
                print(f"  Preparation: {fact['preparation']}")
            if fact["execution"]:
                print(f"  Execution: {fact['execution']}")
            if fact["common_mistake"]:
                print(f"  Common mistake: {fact['common_mistake']}")
            if fact["cue"]:
                print(f"  Cue: {fact['cue']}")
            print(f"  Source: {fact['source']}")