# glob lets us find all files matching a pattern (e.g. all .vtt files in a folder)
from glob import glob
import re
import os


# ============================================================
# TUNING PARAMETERS
# ============================================================

# Target size for each text chunk in words
# 500 words ≈ a good amount of context for Claude to extract coaching facts from
CHUNK_SIZE_WORDS = 500

# How many words to repeat at the start of each new chunk
# Overlap prevents coaching cues from being cut in half at chunk boundaries
CHUNK_OVERLAP_WORDS = 50


def parse_vtt(vtt_path: str) -> str:
    """
    Reads a single .vtt caption file and returns clean plain text.

    Handles:
    - WEBVTT header and metadata lines
    - Timestamp lines (00:00:00.000 --> 00:00:03.440)
    - &nbsp; non-breaking space characters
    - Line breaks mid-sentence
    - Extra whitespace
    """

    with open(vtt_path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split the file into individual lines
    lines = raw.splitlines()

    clean_lines = []

    for line in lines:

        # Skip the WEBVTT header and metadata lines
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue

        # Skip timestamp lines — they look like "00:00:01.234 --> 00:00:04.567"
        # The regex matches the HH:MM:SS.mmm --> HH:MM:SS.mmm pattern
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d+ --> \d{2}:\d{2}:\d{2}\.\d+", line):
            continue

        # Replace &nbsp; (HTML non-breaking space) with a regular space
        line = line.replace("&nbsp;", " ")

        # Strip leading/trailing whitespace from each line
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        clean_lines.append(line)

    # Join all lines into one continuous block of text
    # Lines that were split mid-sentence get rejoined here
    full_text = " ".join(clean_lines)

    # Collapse any runs of multiple spaces into a single space
    full_text = re.sub(r" +", " ", full_text)

    return full_text.strip()


def chunk_text(text: str, source_file: str) -> list:
    """
    Splits a long text string into overlapping chunks of ~CHUNK_SIZE_WORDS words.

    Each chunk is returned as a dict with:
    - "text": the chunk content
    - "source": the filename this came from (useful for debugging retrieval later)
    - "chunk_index": which chunk this is within the source file

    Overlap is included so that coaching cues that fall near a chunk boundary
    appear in full in at least one chunk.
    """

    # Split text into individual words
    words = text.split()

    chunks = []
    chunk_index = 0

    # Step through the word list in increments of (CHUNK_SIZE - OVERLAP)
    # so each chunk starts CHUNK_OVERLAP words before the previous one ended
    step = CHUNK_SIZE_WORDS - CHUNK_OVERLAP_WORDS

    start = 0

    while start < len(words):

        # Take a slice of CHUNK_SIZE_WORDS words starting at `start`
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]

        # Rejoin the words back into a string
        chunk_text_str = " ".join(chunk_words)

        chunks.append({
            "text": chunk_text_str,
            "source": os.path.basename(source_file),
            "chunk_index": chunk_index
        })

        chunk_index += 1

        # If we've reached the end of the text, stop
        if end >= len(words):
            break

        # Move forward by `step` words for the next chunk
        start += step

    return chunks


def parse_all_vtt_files(transcripts_dir: str) -> list:
    """
    Finds all .vtt files recursively under transcripts_dir,
    parses each one, and returns a flat list of all text chunks
    across all files.

    Args:
        transcripts_dir: Root folder to search — e.g. "data/transcripts"
                         Will find files in subfolders like Beginner_Tutorials/
                         and Intermediate_and_Advanced_Tutorials/ automatically.

    Returns:
        List of chunk dicts: [{"text": ..., "source": ..., "chunk_index": ...}, ...]
    """

    # Find all .vtt files anywhere under transcripts_dir
    # ** means "any subfolder", recursive=True enables it
    pattern = os.path.join(transcripts_dir, "**", "*.vtt")
    vtt_files = glob(pattern, recursive=True)

    if not vtt_files:
        print(f"No .vtt files found under: {transcripts_dir}")
        return []

    print(f"Found {len(vtt_files)} .vtt files")

    all_chunks = []
    skipped = 0

    for vtt_path in vtt_files:

        # Parse the file into clean text
        text = parse_vtt(vtt_path)

        # Skip files that ended up empty after cleaning
        if not text:
            print(f"  WARNING: no text extracted from {vtt_path} — skipping")
            skipped += 1
            continue

        # Split into overlapping chunks
        chunks = chunk_text(text, source_file=vtt_path)

        all_chunks.extend(chunks)

    print(f"Parsed {len(vtt_files) - skipped} files → {len(all_chunks)} chunks total")
    if skipped:
        print(f"Skipped {skipped} empty files")

    return all_chunks


# ============================================================
# This block only runs when you execute this file directly.
# It won't run if this file is imported by another module.
# ============================================================

if __name__ == "__main__":

    # Parse all .vtt files from both subfolders
    chunks = parse_all_vtt_files("data/transcripts")

    # Print a summary
    print(f"\nTotal chunks ready for ChromaDB: {len(chunks)}")

    # Print the first 3 chunks so you can verify the output looks right
    print("\nFirst 3 chunks:")
    for chunk in chunks[:3]:
        print(f"\n--- {chunk['source']} (chunk {chunk['chunk_index']}) ---")
        print(chunk["text"][:300] + "...")  # Print first 300 chars of each chunk
