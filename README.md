# Badminton AI Coach

Turns short (under 1 minute) raw amateur badminton video into timestamped, technique-specific coaching feedback.

**Status: functional end to end — actively refining.**

## Why

For amateur sports enthusiasts, access to a personal coach is often out of reach because it can be too expensive or simply unavailable. As a badminton player myself, I wanted feedback on my technique whenever I played, not just when I could find a coach. This project is meant as a training aid that supplements coaching rather than replaces it. The feedback it produces is a starting point for reflection instead of a substitute for a coach's judgment. It's also deliberately built for short clips: reviewing one or two rallies at a time keeps the feedback focused and actionable, so you can improve a little each session rather than being overwhelmed by everything at once.

## How it works

The project has two pipelines: a computer vision pipeline that turns video into structured per-frame data, and a small pipeline of Claude-powered agents that turns that data into coaching feedback.

**1. Computer vision pipeline** (`src/vision/`)
- `pose_estimator.py` — YOLOv8 detects players in the first frame; you click to select which player to track. MediaPipe Pose then extracts 33 body keypoints per frame, re-detecting and re-cropping to the selected player every frame so tracking survives camera motion and multiple players in frame.
- `shuttle_tracker.py` — a custom-trained YOLOv8 model (fine-tuned on 8,000+ labeled frames) detects the shuttlecock per frame, then groups frames into rally segments based on shuttle movement.
- `pipeline.py` — merges pose, shuttle, and rally data into one frame-indexed JSON (`unified_data.json`).

**2. Coaching agent pipeline** (`src/agents/`)
- `vision_agent.py` — converts raw keypoints into human-readable pose facts per sampled frame (elbow/knee angle, wrist height relative to head and shoulder, shuttle presence) — descriptive only, no judgment about whether an angle is "good."
- `knowledge_agent.py` — sends a rally's pose observations to Claude, which spots notable angle/height-change patterns and translates them into badminton-specific search queries, then retrieves matching coaching facts from a ChromaDB knowledge base.
- `orchestrator.py` — sends the pose observations and retrieved coaching facts to Claude again, which synthesizes grounded, timestamped feedback — naming multiple plausible shot types when the retrieved facts don't clearly point to one, rather than guessing confidently.

**3. Offline knowledge base build** (`src/knowledge/`, run once — not part of the per-video pipeline)
- `caption_parser.py` + `knowledge_extractor.py` — parses caption transcripts from 165 badminton tutorial videos and runs them through a Claude extraction pipeline to pull out ~2,900 structured coaching facts (technique, preparation, execution, common mistake, cue, context).
- `vector_store.py` — embeds and stores those facts in ChromaDB for semantic retrieval.

## Tech stack

Python · OpenCV · YOLOv8 · MediaPipe · NumPy · Google Colab · Claude API · ChromaDB

## Setup

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Add your Anthropic API key to a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```
   (Your `.env` file is protected by `.gitignore` and will never be pushed to GitHub.)
3. Place model files in `models/` (not included in this repo — see `.gitignore`):
   - `best.pt` — custom-trained shuttlecock detector
   - `pose_landmarker_lite.task` — MediaPipe pose landmarker model

   (`yolov8n.pt`, used for player detection, downloads automatically on first run.)
4. Build the coaching knowledge base once (only needed the first time, or after regenerating `coaching_facts_full.json`):
   ```
   python -m src.knowledge.vector_store
   ```

## Usage

```
python main.py --video path/to/your/match.mp4 --racket-arm right
```

This runs the full pipeline — player selection, pose/shuttle/rally extraction, and coaching feedback synthesis — and prints timestamped feedback per rally.

Flags:
- `--video` (required) — path to the match video
- `--racket-arm` — `right` or `left` (default: `right`)
- `--output-dir` — where intermediate files are saved (default: `outputs`)

## Example output

```
FEEDBACK FOR RALLY 1
============================================================
Around t=2.9-3.3s: This could be a clear, smash, or drop shot. If a clear:
you're extending your elbow nicely, but focus on accelerating through with
your elbow coming forwards and bringing your wrist through after. If a smash:
your arm is nearly straight at contact, which is good, but make sure you're
throwing your elbow forwards and letting your whole forearm and wrist come
through -- avoid only rotating the bottom half of your forearm, which
reduces power.
```

## Current status and known limitations

- Tested on clips of ~15-34 seconds so far — longer videos, even within the intended under-1-minute range, are untested
- Pose detection: 100% of frames on self-recorded test footage
- Shuttle detection: ~19% of frames on amateur footage — lower than the 0.608 mAP50 the underlying model achieves on its own validation set, since real-world amateur footage (lighting, background clutter, camera angle) is harder than the training distribution. Improving this is planned as its own follow-up project.
- Shot-type identification is inherently ambiguous from pose data alone, with no shuttle trajectory or court-position context yet, so feedback deliberately names multiple plausible shot types rather than guessing confidently.
- Rally segmentation relies on a fixed stillness threshold (`RALLY_END_FRAMES`, currently 3 seconds) tuned to the original test clip's pacing — longer natural pauses between points in other footage can cause rallies to be split or merged incorrectly, so this threshold may need retuning per video.
- Shuttle detection depends on the camera's shooting angle — on footage where overhead space isn't well captured, shots hit high (e.g. clears, lifts) can go undetected entirely, independent of the model's accuracy.
- Search query generation is non-deterministic — a plausible shot type (e.g. drop shot alongside clear and smash for an overhead pattern) can occasionally be left out of a given run's hedge between multiple shot types, even though the underlying prompt asks for full coverage.

## Roadmap

- Test on additional match footage to check generalization beyond the current test clip
- Improve shuttlecock detection on amateur footage
- Lightweight visualization pairing the pose overlay video with timestamped feedback

## Notes

This project was built with guidance from Claude AI.
