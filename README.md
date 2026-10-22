# Badminton AI Coach

Turns short (under 1 minute) raw amateur badminton video into timestamped, technique-specific coaching feedback.

## Why

For amateur sports enthusiasts, access to a personal coach is often out of reach because it can be too expensive or simply unavailable. As a badminton player myself, I wanted feedback on my technique whenever I played, not just when I could find a coach. This project is meant as a training aid that supplements coaching rather than replaces it. The feedback it produces is a starting point for reflection instead of a substitute for a coach's judgment. It's also deliberately built for short clips: reviewing one or two rallies at a time keeps the feedback focused and actionable, so you can improve a little each session rather than being overwhelmed by everything at once.

## How it works

The project has two pipelines: a computer vision pipeline that turns video into structured per-frame data, and a small pipeline of Claude-powered agents that turns that data into coaching feedback.

**1. Computer vision pipeline** (`src/vision/`)
- `pose_estimator.py` — YOLOv8 detects players in the first frame; you click to select which player to track. Every frame after that, the tracker picks the detected person who could have got there since the last frame (no more than half a body height per frame, growing while the player is out of sight) and whose shirt colour matches the player you clicked, so it stays on your player past opponents, doubles partners and line judges. MediaPipe Pose then extracts 33 body keypoints from a crop around that person.
- `shuttle_tracker.py` — a custom-trained YOLOv8 model (fine-tuned on 8,000+ labeled frames) detects the shuttlecock per frame and filters out static background objects that look like a shuttle.
- `pipeline.py` — merges pose and shuttle data into one frame-indexed JSON (`unified_data.json`).
- `shot_events.py` — finds each shot from the racket wrist's speed (measured relative to the shoulders, so running doesn't count), and records facts about it: wrist height before and at the swing, elbow and knee angles, how high the shuttle climbed afterwards, how fast it was coming in, and the pause since the previous shot (`shot_events.json`). When the shuttle is seen next to the racket, the frame where it comes closest is taken as the moment of contact — on soft shots the wrist is fastest just before or after the hit, not at it.

**2. Coaching agent pipeline** (`src/agents/`)
- `shot_classifier.py` — names each shot (smash, clear, drop, drive, net shot, lift, block/defence, serve, return of serve) with rules on those facts: contact height decides overhead vs. underarm, an overhead hit the shuttle leaves fast is a smash, the shuttle's climb separates clear/lift from drop/net shot, a fast incoming shuttle returned with a short action is a block, a long pause with nothing incoming is a serve. Claude can be asked instead (`--use-claude`, with video frames), but on phone footage the shuttle is too small to see in the frames and it guessed, so the rules are the default.
- `vision_agent.py` — keeps MediaPipe's left/right labels consistent across frames (they can flip on footage shot from behind) and saves a check image of the arm being measured, then converts raw keypoints into human-readable pose facts per sampled frame (elbow/knee angle measured from MediaPipe's 3D landmarks, wrist height relative to head and shoulder, shuttle presence) — descriptive only, no judgment about whether an angle is "good."
- `knowledge_agent.py` — sends the clip's pose observations (in windows of up to 30 seconds) to Claude, which spots notable angle/height-change patterns and translates them into badminton-specific search queries, then retrieves matching coaching facts from a ChromaDB knowledge base.
- `orchestrator.py` — sends the pose observations and retrieved coaching facts to Claude again, which synthesizes grounded, timestamped feedback — naming multiple plausible shot types when the retrieved facts don't clearly point to one, rather than guessing confidently.

**3. Offline knowledge base build** (`src/knowledge/`, run once — not part of the per-video pipeline)
- `caption_parser.py` + `knowledge_extractor.py` — parses caption transcripts from 117 badminton tutorial videos (48 duplicate copies are skipped) and runs them through a Claude extraction pipeline to pull out ~2,100 structured coaching facts (technique, preparation, execution, common mistake, cue, context), each tagged with fixed-choice labels: shot type (the same names the shot classifier uses), forehand/backhand, singles/doubles, body parts, and the video's audience level.
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
   - `pose_landmarker_full.task` — MediaPipe pose landmarker model ([download](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)); the lite version tracks the racket arm noticeably worse from behind

   (`yolov8n.pt`, used for player detection, downloads automatically on first run.)
4. Build the coaching knowledge base once (only needed the first time, or after regenerating `coaching_facts_full.json`):
   ```
   python -m src.knowledge.vector_store
   ```

## Usage

```
python main.py --video path/to/your/match.mp4 --racket-arm right
```

This runs the full pipeline — player selection, pose/shuttle extraction, shot detection and naming, and coaching feedback synthesis — and prints timestamped feedback. It also saves the detected shots (`outputs/shot_events.json`), what Claude was given (`outputs/observations.json`, `outputs/queries_and_facts.json`) and what it wrote (`outputs/feedback.json`).

Flags:
- `--video` (required) — path to the match video
- `--racket-arm` — `right` or `left` (default: `right`). Afterwards, open `outputs/racket_arm_check.jpg` and confirm the yellow arm is your racket arm — MediaPipe can mirror left and right on footage shot from behind
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

## Measuring accuracy

Hand-label the shots in a clip with `python scripts/label_shots.py --video data/<clip>.MOV` (see `data/labels/README.md`), then score a run with `python scripts/evaluate.py --labels data/labels/<clip>.csv`. Results append to `outputs/eval_history.csv`.

On the first labelled clip (34 s, 10 shots): all 10 shots found within ±0.5 s with 1 false detection, and all 10 named correctly — but the rules were tuned on that clip, so this is a fit, not an independent test. On a second clip (12.6 s of a professional match): the near player, 5 shots, all found with 2 false detections and 5 of 5 named; the far player (facing the camera), 6 shots, 5 found with 3 false detections and 3 of 5 named, the misses being two very soft net shots. More labelled clips are needed, especially ones with smashes.

## Current status and known limitations

- Tested on clips of ~15-34 seconds so far — longer videos, even within the intended under-1-minute range, are untested
- Pose detection: 100% of frames on self-recorded test footage
- Shuttle detection: ~19% of frames on amateur footage — lower than the 0.608 mAP50 the underlying model achieves on its own validation set, since real-world amateur footage (lighting, background clutter, camera angle) is harder than the training distribution. Improving this is planned as its own follow-up project.
- Shot naming relies on the shuttle being detected during its flight; when it isn't (≈19% detection rate), overhead shots fall back to a guess between clear and drop based on swing speed, with low confidence. A smash is only recognised when the shuttle is seen leaving fast, and the speed line comes from one labelled smash and one drop so far. Forehand vs backhand is not identified yet.
- Shot naming rules were tuned on three labelled players (21 shots) and have seen two smashes. Very soft net shots by a professional barely move the wrist and can be missed as shots altogether.
- Shuttle detection depends on the camera's shooting angle — on footage where overhead space isn't well captured, shots hit high (e.g. clears, lifts) can go undetected entirely, independent of the model's accuracy.
- Search query generation is non-deterministic — a plausible shot type (e.g. drop shot alongside clear and smash for an overhead pattern) can occasionally be left out of a given run's split between multiple shot types, even though the underlying prompt asks for full coverage.

## Roadmap

- Test on additional match footage to check generalization beyond the current test clip
- Improve shuttlecock detection on amateur footage
- Lightweight visualization pairing the pose overlay video with timestamped feedback

## Notes

This project was built with guidance from Claude AI.
