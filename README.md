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
- `knowledge_agent.py` — turns the named shots into knowledge-base searches: for each shot type in the clip, one query about its preparation, contact point and follow-through, plus one query for each measurement that stands out (elbow straight at contact, deep knee bend before the shot, contact below head height, a short swing on a clear, a big swing on a net shot). Searches are limited to that shot type and to the player's discipline, and no fact is returned twice.
- `orchestrator.py` — one Claude call per clip (Sonnet) that receives the shots in time order with their measurements and the retrieved facts, and writes feedback shot by shot. A shot is named outright when the classifier's confidence is 0.7 or above, and hedged between its candidates below that. Clips much longer than 30 seconds are split into equal windows, one call each.

**3. Offline knowledge base build** (`src/knowledge/`, run once — not part of the per-video pipeline)
- `caption_parser.py` + `knowledge_extractor.py` — parses caption transcripts from 117 badminton tutorial videos (48 duplicate copies are skipped) and runs them through a Claude extraction pipeline to pull out ~2,100 structured coaching facts (technique, preparation, execution, common mistake, cue, context), each tagged with fixed-choice labels: shot type (the same names the shot classifier uses), forehand/backhand, singles/doubles, body parts, and the video's audience level.
- `vector_store.py` — removes near-duplicate facts, then embeds and stores the rest in ChromaDB with their tags. A search can be limited to one shot type and to singles or doubles, and matches too far from the question are dropped instead of being returned as the "nearest" anyway.

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
- `--discipline` — `singles` or `doubles` (default: `singles`); keeps advice meant for the other discipline out of the feedback
- `--output-dir` — where intermediate files are saved (default: `outputs`)

## Example output

```
FEEDBACK — whole clip
============================================================
At 5.6s, lift: your racket hand rose from below the shoulder to overhead
through the stroke and the shuttle climbed 8.8 shoulder widths above head
height, showing you got well underneath the shuttle with your strings to
generate height, which is exactly the aim on this shot. Nothing to flag here.

At 7.6s, most likely a block/defence, possibly a lift or net shot: your elbow
was 163° at contact, nearly straight, which matches the common mistake of
standing with the racket too outstretched rather than keeping the elbow out
in front of you at the correct angle. Your knee bend beforehand was deep,
which is good, so focus on keeping that elbow in rather than fully extending it.

At 27.7s, most likely a clear, possibly a smash or drop: your elbow was 166°
at contact, nearly straight, which for a clear prevents the proper
elbow-leading throwing action and limits power. Work on keeping the elbow
slightly bent at contact rather than straightening it fully.

Across the clip, the recurring issue was letting your arm get too straight at
the point of contact, showing up on the block/defence at 7.6s and the clear
at 27.7s — keep that elbow slightly bent through contact on both overhead and
defensive shots.
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

## Roadmap

- Test on additional match footage to check generalization beyond the current test clip
- Improve shuttlecock detection on amateur footage
- Lightweight visualization pairing the pose overlay video with timestamped feedback

## Notes

This project was built with guidance from Claude AI.
