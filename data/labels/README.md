# Shot labels

Hand-made ground truth used by `scripts/evaluate.py` to score the coach.

One CSV per clip, named after the video (e.g. `test_clip.csv`), with columns:

| column    | meaning                                                                 |
|-----------|-------------------------------------------------------------------------|
| time_s    | moment of racket–shuttle contact, in seconds                            |
| frame     | the same moment as a frame number                                       |
| shot_type | smash, clear, drop, drive, net shot, lift, block/defence, serve, return of serve, other |
| side      | forehand, backhand, or unknown                                          |

Make them with `python scripts/label_shots.py --video data/<clip>.MOV`.

Tips:
- Label the selected player's shots only (the one you click on when the pipeline runs).
- Pick the frame where the racket meets the shuttle; ±2 frames is fine. If you need to see where the shuttle lands before you can name the shot, press `m` at the contact moment, keep watching, then press the shot number — the label is saved at the marked time.
- If you can't tell the shot type, use `other` rather than guessing.

Also fill in `checklist.md` once per clip after reading the feedback.
