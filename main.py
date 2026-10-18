import argparse
import os

from src.vision.pipeline import run_pipeline as run_cv_pipeline
from src.vision.shot_events import detect_shots
from src.agents.shot_classifier import classify_shots
from src.agents.vision_agent import analyze_video
from src.agents.orchestrator import run_pipeline as run_coaching_pipeline, _save_json


def main():
    parser = argparse.ArgumentParser(
        description="Badminton AI Coach — turns a short badminton clip into timestamped coaching feedback."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--racket-arm", choices=["right", "left"], default="right",
                         help="Which arm holds the racket (default: right). Check "
                              "outputs/racket_arm_check.jpg afterwards to confirm the right arm "
                              "is being measured — MediaPipe can mirror sides on footage shot from behind")
    parser.add_argument("--output-dir", default="outputs",
                         help="Folder for intermediate and output files (default: outputs)")
    parser.add_argument("--no-feedback", action="store_true",
                         help="Stop after finding and naming shots (no Claude calls) — useful when "
                              "scoring detection against hand labels")
    args = parser.parse_args()

    print("Step 1/3 — Running CV pipeline (player selection, pose, shuttle detection)...")
    run_cv_pipeline(video_path=args.video, output_dir=args.output_dir)

    unified_data_path = os.path.join(args.output_dir, "unified_data.json")

    print("\nStep 2/3 — Finding and naming shots...")
    events_path = os.path.join(args.output_dir, "shot_events.json")
    detect_shots(unified_data_path, racket_arm=args.racket_arm, output_path=events_path)
    classify_shots(events_path, args.video, unified_data_path, racket_arm=args.racket_arm)

    if args.no_feedback:
        # Still produce the racket-arm check image and the observations file,
        # so a validation run can be checked and scored like a full run
        observations = analyze_video(unified_data_path, racket_arm=args.racket_arm,
                                     video_path=args.video, output_dir=args.output_dir)
        _save_json(os.path.join(args.output_dir, "observations.json"), observations)
        print(f"\nStopping before feedback (--no-feedback). Shots saved → {events_path}")
        return

    print("\nStep 3/3 — Generating coaching feedback...")
    feedback_by_window = run_coaching_pipeline(unified_data_path, racket_arm=args.racket_arm,
                                               video_path=args.video, output_dir=args.output_dir)

    for label, feedback in feedback_by_window.items():
        print(f"\n{'=' * 60}")
        print(f"FEEDBACK — {label}")
        print('=' * 60)
        print(feedback)


if __name__ == "__main__":
    main()
