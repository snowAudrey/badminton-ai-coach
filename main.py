import argparse
import os

from src.vision.pipeline import run_pipeline as run_cv_pipeline
from src.agents.orchestrator import run_pipeline as run_coaching_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Badminton AI Coach — turns a match video into timestamped coaching feedback."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--racket-arm", choices=["right", "left"], default="right",
                         help="Player's racket arm side (default: right)")
    parser.add_argument("--output-dir", default="outputs",
                         help="Folder for intermediate and output files (default: outputs)")
    args = parser.parse_args()

    print("Step 1/2 — Running CV pipeline (player selection, pose, shuttle, rally detection)...")
    run_cv_pipeline(video_path=args.video, output_dir=args.output_dir)

    unified_data_path = os.path.join(args.output_dir, "unified_data.json")

    print("\nStep 2/2 — Generating coaching feedback...")
    feedback_by_rally = run_coaching_pipeline(unified_data_path, racket_arm=args.racket_arm)

    for rally_id, feedback in feedback_by_rally.items():
        print(f"\n{'=' * 60}")
        print(f"FEEDBACK FOR RALLY {rally_id}")
        print('=' * 60)
        print(feedback)


if __name__ == "__main__":
    main()
