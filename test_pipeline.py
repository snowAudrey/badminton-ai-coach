from src.vision.pipeline import run_pipeline

#VIDEO_PATH = "data/test_clip.MOV"
VIDEO_PATH = "data/test_clip_pro.MOV"

unified = run_pipeline(
    video_path=VIDEO_PATH,
    output_dir="outputs"
)

# ============================================================
# CHECK 1: Did it produce output?
# ============================================================
print(f"\nTotal frames: {len(unified)}")

# ============================================================
# CHECK 2: Does the structure look right?
# ============================================================
print("\nFirst 3 entries:")
for entry in unified[:3]:
    print(f"  frame={entry['frame']}  t={entry['timestamp']}s  "
          f"rally={entry['rally_id']}  "
          f"keypoints={len(entry['player_keypoints'])}  "
          f"shuttle=({entry['shuttle_x']}, {entry['shuttle_y']})")

# ============================================================
# CHECK 3: Are the detection rates reasonable?
# ============================================================
total = len(unified)
pose_count = sum(1 for f in unified if f["player_keypoints"])
shuttle_count = sum(1 for f in unified if f["shuttle_detected"])
rally_count = sum(1 for f in unified if f["rally_id"] is not None)

print(f"\nDetection rates:")
print(f"  Pose:    {pose_count}/{total} frames ({100*pose_count//total}%)")
print(f"  Shuttle: {shuttle_count}/{total} frames ({100*shuttle_count//total}%)")
print(f"  Rally:   {rally_count}/{total} frames ({100*rally_count//total}%)")