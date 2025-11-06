"""
Targeted densify for BRANDI's largest gap (62.4s - 95.9s).
"""

from jobs.tasks.local_densify import local_densify_task

# Manually specify BRANDI's largest gap window
gap_windows = {
    "BRANDI": [
        {
            "start_ms": 62400,
            "end_ms": 95900,
            "gap_duration_s": 33.5,
        }
    ]
}

print("=" * 60)
print("Targeted Densify: BRANDI Largest Gap (62.4s - 95.9s)")
print("=" * 60)

result = local_densify_task(
    job_id="brandi_densify",
    episode_id="RHOBH-TEST-10-28",
    gap_windows=gap_windows
)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)
print(f"Seconds recovered: {result.get('seconds_recovered', 0):.2f}s")
print(f"Tracklets created: {result.get('stats', {}).get('tracklets_created', 0)}")
print(f"Detections verified: {result.get('stats', {}).get('detections_verified', 0)}")

# Check recall stats
import json
from pathlib import Path

recall_stats_path = Path("data/harvest/RHOBH-TEST-10-28/diagnostics/recall_stats.json")
if recall_stats_path.exists():
    with open(recall_stats_path) as f:
        recall_stats = json.load(f)

    print("\n" + "=" * 60)
    print("Recall Stats (BRANDI)")
    print("=" * 60)
    if "BRANDI" in recall_stats:
        brandi_stats = recall_stats["BRANDI"]
        print(f"Windows scanned: {brandi_stats.get('windows_scanned', 0)}")
        print(f"Frames with detections: {brandi_stats.get('frames_with_detections', 0)}")
        print(f"Verified detections: {brandi_stats.get('detections_verified', 0)}")
        print(f"Seconds recovered: {brandi_stats.get('seconds_recovered', 0):.2f}s")
