"""
End-to-end test script for Screenalyzer pipeline.

Tests full pipeline from video upload through analytics generation.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.jobs import job_manager
from screentime.utils import normalize_episode_id


def test_e2e_pipeline(video_path: Path, episode_id: str):
    """
    Test full E2E pipeline.

    Args:
        video_path: Path to test video
        episode_id: Episode identifier
    """
    print(f"\n{'='*60}")
    print(f"E2E Pipeline Test")
    print(f"{'='*60}")
    print(f"Video: {video_path}")
    print(f"Episode: {episode_id}\n")

    # Normalize episode ID
    episode_id = normalize_episode_id(episode_id)
    print(f"Normalized episode ID: {episode_id}")

    # Verify video exists
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return False

    print(f"✅ Video found ({video_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Enqueue processing job
    print(f"\n{'─'*60}")
    print("Enqueueing processing job...")
    try:
        result = job_manager.enqueue_processing_job(episode_id, video_path)
        job_id = result["job_id"]
        print(f"✅ Job enqueued: {job_id}")
    except Exception as e:
        print(f"❌ Failed to enqueue job: {e}")
        return False

    # Monitor job progress
    print(f"\n{'─'*60}")
    print("Monitoring job progress...")
    print("(Check that harvest.q, inference.q, tracking.q, cluster.q workers are running)\n")

    last_stage = None
    last_progress = 0.0

    for i in range(300):  # 5 minutes max
        try:
            status = job_manager.get_job_status(job_id)
            stage = status.get("stage", "unknown")
            progress = status.get("progress_pct", 0.0)
            job_status = status.get("status", "unknown")

            # Print update if stage or progress changed
            if stage != last_stage or abs(progress - last_progress) > 5.0:
                print(f"  [{job_status}] {stage}: {progress:.1f}%")
                last_stage = stage
                last_progress = progress

            # Check if completed
            if job_status == "completed":
                print(f"\n✅ Job completed!")
                break
            elif job_status == "failed":
                print(f"\n❌ Job failed: {status.get('message', 'Unknown error')}")
                return False

        except Exception as e:
            print(f"⚠️  Status check error: {e}")

        time.sleep(1)
    else:
        print(f"\n❌ Timeout waiting for job completion")
        return False

    # Verify outputs
    print(f"\n{'─'*60}")
    print("Verifying outputs...")

    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    reports_dir = harvest_dir / "diagnostics" / "reports"

    checks = [
        ("Tracks", harvest_dir / "tracks.json"),
        ("Clusters", harvest_dir / "clusters.json"),
        ("Merge suggestions", harvest_dir / "assist" / "merge_suggestions.parquet"),
        ("Low-conf queue", harvest_dir / "assist" / "lowconf_queue.parquet"),
        ("Detection stats", reports_dir / "det_stats.json"),
        ("Tracking stats", reports_dir / "track_stats.json"),
        ("Clustering stats", reports_dir / "cluster_stats.json"),
    ]

    all_passed = True
    for name, path in checks:
        if path.exists():
            size = path.stat().st_size
            print(f"  ✅ {name}: {path.name} ({size} bytes)")
        else:
            print(f"  ❌ {name}: Not found at {path}")
            all_passed = False

    if not all_passed:
        print(f"\n❌ Some outputs missing")
        return False

    # Print summary
    print(f"\n{'='*60}")
    print("E2E Pipeline Test: ✅ PASSED")
    print(f"{'='*60}\n")

    print("Next steps:")
    print(f"  1. Start Streamlit UI: streamlit run app/labeler.py")
    print(f"  2. Navigate to Review tab")
    print(f"  3. Select episode: {episode_id}")
    print(f"  4. Verify UI displays clusters, suggestions, and thumbnails")
    print(f"  5. Test merge/assign actions")
    print(f"  6. Navigate to Analytics tab and generate analytics")
    print("")

    return True


if __name__ == "__main__":
    # Test configuration
    VIDEO_PATH = Path("/Volumes/HardDrive/SCREENALYZER/data/videos/RHOBH-TEST-10-28.mp4")
    EPISODE_ID = "RHOBH-TEST-10-28"

    # Run test
    success = test_e2e_pipeline(VIDEO_PATH, EPISODE_ID)

    sys.exit(0 if success else 1)
