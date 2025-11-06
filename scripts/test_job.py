#!/usr/bin/env python3
"""
Test script for job processing.

Demonstrates job enqueue and status tracking.
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.jobs import job_manager


def test_job_enqueue():
    """Test job enqueue and status tracking."""
    print("\n=== Testing Job Enqueue ===\n")

    # Video path
    video_path = Path("/Volumes/HardDrive/SCREENALYZER/data/videos/RHOBH-TEST-10-28.mp4")

    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    print(f"Video: {video_path}")
    print(f"Size: {video_path.stat().st_size / (1024**2):.2f} MB\n")

    # Enqueue job
    print("1. Enqueuing job...")
    result = job_manager.enqueue_processing_job(episode_id="RHOBH_TEST", video_path=video_path)

    job_id = result["job_id"]
    print(f"   Job ID: {job_id}")
    print(f"   RQ Job ID: {result['rq_job_id']}")
    print(f"   Status: {result['status']}\n")

    # Poll status
    print("2. Polling job status (5 seconds)...")
    for i in range(5):
        time.sleep(1)
        try:
            status = job_manager.get_job_status(job_id)
            print(
                f"   [{i+1}] Stage: {status['stage']}, "
                f"Progress: {status['progress_pct']:.1f}%, "
                f"Status: {status['status']}"
            )
        except Exception as e:
            print(f"   Error getting status: {e}")

    print("\n✅ Test complete!")
    print(f"\nTo check results later:")
    print(f"  ls data/harvest/RHOBH_TEST/")
    print(f"  cat data/harvest/RHOBH_TEST/det_stats.json")


if __name__ == "__main__":
    test_job_enqueue()
