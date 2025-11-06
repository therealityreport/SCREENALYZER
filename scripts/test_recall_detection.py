"""
Test recall detection on RHOBH-TEST-10-28.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.tasks.post_label_recall import post_label_recall_task
from api.jobs import job_manager

def test_recall():
    """Test recall detection."""

    episode_id = "RHOBH-TEST-10-28"
    job_id = f"recall_test_{int(datetime.utcnow().timestamp())}"

    print("=" * 80)
    print("POST-LABEL RECALL DETECTION TEST")
    print("=" * 80)
    print(f"\nEpisode: {episode_id}")
    print(f"Job ID: {job_id}")
    print()

    # Load cluster assignments
    clusters_file = Path(f'data/harvest/{episode_id}/clusters.json')
    with open(clusters_file) as f:
        clusters_data = json.load(f)

    # Build cluster assignments (cluster_id -> person_name)
    cluster_assignments = {}
    for cluster in clusters_data['clusters']:
        cluster_id = cluster['cluster_id']
        person_name = cluster.get('name')
        if person_name:
            cluster_assignments[cluster_id] = person_name

    print(f"Loaded {len(cluster_assignments)} cluster assignments:")
    for cluster_id, person_name in sorted(cluster_assignments.items()):
        print(f"  Cluster {cluster_id}: {person_name}")
    print()

    # Register job with metadata
    job_manager._save_job_metadata(job_id, {
        'job_id': job_id,
        'episode_id': episode_id,
        'video_path': f'data/videos/{episode_id}.mp4',
        'status': 'running',
        'created_at': datetime.utcnow().isoformat(),
    })

    # Run recall detection
    print("Running identity-guided recall detection...")
    print()

    try:
        result = post_label_recall_task(job_id, episode_id, cluster_assignments)

        print("\n" + "=" * 80)
        print("RECALL DETECTION RESULTS")
        print("=" * 80)
        print()

        if not result.get('enabled'):
            print("❌ Recall detection is disabled in config")
            return False

        stats = result.get('stats', {})

        print(f"People processed: {stats.get('people_processed', 0)}")
        print(f"Gap windows scanned: {stats.get('total_windows_scanned', 0)}")
        print(f"Verified detections: {stats.get('recall_detections_verified', 0)}")
        print(f"New tracklets created: {stats.get('new_tracklets_created', 0)}")
        print(f"Processing time: {stats.get('stage_time_ms', 0) / 1000:.1f}s")
        print()

        if stats.get('new_tracklets_created', 0) > 0:
            print("✓ Recall detection found missing faces!")
            print("\nRe-run baseline report to see updated accuracy:")
            print("  python scripts/generate_baseline_report.py")
        else:
            print("ℹ No new tracklets created")
            print("\nThis could mean:")
            print("  • Gap windows too small or sparse")
            print("  • Detection thresholds still too strict")
            print("  • Identity verification too strict")
            print("  • All faces already detected")

        print()
        return True

    except Exception as e:
        print(f"\n❌ ERROR: Recall detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_recall()
    sys.exit(0 if success else 1)
