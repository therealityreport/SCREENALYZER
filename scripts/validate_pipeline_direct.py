"""
Direct pipeline validation without RQ workers.
Runs tasks sequentially to validate Phase 1 implementation.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from datetime import datetime

# Import tasks
from jobs.tasks.harvest import harvest_task
from jobs.tasks.detect_embed import detect_embed_task
from jobs.tasks.track import track_task
from jobs.tasks.cluster import cluster_task
from screentime.utils import normalize_episode_id
from api.jobs import job_manager
from screentime.types import JobStatus

def main():
    video_path = "/Volumes/HardDrive/SCREENALYZER/data/videos/RHOBH-TEST-10-28.mp4"
    episode_id = "RHOBH-TEST-10-28"
    job_id = f"direct_{int(time.time())}"

    print("=" * 60)
    print("Direct Pipeline Validation")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Episode: {episode_id}")
    print(f"Job ID: {job_id}")
    print()

    # Normalize episode ID
    episode_id = normalize_episode_id(episode_id)
    print(f"Normalized episode ID: {episode_id}")
    print()

    # Verify video exists
    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        print(f"❌ Video not found: {video_path}")
        return 1

    print(f"✅ Video found ({video_path_obj.stat().st_size / 1024 / 1024:.1f} MB)")
    print()

    # Initialize minimal job metadata in Redis
    job_data = {
        "job_id": job_id,
        "episode_id": episode_id,
        "video_path": video_path,
        "status": JobStatus.RUNNING.value,
        "created_at": datetime.utcnow().isoformat(),
    }
    job_manager._save_job_metadata(job_id, job_data)
    print(f"✅ Job metadata initialized in Redis")
    print()

    try:
        # Stage 1: Harvest
        print("─" * 60)
        print("Stage 1: Harvest")
        print("─" * 60)
        start_time = time.time()
        harvest_result = harvest_task(job_id, episode_id, video_path)
        elapsed = time.time() - start_time
        print(f"✅ Harvest complete in {elapsed:.1f}s")
        print(f"   Frames sampled: {harvest_result['frames_sampled']}")
        print(f"   Manifest: {harvest_result['manifest_path']}")
        print()

        # Stage 2: Detect & Embed
        print("─" * 60)
        print("Stage 2: Detection & Embeddings")
        print("─" * 60)
        start_time = time.time()
        detect_result = detect_embed_task(job_id, episode_id)
        elapsed = time.time() - start_time
        print(f"✅ Detection complete in {elapsed:.1f}s")
        print(f"   Faces detected: {detect_result['stats'].get('faces_detected', 'N/A')}")
        print(f"   Embeddings: {detect_result['embeddings_path']}")
        print()

        # Stage 3: Tracking
        print("─" * 60)
        print("Stage 3: Tracking")
        print("─" * 60)
        start_time = time.time()
        track_result = track_task(job_id, episode_id)
        elapsed = time.time() - start_time
        print(f"✅ Tracking complete in {elapsed:.1f}s")
        print(f"   Tracks built: {track_result['stats'].get('tracks_built', 'N/A')}")
        print(f"   Tracks file: {track_result['tracks_path']}")
        print()

        # Stage 4: Clustering
        print("─" * 60)
        print("Stage 4: Clustering")
        print("─" * 60)
        start_time = time.time()
        cluster_result = cluster_task(job_id, episode_id)
        elapsed = time.time() - start_time
        print(f"✅ Clustering complete in {elapsed:.1f}s")
        print(f"   Clusters: {cluster_result['stats'].get('clusters_created', 'N/A')}")
        print(f"   Merge suggestions: {cluster_result['stats'].get('merge_suggestions', 'N/A')}")
        print(f"   Low-conf clusters: {cluster_result['stats'].get('lowconf_clusters', 'N/A')}")
        print()

        # Verify all artifacts
        print("=" * 60)
        print("Artifact Verification")
        print("=" * 60)

        harvest_dir = Path("data/harvest") / episode_id
        artifacts = [
            ("Manifest", harvest_dir / "manifest.parquet"),
            ("Embeddings", harvest_dir / "embeddings.parquet"),
            ("Tracks", harvest_dir / "tracks.json"),
            ("Clusters", harvest_dir / "clusters.json"),
            ("Merge suggestions", harvest_dir / "assist" / "merge_suggestions.parquet"),
            ("Low-conf queue", harvest_dir / "assist" / "lowconf_queue.parquet"),
            ("Det stats", harvest_dir / "diagnostics" / "reports" / "det_stats.json"),
            ("Track stats", harvest_dir / "diagnostics" / "reports" / "track_stats.json"),
            ("Cluster stats", harvest_dir / "diagnostics" / "reports" / "cluster_stats.json"),
        ]

        all_exist = True
        for name, path in artifacts:
            exists = path.exists()
            status = "✅" if exists else "❌"
            size_str = f"({path.stat().st_size / 1024:.1f} KB)" if exists else ""
            print(f"{status} {name}: {path} {size_str}")
            if not exists:
                all_exist = False

        print()
        if all_exist:
            print("✅ All artifacts verified!")
            print()
            print("=" * 60)
            print("VALIDATION SUCCESSFUL")
            print("=" * 60)
            return 0
        else:
            print("❌ Some artifacts missing!")
            return 1

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
