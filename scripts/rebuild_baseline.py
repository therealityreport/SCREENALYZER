"""
Rebuild RHOBH-TEST-10-28 pipeline with baseline configuration.
Clears existing data and runs full pipeline from scratch.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.tasks.harvest import harvest_task
from jobs.tasks.detect_embed import detect_embed_task
from jobs.tasks.track import track_task
from jobs.tasks.cluster import cluster_task
from jobs.tasks.analytics import analytics_task
from api.jobs import job_manager
from screentime.types import JobStatus
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def rebuild_pipeline():
    """Rebuild full pipeline for RHOBH-TEST-10-28."""

    episode_id = "RHOBH-TEST-10-28"
    job_id = "baseline_rebuild"

    # Data paths
    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    video_path = str(data_root / "videos" / f"{episode_id}.mp4")

    # Clear existing data
    import shutil
    if harvest_dir.exists():
        logger.info(f"Clearing existing data: {harvest_dir}")
        shutil.rmtree(harvest_dir)

    output_dir = data_root / "outputs" / episode_id
    if output_dir.exists():
        logger.info(f"Clearing existing outputs: {output_dir}")
        shutil.rmtree(output_dir)

    # Register job
    job_data = {
        "job_id": job_id,
        "episode_id": episode_id,
        "video_path": video_path,
        "status": JobStatus.RUNNING.value,
        "created_at": datetime.utcnow().isoformat(),
    }
    job_manager._save_job_metadata(job_id, job_data)

    logger.info("=" * 60)
    logger.info("REBUILDING BASELINE PIPELINE")
    logger.info("=" * 60)

    # Stage 0: Harvest
    logger.info("\nStage 0: Harvest (frame sampling)...")
    harvest_result = harvest_task(job_id, episode_id, video_path)
    logger.info(f"  ✓ Frames sampled: {harvest_result.get('frames_sampled', 0)}")
    logger.info(f"  ✓ Duration: {harvest_result.get('duration_sec', 0):.1f}s")

    # Stage 1: Detection + Embedding
    logger.info("\nStage 1: Detection + Embedding...")
    detect_result = detect_embed_task(job_id, episode_id)
    logger.info(f"  ✓ Faces detected: {detect_result['stats'].get('faces_detected')}")
    logger.info(f"  ✓ Embeddings computed: {detect_result['stats'].get('embeddings_computed')}")

    # Stage 2: Tracking
    logger.info("\nStage 2: Tracking...")
    track_result = track_task(job_id, episode_id)
    logger.info(f"  ✓ Tracks built: {track_result['stats'].get('tracks_built')}")

    reid_meta = track_result.get('reid_metadata', {})
    if reid_meta:
        logger.info(f"  ✓ Re-ID links: {reid_meta.get('links_created', 0)}/{reid_meta.get('total_attempts', 0)}")

    # Stage 3: Clustering
    logger.info("\nStage 3: Clustering...")
    cluster_result = cluster_task(job_id, episode_id)
    logger.info(f"  ✓ Clusters found: {cluster_result['stats'].get('clusters_built')}")

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE REBUILD COMPLETE")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Review clusters in UI and assign names")
    logger.info("2. Run analytics to generate screen time totals")

    return {
        "status": "success",
        "episode_id": episode_id,
        "job_id": job_id,
        "stages_complete": ["detect_embed", "track", "cluster"]
    }


if __name__ == "__main__":
    try:
        result = rebuild_pipeline()
        print(f"\n✓ SUCCESS: {result}")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
