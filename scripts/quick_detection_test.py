"""
Quick detection threshold test.

Tests one or two promising combinations to validate approach
before running full sweep.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from app.lib.data import load_clusters
from jobs.tasks.analytics import analytics_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_detection_thresholds(min_conf: float, min_px: int):
    """Test detection with specific thresholds."""
    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: min_confidence={min_conf}, min_face_px={min_px}")
    logger.info(f"{'='*60}\n")

    # Update config temporarily
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    original_conf = config["detection"]["min_confidence"]
    original_px = config["video"]["min_face_px"]

    config["detection"]["min_confidence"] = min_conf
    config["video"]["min_face_px"] = min_px

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        # Run pipeline stages
        import sys
        import time

        sys.path.insert(0, str(Path.cwd()))

        from jobs.tasks.detect_embed import detect_embed_task
        from jobs.tasks.track import track_task
        from jobs.tasks.cluster import cluster_task
        from api.jobs import job_manager
        from screentime.types import JobStatus
        from datetime import datetime

        job_id = f"test_{int(time.time())}"

        # Register job in Redis
        video_path = str(data_root / "videos" / f"{episode_id}.mp4")
        job_data = {
            "job_id": job_id,
            "episode_id": episode_id,
            "video_path": video_path,
            "status": JobStatus.RUNNING.value,
            "created_at": datetime.utcnow().isoformat(),
        }
        job_manager._save_job_metadata(job_id, job_data)

        logger.info("Stage 1: Detection...")
        detect_result = detect_embed_task(job_id, episode_id)
        logger.info(f"  Faces detected: {detect_result['stats'].get('faces_detected')}")

        logger.info("\nStage 2: Tracking...")
        track_result = track_task(job_id, episode_id)
        logger.info(f"  Tracks built: {track_result['stats'].get('tracks_built')}")
        logger.info(f"  Re-ID links: {track_result['stats'].get('relink_accepted', 0)}")

        logger.info("\nStage 3: Clustering...")
        cluster_result = cluster_task(job_id, episode_id)
        logger.info(f"  Clusters: {cluster_result['stats'].get('clusters_built')}")

        # Assign names (same mapping as before)
        import json

        clusters_path = data_root / "harvest" / episode_id / "clusters.json"
        with open(clusters_path) as f:
            clusters_data = json.load(f)

        name_map = {12: "RINNA", 53: "KIM", 17: "KYLE", 8: "EILEEN", 5: "YOLANDA", 9: "BRANDI", 3: "LVP"}

        for cluster in clusters_data["clusters"]:
            size = cluster["size"]
            # Find closest size match (in case size changed slightly)
            closest_size = min(name_map.keys(), key=lambda k: abs(k - size))
            if abs(closest_size - size) <= 3:  # Allow ±3 tracks variation
                cluster["name"] = name_map[closest_size]

        with open(clusters_path, "w") as f:
            json.dump(clusters_data, f, indent=2)

        logger.info("\nStage 4: Analytics...")
        cluster_assignments = {
            c["cluster_id"]: c["name"]
            for c in clusters_data["clusters"]
            if "name" in c
        }

        analytics_result = analytics_task(job_id, episode_id, cluster_assignments)

        # Load and display results
        totals_df = pd.read_csv(analytics_result["totals_path"])

        logger.info("\n=== RESULTS ===\n")

        gt = {
            "KIM": 48004,
            "KYLE": 21017,
            "RINNA": 25015,
            "EILEEN": 10001,
            "BRANDI": 10014,
            "YOLANDA": 16002,
            "LVP": 2018,
        }

        logger.info(f"{'Person':<10s} {'Auto':>8s} {'GT':>8s} {'Delta':>8s} {'%':>6s} Status")
        logger.info("-" * 60)

        for person in sorted(gt.keys()):
            row = totals_df[totals_df["person_name"] == person]
            if len(row) > 0:
                auto_ms = int(row["total_ms"].values[0])
            else:
                auto_ms = 0

            gt_ms = gt[person]
            delta_ms = gt_ms - auto_ms
            delta_pct = (delta_ms / gt_ms * 100) if gt_ms > 0 else 0

            status = "✓" if abs(delta_ms) <= 1000 else "⚠" if abs(delta_ms) <= 2000 else "✗"

            logger.info(
                f"{person:<10s} {auto_ms:8d} {gt_ms:8d} {delta_ms:+8d} {delta_pct:+6.1f}% {status}"
            )

        return totals_df

    finally:
        # Restore config
        config["detection"]["min_confidence"] = original_conf
        config["video"]["min_face_px"] = original_px

        with open(config_path, "w") as f:
            yaml.dump(config, f)


if __name__ == "__main__":
    # Test the most promising combination first
    logger.info("Testing relaxed thresholds (min_conf=0.65, min_px=64)")
    test_detection_thresholds(0.65, 64)
