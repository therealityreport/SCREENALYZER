#!/usr/bin/env python3
"""Test local densify task for YOLANDA."""

import logging
from pathlib import Path
from jobs.tasks.local_densify import local_densify_task

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configuration
episode_id = "RHOBH-TEST-10-28"
video_path = Path("data/videos/RHOBH-TEST-10-28.mp4")

# Cluster assignments (from clusters.json)
cluster_assignments = {
    0: "RINNA",
    1: "KIM",
    2: "KYLE",
    3: "EILEEN",
    5: "YOLANDA",
    6: "BRANDI",
    7: "LVP",
}

# Target identities for densify (undercounted cast)
target_identities = ["YOLANDA", "RINNA", "BRANDI"]

logger.info(f"Running local densify for {episode_id}")
logger.info(f"Target identities: {target_identities}")

result = local_densify_task(
    job_id="test_densify",
    episode_id=episode_id,
    video_path=video_path,
    target_identities=target_identities,
    cluster_assignments=cluster_assignments,
)

logger.info(f"\n=== RESULTS ===")
logger.info(f"Tracklets created: {result['tracklets_created']}")
logger.info(f"Segments scanned: {result['segments_scanned']}")
logger.info(f"Stats: {result['stats']}")
