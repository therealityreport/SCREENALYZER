#!/usr/bin/env python3
"""Test analytics after local densify."""

import logging
from jobs.tasks.analytics import analytics_task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Configuration
episode_id = "RHOBH-TEST-10-28"

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

logger.info(f"Running analytics for {episode_id}")

result = analytics_task(
    job_id="test_analytics",
    episode_id=episode_id,
    cluster_assignments=cluster_assignments,
)

logger.info(f"\n=== ANALYTICS RESULTS ===")
logger.info(f"Timeline path: {result['timeline_path']}")
logger.info(f"Totals path: {result['totals_path']}")
