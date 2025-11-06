"""
Test script for re-ID track stitching and adaptive gap-merge.

Runs analytics on RHOBH-TEST-10-28 with new algorithm and compares to GT.
"""

import logging
from pathlib import Path

import yaml

from app.lib.data import load_clusters, load_tracks
from jobs.tasks.analytics import analytics_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test re-ID and adaptive merge on RHOBH-TEST-10-28."""
    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")

    logger.info(f"Testing new algorithms on {episode_id}")

    # Load clusters to get assignments
    clusters_data = load_clusters(episode_id, data_root)
    cluster_assignments = {}

    for cluster in clusters_data.get("clusters", []):
        if "name" in cluster:
            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

    logger.info(f"Loaded {len(cluster_assignments)} cluster assignments")

    # Run analytics with new algorithms
    result = analytics_task("test-reid", episode_id, cluster_assignments)

    logger.info("Analytics complete!")
    logger.info(f"Results: {result['stats']}")

    # Load and display results
    import pandas as pd

    totals_path = Path(result["totals_path"])
    totals_df = pd.read_csv(totals_path)

    logger.info("\n=== NEW AUTO TOTALS ===")
    for _, row in totals_df.iterrows():
        logger.info(
            f"{row['person_name']:10s} {row['total_ms']:6d} ms "
            f"({row['appearances']:2d} intervals, conf={row['mean_confidence']:.2f})"
        )

    # Compare to GT
    gt_totals = {
        "KIM": 48004,
        "KYLE": 21017,
        "RINNA": 25015,
        "EILEEN": 10001,
        "BRANDI": 10014,
        "YOLANDA": 16002,
        "LVP": 2018,
    }

    logger.info("\n=== COMPARISON TO GT ===")
    logger.info(f"{'Person':<10s} {'Auto':>8s} {'GT':>8s} {'Delta':>8s} {'%':>6s}")
    logger.info("-" * 50)

    for person in sorted(gt_totals.keys()):
        auto_ms = totals_df[totals_df["person_name"] == person]["total_ms"].values[0]
        gt_ms = gt_totals[person]
        delta_ms = gt_ms - auto_ms
        delta_pct = (delta_ms / gt_ms * 100) if gt_ms > 0 else 0

        status = "✓" if abs(delta_ms) <= 1000 else "⚠" if abs(delta_ms) <= 3000 else "✗"

        logger.info(
            f"{person:<10s} {auto_ms:8d} {gt_ms:8d} {delta_ms:+8d} {delta_pct:+6.1f}% {status}"
        )


if __name__ == "__main__":
    main()
