"""
Detection sensitivity sweep for RHOBH-TEST-10-28.

Tests different detection thresholds to improve recall for
underperforming cast members (YOLANDA, RINNA, BRANDI, EILEEN).
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yaml

from app.lib.data import load_clusters, load_tracks
from jobs.tasks.analytics import analytics_task

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ground_truth() -> dict[str, int]:
    """Load ground truth timings."""
    return {
        "KIM": 48004,
        "KYLE": 21017,
        "RINNA": 25015,
        "EILEEN": 10001,
        "BRANDI": 10014,
        "YOLANDA": 16002,
        "LVP": 2018,
    }


def run_sweep_combo(
    episode_id: str,
    min_confidence: float,
    min_face_px: int,
    data_root: Path,
) -> dict:
    """
    Run detection with specific threshold combo.

    Args:
        episode_id: Episode ID
        min_confidence: Detection confidence threshold
        min_face_px: Minimum face pixel threshold
        data_root: Data root path

    Returns:
        Dict with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: min_conf={min_confidence}, min_face_px={min_face_px}")
    logger.info(f"{'='*60}")

    # Temporarily update config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    original_conf = config["detection"]["min_confidence"]
    original_px = config["video"]["min_face_px"]

    config["detection"]["min_confidence"] = min_confidence
    config["video"]["min_face_px"] = min_face_px

    # Save temporarily
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    try:
        # Run detection + tracking + clustering
        from jobs.tasks.detect_embed import detect_embed_task
        from jobs.tasks.track import track_task
        from jobs.tasks.cluster import cluster_task

        job_id = f"sweep_{int(time.time())}"

        # Detect
        logger.info("Running detection...")
        detect_result = detect_embed_task(job_id, episode_id)
        faces_detected = detect_result["stats"].get("faces_detected", 0)

        # Track
        logger.info("Running tracking...")
        track_result = track_task(job_id, episode_id)
        tracks_built = track_result["stats"].get("tracks_built", 0)

        # Cluster
        logger.info("Running clustering...")
        cluster_result = cluster_task(job_id, episode_id)
        clusters_built = cluster_result["stats"].get("clusters_built", 0)

        # Analytics
        logger.info("Running analytics...")
        clusters_data = load_clusters(episode_id, data_root)
        cluster_assignments = {
            c["cluster_id"]: c["name"]
            for c in clusters_data.get("clusters", [])
            if "name" in c
        }

        analytics_result = analytics_task(job_id, episode_id, cluster_assignments)

        # Load results
        totals_path = Path(analytics_result["totals_path"])
        totals_df = pd.read_csv(totals_path)

        # Build results
        result = {
            "min_confidence": min_confidence,
            "min_face_px": min_face_px,
            "faces_detected": faces_detected,
            "tracks_built": tracks_built,
            "clusters_built": clusters_built,
        }

        # Add per-person totals
        for _, row in totals_df.iterrows():
            person = row["person_name"]
            result[f"auto_ms_{person}"] = int(row["total_ms"])

        # Add deltas vs GT
        gt_totals = load_ground_truth()
        for person, gt_ms in gt_totals.items():
            auto_ms = result.get(f"auto_ms_{person}", 0)
            result[f"delta_ms_{person}"] = gt_ms - auto_ms
            result[f"delta_pct_{person}"] = round((gt_ms - auto_ms) / gt_ms * 100, 1)

        return result

    finally:
        # Restore original config
        config["detection"]["min_confidence"] = original_conf
        config["video"]["min_face_px"] = original_px

        with open(config_path, "w") as f:
            yaml.dump(config, f)


def main():
    """Run detection sensitivity sweep."""
    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")

    logger.info(f"Starting detection sensitivity sweep for {episode_id}")

    # Define sweep parameters
    confidence_levels = [0.70, 0.65, 0.60]
    face_px_levels = [80, 64, 50]

    results = []

    # Run baseline first (current settings)
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (current settings)")
    logger.info("=" * 60)

    baseline = run_sweep_combo(episode_id, 0.70, 80, data_root)
    results.append(baseline)

    # Run sweep (skip baseline combo)
    for min_conf in confidence_levels:
        for min_px in face_px_levels:
            if min_conf == 0.70 and min_px == 80:
                continue  # Skip baseline, already done

            result = run_sweep_combo(episode_id, min_conf, min_px, data_root)
            results.append(result)

    # Save results
    output_dir = data_root / "diagnostics" / "reports" / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_path = output_dir / "detect_sweep.csv"
    sweep_df = pd.DataFrame(results)
    sweep_df.to_csv(sweep_path, index=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Sweep complete! Results saved to: {sweep_path}")
    logger.info(f"{'='*60}")

    # Print summary
    logger.info("\n=== SWEEP SUMMARY ===\n")

    # Focus on underperforming cast
    focus_cast = ["YOLANDA", "RINNA", "BRANDI", "EILEEN"]

    for _, row in sweep_df.iterrows():
        logger.info(
            f"min_conf={row['min_confidence']:.2f}, min_px={int(row['min_face_px']):3d} | "
            f"Faces={int(row['faces_detected']):3d}, Tracks={int(row['tracks_built']):3d}"
        )

        for person in focus_cast:
            delta_col = f"delta_ms_{person}"
            pct_col = f"delta_pct_{person}"

            if delta_col in row:
                delta_ms = int(row[delta_col])
                delta_pct = row[pct_col]
                status = "✓" if abs(delta_ms) <= 2000 else "⚠" if abs(delta_ms) <= 5000 else "✗"

                logger.info(f"  {person:8s}: Δ={delta_ms:+6d}ms ({delta_pct:+5.1f}%) {status}")

        logger.info("")

    # Find best combo
    logger.info("\n=== BEST COMBINATION ===\n")

    # Score: minimize total absolute delta for focus cast
    def score_combo(row):
        total_delta = 0
        for person in focus_cast:
            delta_col = f"delta_ms_{person}"
            if delta_col in row:
                total_delta += abs(row[delta_col])
        return total_delta

    sweep_df["score"] = sweep_df.apply(score_combo, axis=1)
    best_row = sweep_df.loc[sweep_df["score"].idxmin()]

    logger.info(
        f"Best: min_confidence={best_row['min_confidence']:.2f}, "
        f"min_face_px={int(best_row['min_face_px'])}"
    )
    logger.info(f"Total delta (focus cast): {int(best_row['score'])}ms")

    for person in focus_cast:
        delta_col = f"delta_ms_{person}"
        if delta_col in best_row:
            logger.info(
                f"  {person}: Δ={int(best_row[delta_col]):+6d}ms "
                f"({best_row[f'delta_pct_{person}']:+5.1f}%)"
            )


if __name__ == "__main__":
    main()
