#!/usr/bin/env python3
"""
Two-Pass Local Densify Task.

Pass 1: Conservative thresholds on all non-frozen identities
Pass 2: Aggressive thresholds only on identities still >4.5s error
"""

import json
import logging
from pathlib import Path
from typing import Dict
import pandas as pd
import yaml

from jobs.tasks.local_densify import local_densify_task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Ground truth values (ms)
GROUND_TRUTH_MS = {
    "KIM": 48_004,
    "KYLE": 21_017,
    "RINNA": 25_015,
    "EILEEN": 10_001,
    "BRANDI": 10_014,
    "YOLANDA": 16_002,
    "LVP": 2_018,
}


def load_config() -> dict:
    """Load pipeline configuration."""
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_clusters(episode_id: str, data_root: Path) -> dict:
    """Load clusters.json to get cluster assignments."""
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    # Build cluster_id -> name mapping
    cluster_assignments = {}
    for cluster in clusters_data.get("clusters", []):
        if "name" in cluster:
            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

    return cluster_assignments


def compute_current_delta(episode_id: str, data_root: Path, ground_truth: dict) -> dict:
    """
    Compute current delta (error) for each identity.

    Returns: {identity: delta_seconds}
    """
    # Load timeline to get current auto totals
    timeline_path = data_root / "outputs" / episode_id / "timeline.csv"
    if not timeline_path.exists():
        logger.warning(f"Timeline not found at {timeline_path}, using placeholder deltas")
        return {name: 10.0 for name in ground_truth.keys()}

    timeline_df = pd.read_csv(timeline_path)

    deltas = {}
    for identity, gt_ms in ground_truth.items():
        identity_timeline = timeline_df[timeline_df['person_name'] == identity]
        auto_ms = identity_timeline['duration_ms'].sum() if len(identity_timeline) > 0 else 0
        delta_s = (auto_ms - gt_ms) / 1000.0
        deltas[identity] = delta_s
        logger.info(f"{identity}: auto={auto_ms/1000:.2f}s, gt={gt_ms/1000:.2f}s, delta={delta_s:+.2f}s (error={abs(delta_s):.2f}s)")

    return deltas


def run_densify_pass(
    job_id: str,
    episode_id: str,
    video_path: Path,
    target_identities: list[str],
    cluster_assignments: dict,
    pass_name: str,
    config_overrides: dict = None
) -> dict:
    """
    Run densify with specific configuration.

    Args:
        job_id: Unique job identifier
        episode_id: Episode ID
        video_path: Path to video file
        target_identities: List of identities to densify
        cluster_assignments: Mapping of cluster_id -> name
        pass_name: "pass1" or "pass2"
        config_overrides: Optional config overrides for this pass

    Returns:
        Result dict from local_densify_task
    """
    logger.info(f"\n=== Starting {pass_name.upper()} ===")
    logger.info(f"Target identities: {target_identities}")

    # TODO: Apply config_overrides to densify detection params
    # For now, the local_densify_task will use values from pipeline.yaml

    result = local_densify_task(
        job_id=job_id,
        episode_id=episode_id,
        video_path=video_path,
        target_identities=target_identities,
        cluster_assignments=cluster_assignments,
    )

    logger.info(f"\n=== {pass_name.upper()} RESULTS ===")
    logger.info(f"Tracklets created: {result.get('tracklets_created', 0)}")
    logger.info(f"Segments scanned: {result.get('segments_scanned', 0)}")

    # Extract per-identity stats
    stats = result.get('stats', {})
    per_identity = stats.get('per_identity', {})

    for identity in target_identities:
        if identity in per_identity:
            id_stats = per_identity[identity]
            faces_verified = id_stats.get('faces_verified', 0)
            tracklets = id_stats.get('tracklets_accepted', 0)
            logger.info(f"  {identity}: {faces_verified} faces verified, {tracklets} tracklets")

    return result


def save_pass_audit(episode_id: str, data_root: Path, pass_name: str, result: dict):
    """Save densify pass audit report."""
    reports_dir = data_root / "harvest" / episode_id / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / f"densify_{pass_name}_audit.json"

    # Extract per-identity seconds_recovered
    audit = {
        "episode_id": episode_id,
        "pass": pass_name,
        "tracklets_created": result.get('tracklets_created', 0),
        "segments_scanned": result.get('segments_scanned', 0),
        "per_identity": {}
    }

    stats = result.get('stats', {})
    per_identity = stats.get('per_identity', {})

    for identity, id_stats in per_identity.items():
        # Calculate seconds_recovered from tracklets
        tracklets = id_stats.get('tracklet_summaries', [])
        total_duration_ms = sum(t.get('duration_ms', 0) for t in tracklets)
        seconds_recovered = total_duration_ms / 1000.0

        audit['per_identity'][identity] = {
            "seconds_recovered": round(seconds_recovered, 2),
            "faces_verified": id_stats.get('faces_verified', 0),
            "tracklets_accepted": id_stats.get('tracklets_accepted', 0),
            "tracklets_rejected": id_stats.get('tracklets_rejected', 0),
            "windows_scanned": id_stats.get('windows_scanned', 0)
        }

    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)

    logger.info(f"Saved {pass_name} audit to {audit_path}")


def main():
    """Run 2-pass densify on RHOBH-TEST-10-28."""

    # Configuration
    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")
    video_path = data_root / "videos" / f"{episode_id}.mp4"

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return

    # Load config
    config = load_config()
    frozen_identities = []
    for identity, overrides in config.get("timeline", {}).get("per_identity", {}).items():
        if overrides.get("freeze", False):
            frozen_identities.append(identity)

    logger.info(f"Frozen identities (will skip): {frozen_identities}")

    # Load cluster assignments
    cluster_assignments = load_clusters(episode_id, data_root)
    logger.info(f"Cluster assignments: {cluster_assignments}")

    # Determine all eligible identities (not frozen)
    all_identities = list(GROUND_TRUTH_MS.keys())
    pass1_targets = [i for i in all_identities if i not in frozen_identities]

    logger.info(f"\n=== TWO-PASS DENSIFY PIPELINE ===")
    logger.info(f"Episode: {episode_id}")
    logger.info(f"Frozen (skip): {frozen_identities}")
    logger.info(f"Pass 1 targets: {pass1_targets}")

    # ========================================
    # PASS 1: Conservative densify on all
    # ========================================

    pass1_result = run_densify_pass(
        job_id="densify_pass1",
        episode_id=episode_id,
        video_path=video_path,
        target_identities=pass1_targets,
        cluster_assignments=cluster_assignments,
        pass_name="pass1"
    )

    # Save pass1 audit
    save_pass_audit(episode_id, data_root, "pass1", pass1_result)

    # ========================================
    # Compute deltas after Pass 1
    # ========================================

    logger.info(f"\n=== Computing Delta After Pass 1 ===")
    deltas_after_pass1 = compute_current_delta(episode_id, data_root, GROUND_TRUTH_MS)

    # Determine Pass 2 targets (identities still >4.5s error)
    pass2_threshold = config.get("local_densify_pass2", {}).get("trigger_threshold", 4.5)
    pass2_targets = []

    for identity in pass1_targets:
        abs_error = abs(deltas_after_pass1.get(identity, 999))
        if abs_error > pass2_threshold:
            pass2_targets.append(identity)
            logger.info(f"  {identity}: {abs_error:.2f}s error > {pass2_threshold}s → TRIGGER PASS 2")
        else:
            logger.info(f"  {identity}: {abs_error:.2f}s error ≤ {pass2_threshold}s → PASS (skip pass 2)")

    if not pass2_targets:
        logger.info(f"\n✅ All identities ≤{pass2_threshold}s error after Pass 1, skipping Pass 2")
        logger.info(f"\n=== FINAL RESULTS ===")
        logger.info(f"Pass 1 tracklets: {pass1_result.get('tracklets_created', 0)}")
        logger.info(f"Pass 2 tracklets: 0 (not needed)")
        logger.info(f"Total tracklets: {pass1_result.get('tracklets_created', 0)}")
        return

    # ========================================
    # PASS 2: Aggressive densify on residuals
    # ========================================

    logger.info(f"\n=== Pass 2 targets (>{pass2_threshold}s error): {pass2_targets} ===")

    pass2_result = run_densify_pass(
        job_id="densify_pass2",
        episode_id=episode_id,
        video_path=video_path,
        target_identities=pass2_targets,
        cluster_assignments=cluster_assignments,
        pass_name="pass2"
    )

    # Save pass2 audit
    save_pass_audit(episode_id, data_root, "pass2", pass2_result)

    # ========================================
    # Final Summary
    # ========================================

    logger.info(f"\n=== FINAL RESULTS ===")
    logger.info(f"Pass 1 tracklets: {pass1_result.get('tracklets_created', 0)}")
    logger.info(f"Pass 2 tracklets: {pass2_result.get('tracklets_created', 0)}")
    logger.info(f"Total tracklets: {pass1_result.get('tracklets_created', 0) + pass2_result.get('tracklets_created', 0)}")

    logger.info(f"\n=== Final Delta (after both passes) ===")
    final_deltas = compute_current_delta(episode_id, data_root, GROUND_TRUTH_MS)

    pass_count = 0
    for identity, delta_s in final_deltas.items():
        abs_error = abs(delta_s)
        status = "✅ PASS" if abs_error <= 4.5 else "❌ FAIL"
        logger.info(f"  {identity}: {delta_s:+.2f}s ({abs_error:.2f}s error) {status}")
        if abs_error <= 4.5:
            pass_count += 1

    logger.info(f"\nPass rate: {pass_count}/{len(final_deltas)} ({pass_count/len(final_deltas)*100:.0f}%)")


if __name__ == "__main__":
    main()
