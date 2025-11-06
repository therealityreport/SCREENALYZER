"""
Auto-Caps Computation - Identity-Agnostic Timeline Merge Limits.

Computes safe merge caps per identity from episode data (no hardcoding).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_auto_caps(
    episode_id: str,
    data_root: Path,
    config: dict,
    tracks_data: dict,
    clusters_data: dict
) -> Dict[str, Dict]:
    """
    Compute auto-caps for all identities based on episode data.

    Args:
        episode_id: Episode ID
        data_root: Data root path
        config: Pipeline configuration
        tracks_data: Loaded tracks.json
        clusters_data: Loaded clusters.json

    Returns:
        Dictionary: {identity_name: {auto_cap_ms, safe_gap_count, stats}}
    """
    auto_caps_config = config.get("timeline", {}).get("auto_caps", {})

    if not auto_caps_config.get("enabled", False):
        logger.info("Auto-caps disabled in config")
        return {}

    safe_gap_percentile = auto_caps_config.get("safe_gap_percentile", 0.80)
    cap_min_ms = auto_caps_config.get("cap_min_ms", 1200)
    cap_max_ms = auto_caps_config.get("cap_max_ms", 2500)

    min_visible_frac = config.get("timeline", {}).get("min_visible_frac", 0.60)
    conflict_guard_ms = config.get("timeline", {}).get("conflict_guard_ms", 500)

    logger.info(f"Computing auto-caps: P{safe_gap_percentile*100:.0f} of safe gaps, range [{cap_min_ms}, {cap_max_ms}]ms")

    # Build identity -> tracks mapping
    identity_tracks = {}
    for cluster in clusters_data.get("clusters", []):
        if "name" not in cluster:
            continue
        identity = cluster["name"]
        track_ids = cluster.get("track_ids", [])
        identity_tracks[identity] = [
            t for t in tracks_data.get("tracks", []) if t["track_id"] in track_ids
        ]

    # Compute caps for each identity
    auto_caps_results = {}

    for identity, tracks in identity_tracks.items():
        if not tracks:
            continue

        # Sort tracks by start time
        tracks = sorted(tracks, key=lambda t: t.get("start_ms", 0))

        # Identify safe gaps between consecutive tracks
        safe_gaps = []

        for i in range(len(tracks) - 1):
            track_a = tracks[i]
            track_b = tracks[i + 1]

            gap_start_ms = track_a.get("end_ms", 0)
            gap_end_ms = track_b.get("start_ms", 0)
            gap_duration_ms = gap_end_ms - gap_start_ms

            if gap_duration_ms <= 0:
                continue  # No gap or overlap

            # Check if gap is "safe" based on criteria:
            # 1. Both sides have sufficient visible frames
            # 2. No strong conflict from other identities in the gap

            # Criterion 1: Visible fraction (simplified - use mean_confidence as proxy)
            track_a_quality = track_a.get("mean_confidence", 0.0)
            track_b_quality = track_b.get("mean_confidence", 0.0)

            # Simplified visible_frac check (would need frame-level data for exact)
            # Using mean_confidence ≥ 0.70 as proxy for visible_frac ≥ 0.60
            if track_a_quality < 0.70 or track_b_quality < 0.70:
                continue  # Low quality, skip this gap

            # Criterion 2: Conflict guard (check if other identities overlap gap)
            # Simplified: check if any other identity has tracks overlapping [gap_start - guard, gap_end + guard]
            conflict_window = (gap_start_ms - conflict_guard_ms, gap_end_ms + conflict_guard_ms)
            has_conflict = False

            for other_identity, other_tracks in identity_tracks.items():
                if other_identity == identity:
                    continue

                for other_track in other_tracks:
                    other_start = other_track.get("start_ms", 0)
                    other_end = other_track.get("end_ms", 0)

                    # Check overlap with conflict window
                    if not (other_end < conflict_window[0] or other_start > conflict_window[1]):
                        has_conflict = True
                        break

                if has_conflict:
                    break

            if has_conflict:
                continue  # Other identity present in conflict window, skip

            # This is a safe gap
            safe_gaps.append(gap_duration_ms)

        # Compute auto-cap from safe gaps
        if not safe_gaps:
            # No safe gaps found, use default
            auto_cap_ms = cap_min_ms
            logger.warning(f"{identity}: No safe gaps found, using min_cap={cap_min_ms}ms")
        else:
            # Compute P80 (or configured percentile)
            p_value = np.percentile(safe_gaps, safe_gap_percentile * 100)
            # Apply 1.2x multiplier and clamp to [cap_min, cap_max]
            auto_cap_ms = int(np.clip(p_value * 1.2, cap_min_ms, cap_max_ms))
            logger.info(f"{identity}: P{safe_gap_percentile*100:.0f}={p_value:.0f}ms → auto_cap={auto_cap_ms}ms (from {len(safe_gaps)} safe gaps)")

        auto_caps_results[identity] = {
            "auto_cap_ms": auto_cap_ms,
            "safe_gap_count": len(safe_gaps),
            "safe_gap_p80": int(np.percentile(safe_gaps, 80)) if safe_gaps else 0,
            "safe_gap_median": int(np.median(safe_gaps)) if safe_gaps else 0,
            "safe_gap_mean": int(np.mean(safe_gaps)) if safe_gaps else 0,
        }

    return auto_caps_results


def save_auto_caps(episode_id: str, data_root: Path, auto_caps: Dict[str, Dict]):
    """Save auto-caps telemetry to JSON."""
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "per_identity_caps.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "episode_id": episode_id,
        "auto_caps": auto_caps
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved auto-caps to {output_path}")


def load_auto_caps(episode_id: str, data_root: Path) -> Optional[Dict[str, Dict]]:
    """Load auto-caps from JSON if available."""
    caps_path = data_root / "harvest" / episode_id / "diagnostics" / "per_identity_caps.json"

    if not caps_path.exists():
        return None

    with open(caps_path) as f:
        data = json.load(f)

    return data.get("auto_caps", {})
