"""
Auto-Recall: Identity-agnostic under-recall recovery using season bank proximity.

Automatically fills gaps using "largest gaps first" policy, gated by season bank
proximity. No ground truth or per-person tuning required.

Budget-limited densify runs verify candidates using season bank before creating tracklets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import json
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class GapCandidate:
    """A gap candidate for densify/recall recovery."""
    identity_name: str
    gap_start_ms: int
    gap_end_ms: int
    gap_duration_ms: int
    bank_proximity_score: float  # Max similarity to season bank at boundaries
    priority_score: float  # gap_duration_ms × bank_proximity_score
    boundary_track_ids: Tuple[int, int]  # (before_track_id, after_track_id)


@dataclass
class AutoRecallConfig:
    """Identity-agnostic configuration for under-recall recovery."""
    # Gap ranking
    max_gaps_per_episode: int = 4  # Budget: top K gaps total
    max_gaps_per_identity: int = 1  # Max gaps per identity
    min_gap_duration_ms: int = 1000  # Only consider gaps ≥ 1 second

    # Season bank proximity thresholds
    min_bank_proximity: float = 0.60  # Skip identities with proximity < 0.60
    bank_probe_window_frames: int = 5  # Check last 5 frames before gap, first 5 after

    # Densify parameters
    max_window_duration_s: float = 10.0  # Each densify window ≤ 10 seconds
    densify_fps: int = 30  # Sample at 30 fps
    densify_min_conf: float = 0.58  # Pass-1 threshold
    densify_min_face_px: int = 44  # Pass-1 threshold

    # Verification thresholds (season bank)
    verify_min_sim: float = 0.86  # Sim to season bank ≥ 0.86
    verify_min_margin: float = 0.12  # Margin over next-best ≥ 0.12
    verify_min_consecutive_frames: int = 4  # ≥4 consecutive frames to birth tracklet

    # Output
    log_per_identity: bool = True  # Log stats per identity


def compute_bank_proximity(
    track_embeddings: List[np.ndarray],
    season_bank: dict,
    identity_name: str
) -> float:
    """
    Compute max similarity between track embeddings and season bank prototypes.

    Args:
        track_embeddings: List of embeddings from track boundary
        season_bank: Season bank dict
        identity_name: Identity to check

    Returns:
        Max similarity score
    """
    if not season_bank or not track_embeddings:
        return 0.0

    identities = season_bank.get('identities', {})
    if identity_name not in identities:
        return 0.0

    identity_data = identities[identity_name]
    prototypes = identity_data.get('prototypes', {})

    if not prototypes:
        return 0.0

    max_sim = 0.0
    for track_emb in track_embeddings:
        for proto_data in prototypes.values():
            proto_emb = np.array(proto_data['embedding'], dtype=np.float32)
            sim = np.dot(track_emb, proto_emb)
            max_sim = max(max_sim, sim)

    return max_sim


def find_gap_candidates(
    clusters_data: dict,
    tracks_data: dict,
    picked_samples_df,
    season_bank: dict,
    config: AutoRecallConfig
) -> List[GapCandidate]:
    """
    Find gap candidates ranked by (gap_duration × bank_proximity).

    For each named identity:
    1. Find temporal gaps between tracks
    2. Compute bank proximity at gap boundaries
    3. Rank by priority score
    """
    candidates = []

    # Group clusters by identity
    identity_clusters = {}
    for cluster in clusters_data.get('clusters', []):
        name = cluster.get('name', 'Unknown')
        if name == 'Unknown':
            continue  # Skip unknown

        if name not in identity_clusters:
            identity_clusters[name] = []
        identity_clusters[name].append(cluster)

    # For each identity, find gaps
    for identity_name, clusters in identity_clusters.items():
        # Collect all tracks for this identity (across all clusters)
        all_track_ids = []
        for cluster in clusters:
            all_track_ids.extend(cluster.get('track_ids', []))

        if len(all_track_ids) < 2:
            continue  # Need at least 2 tracks to have gaps

        # Get temporal ranges for each track
        track_ranges = []
        for track_id in all_track_ids:
            track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
            if not track:
                continue

            frame_refs = track.get('frame_refs', [])
            if not frame_refs:
                continue

            start_frame = frame_refs[0]['frame_id']
            end_frame = frame_refs[-1]['frame_id']
            track_ranges.append((start_frame, end_frame, track_id))

        if len(track_ranges) < 2:
            continue

        # Sort by start frame
        track_ranges.sort()

        # Find gaps between consecutive tracks
        for i in range(len(track_ranges) - 1):
            gap_start_frame = track_ranges[i][1]  # End of current track
            gap_end_frame = track_ranges[i+1][0]  # Start of next track
            gap_duration_ms = (gap_end_frame - gap_start_frame) * 33  # ~30fps

            if gap_duration_ms < config.min_gap_duration_ms:
                continue  # Gap too short

            # Compute bank proximity at gap boundaries
            # Get embeddings from last N frames before gap
            before_track_id = track_ranges[i][2]
            after_track_id = track_ranges[i+1][2]

            before_samples = picked_samples_df[
                (picked_samples_df['track_id'] == before_track_id) &
                (picked_samples_df['frame_id'] >= gap_start_frame - config.bank_probe_window_frames)
            ]

            after_samples = picked_samples_df[
                (picked_samples_df['track_id'] == after_track_id) &
                (picked_samples_df['frame_id'] <= gap_end_frame + config.bank_probe_window_frames)
            ]

            # Collect embeddings
            boundary_embeddings = []
            for idx, row in before_samples.iterrows():
                boundary_embeddings.append(np.array(row['embedding'], dtype=np.float32))
            for idx, row in after_samples.iterrows():
                boundary_embeddings.append(np.array(row['embedding'], dtype=np.float32))

            if not boundary_embeddings:
                continue  # No embeddings at boundaries

            # Compute bank proximity
            bank_proximity = compute_bank_proximity(boundary_embeddings, season_bank, identity_name)

            if bank_proximity < config.min_bank_proximity:
                continue  # Proximity too low

            # Compute priority score
            priority_score = gap_duration_ms * bank_proximity

            candidates.append(GapCandidate(
                identity_name=identity_name,
                gap_start_ms=gap_start_frame * 33,
                gap_end_ms=gap_end_frame * 33,
                gap_duration_ms=gap_duration_ms,
                bank_proximity_score=bank_proximity,
                priority_score=priority_score,
                boundary_track_ids=(before_track_id, after_track_id)
            ))

    # Sort by priority score (descending)
    candidates.sort(key=lambda c: c.priority_score, reverse=True)

    logger.info(f"Found {len(candidates)} gap candidates across {len(identity_clusters)} identities")

    return candidates


def select_gaps_for_densify(
    candidates: List[GapCandidate],
    config: AutoRecallConfig
) -> List[GapCandidate]:
    """
    Select top K gaps for densify, respecting budget constraints.

    Returns:
        List of gaps to process (≤ max_gaps_per_episode, ≤ max_gaps_per_identity per identity)
    """
    selected = []
    identity_counts = {}

    for candidate in candidates:
        # Check identity budget
        identity_count = identity_counts.get(candidate.identity_name, 0)
        if identity_count >= config.max_gaps_per_identity:
            continue  # Already processed max for this identity

        # Check total budget
        if len(selected) >= config.max_gaps_per_episode:
            break  # Reached total budget

        # Select this gap
        selected.append(candidate)
        identity_counts[candidate.identity_name] = identity_count + 1

    logger.info(f"Selected {len(selected)} gaps for densify (budget: {config.max_gaps_per_episode})")

    return selected


def apply_auto_recall(
    clusters_data: dict,
    tracks_data: dict,
    picked_samples_df,
    season_bank: dict,
    config: AutoRecallConfig = None
) -> Tuple[dict, dict]:
    """
    Apply auto-recall policies to find and rank gaps for densify.

    NOTE: This function only IDENTIFIES gaps. Actual densify runs would be
    executed separately (requires video processing).

    Returns:
        Updated clusters_data (for now, unchanged - densify runs happen separately)
        Recall stats dict (for logging)
    """
    if config is None:
        config = AutoRecallConfig()

    if not season_bank:
        logger.warning("No season bank available - skipping Auto-Recall")
        return clusters_data, {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'status': 'skipped',
            'reason': 'no_season_bank'
        }

    logger.info("Running Auto-Recall gap analysis...")

    # Find gap candidates
    candidates = find_gap_candidates(clusters_data, tracks_data, picked_samples_df, season_bank, config)

    # Select gaps for densify (budget-limited)
    selected_gaps = select_gaps_for_densify(candidates, config)

    # Build stats
    stats = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'status': 'completed',
        'total_candidates_found': len(candidates),
        'gaps_selected_for_densify': len(selected_gaps),
        'total_seconds_to_recover': sum(g.gap_duration_ms for g in selected_gaps) / 1000.0,
        'gaps_by_identity': {},
        'selected_gaps': []
    }

    # Group by identity
    for gap in selected_gaps:
        if gap.identity_name not in stats['gaps_by_identity']:
            stats['gaps_by_identity'][gap.identity_name] = {
                'count': 0,
                'total_duration_ms': 0
            }

        stats['gaps_by_identity'][gap.identity_name]['count'] += 1
        stats['gaps_by_identity'][gap.identity_name]['total_duration_ms'] += gap.gap_duration_ms

        stats['selected_gaps'].append({
            'identity_name': gap.identity_name,
            'gap_start_ms': gap.gap_start_ms,
            'gap_end_ms': gap.gap_end_ms,
            'gap_duration_ms': gap.gap_duration_ms,
            'bank_proximity_score': gap.bank_proximity_score,
            'priority_score': gap.priority_score
        })

    logger.info(f"Auto-Recall summary: {len(selected_gaps)} gaps selected, {stats['total_seconds_to_recover']:.1f}s to recover")

    return clusters_data, stats
