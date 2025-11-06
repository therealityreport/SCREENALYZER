"""
Bootstrap cluster labels from known-good baseline using embedding similarity.

Eliminates unreliable count-based autolabeling by matching track centroids
to baseline person templates via cosine similarity with strict margins.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

from screentime.diagnostics.telemetry import telemetry, TelemetryEvent

logger = logging.getLogger(__name__)


def bootstrap_labels_task(
    job_id: str,
    episode_id: str,
    baseline_episode_id: str | None = None,
    min_sim: float = 0.82,
    min_margin: float = 0.08,
    min_frames: int = 5,
) -> dict:
    """
    Bootstrap cluster labels from baseline using embedding similarity.

    Args:
        job_id: Job ID
        episode_id: Current episode to label
        baseline_episode_id: Baseline episode with known-good labels (defaults to same episode)
        min_sim: Minimum similarity to baseline person
        min_margin: Minimum margin over second-best match
        min_frames: Minimum frames in track to consider for bootstrap

    Returns:
        Dict with bootstrap results
    """
    logger.info(f"[{job_id}] Starting bootstrap labels for {episode_id}")

    baseline_ep = baseline_episode_id or episode_id
    data_root = Path("data")

    # Load current state
    current_harvest = data_root / "harvest" / episode_id
    current_clusters_path = current_harvest / "clusters.json"
    current_tracks_path = current_harvest / "tracks.json"
    current_embeddings_path = current_harvest / "embeddings.parquet"

    # Load baseline state
    baseline_harvest = data_root / "harvest" / baseline_ep
    baseline_clusters_path = baseline_harvest / "clusters.json"
    baseline_tracks_path = baseline_harvest / "tracks.json"
    baseline_embeddings_path = baseline_harvest / "embeddings.parquet"

    if not baseline_clusters_path.exists():
        raise ValueError(f"Baseline clusters not found: {baseline_clusters_path}")
    if not baseline_tracks_path.exists():
        raise ValueError(f"Baseline tracks not found: {baseline_tracks_path}")
    if not baseline_embeddings_path.exists():
        raise ValueError(f"Baseline embeddings not found: {baseline_embeddings_path}")

    # Load data
    with open(current_clusters_path, "r") as f:
        current_clusters_data = json.load(f)
    with open(current_tracks_path, "r") as f:
        current_tracks_data = json.load(f)
    with open(baseline_clusters_path, "r") as f:
        baseline_clusters_data = json.load(f)
    with open(baseline_tracks_path, "r") as f:
        baseline_tracks_data = json.load(f)

    current_embeddings_df = pd.read_parquet(current_embeddings_path)
    baseline_embeddings_df = pd.read_parquet(baseline_embeddings_path)

    logger.info(f"[{job_id}] Loaded {len(current_clusters_data['clusters'])} current clusters")
    logger.info(f"[{job_id}] Loaded {len(baseline_clusters_data['clusters'])} baseline clusters")

    # Build baseline person templates (named clusters only)
    baseline_templates = _build_baseline_templates(
        baseline_clusters_data, baseline_embeddings_df, baseline_tracks_data
    )

    if not baseline_templates:
        logger.warning(f"[{job_id}] No named baseline clusters found - skipping bootstrap")
        return {
            "job_id": job_id,
            "episode_id": episode_id,
            "bootstrap_assigned_count": 0,
            "autolink_candidates_count": 0,
        }

    logger.info(f"[{job_id}] Built templates for {len(baseline_templates)} people")

    # Compute track centroids for current episode
    current_track_centroids = _compute_track_centroids(
        current_tracks_data, current_embeddings_df
    )

    logger.info(f"[{job_id}] Computed centroids for {len(current_track_centroids)} tracks")

    # Match clusters to baseline templates
    assignments = {}  # cluster_id -> person_name
    autolink_candidates = []  # Ambiguous matches for manual review

    for cluster in current_clusters_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        track_ids = cluster["track_ids"]

        # Skip if already named
        if cluster.get("name"):
            logger.debug(f"Cluster {cluster_id} already named: {cluster['name']}")
            continue

        # Compute cluster centroid from track centroids
        track_centroids = [
            current_track_centroids[tid]
            for tid in track_ids
            if tid in current_track_centroids
        ]

        if not track_centroids:
            continue

        # Filter by minimum frames
        high_quality_centroids = [
            tc for tc in track_centroids if tc["frame_count"] >= min_frames
        ]

        if not high_quality_centroids:
            logger.debug(f"Cluster {cluster_id}: No tracks with >={min_frames} frames")
            continue

        # Average embeddings to get cluster centroid
        cluster_embedding = np.mean(
            [tc["embedding"] for tc in high_quality_centroids], axis=0
        )
        cluster_embedding = cluster_embedding / (np.linalg.norm(cluster_embedding) + 1e-8)

        # Compare to all baseline templates
        similarities = {}
        for person_name, template in baseline_templates.items():
            sim = 1.0 - cosine(cluster_embedding, template)
            similarities[person_name] = sim

        # Sort by similarity
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        if not sorted_sims:
            continue

        best_name, best_sim = sorted_sims[0]

        # Check minimum similarity
        if best_sim < min_sim:
            logger.debug(
                f"Cluster {cluster_id}: Best match {best_name} too low (sim={best_sim:.3f} < {min_sim})"
            )
            continue

        # Check margin over second-best
        if len(sorted_sims) > 1:
            second_sim = sorted_sims[1][1]
            margin = best_sim - second_sim

            if margin < min_margin:
                # Ambiguous match - add to autolink candidates
                autolink_candidates.append(
                    {
                        "cluster_id": cluster_id,
                        "track_count": len(track_ids),
                        "best_match": best_name,
                        "best_sim": float(best_sim),
                        "second_match": sorted_sims[1][0],
                        "second_sim": float(second_sim),
                        "margin": float(margin),
                    }
                )
                logger.debug(
                    f"Cluster {cluster_id}: Ambiguous match - {best_name} vs {sorted_sims[1][0]} "
                    f"(margin={margin:.3f} < {min_margin})"
                )
                continue

        # Assign cluster to person
        assignments[cluster_id] = best_name
        logger.info(
            f"Cluster {cluster_id} -> {best_name} (sim={best_sim:.3f}, "
            f"{len(high_quality_centroids)} tracks, "
            f"{sum(tc['frame_count'] for tc in high_quality_centroids)} frames)"
        )

    # Apply assignments to clusters
    for cluster in current_clusters_data["clusters"]:
        if cluster["cluster_id"] in assignments:
            cluster["name"] = assignments[cluster["cluster_id"]]
            cluster["label_source"] = "bootstrap"

    # Save updated clusters
    with open(current_clusters_path, "w") as f:
        json.dump(current_clusters_data, f, indent=2)

    logger.info(f"[{job_id}] Assigned {len(assignments)} clusters via bootstrap")

    # Save autolink candidates
    if autolink_candidates:
        assist_dir = current_harvest / "assist"
        assist_dir.mkdir(parents=True, exist_ok=True)
        autolink_path = assist_dir / "autolink_candidates.parquet"
        autolink_df = pd.DataFrame(autolink_candidates)
        autolink_df.to_parquet(autolink_path, index=False)
        logger.info(
            f"[{job_id}] Saved {len(autolink_candidates)} autolink candidates to {autolink_path}"
        )

    # Telemetry
    telemetry.log(
        TelemetryEvent.JOB_STAGE_COMPLETE,
        metadata={
            "job_id": job_id,
            "stage": "bootstrap_labels",
            "bootstrap_assigned_count": len(assignments),
            "autolink_candidates_count": len(autolink_candidates),
        },
    )

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "bootstrap_assigned_count": len(assignments),
        "autolink_candidates_count": len(autolink_candidates),
        "assignments": assignments,
    }


def _build_baseline_templates(
    clusters_data: dict, embeddings_df: pd.DataFrame, tracks_data: dict | None = None
) -> dict[str, np.ndarray]:
    """
    Build person templates from baseline clusters.

    Args:
        clusters_data: Baseline clusters data
        embeddings_df: Baseline embeddings DataFrame
        tracks_data: Optional tracks data for mapping embeddings to clusters

    Returns:
        Dict mapping person_name -> template_embedding
    """
    templates = {}

    # Build track_id -> frame_refs mapping if tracks provided
    track_frames = {}
    if tracks_data:
        for track in tracks_data.get("tracks", []):
            track_id = track["track_id"]
            track_frames[track_id] = track.get("frame_refs", [])

    for cluster in clusters_data["clusters"]:
        person_name = cluster.get("name")
        if not person_name or person_name == "SKIP":
            continue

        track_ids = set(cluster.get("track_ids", []))

        # Get all embeddings for tracks in this cluster
        cluster_embeddings = []

        if track_frames:
            # Use track mapping to filter embeddings
            for track_id in track_ids:
                frame_refs = track_frames.get(track_id, [])
                for frame_ref in frame_refs:
                    frame_id = frame_ref["frame_id"]
                    det_idx = frame_ref["det_idx"]

                    # Find matching embedding
                    match = embeddings_df[
                        (embeddings_df["frame_id"] == frame_id)
                        & (embeddings_df["det_idx"] == det_idx)
                    ]

                    if len(match) > 0:
                        cluster_embeddings.append(np.array(match.iloc[0]["embedding"]))
        else:
            # Fallback: use all embeddings (less accurate)
            logger.warning(f"No tracks provided - using all embeddings for {person_name}")
            cluster_embeddings = [np.array(row["embedding"]) for _, row in embeddings_df.iterrows()]

        if not cluster_embeddings:
            logger.warning(f"No embeddings found for baseline cluster {person_name}")
            continue

        # Compute mean embedding as template
        template = np.mean(cluster_embeddings, axis=0)
        template = template / (np.linalg.norm(template) + 1e-8)

        templates[person_name] = template
        logger.debug(
            f"Built template for {person_name} from {len(cluster_embeddings)} embeddings"
        )

    return templates


def _compute_track_centroids(
    tracks_data: dict, embeddings_df: pd.DataFrame
) -> dict[int, dict]:
    """
    Compute centroid embedding for each track.

    Args:
        tracks_data: Tracks data
        embeddings_df: Embeddings DataFrame

    Returns:
        Dict mapping track_id -> {embedding, frame_count}
    """
    centroids = {}

    for track in tracks_data["tracks"]:
        track_id = track["track_id"]
        frame_refs = track.get("frame_refs", [])

        if not frame_refs:
            continue

        # Collect embeddings for this track
        track_embeddings = []
        for frame_ref in frame_refs:
            frame_id = frame_ref["frame_id"]
            det_idx = frame_ref["det_idx"]

            # Find matching embedding
            match = embeddings_df[
                (embeddings_df["frame_id"] == frame_id) & (embeddings_df["det_idx"] == det_idx)
            ]

            if len(match) > 0:
                track_embeddings.append(np.array(match.iloc[0]["embedding"]))

        if not track_embeddings:
            continue

        # Compute mean embedding
        centroid = np.mean(track_embeddings, axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-8)

        centroids[track_id] = {"embedding": centroid, "frame_count": len(track_embeddings)}

    return centroids
