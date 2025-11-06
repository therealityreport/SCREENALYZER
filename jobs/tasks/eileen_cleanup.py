"""
EILEEN overcount cleanup - Auto-detect and remove contaminated spans.

Identifies track segments where EILEEN's identity confidence is weaker than
another cast member, then auto-splits and moves those segments to correct clusters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


def eileen_cleanup_task(
    job_id: str,
    episode_id: str,
    cluster_assignments: dict[int, str],
    min_contamination_margin: float = 0.10,
    min_contamination_frames: int = 5,
) -> dict:
    """
    Auto-cleanup EILEEN contamination by detecting and removing misattributed spans.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        cluster_assignments: Map of cluster_id -> person_name
        min_contamination_margin: Minimum margin for other identity to be considered contamination
        min_contamination_frames: Minimum consecutive frames to flag contamination

    Returns:
        Dict with cleanup results
    """
    logger.info(f"[{job_id}] Starting EILEEN contamination cleanup for {episode_id}")

    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    clusters_path = harvest_dir / "clusters.json"
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"

    with open(clusters_path) as f:
        clusters_data = json.load(f)
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    embeddings_df = pd.read_parquet(embeddings_path)

    # Find EILEEN cluster
    eileen_cluster_id = None
    for cluster_id, name in cluster_assignments.items():
        if name == "EILEEN":
            eileen_cluster_id = cluster_id
            break

    if not eileen_cluster_id:
        logger.warning("No EILEEN cluster found")
        return {"contaminated_spans": [], "net_correction_ms": 0}

    # Build person templates
    person_templates = _build_person_templates(
        clusters_data, tracks_data, embeddings_df, cluster_assignments
    )

    if "EILEEN" not in person_templates:
        logger.warning("No EILEEN template found")
        return {"contaminated_spans": [], "net_correction_ms": 0}

    # Find EILEEN cluster
    eileen_cluster = next(
        (c for c in clusters_data["clusters"] if c["cluster_id"] == eileen_cluster_id),
        None
    )

    if not eileen_cluster:
        logger.warning("EILEEN cluster not found")
        return {"contaminated_spans": [], "net_correction_ms": 0}

    # Check each track in EILEEN cluster for contamination
    contaminated_spans = []
    total_correction_ms = 0

    tracks_by_id = {t["track_id"]: t for t in tracks_data["tracks"]}

    for track_id in eileen_cluster["track_ids"][:]:  # Copy list since we'll modify
        track = tracks_by_id.get(track_id)
        if not track:
            continue

        # Check each frame in track
        frame_scores = []
        for frame_ref in track.get("frame_refs", []):
            frame_id = frame_ref["frame_id"]
            det_idx = frame_ref["det_idx"]

            # Find embedding
            match = embeddings_df[
                (embeddings_df["frame_id"] == frame_id)
                & (embeddings_df["det_idx"] == det_idx)
            ]

            if len(match) == 0:
                continue

            embedding = np.array(match.iloc[0]["embedding"])

            # Compare to all person templates
            similarities = {}
            for person_name, template in person_templates.items():
                sim = 1.0 - cosine(embedding, template)
                similarities[person_name] = sim

            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            best_person, best_sim = sorted_sims[0]
            eileen_sim = similarities.get("EILEEN", 0.0)

            # Calculate margin
            if best_person != "EILEEN":
                margin = best_sim - eileen_sim
                frame_scores.append({
                    "frame_id": frame_id,
                    "best_person": best_person,
                    "margin": margin,
                    "is_contaminated": margin >= min_contamination_margin,
                })

        # Find contaminated spans (consecutive frames)
        if not frame_scores:
            continue

        contaminated_count = sum(1 for fs in frame_scores if fs["is_contaminated"])
        contamination_pct = (contaminated_count / len(frame_scores)) * 100

        # Flag if significant contamination
        if contaminated_count >= min_contamination_frames or contamination_pct >= 25:
            # This track is contaminated
            best_other_person = max(
                set(fs["best_person"] for fs in frame_scores if fs["is_contaminated"]),
                key=lambda p: sum(1 for fs in frame_scores if fs.get("best_person") == p)
            )

            span_info = {
                "track_id": track_id,
                "start_ms": track["start_ms"],
                "end_ms": track["end_ms"],
                "duration_ms": track["duration_ms"],
                "contaminated_frames": contaminated_count,
                "total_frames": len(frame_scores),
                "contamination_pct": contamination_pct,
                "best_other_person": best_other_person,
            }

            contaminated_spans.append(span_info)
            total_correction_ms += track["duration_ms"]

            logger.info(
                f"[{job_id}] Found contaminated track {track_id}: "
                f"{contaminated_count}/{len(frame_scores)} frames contaminated with {best_other_person}"
            )

            # Remove from EILEEN cluster
            eileen_cluster["track_ids"].remove(track_id)
            eileen_cluster["size"] -= 1

            # Try to move to correct cluster
            target_cluster_id = next(
                (cid for cid, name in cluster_assignments.items() if name == best_other_person),
                None
            )

            if target_cluster_id:
                target_cluster = next(
                    (c for c in clusters_data["clusters"] if c["cluster_id"] == target_cluster_id),
                    None
                )

                if target_cluster:
                    target_cluster["track_ids"].append(track_id)
                    target_cluster["size"] += 1
                    logger.info(f"[{job_id}] Moved track {track_id} to {best_other_person}")

    # Save updated clusters
    with open(clusters_path, "w") as f:
        json.dump(clusters_data, f, indent=2)

    # Save overcount windows report
    if contaminated_spans:
        overcount_df = pd.DataFrame(contaminated_spans)
        overcount_path = reports_dir / "overcount_windows_EILEEN.csv"
        overcount_df.to_csv(overcount_path, index=False)
        logger.info(f"[{job_id}] Saved overcount report to {overcount_path}")

    logger.info(
        f"[{job_id}] EILEEN cleanup complete: "
        f"{len(contaminated_spans)} tracks moved, {total_correction_ms}ms corrected"
    )

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "contaminated_spans": contaminated_spans,
        "net_correction_ms": total_correction_ms,
        "tracks_moved": len(contaminated_spans),
    }


def _build_person_templates(
    clusters_data: dict,
    tracks_data: dict,
    embeddings_df: pd.DataFrame,
    cluster_assignments: dict[int, str],
) -> dict[str, np.ndarray]:
    """Build embedding templates for each named person."""
    templates = {}
    tracks_by_id = {t["track_id"]: t for t in tracks_data["tracks"]}

    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        person_name = cluster_assignments.get(cluster_id)

        if not person_name or person_name == "SKIP":
            continue

        # Collect all embeddings for this person's tracks
        person_embeddings = []
        for track_id in cluster["track_ids"]:
            track = tracks_by_id.get(track_id)
            if not track:
                continue

            for frame_ref in track.get("frame_refs", []):
                frame_id = frame_ref["frame_id"]
                det_idx = frame_ref["det_idx"]

                match = embeddings_df[
                    (embeddings_df["frame_id"] == frame_id)
                    & (embeddings_df["det_idx"] == det_idx)
                ]

                if len(match) > 0:
                    person_embeddings.append(np.array(match.iloc[0]["embedding"]))

        if person_embeddings:
            template = np.mean(person_embeddings, axis=0)
            template = template / (np.linalg.norm(template) + 1e-8)
            templates[person_name] = template
            logger.debug(f"Built template for {person_name} from {len(person_embeddings)} embeddings")

    return templates
