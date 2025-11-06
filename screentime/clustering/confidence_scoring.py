"""
Frame-level and track-level confidence scoring.

Computes identity similarity scores for individual frames and robust
track-level aggregates for sorting/filtering low-confidence tracks.
Also surfaces secondary metrics (conflict fraction, intra-track variance)
required for Workspace confidence queues.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize vector with numerical guard."""
    return vec / (np.linalg.norm(vec) + 1e-9)


def _build_prototype_lookup(season_bank: Dict) -> Tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Build lookup tables for season bank prototypes.

    Returns:
        proto_matrix: Array of shape (N, 512) with normalized prototypes.
        proto_identity: Array of identity names per prototype row.
        identity_to_indices: Mapping of identity -> prototype row indices.
    """
    identities = season_bank.get("identities", {}) or {}

    proto_rows: List[np.ndarray] = []
    proto_identity: List[str] = []
    identity_to_indices: dict[str, np.ndarray] = {}

    for identity, bins in identities.items():
        indices: List[int] = []
        for proto_list in bins.values():
            for proto in proto_list:
                emb = np.asarray(proto.get("embedding", []), dtype=np.float32)
                if emb.size == 0:
                    continue
                proto_rows.append(_normalize(emb))
                indices.append(len(proto_rows) - 1)
                proto_identity.append(identity)

        if indices:
            identity_to_indices[identity] = np.asarray(indices, dtype=np.int32)

    if proto_rows:
        proto_matrix = np.vstack(proto_rows)
        proto_identity_arr = np.asarray(proto_identity, dtype=object)
    else:
        proto_matrix = np.empty((0, 512), dtype=np.float32)
        proto_identity_arr = np.empty((0,), dtype=object)

    return proto_matrix, proto_identity_arr, identity_to_indices


def score_all_tracks_and_frames(
    picked_samples_df: pd.DataFrame,
    clusters_data: Dict,
    season_bank: Dict,
    thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Score all frames and compute track-level confidence metrics.

    Args:
        picked_samples_df: DataFrame with picked samples (must have 'embedding', 'track_id' columns)
        clusters_data: Clusters dict with assignments
        season_bank: Season bank with identity prototypes
        thresholds: Optional overrides for confidence thresholds

    Returns:
        Tuple of:
            - Updated DataFrame with frame-level diagnostics columns
            - Dict mapping track_id -> track confidence metrics
    """
    thresholds = thresholds or {}
    frame_low = float(thresholds.get("frame_low", 0.55))
    top2_margin_low = float(thresholds.get("top2_margin_low", 0.08))

    # Build track_id -> (cluster_id, identity_name, centroid) mapping
    track_to_cluster: dict[int, tuple[Optional[int], str, Optional[np.ndarray]]] = {}

    for cluster in clusters_data.get("clusters", []):
        cluster_id = int(cluster["cluster_id"])
        identity_name = cluster.get("name") or "Unknown"
        centroid = cluster.get("centroid")
        centroid_vec = _normalize(np.asarray(centroid, dtype=np.float32)) if centroid is not None else None

        for track_id in cluster.get("track_ids", []):
            track_to_cluster[int(track_id)] = (cluster_id, identity_name, centroid_vec)

    logger.info(
        "Scoring confidence for %s frames across %s tracks",
        len(picked_samples_df),
        len(track_to_cluster),
    )

    proto_matrix, proto_identity, identity_to_indices = _build_prototype_lookup(season_bank)

    # Accumulate per-frame diagnostics
    frame_conf_vals: List[float] = []
    frame_margin_vals: List[float] = []
    frame_conflict_flags: List[bool] = []
    frame_low_flags: List[bool] = []

    # Accumulate per-track aggregates
    per_track_data: dict[int, dict[str, object]] = {}

    for _, row in picked_samples_df.iterrows():
        track_id = int(row["track_id"])
        embedding = np.asarray(row["embedding"], dtype=np.float32)
        if embedding.size == 0:
            # Defensive guard — treat as zero-confidence sample
            embedding = np.zeros(512, dtype=np.float32)
        embedding_norm = _normalize(embedding)

        cluster_id, identity_name, centroid_vec = track_to_cluster.get(
            track_id, (None, "Unknown", None)
        )

        assigned_indices = identity_to_indices.get(identity_name)

        # Compute similarity to assigned identity (if available)
        best_sim = 0.0
        if assigned_indices is not None and assigned_indices.size > 0:
            sims = proto_matrix[assigned_indices] @ embedding_norm
            best_sim = float(sims.max())
        elif centroid_vec is not None:
            best_sim = float(np.dot(centroid_vec, embedding_norm))

        # Compute best competing identity similarity
        has_other = False
        best_other = 0.0
        if proto_matrix.size > 0:
            sims_all = proto_matrix @ embedding_norm
            if assigned_indices is not None and assigned_indices.size > 0:
                mask = np.ones(sims_all.shape[0], dtype=bool)
                mask[assigned_indices] = False
                if mask.any():
                    best_other = float(sims_all[mask].max())
                    has_other = True
            else:
                best_other = float(sims_all.max())
                has_other = sims_all.size > 0

        margin = best_sim - best_other
        is_low_frame = best_sim < frame_low
        is_conflict = has_other and (margin < top2_margin_low)

        frame_conf_vals.append(best_sim)
        frame_margin_vals.append(margin)
        frame_conflict_flags.append(bool(is_conflict))
        frame_low_flags.append(bool(is_low_frame))

        track_entry = per_track_data.setdefault(
            track_id,
            {
                "conf": [],
                "margins": [],
                "conflict_count": 0,
                "low_count": 0,
                "identity": identity_name,
                "cluster_id": cluster_id,
            },
        )
        track_entry["conf"].append(best_sim)
        track_entry["margins"].append(margin)
        if is_conflict:
            track_entry["conflict_count"] += 1
        if is_low_frame:
            track_entry["low_count"] += 1

    picked_samples_df = picked_samples_df.copy()
    picked_samples_df["frame_conf"] = frame_conf_vals
    picked_samples_df["frame_margin"] = frame_margin_vals
    picked_samples_df["frame_conflict"] = frame_conflict_flags
    picked_samples_df["frame_low"] = frame_low_flags

    track_metrics: Dict[int, Dict[str, float]] = {}

    # Ensure every clustered track gets a metrics entry, even if it had zero picked samples
    for track_id, (cluster_id, identity_name, _) in track_to_cluster.items():
        entry = per_track_data.get(
            track_id,
            {
                "conf": [],
                "margins": [],
                "conflict_count": 0,
                "low_count": 0,
                "identity": identity_name,
                "cluster_id": cluster_id,
            },
        )
        conf_values = np.asarray(entry["conf"], dtype=np.float32)
        margin_values = np.asarray(entry["margins"], dtype=np.float32)
        n_frames = int(conf_values.size)

        if n_frames > 0:
            conf_p25 = float(np.percentile(conf_values, 25))
            conf_mean = float(np.mean(conf_values))
            conf_min = float(np.min(conf_values))
            intra_var = float(np.var(conf_values))
            avg_margin = float(np.mean(margin_values))
            margin_p25 = float(np.percentile(margin_values, 25))
            conflict_frac = float(entry["conflict_count"] / n_frames)
            n_low = int(entry["low_count"])
        else:
            conf_p25 = conf_mean = conf_min = 0.0
            intra_var = 0.0
            avg_margin = 0.0
            margin_p25 = 0.0
            conflict_frac = 0.0
            n_low = 0

        track_metrics[int(track_id)] = {
            "identity": entry.get("identity", "Unknown"),
            "cluster_id": entry.get("cluster_id"),
            "n_frames": n_frames,
            "conf_p25": conf_p25,
            "conf_mean": conf_mean,
            "conf_min": conf_min,
            "conflict_frac": conflict_frac,
            "intra_var": intra_var,
            "n_low": n_low,
            "avg_margin": avg_margin,
            "margin_p25": margin_p25,
            "conflict_count": int(entry["conflict_count"]),
            "low_count": int(entry["low_count"]),
        }

    logger.info("Computed confidence metrics for %s tracks", len(track_metrics))

    if track_metrics:
        p25_scores = [m["conf_p25"] for m in track_metrics.values()]
        logger.info(
            "Track confidence (p25) - mean: %.3f, min: %.3f, max: %.3f",
            float(np.mean(p25_scores)),
            float(np.min(p25_scores)),
            float(np.max(p25_scores)),
        )

    return picked_samples_df, track_metrics


def update_clusters_with_track_metrics(
    clusters_data: Dict,
    track_metrics: Dict[int, Dict[str, float]],
) -> Dict:
    """
    Update clusters.json with track-level confidence metrics.

    Args:
        clusters_data: Clusters dict
        track_metrics: Dict mapping track_id -> confidence metrics

    Returns:
        Updated clusters_data with track_metrics embedded
    """
    updated_clusters = clusters_data.copy()

    for cluster in updated_clusters.get("clusters", []):
        metrics_list = []
        for track_id in cluster.get("track_ids", []):
            tm = track_metrics.get(int(track_id))
            if not tm:
                continue
            metrics_list.append(
                {
                    "track_id": int(track_id),
                    "identity": tm.get("identity", "Unknown"),
                    "n_frames": tm.get("n_frames", 0),
                    "conf_p25": tm.get("conf_p25", 0.0),
                    "conf_mean": tm.get("conf_mean", 0.0),
                    "conf_min": tm.get("conf_min", 0.0),
                    "conflict_frac": tm.get("conflict_frac", 0.0),
                    "intra_var": tm.get("intra_var", 0.0),
                    "n_low": tm.get("n_low", 0),
                    "avg_margin": tm.get("avg_margin", 0.0),
                    "margin_p25": tm.get("margin_p25", 0.0),
                }
            )

        metrics_list.sort(key=lambda x: x.get("conf_p25", 0.0), reverse=True)
        cluster["track_metrics"] = metrics_list
        cluster["track_ids"] = [tm["track_id"] for tm in metrics_list]

    return updated_clusters
