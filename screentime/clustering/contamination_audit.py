"""
Contamination Audit - Detect and Auto-Split Mixed Clusters.

Identity-agnostic detection of:
1. Intra-cluster outliers (doesn't match cluster medoid)
2. Cross-identity contamination (better match to different cluster)

Auto-splits contaminated spans, no per-person tuning.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

logger = logging.getLogger(__name__)


@dataclass
class ContaminationConfig:
    """
    Contamination detection configuration (uniform for all identities).
    """
    # Intra-cluster outlier detection
    outlier_mad_threshold: float = 3.0       # MAD (median absolute deviation) threshold
    outlier_sim_threshold: float = 0.75      # Min similarity to cluster medoid

    # Cross-identity contamination detection
    cross_id_margin: float = 0.10            # Margin: best_other - current ≥ this
    min_contiguous_frames: int = 4           # Minimum contiguous frames for contamination span

    # Auto-split behavior
    auto_split_enabled: bool = True          # Enable automatic splitting
    min_evidence_strength: float = 0.12      # Minimum margin for auto-split assignment


@dataclass
class ContaminationSpan:
    """A contiguous span of contaminated samples within a cluster."""
    cluster_id: int
    cluster_name: str
    track_id: int
    start_idx: int
    end_idx: int
    frame_ids: List[int]
    timestamps_ms: List[int]
    reason: str                              # "outlier" or "cross_identity"
    best_match_cluster: Optional[str] = None  # For cross-identity contamination
    evidence_strength: float = 0.0           # Similarity margin
    action: str = "flag"                     # "flag", "split_to_unknown", "split_to_cluster"


def compute_cluster_medoid(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Compute medoid (most representative embedding) for a cluster.

    Args:
        embeddings: List of embeddings

    Returns:
        Medoid embedding
    """
    if len(embeddings) == 0:
        raise ValueError("Cannot compute medoid of empty cluster")

    if len(embeddings) == 1:
        return embeddings[0]

    # Compute pairwise distances
    n = len(embeddings)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Cosine distance = 1 - similarity
            sim = np.dot(embeddings[i], embeddings[j])
            dist = 1.0 - sim
            distances[i, j] = dist
            distances[j, i] = dist

    # Medoid = embedding with minimum sum of distances
    sum_distances = distances.sum(axis=1)
    medoid_idx = np.argmin(sum_distances)

    return embeddings[medoid_idx]


def detect_outliers(
    cluster_embeddings: List[np.ndarray],
    cluster_medoid: np.ndarray,
    config: ContaminationConfig
) -> List[int]:
    """
    Detect outlier embeddings within a cluster.

    Args:
        cluster_embeddings: All embeddings in cluster
        cluster_medoid: Cluster medoid
        config: Contamination config

    Returns:
        List of outlier indices
    """
    # Compute similarities to medoid
    similarities = [np.dot(emb, cluster_medoid) for emb in cluster_embeddings]

    # Method 1: Direct similarity threshold
    outliers_sim = [i for i, sim in enumerate(similarities) if sim < config.outlier_sim_threshold]

    # Method 2: MAD (Median Absolute Deviation) threshold
    median_sim = np.median(similarities)
    mad = median_abs_deviation(similarities)

    if mad > 0:
        outliers_mad = [
            i for i, sim in enumerate(similarities)
            if abs(sim - median_sim) > config.outlier_mad_threshold * mad
        ]
    else:
        outliers_mad = []

    # Union of both methods
    outliers = list(set(outliers_sim + outliers_mad))

    return sorted(outliers)


def detect_cross_identity_contamination(
    sample_embeddings: List[np.ndarray],
    current_cluster_medoid: np.ndarray,
    all_cluster_medoids: Dict[str, np.ndarray],
    current_cluster_name: str,
    config: ContaminationConfig
) -> List[Tuple[int, str, float]]:
    """
    Detect samples that match a different cluster better than current cluster.

    Args:
        sample_embeddings: Embeddings to check
        current_cluster_medoid: Current cluster's medoid
        all_cluster_medoids: Dict of {cluster_name: medoid} for all clusters
        current_cluster_name: Name of current cluster
        config: Contamination config

    Returns:
        List of (sample_idx, best_match_cluster_name, margin) for contaminated samples
    """
    contaminated = []

    for idx, emb in enumerate(sample_embeddings):
        # Similarity to current cluster
        sim_to_current = np.dot(emb, current_cluster_medoid)

        # Similarity to all other clusters
        best_other_sim = 0.0
        best_other_name = None

        for other_name, other_medoid in all_cluster_medoids.items():
            if other_name == current_cluster_name:
                continue

            sim_to_other = np.dot(emb, other_medoid)
            if sim_to_other > best_other_sim:
                best_other_sim = sim_to_other
                best_other_name = other_name

        # Check if other cluster is significantly better
        margin = best_other_sim - sim_to_current

        if margin >= config.cross_id_margin:
            contaminated.append((idx, best_other_name, margin))

    return contaminated


def find_contiguous_spans(
    contaminated_indices: List[int],
    min_contiguous: int = 4
) -> List[Tuple[int, int]]:
    """
    Find contiguous spans of contaminated samples.

    Args:
        contaminated_indices: List of sample indices flagged as contaminated
        min_contiguous: Minimum span length

    Returns:
        List of (start_idx, end_idx) tuples for contiguous spans
    """
    if not contaminated_indices:
        return []

    spans = []
    current_span_start = contaminated_indices[0]
    current_span_end = contaminated_indices[0]

    for i in range(1, len(contaminated_indices)):
        if contaminated_indices[i] == current_span_end + 1:
            # Extend current span
            current_span_end = contaminated_indices[i]
        else:
            # End current span, start new one
            if (current_span_end - current_span_start + 1) >= min_contiguous:
                spans.append((current_span_start, current_span_end))

            current_span_start = contaminated_indices[i]
            current_span_end = contaminated_indices[i]

    # Don't forget last span
    if (current_span_end - current_span_start + 1) >= min_contiguous:
        spans.append((current_span_start, current_span_end))

    return spans


def audit_cluster_contamination(
    cluster: dict,
    picked_samples_df: pd.DataFrame,
    all_cluster_medoids: Dict[str, np.ndarray],
    config: ContaminationConfig
) -> List[ContaminationSpan]:
    """
    Audit a single cluster for contamination.

    Args:
        cluster: Cluster data from clusters.json
        picked_samples_df: Picked samples (faces-only, top-K per track)
        all_cluster_medoids: Dict of {cluster_name: medoid} for all clusters
        config: Contamination config

    Returns:
        List of contamination spans detected
    """
    cluster_id = cluster['cluster_id']
    cluster_name = cluster.get('name', f"Cluster {cluster_id}")

    # Get all samples for this cluster
    track_ids = cluster.get('track_ids', [])
    cluster_samples = picked_samples_df[picked_samples_df['track_id'].isin(track_ids)]

    if len(cluster_samples) == 0:
        logger.warning(f"Cluster {cluster_name}: No picked samples found")
        return []

    # Compute cluster medoid
    cluster_embeddings = [np.array(row['embedding']) for _, row in cluster_samples.iterrows()]
    cluster_medoid = compute_cluster_medoid(cluster_embeddings)

    contamination_spans = []

    # Check each track in the cluster
    for track_id in track_ids:
        track_samples = cluster_samples[cluster_samples['track_id'] == track_id]

        if len(track_samples) == 0:
            continue

        track_embeddings = [np.array(row['embedding']) for _, row in track_samples.iterrows()]

        # 1. Detect intra-cluster outliers
        outlier_indices = detect_outliers(track_embeddings, cluster_medoid, config)

        # 2. Detect cross-identity contamination
        cross_id_contamination = detect_cross_identity_contamination(
            track_embeddings,
            cluster_medoid,
            all_cluster_medoids,
            cluster_name,
            config
        )

        # Combine both types
        all_contaminated_indices = set(outlier_indices + [idx for idx, _, _ in cross_id_contamination])

        if not all_contaminated_indices:
            continue  # This track is clean

        # Find contiguous spans
        spans = find_contiguous_spans(
            sorted(all_contaminated_indices),
            min_contiguous=config.min_contiguous_frames
        )

        # Create ContaminationSpan objects
        for start_idx, end_idx in spans:
            span_samples = track_samples.iloc[start_idx:end_idx+1]

            # Determine reason and best match
            reason = "outlier"
            best_match = None
            evidence_strength = 0.0

            # Check if any samples in span have strong cross-identity match
            for idx, best_other, margin in cross_id_contamination:
                if start_idx <= idx <= end_idx and margin > evidence_strength:
                    reason = "cross_identity"
                    best_match = best_other
                    evidence_strength = margin

            # Determine action
            action = "flag"
            if config.auto_split_enabled:
                if evidence_strength >= config.min_evidence_strength and best_match:
                    action = f"split_to_{best_match}"
                else:
                    action = "split_to_unknown"

            span = ContaminationSpan(
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                track_id=int(track_id),
                start_idx=int(start_idx),
                end_idx=int(end_idx),
                frame_ids=[int(row['frame_id']) for _, row in span_samples.iterrows()],
                timestamps_ms=[int(row.get('ts_ms', 0)) for _, row in span_samples.iterrows()],
                reason=reason,
                best_match_cluster=best_match,
                evidence_strength=float(evidence_strength),
                action=action
            )

            contamination_spans.append(span)

    return contamination_spans


def audit_all_clusters(
    clusters_data: dict,
    picked_samples_df: pd.DataFrame,
    config: ContaminationConfig
) -> Dict[str, List[ContaminationSpan]]:
    """
    Audit all clusters for contamination.

    Args:
        clusters_data: Loaded clusters.json
        picked_samples_df: Picked samples (faces-only, top-K per track)
        config: Contamination config

    Returns:
        Dict of {cluster_name: [contamination_spans]}
    """
    logger.info("Starting contamination audit on all clusters...")

    # Compute medoids for all clusters first
    all_cluster_medoids = {}

    for cluster in clusters_data.get("clusters", []):
        if "name" not in cluster:
            continue

        cluster_name = cluster["name"]
        track_ids = cluster.get("track_ids", [])

        cluster_samples = picked_samples_df[picked_samples_df['track_id'].isin(track_ids)]

        if len(cluster_samples) == 0:
            continue

        cluster_embeddings = [np.array(row['embedding']) for _, row in cluster_samples.iterrows()]
        all_cluster_medoids[cluster_name] = compute_cluster_medoid(cluster_embeddings)

    logger.info(f"Computed medoids for {len(all_cluster_medoids)} clusters")

    # Audit each cluster
    results = {}

    for cluster in clusters_data.get("clusters", []):
        if "name" not in cluster:
            continue

        cluster_name = cluster["name"]
        spans = audit_cluster_contamination(cluster, picked_samples_df, all_cluster_medoids, config)

        if spans:
            results[cluster_name] = spans
            logger.info(f"  {cluster_name}: Found {len(spans)} contamination spans")

    logger.info(f"Contamination audit complete: {len(results)} clusters with contamination")

    return results


def save_contamination_audit(
    episode_id: str,
    data_root: Path,
    contamination_results: Dict[str, List[ContaminationSpan]]
):
    """Save contamination audit results to JSON."""
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "contamination_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    output = {
        "episode_id": episode_id,
        "clusters": {}
    }

    for cluster_name, spans in contamination_results.items():
        output["clusters"][cluster_name] = [
            {
                "track_id": span.track_id,
                "start_idx": span.start_idx,
                "end_idx": span.end_idx,
                "frame_count": len(span.frame_ids),
                "frame_ids": span.frame_ids,
                "timestamps_ms": span.timestamps_ms,
                "time_range": f"{span.timestamps_ms[0]/1000:.1f}s - {span.timestamps_ms[-1]/1000:.1f}s" if span.timestamps_ms else "N/A",
                "reason": span.reason,
                "best_match_cluster": span.best_match_cluster,
                "evidence_strength": round(span.evidence_strength, 3),
                "action": span.action
            }
            for span in spans
        ]

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved contamination audit to {output_path}")
