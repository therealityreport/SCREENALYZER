"""
Auto-tune DBSCAN epsilon threshold per episode.

Uses k-NN distance curve with knee detection to find optimal eps.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import json

import numpy as np
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


@dataclass
class ThresholdConfig:
    """Configuration for auto-threshold tuning."""
    k_neighbors: int = 5  # k for k-NN distance curve
    eps_min: float = 0.40  # Minimum safe eps (looser to separate similar faces)
    eps_max: float = 0.50  # Maximum safe eps (allow more separation)
    knee_curve: str = "convex"  # Knee detection curve type
    knee_direction: str = "increasing"  # Knee detection direction


def compute_knn_distances(
    embeddings: np.ndarray,
    k: int = 5,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Compute k-nearest-neighbor distances for each embedding.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        k: Number of nearest neighbors
        metric: Distance metric

    Returns:
        Array of k-NN distances sorted ascending (n_samples,)
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nbrs.fit(embeddings)

    # Get k+1 neighbors (includes self as first neighbor)
    distances, _ = nbrs.kneighbors(embeddings)

    # Take k-th neighbor distance (skip self)
    kth_distances = distances[:, k]

    # Sort ascending
    kth_distances_sorted = np.sort(kth_distances)

    return kth_distances_sorted


def find_knee_point(
    distances: np.ndarray,
    config: ThresholdConfig
) -> Tuple[float, int]:
    """
    Find knee/elbow point in k-NN distance curve using percentile.

    Uses P75 as a simple, robust heuristic for the elbow point.
    This represents a point where 75% of tracks are closer than this distance,
    which is a good balance for DBSCAN eps.

    Args:
        distances: Sorted k-NN distances
        config: Threshold configuration

    Returns:
        Tuple of (knee_distance, knee_index)
    """
    # Use P75 as knee point (simple and robust)
    knee_dist = np.percentile(distances, 75)
    knee_idx = int(np.searchsorted(distances, knee_dist))

    logger.info(f"Knee point (P75): distance={knee_dist:.4f}, index={knee_idx}/{len(distances)}")

    return knee_dist, knee_idx


def auto_tune_eps(
    track_embeddings: np.ndarray,
    config: ThresholdConfig = None
) -> dict:
    """
    Auto-tune DBSCAN eps from k-NN distance curve.

    Args:
        track_embeddings: Track centroid embeddings (n_tracks, embedding_dim)
        config: Threshold configuration

    Returns:
        Dict with tuning results:
        {
            "eps_tuned": float,
            "eps_raw": float,  # Before clamping
            "knee_index": int,
            "k_neighbors": int,
            "eps_min": float,
            "eps_max": float,
            "n_tracks": int,
            "distances_p25": float,
            "distances_p50": float,
            "distances_p75": float,
            "distances_p90": float,
        }
    """
    if config is None:
        config = ThresholdConfig()

    logger.info(f"Auto-tuning DBSCAN eps for {len(track_embeddings)} tracks")

    # Compute k-NN distances
    knn_distances = compute_knn_distances(
        track_embeddings,
        k=config.k_neighbors,
        metric="cosine"
    )

    # Find knee point
    knee_dist, knee_idx = find_knee_point(knn_distances, config)

    logger.info(
        f"Knee point: idx={knee_idx}/{len(knn_distances)}, "
        f"distance={knee_dist:.4f}"
    )

    # Clamp to safe range
    eps_tuned = np.clip(knee_dist, config.eps_min, config.eps_max)

    if eps_tuned != knee_dist:
        logger.info(
            f"Clamped eps from {knee_dist:.4f} to {eps_tuned:.4f} "
            f"(range: [{config.eps_min}, {config.eps_max}])"
        )

    # Compute distance percentiles for diagnostics
    percentiles = {
        "distances_p25": float(np.percentile(knn_distances, 25)),
        "distances_p50": float(np.percentile(knn_distances, 50)),
        "distances_p75": float(np.percentile(knn_distances, 75)),
        "distances_p90": float(np.percentile(knn_distances, 90)),
    }

    result = {
        "eps_tuned": float(eps_tuned),
        "eps_raw": float(knee_dist),
        "knee_index": int(knee_idx),
        "k_neighbors": config.k_neighbors,
        "eps_min": config.eps_min,
        "eps_max": config.eps_max,
        "n_tracks": len(track_embeddings),
        **percentiles,
    }

    logger.info(
        f"Auto-tuned eps: {eps_tuned:.4f} "
        f"(P50={percentiles['distances_p50']:.4f}, "
        f"P75={percentiles['distances_p75']:.4f})"
    )

    return result


def save_threshold_diagnostics(
    episode_id: str,
    data_root: Path,
    threshold_results: dict
) -> None:
    """Save threshold tuning diagnostics to JSON."""
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "cluster_threshold.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "episode_id": episode_id,
        "threshold_results": threshold_results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved cluster threshold diagnostics to {output_path}")
