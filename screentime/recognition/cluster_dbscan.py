"""
DBSCAN clustering for face embeddings.

Clusters face tracks by embedding similarity with quality scoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


@dataclass
class ClusterMetadata:
    """Metadata for a face cluster."""

    cluster_id: int
    track_ids: list[int]
    size: int
    centroid: np.ndarray
    variance: float
    silhouette_score: float
    quality_score: float
    is_lowconf: bool


class DBSCANClusterer:
    """DBSCAN-based face clustering with quality scoring."""

    def __init__(
        self,
        eps: float = 0.45,
        min_samples: int = 3,
        lowconf_threshold: float = 0.6,
    ):
        """
        Initialize clusterer.

        Args:
            eps: DBSCAN epsilon (max distance between samples)
            min_samples: Minimum samples to form a cluster
            lowconf_threshold: Quality score threshold for low-confidence flag
        """
        self.eps = eps
        self.min_samples = min_samples
        self.lowconf_threshold = lowconf_threshold

    def cluster(
        self,
        embeddings: np.ndarray,
        track_ids: list[int],
    ) -> tuple[list[ClusterMetadata], dict[int, int]]:
        """
        Cluster face embeddings.

        Args:
            embeddings: Array of face embeddings (N x 512)
            track_ids: List of track IDs corresponding to embeddings

        Returns:
            Tuple of (cluster_metadata, track_to_cluster_map)
        """
        if len(embeddings) == 0:
            return [], {}

        logger.info(
            f"Clustering {len(embeddings)} embeddings with eps={self.eps}, min_samples={self.min_samples}"
        )

        # Run DBSCAN
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine")
        labels = clusterer.fit_predict(embeddings)

        # Build track-to-cluster map
        track_to_cluster = {track_id: int(label) for track_id, label in zip(track_ids, labels)}

        # Get unique cluster IDs (excluding noise: -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        logger.info(f"Found {len(unique_labels)} clusters, {sum(labels == -1)} noise points")

        # Calculate silhouette score (if possible)
        global_silhouette = None
        if len(unique_labels) > 1 and sum(labels != -1) > 1:
            try:
                # Only compute on non-noise points
                non_noise_mask = labels != -1
                global_silhouette = silhouette_score(
                    embeddings[non_noise_mask],
                    labels[non_noise_mask],
                    metric="cosine",
                )
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")

        # Build cluster metadata
        cluster_metadata = []

        for cluster_id in sorted(unique_labels):
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            cluster_track_ids = [
                tid for tid, label in zip(track_ids, labels) if label == cluster_id
            ]

            # Compute centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # Compute variance (mean distance from centroid)
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            variance = float(np.mean(distances))

            # Compute silhouette score for this cluster
            cluster_silhouette = None
            if global_silhouette is not None:
                try:
                    non_noise_mask = labels != -1
                    cluster_sample_indices = np.where(mask & non_noise_mask)[0]
                    if len(cluster_sample_indices) > 0:
                        # Get silhouette samples for this cluster
                        from sklearn.metrics import silhouette_samples

                        silhouette_samples_all = silhouette_samples(
                            embeddings[non_noise_mask],
                            labels[non_noise_mask],
                            metric="cosine",
                        )
                        cluster_silhouette = float(
                            np.mean(silhouette_samples_all[labels[non_noise_mask] == cluster_id])
                        )
                except Exception as e:
                    logger.warning(f"Failed to compute cluster silhouette score: {e}")

            # Calculate quality score
            # Quality = (1 - normalized_variance) * silhouette
            # Higher is better
            if cluster_silhouette is not None:
                quality_score = (1.0 - min(variance, 1.0)) * max(0, cluster_silhouette)
            else:
                quality_score = 1.0 - min(variance, 1.0)

            # Flag low-confidence clusters
            is_lowconf = quality_score < self.lowconf_threshold

            metadata = ClusterMetadata(
                cluster_id=int(cluster_id),
                track_ids=cluster_track_ids,
                size=len(cluster_track_ids),
                centroid=centroid,
                variance=variance,
                silhouette_score=cluster_silhouette if cluster_silhouette is not None else 0.0,
                quality_score=quality_score,
                is_lowconf=is_lowconf,
            )

            cluster_metadata.append(metadata)

        return cluster_metadata, track_to_cluster

    def get_noise_tracks(self, track_to_cluster: dict[int, int]) -> list[int]:
        """Get list of track IDs assigned to noise (cluster -1)."""
        return [track_id for track_id, cluster_id in track_to_cluster.items() if cluster_id == -1]
