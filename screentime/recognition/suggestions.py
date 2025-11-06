"""
Merge suggestion generation.

Generates pairwise merge suggestions for similar clusters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class MergeSuggestion:
    """Pairwise merge suggestion."""

    cluster_a_id: int
    cluster_b_id: int
    similarity: float
    cluster_a_size: int
    cluster_b_size: int
    combined_size: int
    rank: int


class MergeSuggester:
    """Generate merge suggestions for similar clusters."""

    def __init__(
        self,
        similarity_threshold: float = 0.75,
        max_suggestions: int = 100,
    ):
        """
        Initialize suggester.

        Args:
            similarity_threshold: Minimum cosine similarity for suggestions
            max_suggestions: Maximum number of suggestions to generate
        """
        self.similarity_threshold = similarity_threshold
        self.max_suggestions = max_suggestions

    def generate_suggestions(
        self,
        cluster_metadata: list,
    ) -> list[MergeSuggestion]:
        """
        Generate merge suggestions.

        Args:
            cluster_metadata: List of ClusterMetadata objects

        Returns:
            List of MergeSuggestion objects, sorted by similarity (descending)
        """
        if len(cluster_metadata) < 2:
            logger.info("Not enough clusters for merge suggestions")
            return []

        logger.info(f"Generating merge suggestions for {len(cluster_metadata)} clusters")

        # Build pairwise similarity matrix
        suggestions = []

        for i, cluster_a in enumerate(cluster_metadata):
            for j, cluster_b in enumerate(cluster_metadata):
                if i >= j:
                    # Skip self-comparisons and duplicates
                    continue

                # Calculate cosine similarity between centroids
                similarity = 1.0 - cosine(cluster_a.centroid, cluster_b.centroid)

                # Filter by threshold
                if similarity >= self.similarity_threshold:
                    suggestion = MergeSuggestion(
                        cluster_a_id=cluster_a.cluster_id,
                        cluster_b_id=cluster_b.cluster_id,
                        similarity=similarity,
                        cluster_a_size=cluster_a.size,
                        cluster_b_size=cluster_b.size,
                        combined_size=cluster_a.size + cluster_b.size,
                        rank=0,  # Will be set later
                    )
                    suggestions.append(suggestion)

        # Sort by similarity (descending)
        suggestions.sort(key=lambda s: s.similarity, reverse=True)

        # Limit to max_suggestions
        suggestions = suggestions[: self.max_suggestions]

        # Assign ranks
        for rank, suggestion in enumerate(suggestions, start=1):
            suggestion.rank = rank

        logger.info(f"Generated {len(suggestions)} merge suggestions")

        return suggestions

    def filter_by_quality(
        self,
        suggestions: list[MergeSuggestion],
        cluster_metadata: list,
        min_quality: float = 0.6,
    ) -> list[MergeSuggestion]:
        """
        Filter suggestions to only include high-quality clusters.

        Args:
            suggestions: List of merge suggestions
            cluster_metadata: List of cluster metadata
            min_quality: Minimum quality score for both clusters

        Returns:
            Filtered list of suggestions
        """
        # Build cluster quality map
        quality_map = {c.cluster_id: c.quality_score for c in cluster_metadata}

        # Filter suggestions
        filtered = []
        for suggestion in suggestions:
            quality_a = quality_map.get(suggestion.cluster_a_id, 0.0)
            quality_b = quality_map.get(suggestion.cluster_b_id, 0.0)

            if quality_a >= min_quality and quality_b >= min_quality:
                filtered.append(suggestion)

        logger.info(
            f"Filtered {len(suggestions)} suggestions to {len(filtered)} high-quality suggestions"
        )

        return filtered
