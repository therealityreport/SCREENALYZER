"""
Open-set cluster assignment using season bank prototypes.

No forced guesses - only assigns when confidence and margin are sufficient.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def assign_clusters_open_set(
    clusters_data: Dict,
    season_bank: Dict,
    min_sim: float = 0.60,
    min_margin: float = 0.08
) -> Dict[int, Dict]:
    """
    Assign clusters to identities using open-set matching.

    Args:
        clusters_data: Clusters dict with cluster_id, embeddings, etc.
        season_bank: Season bank with identity prototypes
        min_sim: Minimum similarity threshold for assignment
        min_margin: Minimum margin over second-best identity

    Returns:
        Dict mapping cluster_id -> {'name': str, 'confidence': float, 'margin': float}
    """
    assignments = {}
    identities = season_bank.get('identities', {})

    if not identities:
        logger.warning("Season bank has no identities")
        return assignments

    # Build prototype lookup: identity_name -> list of embeddings
    identity_prototypes = {}
    for identity_name, bins in identities.items():
        protos = []
        for bin_key, proto_list in bins.items():
            for proto in proto_list:
                emb = np.array(proto['embedding'], dtype=np.float32)
                protos.append(emb)
        if protos:
            identity_prototypes[identity_name] = protos

    logger.info(f"Loaded {len(identity_prototypes)} identities with prototypes")

    # Assign each cluster
    for cluster in clusters_data.get('clusters', []):
        cluster_id = cluster['cluster_id']

        # Get cluster centroid (mean of all embeddings)
        # In practice, we'd load embeddings from picked_samples.parquet
        # For now, skip if no embeddings available
        if 'centroid' not in cluster:
            continue

        centroid = np.array(cluster['centroid'], dtype=np.float32)

        # Compute similarities to each identity
        identity_sims = {}
        for identity_name, protos in identity_prototypes.items():
            # Max similarity across all prototypes for this identity
            sims = [cosine_similarity(centroid, proto) for proto in protos]
            identity_sims[identity_name] = max(sims) if sims else 0.0

        # Sort by similarity
        sorted_sims = sorted(identity_sims.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_sims) < 1:
            continue

        top_identity, top_sim = sorted_sims[0]
        second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0

        margin = top_sim - second_sim

        # Open-set decision
        if top_sim >= min_sim and margin >= min_margin:
            assignments[cluster_id] = {
                'name': top_identity,
                'confidence': float(top_sim),
                'margin': float(margin),
                'second_best': sorted_sims[1][0] if len(sorted_sims) > 1 else None,
                'second_sim': float(second_sim)
            }
            logger.info(f"Cluster {cluster_id} → {top_identity} (sim={top_sim:.3f}, margin={margin:.3f})")
        else:
            logger.info(f"Cluster {cluster_id} → Unknown (sim={top_sim:.3f} < {min_sim} or margin={margin:.3f} < {min_margin})")

    return assignments
