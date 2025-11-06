"""
Purity-Driven Auto-Threshold for DBSCAN Clustering.

Sweeps eps candidates and scores each by silhouette + impurity.
Chooses eps that maximizes cluster quality without manual tuning.
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

IMPURITY_GUARD_MAX = 0.30
EPS_FLOOR_RANGE = (0.28, 0.30)
EPS_FLOOR_DELTA = 0.02
EPS_FLOOR_MIN = 0.31


@dataclass
class PurityConfig:
    """Configuration for purity-driven eps selection."""
    eps_step: float = 0.02  # Step size for eps candidates
    eps_range_offset: float = 0.10  # Range around knee: [knee-offset, knee+offset]
    impurity_weight: float = 0.75  # Weight for impurity penalty (λ)
    intra_sim_threshold: float = 0.75  # Min sim to cluster medoid
    cross_margin_threshold: float = 0.10  # Margin for cross-identity detection
    min_cluster_constraint: int = 3  # Minimum number of clusters
    max_cluster_size_percentile: int = 95  # Max cluster size (P95 track count)


def compute_cluster_medoid(embeddings: List[np.ndarray]) -> np.ndarray:
    """Compute medoid (most representative embedding) for a cluster."""
    if len(embeddings) == 0:
        raise ValueError("Cannot compute medoid of empty cluster")

    if len(embeddings) == 1:
        return embeddings[0]

    # Compute pairwise cosine similarities
    n = len(embeddings)
    similarities = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            sim = np.dot(embeddings[i], embeddings[j])
            similarities[i, j] = sim
            similarities[j, i] = sim

    # Medoid = embedding with maximum sum of similarities
    sum_similarities = similarities.sum(axis=1)
    medoid_idx = np.argmax(sum_similarities)

    return embeddings[medoid_idx]


def compute_impurity_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    config: PurityConfig
) -> Tuple[float, Dict]:
    """
    Compute impurity score for clustering result.

    Impurity = % of samples that are:
    - Intra-cluster outliers (sim_to_medoid < threshold)
    - Cross-cluster candidates (best_other - current ≥ margin)

    Args:
        embeddings: All embeddings
        labels: Cluster labels from DBSCAN
        config: Purity configuration

    Returns:
        Tuple of (impurity_score, diagnostics_dict)
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    if len(unique_labels) == 0:
        return 1.0, {"reason": "no_clusters"}

    # Compute medoids for all clusters
    cluster_medoids = {}
    for label in unique_labels:
        cluster_mask = labels == label
        cluster_embeddings = embeddings[cluster_mask]
        cluster_medoids[label] = compute_cluster_medoid(list(cluster_embeddings))

    # Check each sample for impurity
    intra_outliers = 0
    cross_candidates = 0
    total_samples = 0

    for label in unique_labels:
        cluster_mask = labels == label
        cluster_embeddings = embeddings[cluster_mask]
        cluster_medoid = cluster_medoids[label]

        for emb in cluster_embeddings:
            total_samples += 1

            # Check intra-cluster outlier
            sim_to_medoid = np.dot(emb, cluster_medoid)
            if sim_to_medoid < config.intra_sim_threshold:
                intra_outliers += 1

            # Check cross-cluster contamination
            best_other_sim = 0.0
            for other_label, other_medoid in cluster_medoids.items():
                if other_label != label:
                    sim_to_other = np.dot(emb, other_medoid)
                    best_other_sim = max(best_other_sim, sim_to_other)

            cross_margin = best_other_sim - sim_to_medoid
            if cross_margin >= config.cross_margin_threshold:
                cross_candidates += 1

    if total_samples == 0:
        return 1.0, {"reason": "no_samples"}

    # Impurity = fraction of outliers + cross-candidates
    impurity_rate = (intra_outliers + cross_candidates) / total_samples

    diagnostics = {
        "intra_outliers": intra_outliers,
        "cross_candidates": cross_candidates,
        "total_samples": total_samples,
        "impurity_rate": impurity_rate,
    }

    return impurity_rate, diagnostics


def evaluate_eps_candidate(
    embeddings: np.ndarray,
    eps: float,
    min_samples: int,
    config: PurityConfig
) -> Dict:
    """
    Evaluate a single eps candidate by clustering and scoring.

    Args:
        embeddings: Track centroid embeddings
        eps: Epsilon value to test
        min_samples: DBSCAN min_samples parameter
        config: Purity configuration

    Returns:
        Dict with evaluation results
    """
    # Run DBSCAN with this eps
    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    labels = clusterer.fit_predict(embeddings)

    # Count clusters (excluding noise label -1)
    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)
    n_noise = np.sum(labels == -1)

    # Compute silhouette score (only if ≥2 clusters and not all noise)
    if n_clusters >= 2 and n_noise < len(labels):
        try:
            silhouette = silhouette_score(embeddings, labels, metric='cosine')
        except Exception as e:
            logger.warning(f"Silhouette calculation failed for eps={eps:.4f}: {e}")
            silhouette = 0.0
    else:
        silhouette = 0.0

    # Compute impurity score
    impurity, impurity_diag = compute_impurity_score(embeddings, labels, config)

    # Combined quality score
    quality_score = silhouette - config.impurity_weight * impurity

    # Check constraints
    passes_min_clusters = n_clusters >= config.min_cluster_constraint

    # Max cluster size constraint (no cluster should have >P95 of tracks)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    max_allowed_size = int(np.percentile(range(len(embeddings)), config.max_cluster_size_percentile))
    passes_max_size = max_cluster_size <= max_allowed_size if max_allowed_size > 0 else True

    passes_constraints = passes_min_clusters and passes_max_size

    result = {
        "eps": eps,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "silhouette": silhouette,
        "impurity": impurity,
        "quality_score": quality_score,
        "passes_constraints": passes_constraints,
        "max_cluster_size": max_cluster_size,
        "impurity_diagnostics": impurity_diag,
    }

    return result


def purity_driven_eps_selection(
    track_embeddings: np.ndarray,
    min_samples: int = 3,
    config: PurityConfig = None
) -> Dict:
    """
    Select optimal DBSCAN eps using purity-driven quality sweep.

    Algorithm:
    1. Compute k-NN knee as starting point
    2. Generate eps candidates around knee
    3. For each eps: cluster and compute silhouette - λ * impurity
    4. Choose eps that maximizes quality score while passing constraints

    Args:
        track_embeddings: Track centroid embeddings (n_tracks, embedding_dim)
        min_samples: DBSCAN min_samples parameter
        config: Purity configuration

    Returns:
        Dict with selection results and candidate table
    """
    if config is None:
        config = PurityConfig()

    logger.info(f"Starting purity-driven eps selection for {len(track_embeddings)} tracks")

    # Step 1: Compute k-NN knee as starting point
    n_samples = len(track_embeddings)

    # Cap k by available samples (need at least 2 neighbors for knn)
    k_neighbors = min(6, max(2, n_samples))  # k=5+self ideally, but cap at n_samples

    # If too few samples, use fallback
    if n_samples < 6:
        logger.warning(
            f"Only {n_samples} tracks available (< 6), using fallback eps selection. "
            f"Using k={k_neighbors} neighbors instead of ideal k=6."
        )

        # Fallback: use median distance as knee estimate
        if n_samples >= 2:
            nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')
            nbrs.fit(track_embeddings)
            distances, _ = nbrs.kneighbors(track_embeddings)
            # Use last neighbor (furthest)
            kth_distances = distances[:, -1]
            kth_distances_sorted = np.sort(kth_distances)
            knee_dist = np.percentile(kth_distances_sorted, 75)  # P75 as knee
        else:
            # Single track: use default eps
            logger.warning("Only 1 track, using default eps=0.35")
            knee_dist = 0.35
    else:
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, metric='cosine')  # k=5 + self
        nbrs.fit(track_embeddings)
        distances, _ = nbrs.kneighbors(track_embeddings)
        kth_distances = distances[:, -1]  # Take last neighbor
        kth_distances_sorted = np.sort(kth_distances)
        knee_dist = np.percentile(kth_distances_sorted, 75)  # P75 as knee

    logger.info(f"k-NN knee point: {knee_dist:.4f} (P75 of k={k_neighbors-1}-NN distances)")

    # Step 2: Generate eps candidates around knee
    eps_min = max(0.25, knee_dist - config.eps_range_offset)
    eps_max = min(0.55, knee_dist + config.eps_range_offset)

    eps_candidates = np.arange(eps_min, eps_max + config.eps_step, config.eps_step)

    logger.info(
        f"Evaluating {len(eps_candidates)} eps candidates "
        f"in range [{eps_min:.4f}, {eps_max:.4f}]"
    )

    # Step 3: Evaluate each candidate
    candidate_results = []
    for eps in eps_candidates:
        result = evaluate_eps_candidate(track_embeddings, eps, min_samples, config)
        candidate_results.append(result)

        logger.info(
            f"  eps={eps:.4f}: {result['n_clusters']} clusters, "
            f"silhouette={result['silhouette']:.3f}, "
            f"impurity={result['impurity']:.3f}, "
            f"quality={result['quality_score']:.3f}, "
            f"passes={result['passes_constraints']}"
        )

    # Step 4: Choose best eps applying guard rails
    valid_candidates = [r for r in candidate_results if r['passes_constraints']]
    if not valid_candidates:
        logger.warning("No candidates passed constraints, using best quality score anyway")
        valid_candidates = candidate_results

    candidate_pool = list(valid_candidates)

    selection_steps: List[str] = ["max_quality_score"]
    primary_best = max(candidate_pool, key=lambda r: r['quality_score'])
    final_pool = candidate_pool
    final_best = primary_best
    eps_floor_applied = False
    eps_floor_value: float | None = None

    if (
        primary_best['impurity'] > IMPURITY_GUARD_MAX
        and EPS_FLOOR_RANGE[0] <= knee_dist <= EPS_FLOOR_RANGE[1]
    ):
        eps_floor_value = max(knee_dist + EPS_FLOOR_DELTA, EPS_FLOOR_MIN)
        filtered = [c for c in candidate_pool if c['eps'] >= eps_floor_value]
        if not filtered:
            filtered = [c for c in candidate_results if c['eps'] >= eps_floor_value]
        if filtered:
            final_pool = filtered
            final_best = max(filtered, key=lambda r: r['quality_score'])
            selection_steps.append(f"eps_floor>= {eps_floor_value:.2f}")
            eps_floor_applied = True

    if final_best['impurity'] > IMPURITY_GUARD_MAX:
        guard_candidates = [c for c in final_pool if c['passes_constraints']]
        if not guard_candidates:
            guard_candidates = final_pool
        final_best = min(
            guard_candidates,
            key=lambda c: (c['impurity'], -c['quality_score'])
        )
        selection_steps.append(f"impurity_guard<= {IMPURITY_GUARD_MAX:.2f}")

    best_result = final_best
    eps_chosen = best_result['eps']

    logger.info(
        f"\n✅ Chosen eps={eps_chosen:.4f}: "
        f"{best_result['n_clusters']} clusters, "
        f"silhouette={best_result['silhouette']:.3f}, "
        f"impurity={best_result['impurity']:.3f}, "
        f"quality={best_result['quality_score']:.3f}"
    )
    logger.info("Selection path: %s", " -> ".join(selection_steps))

    selection_details = {
        "reasons": selection_steps,
        "impurity_max": IMPURITY_GUARD_MAX,
        "eps_floor_applied": eps_floor_applied,
        "eps_floor_min": float(eps_floor_value) if eps_floor_value is not None else None,
        "initial_best": _simplify_candidate(primary_best),
        "final_best": _simplify_candidate(best_result),
    }

    return {
        "eps_chosen": eps_chosen,
        "knee_dist": knee_dist,
        "n_candidates": len(eps_candidates),
        "best_result": best_result,
        "all_candidates": candidate_results,
        "config": {
            "impurity_weight": config.impurity_weight,
            "intra_sim_threshold": config.intra_sim_threshold,
            "cross_margin_threshold": config.cross_margin_threshold,
            "min_cluster_constraint": config.min_cluster_constraint,
        },
        "selection": selection_details,
    }


def _simplify_candidate(candidate: dict) -> dict:
    return {
        "eps": float(candidate["eps"]),
        "n_clusters": int(candidate["n_clusters"]),
        "n_noise": int(candidate["n_noise"]),
        "impurity": float(candidate["impurity"]),
        "quality_score": float(candidate["quality_score"]),
        "passes_constraints": bool(candidate["passes_constraints"]),
    }


def save_purity_diagnostics(
    episode_id: str,
    data_root: Path,
    purity_results: dict
) -> None:
    """Save purity-driven threshold selection diagnostics to JSON."""
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "cluster_threshold.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Simplify candidate results for JSON serialization
    candidates_simplified = []
    for c in purity_results["all_candidates"]:
        candidates_simplified.append({
            "eps": float(c["eps"]),
            "n_clusters": int(c["n_clusters"]),
            "n_noise": int(c["n_noise"]),
            "silhouette": float(c["silhouette"]),
            "impurity": float(c["impurity"]),
            "quality_score": float(c["quality_score"]),
            "passes_constraints": bool(c["passes_constraints"]),
            "max_cluster_size": int(c["max_cluster_size"]),
        })

    output_data = {
        "episode_id": episode_id,
        "method": "purity_driven",
        "eps_chosen": float(purity_results["eps_chosen"]),
        "knee_dist": float(purity_results["knee_dist"]),
        "n_candidates_evaluated": purity_results["n_candidates"],
        "best_result": {
            "eps": float(purity_results["best_result"]["eps"]),
            "n_clusters": int(purity_results["best_result"]["n_clusters"]),
            "n_noise": int(purity_results["best_result"]["n_noise"]),
            "silhouette": float(purity_results["best_result"]["silhouette"]),
            "impurity": float(purity_results["best_result"]["impurity"]),
            "quality_score": float(purity_results["best_result"]["quality_score"]),
            "impurity_diagnostics": purity_results["best_result"]["impurity_diagnostics"],
        },
        "config": purity_results["config"],
        "all_candidates": candidates_simplified,
        "selection": purity_results.get("selection", {}),
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Saved purity-driven threshold diagnostics to {output_path}")
