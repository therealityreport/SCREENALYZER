"""
Clean Clustering Task - Identity-Agnostic with Face-Only Gating.

Integrates:
1. Face-only filtering (removes non-face chips)
2. Top-K selection per track (best embeddings for centroids)
3. DBSCAN clustering on clean centroids
4. Contamination audit + auto-split

No per-person tuning - uniform for all identities.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from screentime.clustering.face_quality import (
    FaceQualityFilter,
    filter_face_samples,
    pick_top_k_per_track,
    save_picked_samples,
)
from screentime.clustering.contamination_audit import (
    ContaminationConfig,
    audit_all_clusters,
    save_contamination_audit,
)
from screentime.recognition.cluster_dbscan import DBSCANClusterer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_track_centroid_from_picked(
    track_id: int, picked_samples: pd.DataFrame
) -> np.ndarray:
    """
    Compute track centroid from picked samples (top-K embeddings).

    Args:
        track_id: Track ID
        picked_samples: Picked samples DataFrame (faces-only, top-K per track)

    Returns:
        Centroid embedding (mean of picked embeddings)
    """
    track_samples = picked_samples[picked_samples["track_id"] == track_id]

    if len(track_samples) == 0:
        raise ValueError(f"No picked samples for track {track_id}")

    embeddings = [np.array(row["embedding"]) for _, row in track_samples.iterrows()]

    # Return mean (normalized)
    centroid = np.mean(embeddings, axis=0)
    centroid = centroid / np.linalg.norm(centroid)  # L2 normalize

    return centroid


def cluster_clean_task(job_id: str, episode_id: str, data_root: Path = None) -> dict:
    """
    Run clean clustering with face-only gating and contamination audit.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        data_root: Data root path (defaults to "data")

    Returns:
        Dict with clustering results
    """
    logger.info(f"[{job_id}] Starting CLEAN cluster task for {episode_id}")
    start_time = time.time()

    if data_root is None:
        data_root = Path("data")

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract configs with defaults
    clustering_config = config.get("clustering", {})
    face_quality_config = clustering_config.get(
        "face_quality",
        {
            "min_face_conf": 0.65,
            "min_face_px": 72,
            "max_co_face_iou": 0.10,
            "top_k_per_track": 10,
        },
    )
    contamination_config = clustering_config.get(
        "contamination_audit",
        {
            "enabled": True,
            "outlier_mad_threshold": 3.0,
            "outlier_sim_threshold": 0.75,
            "cross_id_margin": 0.10,
            "min_contiguous_frames": 4,
            "auto_split_enabled": True,
            "min_evidence_strength": 0.12,
        },
    )

    # Setup paths
    harvest_dir = data_root / "harvest" / episode_id
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"

    if not tracks_path.exists():
        raise ValueError(f"Tracks not found: {tracks_path}")

    if not embeddings_path.exists():
        raise ValueError(f"Embeddings not found: {embeddings_path}")

    # Load data
    logger.info(f"[{job_id}] Loading tracks and embeddings...")
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data["tracks"]
    embeddings_df = pd.read_parquet(embeddings_path)

    logger.info(f"[{job_id}] Loaded {len(tracks)} tracks, {len(embeddings_df)} embeddings")

    # ========================================
    # STEP 1: Face-Only Filtering
    # ========================================
    logger.info(f"\n[{job_id}] === STEP 1: Face-Only Filtering ===")

    face_filter = FaceQualityFilter(
        min_face_conf=face_quality_config["min_face_conf"],
        min_face_px=face_quality_config["min_face_px"],
        max_co_face_iou=face_quality_config["max_co_face_iou"],
        require_embedding=True,
    )

    faces_only_df = filter_face_samples(embeddings_df, tracks_data, face_filter)

    logger.info(
        f"[{job_id}] Faces-only: {len(faces_only_df)} / {len(embeddings_df)} "
        f"({len(faces_only_df)/len(embeddings_df)*100:.1f}% retained)"
    )

    # ========================================
    # STEP 2: Top-K Selection Per Track
    # ========================================
    logger.info(f"\n[{job_id}] === STEP 2: Top-K Selection Per Track ===")

    top_k = face_quality_config["top_k_per_track"]
    picked_df = pick_top_k_per_track(faces_only_df, k=top_k)

    logger.info(
        f"[{job_id}] Picked {len(picked_df)} samples for centroids "
        f"({len(picked_df)/len(faces_only_df)*100:.1f}% of faces-only)"
    )

    # Save picked samples for gallery
    save_picked_samples(episode_id, data_root, picked_df)

    # ========================================
    # STEP 3: Compute Track Centroids from Picked Samples
    # ========================================
    logger.info(f"\n[{job_id}] === STEP 3: Compute Track Centroids ===")

    track_centroids = []
    track_ids_for_clustering = []

    for track in tracks:
        track_id = track["track_id"]

        try:
            centroid = compute_track_centroid_from_picked(track_id, picked_df)
            track_centroids.append(centroid)
            track_ids_for_clustering.append(track_id)
        except ValueError as e:
            logger.warning(f"[{job_id}] {e} - skipping")

    logger.info(f"[{job_id}] Computed {len(track_centroids)} track centroids")

    if len(track_centroids) == 0:
        logger.error(f"[{job_id}] No track centroids - cannot cluster")
        return {"error": "No track centroids"}

    track_centroids_array = np.array(track_centroids)

    # ========================================
    # STEP 4: DBSCAN Clustering on Clean Centroids
    # ========================================
    logger.info(f"\n[{job_id}] === STEP 4: DBSCAN Clustering ===")

    dbscan_eps = clustering_config.get("eps", 0.45)
    dbscan_min_samples = clustering_config.get("min_samples", 3)

    clusterer = DBSCANClusterer(eps=dbscan_eps, min_samples=dbscan_min_samples)

    cluster_metadata, track_to_cluster = clusterer.cluster(
        track_centroids_array, track_ids_for_clustering
    )

    logger.info(
        f"[{job_id}] Clustering complete: {len(cluster_metadata)} clusters, "
        f"{sum(1 for c in track_to_cluster.values() if c == -1)} noise tracks"
    )

    # Build clusters.json structure
    clusters_output = {"clusters": []}

    for cluster_meta in cluster_metadata:
        cluster_dict = {
            "cluster_id": cluster_meta.cluster_id,
            "track_ids": cluster_meta.track_ids,
            "size": cluster_meta.size,
            "quality_score": float(cluster_meta.quality_score),
            "is_lowconf": cluster_meta.is_lowconf,
        }
        clusters_output["clusters"].append(cluster_dict)

    # Save clusters
    clusters_path = harvest_dir / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(clusters_output, f, indent=2)

    logger.info(f"[{job_id}] Saved clusters to {clusters_path}")

    # ========================================
    # STEP 5: Contamination Audit + Auto-Split
    # ========================================
    if contamination_config.get("enabled", True):
        logger.info(f"\n[{job_id}] === STEP 5: Contamination Audit ===")

        contam_config = ContaminationConfig(
            outlier_mad_threshold=contamination_config["outlier_mad_threshold"],
            outlier_sim_threshold=contamination_config["outlier_sim_threshold"],
            cross_id_margin=contamination_config["cross_id_margin"],
            min_contiguous_frames=contamination_config["min_contiguous_frames"],
            auto_split_enabled=contamination_config["auto_split_enabled"],
            min_evidence_strength=contamination_config["min_evidence_strength"],
        )

        contamination_results = audit_all_clusters(
            clusters_output, picked_df, contam_config
        )

        save_contamination_audit(episode_id, data_root, contamination_results)

        total_spans = sum(len(spans) for spans in contamination_results.values())
        logger.info(
            f"[{job_id}] Contamination audit complete: "
            f"{total_spans} spans across {len(contamination_results)} clusters"
        )

        # TODO: Apply auto-splits to clusters.json
        # For now, just log what would be split
        for cluster_name, spans in contamination_results.items():
            for span in spans:
                logger.info(
                    f"[{job_id}]   {cluster_name} - Track {span.track_id}: "
                    f"{span.reason} → {span.action}"
                )

    elapsed = time.time() - start_time
    logger.info(f"\n[{job_id}] Clean clustering complete in {elapsed:.1f}s")

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "num_clusters": len(cluster_metadata),
        "num_tracks_clustered": len(track_ids_for_clustering),
        "num_noise_tracks": sum(1 for c in track_to_cluster.values() if c == -1),
        "faces_only_retention": len(faces_only_df) / len(embeddings_df),
        "picked_samples": len(picked_df),
        "contamination_audit_enabled": contamination_config.get("enabled", True),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cluster_clean.py <EPISODE_ID>")
        sys.exit(1)

    episode_id = sys.argv[1]
    result = cluster_clean_task(f"clean_cluster_{episode_id}", episode_id)

    print(f"\n=== RESULTS ===")
    print(f"Clusters: {result['num_clusters']}")
    print(f"Tracks clustered: {result['num_tracks_clustered']}")
    print(f"Noise tracks: {result['num_noise_tracks']}")
    print(f"Faces-only retention: {result['faces_only_retention']*100:.1f}%")
    print(f"Picked samples: {result['picked_samples']}")
    print(f"Elapsed: {result['elapsed_seconds']:.1f}s")
