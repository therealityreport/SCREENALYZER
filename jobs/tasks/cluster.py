"""
Clustering task.

Clusters face tracks and generates merge suggestions.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from screentime.diagnostics.telemetry import telemetry, TelemetryEvent
from screentime.recognition.cluster_dbscan import DBSCANClusterer
from screentime.recognition.suggestions import MergeSuggester
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
from screentime.clustering.purity_driven_eps import (
    purity_driven_eps_selection,
    save_purity_diagnostics,
    PurityConfig,
)
from jobs.tasks.generate_face_stills import generate_face_stills_task

logger = logging.getLogger(__name__)


def cluster_task(job_id: str, episode_id: str) -> dict:
    """
    Cluster face tracks and generate merge suggestions.

    Args:
        job_id: Job ID
        episode_id: Episode ID

    Returns:
        Dict with clustering results
    """
    logger.info(f"[{job_id}] Starting cluster task for {episode_id}")

    start_time = time.time()

    # Emit initial progress
    from screentime.diagnostics.utils import emit_progress
    emit_progress(
        episode_id=episode_id,
        step="3. Agglomerative Clustering (Group Tracks)",
        step_index=3,
        total_steps=4,
        status="running",
        message="Loading tracks and embeddings...",
        pct=0.0,
    )

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    clustering_config = config["clustering"]

    # Setup paths
    harvest_dir = Path("data/harvest") / episode_id
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"

    if not tracks_path.exists():
        raise ValueError(f"Tracks not found: {tracks_path}")

    if not embeddings_path.exists():
        raise ValueError(f"Embeddings not found: {embeddings_path}")

    # Load tracks
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data["tracks"]
    logger.info(f"[{job_id}] Loaded {len(tracks)} tracks")

    # Load embeddings
    embeddings_df = pd.read_parquet(embeddings_path)
    logger.info(f"[{job_id}] Loaded {len(embeddings_df)} embeddings")

    # Add track_id to embeddings by matching frame_id + det_idx
    logger.info(f"[{job_id}] Adding track_id to embeddings...")
    embeddings_df['track_id'] = None

    for track in tracks:
        track_id = track["track_id"]
        frame_refs = track["frame_refs"]

        for ref in frame_refs:
            frame_id = ref["frame_id"]
            det_idx = ref["det_idx"]

            # Find matching rows in embeddings_df
            mask = (embeddings_df["frame_id"] == frame_id) & (embeddings_df["det_idx"] == det_idx)
            embeddings_df.loc[mask, 'track_id'] = track_id

    # Drop embeddings without track_id (orphaned detections)
    embeddings_with_tracks = embeddings_df[embeddings_df['track_id'].notna()].copy()
    logger.info(
        f"[{job_id}] Mapped {len(embeddings_with_tracks)} / {len(embeddings_df)} embeddings to tracks"
    )

    # Convert bbox columns to bbox list for face quality filter
    if 'bbox_x1' in embeddings_with_tracks.columns:
        embeddings_with_tracks['bbox'] = embeddings_with_tracks.apply(
            lambda row: [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
            axis=1
        )

    # ========================================
    # STEP 1: Face-Only Filtering
    # ========================================
    logger.info(f"[{job_id}] === STEP 1: Face-Only Filtering ===")

    face_quality_config = clustering_config.get(
        "face_quality",
        {
            "min_face_conf": 0.65,
            "min_face_px": 72,
            "max_co_face_iou": 0.10,
            "top_k_per_track": 10,
            "require_embedding": True,
        },
    )

    face_filter = FaceQualityFilter(
        min_face_conf=face_quality_config["min_face_conf"],
        min_face_px=face_quality_config["min_face_px"],
        max_co_face_iou=face_quality_config["max_co_face_iou"],
        require_embedding=face_quality_config["require_embedding"],
    )

    faces_only_df = filter_face_samples(embeddings_with_tracks, tracks_data, face_filter)

    logger.info(
        f"[{job_id}] Faces-only: {len(faces_only_df)} / {len(embeddings_with_tracks)} "
        f"({len(faces_only_df)/len(embeddings_with_tracks)*100:.1f}% retained)"
    )

    # ========================================
    # STEP 2: Top-K Selection Per Track
    # ========================================
    logger.info(f"[{job_id}] === STEP 2: Top-K Selection Per Track ===")

    top_k = face_quality_config["top_k_per_track"]
    picked_df = pick_top_k_per_track(faces_only_df, k=top_k)

    logger.info(
        f"[{job_id}] Picked {len(picked_df)} samples for centroids "
        f"({len(picked_df)/len(faces_only_df)*100:.1f}% of faces-only)"
    )

    # Save picked samples for gallery
    save_picked_samples(episode_id, Path("data"), picked_df)

    # ========================================
    # STEP 3: Compute Track Centroids from Picked Samples
    # ========================================
    logger.info(f"[{job_id}] === STEP 3: Compute Track Centroids ===")

    track_embeddings = []
    track_ids = []

    for track in tracks:
        track_id = track["track_id"]

        # Get picked samples for this track
        track_samples = picked_df[picked_df["track_id"] == track_id]

        if len(track_samples) == 0:
            logger.warning(f"[{job_id}] No picked samples for track {track_id} - skipping")
            continue

        # Compute centroid from picked embeddings
        embeddings = [np.array(row["embedding"]) for _, row in track_samples.iterrows()]
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)  # L2 normalize

        track_embeddings.append(centroid)
        track_ids.append(track_id)

    track_embeddings = np.array(track_embeddings)
    logger.info(f"[{job_id}] Computed {len(track_embeddings)} track centroids from picked samples")

    # Emit progress after centroids computed
    emit_progress(
        episode_id=episode_id,
        step="3. Agglomerative Clustering (Group Tracks)",
        step_index=3,
        total_steps=4,
        status="running",
        message=f"Computed {len(track_embeddings)} track centroids, finding optimal clustering...",
        pct=0.3,
    )

    # ========================================
    # STEP 3.5: Purity-Driven DBSCAN Epsilon Selection
    # ========================================
    logger.info(f"[{job_id}] === Purity-Driven Eps Selection ===")

    purity_config = PurityConfig(
        eps_step=0.02,
        eps_range_offset=0.10,
        impurity_weight=0.75,
        intra_sim_threshold=0.75,
        cross_margin_threshold=0.10,
        min_cluster_constraint=5,  # Expect at least 5-7 people in RHOBH
        max_cluster_size_percentile=95,
    )

    purity_results = purity_driven_eps_selection(
        track_embeddings,
        min_samples=clustering_config["min_samples"],
        config=purity_config
    )

    eps_chosen = purity_results["eps_chosen"]

    logger.info(
        f"[{job_id}] ✅ Purity-driven eps chosen: {eps_chosen:.4f} "
        f"(knee: {purity_results['knee_dist']:.4f}, "
        f"clusters: {purity_results['best_result']['n_clusters']}, "
        f"silhouette: {purity_results['best_result']['silhouette']:.3f}, "
        f"impurity: {purity_results['best_result']['impurity']:.3f})"
    )

    # Save purity diagnostics
    save_purity_diagnostics(episode_id, Path("data"), purity_results)

    # Emit progress after eps selection
    emit_progress(
        episode_id=episode_id,
        step="3. Agglomerative Clustering (Group Tracks)",
        step_index=3,
        total_steps=4,
        status="running",
        message=f"Eps selected ({eps_chosen:.4f}), running DBSCAN clustering...",
        pct=0.5,
    )

    # Run clustering with purity-driven eps
    clusterer = DBSCANClusterer(
        eps=eps_chosen,  # Use purity-driven eps
        min_samples=clustering_config["min_samples"],
    )

    cluster_metadata, track_to_cluster = clusterer.cluster(track_embeddings, track_ids)

    # Emit progress after clustering
    emit_progress(
        episode_id=episode_id,
        step="3. Agglomerative Clustering (Group Tracks)",
        step_index=3,
        total_steps=4,
        status="running",
        message=f"Built {len(cluster_metadata)} clusters, running contamination audit...",
        pct=0.7,
    )

    clustering_stats = defaultdict(int)
    clustering_stats["eps_chosen"] = eps_chosen
    clustering_stats["knee_dist"] = purity_results["knee_dist"]
    clustering_stats["n_candidates_evaluated"] = purity_results["n_candidates"]
    clustering_stats["silhouette"] = purity_results["best_result"]["silhouette"]
    clustering_stats["impurity"] = purity_results["best_result"]["impurity"]
    clustering_stats["quality_score"] = purity_results["best_result"]["quality_score"]
    clustering_stats["clusters_built"] = len(cluster_metadata)
    clustering_stats["noise_tracks"] = len(clusterer.get_noise_tracks(track_to_cluster))

    # Calculate cluster variance
    if cluster_metadata:
        variances = [c.variance for c in cluster_metadata]
        clustering_stats["cluster_variance_mean"] = float(np.mean(variances))
        clustering_stats["cluster_variance_std"] = float(np.std(variances))

    # Build clusters.json output
    clusters_data = []
    lowconf_clusters = []

    for cluster in cluster_metadata:
        cluster_dict = {
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "track_ids": cluster.track_ids,
            "variance": round(cluster.variance, 4),
            "silhouette_score": round(cluster.silhouette_score, 4),
            "quality_score": round(cluster.quality_score, 4),
            "is_lowconf": cluster.is_lowconf,
        }

        clusters_data.append(cluster_dict)

        if cluster.is_lowconf:
            lowconf_clusters.append(cluster_dict)

    # Sort by cluster_id
    clusters_data.sort(key=lambda c: c["cluster_id"])

    # ========================================
    # STEP 4: Contamination Audit + Auto-Split
    # ========================================
    contamination_config_dict = clustering_config.get(
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

    if contamination_config_dict.get("enabled", True):
        logger.info(f"[{job_id}] === STEP 4: Contamination Audit ===")

        contam_config = ContaminationConfig(
            outlier_mad_threshold=contamination_config_dict["outlier_mad_threshold"],
            outlier_sim_threshold=contamination_config_dict["outlier_sim_threshold"],
            cross_id_margin=contamination_config_dict["cross_id_margin"],
            min_contiguous_frames=contamination_config_dict["min_contiguous_frames"],
            auto_split_enabled=contamination_config_dict["auto_split_enabled"],
            min_evidence_strength=contamination_config_dict["min_evidence_strength"],
        )

        # Build clusters output for audit
        clusters_output = {
            "episode_id": episode_id,
            "total_clusters": len(clusters_data),
            "noise_tracks": clustering_stats["noise_tracks"],
            "clusters": clusters_data,
        }

        contamination_results = audit_all_clusters(
            clusters_output, picked_df, contam_config
        )

        save_contamination_audit(episode_id, Path("data"), contamination_results)

        total_spans = sum(len(spans) for spans in contamination_results.values())
        logger.info(
            f"[{job_id}] Contamination audit complete: "
            f"{total_spans} spans across {len(contamination_results)} clusters"
        )

        # Log what would be split (auto-split implementation TBD)
        for cluster_name, spans in contamination_results.items():
            for span in spans:
                logger.info(
                    f"[{job_id}]   {cluster_name} - Track {span.track_id}: "
                    f"{span.reason} → {span.action}"
                )

        clustering_stats["contamination_spans"] = total_spans
        clustering_stats["contamination_clusters"] = len(contamination_results)

    # Save clusters.json
    clusters_path = harvest_dir / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(
            {
                "episode_id": episode_id,
                "total_clusters": len(clusters_data),
                "noise_tracks": clustering_stats["noise_tracks"],
                "clusters": clusters_data,
            },
            f,
            indent=2,
        )

    logger.info(f"[{job_id}] Saved {len(clusters_data)} clusters to {clusters_path}")

    # Emit progress after saving clusters
    emit_progress(
        episode_id=episode_id,
        step="3. Agglomerative Clustering (Group Tracks)",
        step_index=3,
        total_steps=4,
        status="running",
        message=f"Saved {len(clusters_data)} clusters, generating merge suggestions...",
        pct=0.9,
    )

    # Generate merge suggestions
    suggester = MergeSuggester(
        similarity_threshold=0.75,
        max_suggestions=100,
    )

    suggestions = suggester.generate_suggestions(cluster_metadata)
    clustering_stats["suggestions_enqueued"] = len(suggestions)

    # Save merge suggestions
    assist_dir = harvest_dir / "assist"
    assist_dir.mkdir(parents=True, exist_ok=True)

    if suggestions:
        suggestions_data = [
            {
                "cluster_a_id": s.cluster_a_id,
                "cluster_b_id": s.cluster_b_id,
                "similarity": round(s.similarity, 4),
                "cluster_a_size": s.cluster_a_size,
                "cluster_b_size": s.cluster_b_size,
                "combined_size": s.combined_size,
                "rank": s.rank,
            }
            for s in suggestions
        ]

        suggestions_df = pd.DataFrame(suggestions_data)
        suggestions_path = assist_dir / "merge_suggestions.parquet"
        suggestions_df.to_parquet(suggestions_path, index=False)

        logger.info(f"[{job_id}] Saved {len(suggestions)} merge suggestions to {suggestions_path}")
    else:
        suggestions_path = assist_dir / "merge_suggestions.parquet"
        if not suggestions_path.exists():
            empty_df = pd.DataFrame(
                columns=[
                    "cluster_a_id",
                    "cluster_b_id",
                    "similarity",
                    "cluster_a_size",
                    "cluster_b_size",
                    "combined_size",
                    "rank",
                ]
            )
            empty_df.to_parquet(suggestions_path, index=False)
            logger.info(f"[{job_id}] Created empty merge suggestions file at {suggestions_path}")

    # Save low-confidence queue
    clustering_stats["lowconf_enqueued"] = len(lowconf_clusters)

    if lowconf_clusters:
        lowconf_data = [
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "quality_score": c["quality_score"],
                "variance": c["variance"],
                "track_ids": c["track_ids"],
            }
            for c in lowconf_clusters
        ]

        lowconf_df = pd.DataFrame(lowconf_data)
        lowconf_path = assist_dir / "lowconf_queue.parquet"
        lowconf_df.to_parquet(lowconf_path, index=False)

        logger.info(f"[{job_id}] Saved {len(lowconf_clusters)} low-conf clusters to {lowconf_path}")

    # Calculate stage time
    stage_time_ms = int((time.time() - start_time) * 1000)
    clustering_stats["stage_time_ms_clustering"] = stage_time_ms

    # Save clustering stats
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / "cluster_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(clustering_stats), f, indent=2)

    logger.info(f"[{job_id}] Clustering stats saved to {stats_path}")

    # Save final checkpoint
    _save_checkpoint(
        harvest_dir,
        {
            "job_id": job_id,
            "episode_id": episode_id,
            "last_completed_stage": "cluster",
            "clusters_built": len(clusters_data),
            "suggestions_enqueued": clustering_stats["suggestions_enqueued"],
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

    # Telemetry
    telemetry.log(
        TelemetryEvent.JOB_STAGE_COMPLETE,
        metadata={
            "job_id": job_id,
            "stage": "cluster",
            "clusters_built": len(clusters_data),
            "suggestions_enqueued": clustering_stats["suggestions_enqueued"],
            "lowconf_enqueued": clustering_stats["lowconf_enqueued"],
            "stage_time_ms": stage_time_ms,
        },
    )

    # Update job progress
    from api.jobs import job_manager

    job_manager.update_job_progress(job_id, "cluster", 100.0)

    try:
        job_manager.assist_queue.enqueue(
            generate_face_stills_task,
            args=(episode_id,),
            job_timeout="30m",  # Increased timeout for SER-FIQ processing
        )
        logger.info('[%s] Enqueued face stills generation for %s', job_id, episode_id)
    except Exception as exc:
        logger.warning('[%s] Failed to enqueue face stills generation for %s: %s', job_id, episode_id, exc)

    # Mark job as completed
    # Note: Analytics task is triggered manually from UI after user assigns cluster names
    # (see app/labeler.py Analytics page), not auto-enqueued here
    job_manager.complete_job(job_id)

    logger.info(f"[{job_id}] Cluster task completed in {stage_time_ms}ms")

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "clusters_path": str(clusters_path),
        "stats_path": str(stats_path),
        "stats": dict(clustering_stats),
    }


def _save_checkpoint(harvest_dir: Path, checkpoint_data: dict) -> None:
    """Save checkpoint to disk."""
    checkpoint_dir = harvest_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "job.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_file}")
