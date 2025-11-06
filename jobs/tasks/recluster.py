"""
Re-clustering task: clustering-only (no re-detection/embedding).

Rebuilds picked_samples from existing embeddings, runs purity-driven DBSCAN,
applies open-set assignment with season bank, updates clusters.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from rq import get_current_job

logger = logging.getLogger(__name__)


def update_progress(step: str, percent: int):
    """Update job progress in RQ job.meta."""
    try:
        job = get_current_job()
        if job:
            job.meta['step'] = step
            job.meta['percent'] = percent
            job.meta['status'] = 'running'
            job.save_meta()
            logger.info(f"Progress: {step} ({percent}%)")
    except Exception as e:
        logger.warning(f"Failed to update progress: {e}")


def _compute_pairwise_centroid_sim_mean(
    track_centroids: dict[int, np.ndarray],
    track_ids: list[int],
) -> float:
    """Compute mean pairwise cosine similarity between track centroids in a cluster."""
    vectors = [
        track_centroids[tid]
        for tid in track_ids
        if tid in track_centroids and track_centroids[tid] is not None
    ]

    if not vectors:
        return 0.0
    if len(vectors) == 1:
        return 1.0

    matrix = np.stack(vectors, axis=0)
    sims = matrix @ matrix.T  # Since centroids are L2-normalized
    triu_idx = np.triu_indices(len(vectors), k=1)
    pairwise = sims[triu_idx]
    if pairwise.size == 0:
        return 1.0
    return float(np.mean(pairwise))


def _build_cluster_metrics(
    clusters: list[dict],
    track_metrics: dict[int, dict],
    track_centroids: dict[int, np.ndarray],
) -> list[dict]:
    """Aggregate per-cluster confidence metrics."""
    cluster_metrics = []

    for cluster in clusters:
        cluster_id = int(cluster["cluster_id"])
        name = cluster.get("name", "Unknown")
        track_ids = [int(tid) for tid in cluster.get("track_ids", [])]

        metrics_list = [track_metrics.get(tid) for tid in track_ids if track_metrics.get(tid)]
        n_tracks = len(metrics_list)

        if metrics_list:
            conf_p25_values = [m["conf_p25"] for m in metrics_list]
            total_frames = sum(m["n_frames"] for m in metrics_list)
            conflict_frames = sum(m["conflict_frac"] * m["n_frames"] for m in metrics_list)
            median_p25 = float(np.median(conf_p25_values))
            min_p25 = float(np.min(conf_p25_values))
            contam_rate = float(conflict_frames / total_frames) if total_frames else 0.0
        else:
            median_p25 = 0.0
            min_p25 = 0.0
            contam_rate = 0.0

        pairwise_sim = _compute_pairwise_centroid_sim_mean(track_centroids, track_ids)

        cluster_metrics.append(
            {
                "cluster_id": cluster_id,
                "name": name,
                "n_tracks": n_tracks,
                "tracks_conf_p25_median": median_p25,
                "contam_rate": contam_rate,
                "pairwise_centroid_sim_mean": pairwise_sim,
                "min_track_conf_p25": min_p25,
            }
        )

    return cluster_metrics


def _build_person_metrics(
    clusters: list[dict],
    track_metrics: dict[int, dict],
) -> list[dict]:
    """Aggregate per-person confidence metrics from cluster + track data."""
    identity_to_tracks: dict[str, list[dict]] = {}

    for cluster in clusters:
        identity = cluster.get("name") or "Unknown"
        if identity == "Unknown":
            continue

        track_ids = [int(tid) for tid in cluster.get("track_ids", [])]
        metrics_list = [track_metrics.get(tid) for tid in track_ids if track_metrics.get(tid)]
        if not metrics_list:
            continue
        identity_to_tracks.setdefault(identity, []).extend(metrics_list)

    person_metrics = []

    for identity, metrics_list in identity_to_tracks.items():
        conf_p25_values = [m["conf_p25"] for m in metrics_list]
        margin_values = [m["avg_margin"] for m in metrics_list if m["n_frames"] > 0]
        total_frames = sum(m["n_frames"] for m in metrics_list)
        conflict_frames = sum(m["conflict_frac"] * m["n_frames"] for m in metrics_list)
        n_tracks = len(metrics_list)

        bank_conf_median = float(np.median(conf_p25_values)) if conf_p25_values else 0.0
        bank_contam_rate = float(conflict_frames / total_frames) if total_frames else 0.0
        inter_id_margin = float(np.median(margin_values)) if margin_values else 0.0

        person_metrics.append(
            {
                "person": identity,
                "n_clusters": sum(1 for cluster in clusters if (cluster.get("name") or "Unknown") == identity),
                "n_tracks": n_tracks,
                "bank_conf_median_p25": bank_conf_median,
                "bank_contam_rate": bank_contam_rate,
                "inter_id_margin": inter_id_margin,
            }
        )

    person_metrics.sort(key=lambda x: x["bank_conf_median_p25"], reverse=True)
    return person_metrics


def recluster_task(
    job_id: str,
    episode_id: str,
    show_id: str = None,
    season_id: str = None,
    sources: list[str] = None,
    use_constraints: bool = False
) -> dict:
    """
    Re-cluster episode using existing embeddings.

    Args:
        job_id: Job ID
        episode_id: Episode identifier
        show_id: Show ID (for season bank lookup)
        season_id: Season ID (for season bank lookup)
        sources: Sources to include (baseline, entrance, densify)

    Returns:
        Result dict with cluster count, eps, etc.
    """
    from screentime.clustering.face_quality import filter_face_samples, pick_top_k_per_track
    from screentime.clustering.purity_driven_eps import purity_driven_eps_selection, save_purity_diagnostics
    from screentime.recognition.cluster_dbscan import DBSCANClusterer

    logger.info(f"[{job_id}] Starting re-clustering for {episode_id}")
    update_progress("Loading data", 5)

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    confidence_config = config.get('confidence', {}) or {}

    harvest_dir = Path("data/harvest") / episode_id

    # Load embeddings
    embeddings_path = harvest_dir / "embeddings.parquet"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

    embeddings_df = pd.read_parquet(embeddings_path)
    logger.info(f"[{job_id}] Loaded {len(embeddings_df)} embeddings")

    # Filter sources if specified (only if 'source' column exists)
    if sources and 'source' in embeddings_df.columns:
        embeddings_df = embeddings_df[embeddings_df['source'].isin(sources)]
        logger.info(f"[{job_id}] Filtered to sources {sources}: {len(embeddings_df)} embeddings")
    elif sources:
        logger.info(f"[{job_id}] Source filtering requested but 'source' column not found - using all embeddings")

    # Load tracks to add track_id to embeddings
    tracks_path = harvest_dir / "tracks.json"
    if not tracks_path.exists():
        raise FileNotFoundError(f"Tracks not found: {tracks_path}")

    with open(tracks_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data.get('tracks', [])
    logger.info(f"[{job_id}] Loaded {len(tracks)} tracks")

    # Add track_id to embeddings
    embeddings_df['track_id'] = None
    for track in tracks:
        track_id = track["track_id"]
        frame_refs = track["frame_refs"]
        for ref in frame_refs:
            frame_id = ref["frame_id"]
            det_idx = ref["det_idx"]
            mask = (embeddings_df["frame_id"] == frame_id) & (embeddings_df["det_idx"] == det_idx)
            embeddings_df.loc[mask, 'track_id'] = track_id

    embeddings_with_tracks = embeddings_df[embeddings_df['track_id'].notna()].copy()
    logger.info(f"[{job_id}] Mapped {len(embeddings_with_tracks)} / {len(embeddings_df)} embeddings to tracks")
    update_progress("Preparing embeddings", 15)

    # Filter suppressed items (deleted tracks/clusters)
    from app.lib.episode_status import load_suppress_data
    suppress_data = load_suppress_data(episode_id, Path("data"))
    deleted_tracks = set(suppress_data.get('deleted_tracks', []))

    if deleted_tracks:
        before_count = len(embeddings_with_tracks)
        embeddings_with_tracks = embeddings_with_tracks[~embeddings_with_tracks['track_id'].isin(deleted_tracks)]
        logger.info(f"[{job_id}] Filtered {before_count - len(embeddings_with_tracks)} suppressed tracks from clustering")

    # Convert bbox columns to bbox list if needed
    if 'bbox_x1' in embeddings_with_tracks.columns:
        embeddings_with_tracks['bbox'] = embeddings_with_tracks.apply(
            lambda row: [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']],
            axis=1
        )

    # Step 1: Face-only filtering + Top-K
    logger.info(f"[{job_id}] Applying face-quality filtering...")
    from screentime.clustering.face_quality import FaceQualityFilter

    clustering_config = config.get('clustering', {})
    face_quality_config = clustering_config.get('face_quality', {})

    face_filter = FaceQualityFilter(
        min_face_conf=face_quality_config.get('min_face_conf', 0.65),
        min_face_px=face_quality_config.get('min_face_px', 72),
        max_co_face_iou=face_quality_config.get('max_co_face_iou', 0.10),
        require_embedding=face_quality_config.get('require_embedding', True)
    )

    filtered_df = filter_face_samples(embeddings_with_tracks, tracks_data, face_filter)

    logger.info(f"[{job_id}] Face-quality filter: {len(filtered_df)}/{len(embeddings_with_tracks)} retained ({len(filtered_df)/len(embeddings_with_tracks)*100:.1f}%)")
    update_progress("Filtering faces", 25)

    # Top-K per track
    top_k = face_quality_config.get('top_k_per_track', 10)
    picked_df = pick_top_k_per_track(filtered_df, k=top_k)

    logger.info(f"[{job_id}] Top-K per track (K={top_k}): {len(picked_df)} samples")
    update_progress("Picking samples", 35)

    # Save picked_samples.parquet
    picked_samples_path = harvest_dir / "picked_samples.parquet"
    picked_df.to_parquet(picked_samples_path, index=False)
    logger.info(f"[{job_id}] Saved picked_samples.parquet")

    # Step 2: Prepare for clustering - compute track centroids
    track_embeddings = []
    track_ids = []
    track_embeddings_dict = {}  # Keep full embeddings for later centroid computation
    track_centroids: dict[int, np.ndarray] = {}

    for track_id, group in picked_df.groupby('track_id'):
        # Get all embeddings for this track
        embeddings = [np.array(emb, dtype=np.float32) for emb in group['embedding'].tolist()]
        track_embeddings_dict[track_id] = embeddings

        # Compute centroid for clustering
        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0.0:
            centroid = np.zeros_like(centroid)
        else:
            centroid = centroid / norm  # L2 normalize

        track_centroids[int(track_id)] = centroid

        track_embeddings.append(centroid)
        track_ids.append(track_id)

    track_embeddings = np.array(track_embeddings)
    logger.info(f"[{job_id}] Computed {len(track_embeddings)} track centroids from picked samples")

    # Step 3: Purity-driven eps selection
    logger.info(f"[{job_id}] Running purity-driven eps selection...")
    from screentime.clustering.purity_driven_eps import PurityConfig

    clustering_config = config.get('clustering', {})
    purity_config_dict = clustering_config.get('purity_driven_eps', {})

    purity_config = PurityConfig(
        eps_step=purity_config_dict.get('eps_step', 0.02),
        eps_range_offset=purity_config_dict.get('eps_range_offset', 0.10),
        impurity_weight=purity_config_dict.get('impurity_weight', 0.75),
        intra_sim_threshold=purity_config_dict.get('intra_sim_threshold', 0.75),
        cross_margin_threshold=purity_config_dict.get('cross_margin_threshold', 0.10),
        min_cluster_constraint=purity_config_dict.get('min_cluster_constraint', 5),
        max_cluster_size_percentile=purity_config_dict.get('max_cluster_size_percentile', 95)
    )

    purity_results = purity_driven_eps_selection(
        track_embeddings,
        min_samples=clustering_config.get('min_samples', 3),
        config=purity_config
    )

    eps_chosen = purity_results["eps_chosen"]
    logger.info(f"[{job_id}] Purity-driven eps chosen: {eps_chosen:.4f}")
    update_progress("Auto-tuning EPS", 50)

    # Save purity diagnostics
    save_purity_diagnostics(episode_id, Path("data"), purity_results)

    # Step 4: Run DBSCAN with chosen eps
    clusterer = DBSCANClusterer(
        eps=eps_chosen,
        min_samples=clustering_config.get('min_samples', 3)
    )

    cluster_metadata, track_to_cluster = clusterer.cluster(track_embeddings, track_ids)

    n_clusters = len(cluster_metadata)
    n_noise = len(clusterer.get_noise_tracks(track_to_cluster))

    logger.info(f"[{job_id}] Clustering complete: eps={eps_chosen:.3f}, clusters={n_clusters}, noise={n_noise}")
    update_progress("DBSCAN clustering", 60)

    # Step 4.5: Load existing clusters and extract constraints (if requested)
    constraints = None
    enforcement_diagnostics = {}
    consolidations = {}

    if use_constraints:
        logger.info(f"[{job_id}] Loading and merging constraints...")

        # Load old clusters.json
        old_clusters_path = harvest_dir / "clusters.json"
        if old_clusters_path.exists():
            with open(old_clusters_path) as f:
                old_clusters_data = json.load(f)

            from screentime.clustering.constraints import (
                extract_constraints_from_clusters,
                consolidate_same_name_clusters
            )

            # Extract constraints from current clusters
            constraints = extract_constraints_from_clusters(old_clusters_data)
            logger.info(f"[{job_id}] Extracted {len(constraints.must_link)} ML pairs, {len(constraints.cannot_link)} CL pairs from clusters")

            # Persist: Load additional constraints from track_constraints.jsonl
            track_constraints_path = harvest_dir / "diagnostics" / "track_constraints.jsonl"
            if track_constraints_path.exists():
                logger.info(f"[{job_id}] Loading persisted constraints from track_constraints.jsonl...")
                additional_ml = set()
                additional_cl = set()

                with open(track_constraints_path) as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            constraint_info = entry.get('constraints', {})

                            # Extract ML pairs
                            for ml_list_key in ['must_link_a', 'must_link_b', 'must_link_moved']:
                                for pair in constraint_info.get(ml_list_key, []):
                                    if isinstance(pair, list) and len(pair) == 2:
                                        additional_ml.add((min(pair[0], pair[1]), max(pair[0], pair[1])))

                            # Extract CL pairs
                            for pair in constraint_info.get('cannot_link', []):
                                if isinstance(pair, list) and len(pair) == 2:
                                    additional_cl.add((min(pair[0], pair[1]), max(pair[0], pair[1])))

                # Merge with extracted constraints (de-duplicate)
                merged_ml = list(set(constraints.must_link) | additional_ml)
                merged_cl = list(set(constraints.cannot_link) | additional_cl)

                logger.info(f"[{job_id}] Merged constraints: {len(additional_ml)} additional ML, {len(additional_cl)} additional CL")

                # Rebuild ConstraintSet with merged constraints
                from screentime.clustering.constraints import ConstraintSet, UnionFind
                uf = UnionFind()
                for tid_a, tid_b in merged_ml:
                    uf.union(tid_a, tid_b)
                ml_components = uf.get_components()

                constraints = ConstraintSet(
                    must_link=merged_ml,
                    cannot_link=merged_cl,
                    ml_components=ml_components
                )

                logger.info(f"[{job_id}] Total constraints after merge: ML={len(constraints.must_link)}, CL={len(constraints.cannot_link)}")

            # Apply same-name consolidation (default ON when use_constraints=True)
            logger.info(f"[{job_id}] Applying same-name consolidation...")
            constraints, consolidations = consolidate_same_name_clusters(
                old_clusters_data,
                constraints,
                min_similarity=0.75
            )

            logger.info(f"[{job_id}] Same-name consolidations: {consolidations}")
            update_progress("Consolidating", 90)
            logger.info(f"[{job_id}] Total ML pairs after consolidation: {len(constraints.must_link)}")
        else:
            logger.warning(f"[{job_id}] use_constraints=True but no existing clusters.json found - skipping constraints")

    # Step 5: Build clusters_data with centroids for open-set assignment
    clusters_data = []

    for cluster in cluster_metadata:
        # Compute centroid from all embeddings in this cluster
        cluster_embeddings = []
        for track_id in cluster.track_ids:
            if track_id in track_embeddings_dict:
                cluster_embeddings.extend(track_embeddings_dict[track_id])

        if cluster_embeddings:
            centroid = np.mean(cluster_embeddings, axis=0).astype(np.float32)
        else:
            centroid = None

        cluster_dict = {
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "track_ids": cluster.track_ids,
            "variance": round(cluster.variance, 4),
            "silhouette_score": round(cluster.silhouette_score, 4),
            "quality_score": round(cluster.quality_score, 4),
            "is_lowconf": cluster.is_lowconf,
            "centroid": centroid.tolist() if centroid is not None else None
        }

        clusters_data.append(cluster_dict)

    clusters_data.sort(key=lambda c: c["cluster_id"])

    # Step 5.5: Enforce constraints (if extracted)
    if constraints:
        logger.info(f"[{job_id}] Enforcing constraints on clustering results...")
        from screentime.clustering.constraints import enforce_constraints_post_clustering

        clusters_data, enforcement_diagnostics = enforce_constraints_post_clustering(
            clusters_data,
            constraints
        )

        logger.info(f"[{job_id}] Constraint enforcement: {enforcement_diagnostics.get('cl_violations_repaired', 0)} CL violations repaired")
    update_progress("Enforcing constraints", 70)

    # Step 6: Load season bank if available
    season_bank = None
    if show_id and season_id:
        season_bank_path = Path(f"data/facebank/{show_id}/{season_id}/multi_prototypes.json")
        if season_bank_path.exists():
            with open(season_bank_path) as f:
                season_bank = json.load(f)
            logger.info(f"[{job_id}] Loaded season bank: {len(season_bank.get('identities', {}))} identities")

            # Save diagnostics about loaded identities
            diagnostics_dir = Path(f"data/harvest/{episode_id}/diagnostics")
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            
            bank_diagnostics = {
                "episode_id": episode_id,
                "show_id": show_id,
                "season_id": season_id,
                "season_bank_path": str(season_bank_path),
                "loaded_identities": []
            }
            
            for person_name, prototypes in season_bank.get('identities', {}).items():
                # Count prototypes across all bins
                total_prototypes = 0
                bin_counts = {}
                embed_shape = None
                
                for bin_name, protos in prototypes.items():
                    bin_count = len(protos)
                    total_prototypes += bin_count
                    bin_counts[bin_name] = bin_count
                    if protos and 'embedding' in protos[0] and not embed_shape:
                        embed_shape = len(protos[0]['embedding'])
                
                bank_diagnostics["loaded_identities"].append({
                    "person": person_name,
                    "total_prototypes": total_prototypes,
                    "bin_counts": bin_counts,
                    "embedding_dim": embed_shape
                })
            
            bank_diagnostics["total_identities"] = len(bank_diagnostics["loaded_identities"])
            
            with open(diagnostics_dir / "cluster_bank_load.json", "w") as f:
                json.dump(bank_diagnostics, f, indent=2)
            
            logger.info(f"[{job_id}] Bank diagnostics: {bank_diagnostics['total_identities']} identities with {sum(p['total_prototypes'] for p in bank_diagnostics['loaded_identities'])} total prototypes")


    # Step 7: Open-set assignment
    clusters_output = {
        "episode_id": episode_id,
        "total_clusters": n_clusters,
        "noise_tracks": n_noise,
        "clusters": clusters_data
    }

    track_metrics: dict[int, dict] = {}
    cluster_metrics: list[dict] = []
    person_metrics: list[dict] = []

    if season_bank:
        logger.info(f"[{job_id}] Applying open-set assignment...")
        update_progress("Assigning names", 75)
        from screentime.clustering.open_set_assign import assign_clusters_open_set

        open_set_config = config.get('open_set', {})
        assignments = assign_clusters_open_set(
            clusters_output,
            season_bank,
            min_sim=open_set_config.get('min_sim', 0.60),
            min_margin=open_set_config.get('min_margin', 0.08)
        )

        # Update cluster names
        for cluster in clusters_output['clusters']:
            cluster_id = cluster['cluster_id']
            if cluster_id in assignments:
                cluster['name'] = assignments[cluster_id]['name']
                cluster['assignment_confidence'] = assignments[cluster_id]['confidence']
                cluster['assignment_margin'] = assignments[cluster_id]['margin']
            else:
                cluster['name'] = 'Unknown'
                cluster['assignment_confidence'] = 0.0
                cluster['assignment_margin'] = 0.0

        logger.info(f"[{job_id}] Assigned {len([c for c in clusters_output['clusters'] if c.get('name') != 'Unknown'])} clusters")
    else:
        logger.info(f"[{job_id}] No season bank found - all clusters marked as Unknown")
        for cluster in clusters_output['clusters']:
            cluster['name'] = 'Unknown'
            cluster['assignment_confidence'] = 0.0
            cluster['assignment_margin'] = 0.0

    # Step 7.5: Compute frame, track, cluster, and person confidence metrics
    if season_bank:
        logger.info(f"[{job_id}] Computing confidence metrics...")
        update_progress("Computing confidence", 85)
        from screentime.clustering.confidence_scoring import (
            score_all_tracks_and_frames,
            update_clusters_with_track_metrics
        )

        thresholds = {
            "frame_low": confidence_config.get("frame_low", 0.55),
            "top2_margin_low": confidence_config.get("top2_margin_low", 0.08),
        }

        picked_df_with_conf, track_metrics = score_all_tracks_and_frames(
            picked_df,
            clusters_output,
            season_bank,
            thresholds=thresholds
        )

        # Save updated picked_samples with frame diagnostics
        picked_df_with_conf.to_parquet(picked_samples_path, index=False)
        logger.info(f"[{job_id}] Updated picked_samples.parquet with frame confidence diagnostics")

        # Update clusters with per-track metrics
        clusters_output = update_clusters_with_track_metrics(clusters_output, track_metrics)
        cluster_metrics = _build_cluster_metrics(clusters_output['clusters'], track_metrics, track_centroids)
        person_metrics = _build_person_metrics(clusters_output['clusters'], track_metrics)
        clusters_output['cluster_metrics'] = cluster_metrics

        logger.info(f"[{job_id}] Added track metrics to {len(track_metrics)} tracks, cluster metrics to {len(cluster_metrics)} clusters, person metrics to {len(person_metrics)} people")

        # Log low-confidence tracks for review
        track_low_threshold = confidence_config.get("track_low_p25", 0.55)
        low_conf_tracks = [
            (tid, metrics['conf_p25'], metrics.get('identity', 'Unknown'))
            for tid, metrics in track_metrics.items()
            if metrics['conf_p25'] < track_low_threshold
        ]
        if low_conf_tracks:
            low_conf_tracks.sort(key=lambda x: x[1])
            worst = low_conf_tracks[0]
            logger.warning(
                f"[{job_id}] Found {len(low_conf_tracks)} tracks with conf_p25 < {track_low_threshold:.2f} - lowest: Track {worst[0]} ({worst[2]}) = {worst[1]:.3f}"
            )
    else:
        logger.info(f"[{job_id}] Skipping confidence scoring - no season bank available")

    # Step 8: Save updated clusters.json (remove centroid from saved file)
    clusters_to_save = {
        "episode_id": episode_id,
        "total_clusters": n_clusters,
        "noise_tracks": n_noise,
        "clusters": []
    }
    clusters_to_save["cluster_metrics"] = cluster_metrics

    for cluster in clusters_output['clusters']:
        # Remove centroid before saving (only needed for assignment)
        saved_cluster = {k: v for k, v in cluster.items() if k != 'centroid'}
        clusters_to_save['clusters'].append(saved_cluster)

    clusters_path = harvest_dir / "clusters.json"
    with open(clusters_path, 'w') as f:
        json.dump(clusters_to_save, f, indent=2)

    logger.info(f"[{job_id}] Updated clusters.json")

    diagnostics_dir = harvest_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    cluster_metrics_path = diagnostics_dir / "cluster_metrics.json"
    with open(cluster_metrics_path, 'w') as f:
        json.dump(cluster_metrics, f, indent=2)

    person_metrics_path = diagnostics_dir / "person_metrics.json"
    with open(person_metrics_path, 'w') as f:
        json.dump(person_metrics, f, indent=2)
    logger.info(f"[{job_id}] Persisted cluster/person metrics to diagnostics directory")

    # Save constraint diagnostics (if constraints were used)
    if constraints:
        from screentime.clustering.constraints import save_constraint_diagnostics
        save_constraint_diagnostics(episode_id, Path("data"), constraints, enforcement_diagnostics, consolidations)

    # Mark analytics as dirty (needs rebuild after re-clustering)
    from app.lib.analytics_dirty import mark_analytics_dirty
    mark_analytics_dirty(episode_id, Path("data"), reason="re-cluster completed")
    logger.info(f"[{job_id}] Marked analytics as dirty - rebuild needed")

    update_progress("Complete", 100)

    # Return result
    return {
        'status': 'completed',
        'episode_id': episode_id,
        'eps_chosen': eps_chosen,
        'n_clusters': len(clusters_to_save['clusters']),  # Use final cluster count after enforcement
        'n_noise': n_noise,
        'n_assigned': len([c for c in clusters_output['clusters'] if c.get('name') != 'Unknown']),
        'season_bank_used': season_bank is not None,
        'constraints_used': constraints is not None,
        'constraints': constraints.to_dict() if constraints else {},
        'analytics_marked_dirty': True
    }
