"""
Cluster mixedness audit - detect potential identity contamination.

Analyzes clusters for signs of mixed identities:
- High intra-cluster variance
- Low silhouette scores
- Low centroid separation margin
- Unusually long tracks (potential cross-cut associations)
- High percentage of low-confidence frames
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def cluster_audit_task(episode_id: str) -> dict[str, Any]:
    """
    Audit clusters for potential identity contamination.

    Args:
        episode_id: Episode ID to audit

    Returns:
        Dict with audit results and suspected mixed clusters
    """
    logger.info(f"Starting cluster audit for {episode_id}")

    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id

    # Load data
    clusters_path = harvest_dir / "clusters.json"
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"

    if not all(p.exists() for p in [clusters_path, tracks_path, embeddings_path]):
        raise ValueError(f"Missing data files for {episode_id}")

    with open(clusters_path) as f:
        clusters_data = json.load(f)

    with open(tracks_path) as f:
        tracks_data = json.load(f)

    embeddings_df = pd.read_parquet(embeddings_path)

    # Build track index
    tracks_by_id = {t["track_id"]: t for t in tracks_data["tracks"]}

    # Build track -> embeddings map via frame_refs
    track_embeddings = {}
    for track_id, track in tracks_by_id.items():
        track_embeds = []
        for frame_ref in track.get("frame_refs", []):
            frame_id = frame_ref["frame_id"]
            det_idx = frame_ref["det_idx"]
            # Find matching embedding
            match = embeddings_df[
                (embeddings_df["frame_id"] == frame_id) &
                (embeddings_df["det_idx"] == det_idx)
            ]
            if len(match) > 0:
                track_embeds.append(match.iloc[0])
        if track_embeds:
            track_embeddings[track_id] = track_embeds

    # Audit each cluster
    audit_results = []

    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        track_ids = cluster["track_ids"]
        cluster_name = cluster.get("name", "UNLABELED")

        logger.info(f"Auditing cluster {cluster_id} ({cluster_name}): {len(track_ids)} tracks")

        # Collect all embeddings for this cluster
        cluster_embeddings = []
        cluster_confidences = []
        cluster_face_sizes = []

        for track_id in track_ids:
            if track_id in track_embeddings:
                track_embeds_list = track_embeddings[track_id]
                for row in track_embeds_list:
                    cluster_embeddings.append(np.array(row["embedding"]))
                    cluster_confidences.append(row["confidence"])
                    face_w = row["bbox_x2"] - row["bbox_x1"]
                    face_h = row["bbox_y2"] - row["bbox_y1"]
                    cluster_face_sizes.append(np.sqrt(face_w * face_h))

        if len(cluster_embeddings) < 2:
            logger.warning(f"Cluster {cluster_id} has <2 embeddings, skipping")
            continue

        cluster_embeddings = np.array(cluster_embeddings)

        # Compute intra-cluster variance
        cluster_mean = cluster_embeddings.mean(axis=0)
        intra_variance = np.mean([cosine(emb, cluster_mean) for emb in cluster_embeddings])

        # Compute silhouette score (need at least 2 clusters)
        if len(clusters_data["clusters"]) >= 2:
            # Create labels array for all embeddings
            all_embeddings = []
            all_labels = []
            for c in clusters_data["clusters"]:
                for tid in c["track_ids"]:
                    if tid in track_embeddings:
                        track_embeds_list = track_embeddings[tid]
                        for row in track_embeds_list:
                            all_embeddings.append(np.array(row["embedding"]))
                            all_labels.append(c["cluster_id"])

            if len(all_embeddings) >= len(clusters_data["clusters"]):
                try:
                    silhouette = silhouette_score(
                        np.array(all_embeddings), all_labels, metric="cosine"
                    )
                except:
                    silhouette = None
            else:
                silhouette = None
        else:
            silhouette = None

        # Find centroid separation (distance to nearest other cluster)
        min_centroid_dist = 1.0  # Max cosine distance
        for other_cluster in clusters_data["clusters"]:
            if other_cluster["cluster_id"] == cluster_id:
                continue

            other_embeddings = []
            for tid in other_cluster["track_ids"]:
                if tid in track_embeddings:
                    track_embeds_list = track_embeddings[tid]
                    for row in track_embeds_list:
                        other_embeddings.append(np.array(row["embedding"]))

            if len(other_embeddings) > 0:
                other_mean = np.array(other_embeddings).mean(axis=0)
                dist = cosine(cluster_mean, other_mean)
                min_centroid_dist = min(min_centroid_dist, dist)

        # Analyze tracks
        track_durations = []
        long_tracks = []

        for track_id in track_ids:
            if track_id in tracks_by_id:
                track = tracks_by_id[track_id]
                duration_ms = track["end_ms"] - track["start_ms"]
                track_durations.append(duration_ms)

                # Flag tracks > 10 seconds (potential cross-cut associations)
                if duration_ms > 10000:
                    long_tracks.append(
                        {
                            "track_id": track_id,
                            "duration_ms": duration_ms,
                            "start_ms": track["start_ms"],
                            "end_ms": track["end_ms"],
                        }
                    )

        # Compute quality metrics
        mean_confidence = np.mean(cluster_confidences) if cluster_confidences else 0
        mean_face_size = np.mean(cluster_face_sizes) if cluster_face_sizes else 0
        pct_low_conf = (
            (np.array(cluster_confidences) < 0.7).sum() / len(cluster_confidences) * 100
            if cluster_confidences
            else 0
        )

        # Determine if cluster is suspected mixed
        suspected_mixed = False
        reasons = []

        if intra_variance > 0.35:  # High internal variance
            suspected_mixed = True
            reasons.append(f"high_variance:{intra_variance:.3f}")

        if min_centroid_dist < 0.15:  # Too close to another cluster
            suspected_mixed = True
            reasons.append(f"low_separation:{min_centroid_dist:.3f}")

        if len(long_tracks) > 0:  # Has suspiciously long tracks
            suspected_mixed = True
            reasons.append(f"long_tracks:{len(long_tracks)}")

        if pct_low_conf > 40:  # More than 40% low confidence
            suspected_mixed = True
            reasons.append(f"low_conf:{pct_low_conf:.1f}%")

        audit_result = {
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "size": len(track_ids),
            "total_embeddings": len(cluster_embeddings),
            "intra_variance": float(intra_variance),
            "silhouette_score": float(silhouette) if silhouette is not None else None,
            "min_centroid_distance": float(min_centroid_dist),
            "mean_confidence": float(mean_confidence),
            "mean_face_size": float(mean_face_size),
            "pct_low_confidence": float(pct_low_conf),
            "max_track_duration_ms": max(track_durations) if track_durations else 0,
            "long_tracks_count": len(long_tracks),
            "long_tracks": long_tracks[:5],  # Top 5 longest
            "suspected_mixed": suspected_mixed,
            "reasons": reasons,
        }

        audit_results.append(audit_result)

    # Save audit report
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_path = reports_dir / "mixedness_report.json"

    audit_report = {
        "episode_id": episode_id,
        "total_clusters": len(clusters_data["clusters"]),
        "suspected_mixed": sum(1 for r in audit_results if r["suspected_mixed"]),
        "clusters": audit_results,
    }

    with open(audit_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(
        f"Audit complete: {audit_report['suspected_mixed']}/{audit_report['total_clusters']} "
        f"clusters flagged. Report saved to {audit_path}"
    )

    return audit_report


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m jobs.tasks.cluster_audit <episode_id>")
        sys.exit(1)

    episode_id = sys.argv[1]
    result = cluster_audit_task(episode_id)

    print("\n" + "=" * 70)
    print("CLUSTER MIXEDNESS AUDIT")
    print("=" * 70)
    print(f"\nEpisode: {episode_id}")
    print(f"Total clusters: {result['total_clusters']}")
    print(f"Suspected mixed: {result['suspected_mixed']}")
    print()

    for cluster in result["clusters"]:
        flag = "⚠️ FLAGGED" if cluster["suspected_mixed"] else "✓ Clean"
        print(f"Cluster {cluster['cluster_id']} ({cluster['cluster_name']}): {flag}")

        if cluster["suspected_mixed"]:
            print(f"  Reasons: {', '.join(cluster['reasons'])}")
            if cluster["long_tracks_count"] > 0:
                print(f"  Long tracks: {cluster['long_tracks_count']}")
                for lt in cluster["long_tracks"][:3]:
                    print(
                        f"    Track {lt['track_id']}: {lt['duration_ms']/1000:.1f}s "
                        f"({lt['start_ms']}-{lt['end_ms']}ms)"
                    )
