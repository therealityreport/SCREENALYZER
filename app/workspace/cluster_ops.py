"""
Cluster operations for Workspace P2.

Provides bulk operations, refinement, and state management for clusters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def reassign_tracks(
    track_ids: list[int],
    new_person_id: str,
    episode_id: str,
    data_root: str | Path = Path("data"),
) -> dict:
    """
    Re-assign selected tracks to a different person.

    Args:
        track_ids: List of track IDs to reassign
        new_person_id: Target person ID
        episode_id: Episode ID
        data_root: Data root directory

    Returns:
        Dict with operation results
    """
    data_root = Path(data_root)
    clusters_file = data_root / "harvest" / episode_id / "clusters.json"

    if not clusters_file.exists():
        return {"success": False, "error": "Clusters file not found"}

    try:
        # Load clusters
        with open(clusters_file, "r") as f:
            clusters_data = json.load(f)

        clusters = clusters_data.get("clusters", [])
        tracks_modified = 0
        affected_clusters = set()

        # Find and reassign tracks
        for cluster in clusters:
            original_person = cluster.get("person_id")
            tracks = cluster.get("tracks", [])

            # Check if any tracks in this cluster need reassignment
            tracks_to_reassign = [t for t in tracks if t.get("track_id") in track_ids]

            if tracks_to_reassign:
                affected_clusters.add(cluster.get("cluster_id"))

                # If reassigning ALL tracks in cluster, update cluster person_id
                if len(tracks_to_reassign) == len(tracks):
                    cluster["person_id"] = new_person_id
                    cluster["dirty"] = True
                    tracks_modified += len(tracks_to_reassign)
                else:
                    # Split cluster: move tracks to new/existing cluster
                    # For now, just mark as dirty and require refinement
                    cluster["dirty"] = True
                    tracks_modified += len(tracks_to_reassign)

                    # Update track assignments
                    for track in tracks_to_reassign:
                        track["person_id"] = new_person_id

        # Save updated clusters
        temp_file = clusters_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(clusters_data, f, indent=2)
        temp_file.replace(clusters_file)

        logger.info(
            f"Reassigned {tracks_modified} tracks to {new_person_id} "
            f"(affected {len(affected_clusters)} clusters)"
        )

        return {
            "success": True,
            "tracks_modified": tracks_modified,
            "affected_clusters": len(affected_clusters),
            "new_person_id": new_person_id,
        }

    except Exception as e:
        logger.error(f"Failed to reassign tracks: {e}")
        return {"success": False, "error": str(e)}


def delete_tracks(
    track_ids: list[int],
    episode_id: str,
    data_root: str | Path = Path("data"),
) -> dict:
    """
    Delete selected tracks from clusters.

    Args:
        track_ids: List of track IDs to delete
        episode_id: Episode ID
        data_root: Data root directory

    Returns:
        Dict with operation results
    """
    data_root = Path(data_root)
    clusters_file = data_root / "harvest" / episode_id / "clusters.json"

    if not clusters_file.exists():
        return {"success": False, "error": "Clusters file not found"}

    try:
        # Load clusters
        with open(clusters_file, "r") as f:
            clusters_data = json.load(f)

        clusters = clusters_data.get("clusters", [])
        tracks_deleted = 0
        clusters_to_remove = []

        # Remove tracks from clusters
        for i, cluster in enumerate(clusters):
            tracks = cluster.get("tracks", [])
            original_count = len(tracks)

            # Filter out deleted tracks
            cluster["tracks"] = [t for t in tracks if t.get("track_id") not in track_ids]

            tracks_deleted += original_count - len(cluster["tracks"])

            # Mark cluster as dirty if tracks were removed
            if len(cluster["tracks"]) < original_count:
                cluster["dirty"] = True

            # Mark empty clusters for removal
            if len(cluster["tracks"]) == 0:
                clusters_to_remove.append(i)

        # Remove empty clusters (reverse order to preserve indices)
        for idx in reversed(clusters_to_remove):
            clusters.pop(idx)

        clusters_data["clusters"] = clusters

        # Save updated clusters
        temp_file = clusters_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(clusters_data, f, indent=2)
        temp_file.replace(clusters_file)

        logger.info(
            f"Deleted {tracks_deleted} tracks, removed {len(clusters_to_remove)} empty clusters"
        )

        return {
            "success": True,
            "tracks_deleted": tracks_deleted,
            "clusters_removed": len(clusters_to_remove),
        }

    except Exception as e:
        logger.error(f"Failed to delete tracks: {e}")
        return {"success": False, "error": str(e)}


def refine_clusters(
    episode_id: str,
    data_root: str | Path = Path("data"),
    distance_threshold: float = 0.35,
    outlier_threshold: float = 0.35,
) -> dict:
    """
    Refine clusters by recomputing centroids, ejecting outliers, and merging duplicates.

    Args:
        episode_id: Episode ID
        data_root: Data root directory
        distance_threshold: Max distance for merging clusters
        outlier_threshold: Min similarity to keep track in cluster

    Returns:
        Dict with refinement results
    """
    data_root = Path(data_root)
    clusters_file = data_root / "harvest" / episode_id / "clusters.json"

    if not clusters_file.exists():
        return {"success": False, "error": "Clusters file not found"}

    try:
        # Load clusters
        with open(clusters_file, "r") as f:
            clusters_data = json.load(f)

        clusters = clusters_data.get("clusters", [])
        stats = {
            "clusters_updated": 0,
            "outliers_ejected": 0,
            "clusters_merged": 0,
            "centroids_recomputed": 0,
        }

        # Step 1: Recompute centroids and eject outliers
        for cluster in clusters:
            tracks = cluster.get("tracks", [])

            if len(tracks) == 0:
                continue

            # Extract embeddings
            embeddings = []
            for track in tracks:
                emb = track.get("embedding")
                if emb:
                    embeddings.append(np.array(emb))

            if len(embeddings) == 0:
                continue

            # Recompute centroid
            embeddings_array = np.array(embeddings)
            new_centroid = np.mean(embeddings_array, axis=0)
            cluster["centroid"] = new_centroid.tolist()
            stats["centroids_recomputed"] += 1

            # Eject outliers (tracks with similarity < threshold)
            tracks_to_keep = []
            for track, emb in zip(tracks, embeddings):
                similarity = np.dot(emb, new_centroid) / (
                    np.linalg.norm(emb) * np.linalg.norm(new_centroid)
                )

                if similarity >= outlier_threshold:
                    track["cluster_confidence"] = float(similarity)
                    tracks_to_keep.append(track)
                else:
                    stats["outliers_ejected"] += 1
                    logger.debug(
                        f"Ejected track {track.get('track_id')} "
                        f"from cluster {cluster.get('cluster_id')} (sim={similarity:.3f})"
                    )

            cluster["tracks"] = tracks_to_keep

            if len(tracks_to_keep) != len(tracks):
                cluster["dirty"] = True
                stats["clusters_updated"] += 1

        # Step 2: Merge similar clusters
        i = 0
        while i < len(clusters):
            cluster_i = clusters[i]
            centroid_i = np.array(cluster_i.get("centroid", []))

            if len(centroid_i) == 0:
                i += 1
                continue

            j = i + 1
            while j < len(clusters):
                cluster_j = clusters[j]
                centroid_j = np.array(cluster_j.get("centroid", []))

                if len(centroid_j) == 0:
                    j += 1
                    continue

                # Compute distance between centroids
                similarity = np.dot(centroid_i, centroid_j) / (
                    np.linalg.norm(centroid_i) * np.linalg.norm(centroid_j)
                )

                # Merge if similar enough
                if similarity >= (1.0 - distance_threshold):
                    logger.info(
                        f"Merging cluster {cluster_j.get('cluster_id')} "
                        f"into {cluster_i.get('cluster_id')} (sim={similarity:.3f})"
                    )

                    # Merge tracks
                    cluster_i["tracks"].extend(cluster_j["tracks"])
                    cluster_i["dirty"] = True

                    # Recompute centroid for merged cluster
                    embeddings = [
                        np.array(t.get("embedding"))
                        for t in cluster_i["tracks"]
                        if t.get("embedding")
                    ]
                    if embeddings:
                        new_centroid = np.mean(np.array(embeddings), axis=0)
                        cluster_i["centroid"] = new_centroid.tolist()
                        centroid_i = new_centroid

                    # Remove cluster_j
                    clusters.pop(j)
                    stats["clusters_merged"] += 1
                    stats["clusters_updated"] += 1
                else:
                    j += 1

            i += 1

        clusters_data["clusters"] = clusters

        # Save refined clusters
        temp_file = clusters_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(clusters_data, f, indent=2)
        temp_file.replace(clusters_file)

        logger.info(f"Cluster refinement complete: {stats}")

        return {
            "success": True,
            **stats,
        }

    except Exception as e:
        logger.error(f"Failed to refine clusters: {e}")
        return {"success": False, "error": str(e)}


def get_cluster_stats(
    episode_id: str,
    data_root: str | Path = Path("data"),
) -> dict:
    """
    Get statistics about clusters for an episode.

    Args:
        episode_id: Episode ID
        data_root: Data root directory

    Returns:
        Dict with cluster statistics
    """
    data_root = Path(data_root)
    clusters_file = data_root / "harvest" / episode_id / "clusters.json"

    if not clusters_file.exists():
        return {
            "total_clusters": 0,
            "assigned_clusters": 0,
            "unassigned_clusters": 0,
            "low_confidence_clusters": 0,
            "dirty_clusters": 0,
            "total_tracks": 0,
        }

    try:
        with open(clusters_file, "r") as f:
            clusters_data = json.load(f)

        clusters = clusters_data.get("clusters", [])

        stats = {
            "total_clusters": len(clusters),
            "assigned_clusters": 0,
            "unassigned_clusters": 0,
            "low_confidence_clusters": 0,
            "dirty_clusters": 0,
            "total_tracks": 0,
        }

        for cluster in clusters:
            person_id = cluster.get("person_id")
            tracks = cluster.get("tracks", [])
            stats["total_tracks"] += len(tracks)

            if cluster.get("dirty"):
                stats["dirty_clusters"] += 1

            if person_id and person_id != "unassigned":
                stats["assigned_clusters"] += 1
            else:
                stats["unassigned_clusters"] += 1

            # Low confidence: avg cluster confidence < 0.6
            confidences = [t.get("cluster_confidence", 0) for t in tracks]
            if confidences and np.mean(confidences) < 0.6:
                stats["low_confidence_clusters"] += 1

        return stats

    except Exception as e:
        logger.error(f"Failed to get cluster stats: {e}")
        return {
            "total_clusters": 0,
            "assigned_clusters": 0,
            "unassigned_clusters": 0,
            "low_confidence_clusters": 0,
            "dirty_clusters": 0,
            "total_tracks": 0,
            "error": str(e),
        }
