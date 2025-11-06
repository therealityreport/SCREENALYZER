"""
Cluster mutation operations.

Handles merge, split, and assignment operations on clusters.json.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from app.lib.data import load_clusters, load_tracks
from screentime.recognition.suggestions import MergeSuggester

logger = logging.getLogger(__name__)


class ClusterMutator:
    """Handles cluster mutations with atomic updates."""

    def __init__(self, episode_id: str, data_root: Path = Path("data")):
        """
        Initialize cluster mutator.

        Args:
            episode_id: Episode identifier
            data_root: Data root directory
        """
        self.episode_id = episode_id
        self.data_root = data_root
        self.harvest_dir = data_root / "harvest" / episode_id
        self.clusters_path = self.harvest_dir / "clusters.json"
        self.tracks_path = self.harvest_dir / "tracks.json"

    def merge_clusters(
        self, cluster_a_id: int, cluster_b_id: int, new_name: Optional[str] = None
    ) -> dict:
        """
        Merge two clusters into one.

        Args:
            cluster_a_id: First cluster ID
            cluster_b_id: Second cluster ID
            new_name: Optional name for merged cluster

        Returns:
            Updated clusters data
        """
        logger.info(f"Merging clusters {cluster_a_id} and {cluster_b_id}")

        # Load current data
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Find clusters
        cluster_a = next((c for c in clusters if c["cluster_id"] == cluster_a_id), None)
        cluster_b = next((c for c in clusters if c["cluster_id"] == cluster_b_id), None)

        if not cluster_a or not cluster_b:
            raise ValueError(f"Clusters {cluster_a_id} or {cluster_b_id} not found")

        # Merge track_ids
        merged_track_ids = list(set(cluster_a["track_ids"] + cluster_b["track_ids"]))

        # Calculate new quality score (weighted average)
        total_size = cluster_a["size"] + cluster_b["size"]
        new_quality_score = (
            cluster_a["quality_score"] * cluster_a["size"]
            + cluster_b["quality_score"] * cluster_b["size"]
        ) / total_size

        # Create merged cluster (keep cluster_a_id)
        merged_cluster = {
            "cluster_id": cluster_a_id,
            "size": len(merged_track_ids),
            "track_ids": merged_track_ids,
            "variance": (cluster_a["variance"] + cluster_b["variance"]) / 2,
            "silhouette_score": (cluster_a["silhouette_score"] + cluster_b["silhouette_score"]) / 2,
            "quality_score": round(new_quality_score, 4),
            "is_lowconf": new_quality_score < 0.6,
        }

        if new_name:
            merged_cluster["name"] = new_name

        # Remove old clusters and add merged
        clusters = [c for c in clusters if c["cluster_id"] not in [cluster_a_id, cluster_b_id]]
        clusters.append(merged_cluster)

        # Update clusters data
        clusters_data["clusters"] = clusters
        clusters_data["total_clusters"] = len(clusters)

        # Save atomically
        self._save_clusters_atomic(clusters_data)

        # Regenerate suggestions
        self._regenerate_suggestions(clusters_data)

        logger.info(f"Merged clusters {cluster_a_id} + {cluster_b_id} → {cluster_a_id}")

        return clusters_data

    def split_cluster(
        self, cluster_id: int, track_ids_a: list[int], track_ids_b: list[int]
    ) -> dict:
        """
        Split a cluster into two new clusters.

        Args:
            cluster_id: Cluster ID to split
            track_ids_a: Track IDs for first new cluster
            track_ids_b: Track IDs for second new cluster

        Returns:
            Updated clusters data
        """
        logger.info(f"Splitting cluster {cluster_id}")

        # Load current data
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Find cluster
        cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")

        # Get next cluster ID
        max_cluster_id = max(c["cluster_id"] for c in clusters)
        new_cluster_a_id = max_cluster_id + 1
        new_cluster_b_id = max_cluster_id + 2

        # Create new clusters
        new_cluster_a = {
            "cluster_id": new_cluster_a_id,
            "size": len(track_ids_a),
            "track_ids": track_ids_a,
            "variance": cluster["variance"],  # Inherit from parent
            "silhouette_score": cluster["silhouette_score"],
            "quality_score": cluster["quality_score"],
            "is_lowconf": cluster["is_lowconf"],
        }

        new_cluster_b = {
            "cluster_id": new_cluster_b_id,
            "size": len(track_ids_b),
            "track_ids": track_ids_b,
            "variance": cluster["variance"],
            "silhouette_score": cluster["silhouette_score"],
            "quality_score": cluster["quality_score"],
            "is_lowconf": cluster["is_lowconf"],
        }

        # Remove old cluster and add new ones
        clusters = [c for c in clusters if c["cluster_id"] != cluster_id]
        clusters.extend([new_cluster_a, new_cluster_b])

        # Update clusters data
        clusters_data["clusters"] = clusters
        clusters_data["total_clusters"] = len(clusters)

        # Save atomically
        self._save_clusters_atomic(clusters_data)

        # Regenerate suggestions
        self._regenerate_suggestions(clusters_data)

        logger.info(f"Split cluster {cluster_id} → {new_cluster_a_id}, {new_cluster_b_id}")

        return clusters_data

    def assign_name(self, cluster_id: int, person_name: str, lock: bool = True) -> dict:
        """
        Assign a name to a cluster and optionally lock it.

        Args:
            cluster_id: Cluster ID
            person_name: Person name
            lock: If True, set assignment_confidence=1.0 (locked)

        Returns:
            Dict with cluster_id, name, locked status, and clusters_data
        """
        logger.info(f"Assigning name '{person_name}' to cluster {cluster_id} (lock={lock})")

        # Load current data
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Find cluster
        cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")

        # Update name
        cluster["name"] = person_name

        # Set lock (assignment confidence)
        if lock:
            cluster["assignment_confidence"] = 1.0

        # Save atomically
        self._save_clusters_atomic(clusters_data)

        # Log constraints to track-level diagnostics (if cluster has tracks)
        track_ids = cluster.get("track_ids", [])
        if track_ids:
            self._log_constraints(
                track_ids,
                person_name,
                "assign_name",
                cluster_id=cluster_id,
                lock=lock,
            )

        # Mark analytics dirty
        self._mark_analytics_dirty("cluster assignment")

        logger.info(f"Assigned '{person_name}' to cluster {cluster_id}, locked={lock}")

        return {
            'cluster_id': cluster_id,
            'name': person_name,
            'locked': lock,
            'clusters_data': clusters_data
        }

    def move_track(self, track_id: int, from_cluster_id: int, to_cluster_id: int) -> dict:
        """
        Move a track from one cluster to another.

        Args:
            track_id: Track ID to move
            from_cluster_id: Source cluster ID
            to_cluster_id: Destination cluster ID (or -1 to create new cluster)

        Returns:
            Updated clusters data
        """
        logger.info(f"Moving track {track_id} from cluster {from_cluster_id} to {to_cluster_id}")

        # Load current data
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Find source cluster
        from_cluster = next((c for c in clusters if c["cluster_id"] == from_cluster_id), None)
        if not from_cluster:
            raise ValueError(f"Source cluster {from_cluster_id} not found")

        # Verify track exists in source cluster
        if track_id not in from_cluster["track_ids"]:
            raise ValueError(f"Track {track_id} not found in cluster {from_cluster_id}")

        # Remove track from source cluster
        from_cluster["track_ids"].remove(track_id)
        from_cluster["size"] = len(from_cluster["track_ids"])

        # Handle destination
        if to_cluster_id == -1:
            # Create new cluster
            max_cluster_id = max(c["cluster_id"] for c in clusters)
            new_cluster_id = max_cluster_id + 1

            new_cluster = {
                "cluster_id": new_cluster_id,
                "size": 1,
                "track_ids": [track_id],
                "variance": from_cluster["variance"],
                "silhouette_score": 0.0,
                "quality_score": 0.5,
                "is_lowconf": True,
            }

            clusters.append(new_cluster)
            logger.info(f"Created new cluster {new_cluster_id} with track {track_id}")
        else:
            # Find destination cluster
            to_cluster = next((c for c in clusters if c["cluster_id"] == to_cluster_id), None)
            if not to_cluster:
                raise ValueError(f"Destination cluster {to_cluster_id} not found")

            # Add track to destination cluster
            to_cluster["track_ids"].append(track_id)
            to_cluster["size"] = len(to_cluster["track_ids"])

        # Remove source cluster if empty
        if from_cluster["size"] == 0:
            clusters = [c for c in clusters if c["cluster_id"] != from_cluster_id]
            logger.info(f"Removed empty cluster {from_cluster_id}")

        # Update clusters data
        clusters_data["clusters"] = clusters
        clusters_data["total_clusters"] = len(clusters)

        # Save atomically
        self._save_clusters_atomic(clusters_data)

        logger.info(f"Moved track {track_id} from cluster {from_cluster_id} to {to_cluster_id}")

        return clusters_data

    def delete_track(self, track_id: int, cluster_id: int) -> dict:
        """
        Delete a track from a cluster.

        Args:
            track_id: Track ID to delete
            cluster_id: Cluster ID containing the track

        Returns:
            Updated clusters data
        """
        logger.info(f"Deleting track {track_id} from cluster {cluster_id}")

        # Load current data
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Find cluster
        cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        if not cluster:
            raise ValueError(f"Cluster {cluster_id} not found")

        # Verify track exists
        if track_id not in cluster["track_ids"]:
            raise ValueError(f"Track {track_id} not found in cluster {cluster_id}")

        # Remove track
        cluster["track_ids"].remove(track_id)
        cluster["size"] = len(cluster["track_ids"])

        # Remove cluster if empty
        if cluster["size"] == 0:
            clusters = [c for c in clusters if c["cluster_id"] != cluster_id]
            logger.info(f"Removed empty cluster {cluster_id}")

        # Update clusters data
        clusters_data["clusters"] = clusters
        clusters_data["total_clusters"] = len(clusters)

        # Save atomically
        self._save_clusters_atomic(clusters_data)

        logger.info(f"Deleted track {track_id} from cluster {cluster_id}")

        return clusters_data

    def delete_frame_from_track(self, track_id: int, frame_id: int) -> dict:
        """
        Delete a single frame from a track's frame_refs.

        Args:
            track_id: Track ID
            frame_id: Frame ID to remove

        Returns:
            Updated tracks data
        """
        logger.info(f"Deleting frame {frame_id} from track {track_id}")

        # Load tracks data
        tracks_data = load_tracks(self.episode_id, self.data_root)
        if not tracks_data:
            raise ValueError(f"No tracks data found for {self.episode_id}")

        tracks = tracks_data.get("tracks", [])

        # Find track
        track = next((t for t in tracks if t["track_id"] == track_id), None)
        if not track:
            raise ValueError(f"Track {track_id} not found")

        # Remove frame from frame_refs
        original_frame_count = len(track.get("frame_refs", []))
        track["frame_refs"] = [
            ref for ref in track.get("frame_refs", [])
            if ref["frame_id"] != frame_id
        ]
        new_frame_count = len(track["frame_refs"])

        if new_frame_count == original_frame_count:
            raise ValueError(f"Frame {frame_id} not found in track {track_id}")

        # Update track time boundaries if needed
        if track["frame_refs"]:
            # Track still has frames - update start/end times if necessary
            # (Keep original start_ms/end_ms unless this was first/last frame)
            pass
        else:
            # Track is now empty - should be deleted from cluster
            logger.warning(f"Track {track_id} is now empty after deleting frame {frame_id}")

        # Save tracks atomically
        self._save_tracks_atomic(tracks_data)

        # Regenerate suggestions to reflect updated prototypes
        clusters_data = load_clusters(self.episode_id, self.data_root)
        if clusters_data:
            self._regenerate_suggestions(clusters_data)

        logger.info(f"Deleted frame {frame_id} from track {track_id} ({original_frame_count} → {new_frame_count} frames)")

        return tracks_data

    def move_frame_to_cluster(self, source_track_id: int, frame_id: int, target_cluster_id: int) -> dict:
        """
        Move a single frame from one track/cluster to another cluster.

        Creates a new single-frame track in the target cluster.

        Args:
            source_track_id: Track ID containing the frame
            frame_id: Frame ID to move
            target_cluster_id: Target cluster ID

        Returns:
            Updated clusters data
        """
        logger.info(f"Moving frame {frame_id} from track {source_track_id} to cluster {target_cluster_id}")

        # Load data
        tracks_data = load_tracks(self.episode_id, self.data_root)
        clusters_data = load_clusters(self.episode_id, self.data_root)

        if not tracks_data or not clusters_data:
            raise ValueError(f"No data found for {self.episode_id}")

        tracks = tracks_data.get("tracks", [])
        clusters = clusters_data.get("clusters", [])

        # Find source track
        source_track = next((t for t in tracks if t["track_id"] == source_track_id), None)
        if not source_track:
            raise ValueError(f"Source track {source_track_id} not found")

        # Find frame in source track
        frame_ref = next((ref for ref in source_track.get("frame_refs", []) if ref["frame_id"] == frame_id), None)
        if not frame_ref:
            raise ValueError(f"Frame {frame_id} not found in track {source_track_id}")

        # Find source cluster (the cluster containing source_track_id)
        source_cluster = next((c for c in clusters if source_track_id in c.get("track_ids", [])), None)
        if not source_cluster:
            raise ValueError(f"Source cluster for track {source_track_id} not found")

        # Find target cluster
        target_cluster = next((c for c in clusters if c["cluster_id"] == target_cluster_id), None)
        if not target_cluster:
            raise ValueError(f"Target cluster {target_cluster_id} not found")

        # Create a new track with just this frame
        new_track_id = max(t["track_id"] for t in tracks) + 1
        new_track = {
            "track_id": new_track_id,
            "start_ms": source_track.get("start_ms", 0),  # Approximate - would need frame timestamp
            "end_ms": source_track.get("end_ms", 0),
            "frame_refs": [frame_ref]
        }
        tracks.append(new_track)

        # Add new track to target cluster
        target_cluster["track_ids"].append(new_track_id)
        target_cluster["size"] = len(target_cluster["track_ids"])

        # Remove frame from source track
        source_track["frame_refs"] = [ref for ref in source_track.get("frame_refs", []) if ref["frame_id"] != frame_id]

        # If source track is now empty, remove it from source cluster
        if not source_track["frame_refs"]:
            logger.info(f"Source track {source_track_id} is now empty, removing from cluster")
            source_cluster["track_ids"] = [tid for tid in source_cluster["track_ids"] if tid != source_track_id]
            source_cluster["size"] = len(source_cluster["track_ids"])
            # Remove track from tracks list
            tracks = [t for t in tracks if t["track_id"] != source_track_id]

        # Save updates
        tracks_data["tracks"] = tracks
        self._save_tracks_atomic(tracks_data)
        self._save_clusters_atomic(clusters_data)

        # Regenerate suggestions
        self._regenerate_suggestions(clusters_data)

        logger.info(f"Moved frame {frame_id} to new track {new_track_id} in cluster {target_cluster_id}")

        return clusters_data

    def _save_tracks_atomic(self, tracks_data: dict) -> None:
        """
        Save tracks data atomically.

        Args:
            tracks_data: Tracks data to save
        """
        # Create backup
        backup_path = self.tracks_path.with_suffix(".json.bak")
        if self.tracks_path.exists():
            shutil.copy(self.tracks_path, backup_path)

        # Write to temp file
        temp_path = self.tracks_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(tracks_data, f, indent=2)

        # Atomic rename
        temp_path.rename(self.tracks_path)

        logger.info(f"Saved tracks atomically to {self.tracks_path}")

    def _save_clusters_atomic(self, clusters_data: dict) -> None:
        """
        Save clusters data atomically.

        Args:
            clusters_data: Clusters data to save
        """
        # Create backup
        backup_path = self.clusters_path.with_suffix(".json.bak")
        if self.clusters_path.exists():
            shutil.copy(self.clusters_path, backup_path)

        # Write to temp file
        temp_path = self.clusters_path.with_suffix(".json.tmp")
        with open(temp_path, "w") as f:
            json.dump(clusters_data, f, indent=2)

        # Atomic rename
        temp_path.rename(self.clusters_path)

        logger.info(f"Saved clusters atomically to {self.clusters_path}")

    def _regenerate_suggestions(self, clusters_data: dict) -> None:
        """
        Regenerate merge suggestions after cluster mutations.

        Args:
            clusters_data: Updated clusters data
        """
        try:
            import numpy as np

            from app.lib.data import load_embeddings
            from screentime.recognition.cluster_dbscan import ClusterMetadata

            # Load embeddings to compute real centroids
            embeddings_df = load_embeddings(self.episode_id, self.data_root)
            tracks_data = load_tracks(self.episode_id, self.data_root)

            if embeddings_df is None or tracks_data is None:
                logger.warning("Cannot regenerate suggestions: missing embeddings or tracks data")
                return

            cluster_metadata = []
            for cluster in clusters_data.get("clusters", []):
                # Compute real centroid from track embeddings
                track_ids = cluster["track_ids"]
                track_embeddings = []

                for track_id in track_ids:
                    # Find track
                    track = next(
                        (t for t in tracks_data.get("tracks", []) if t["track_id"] == track_id),
                        None,
                    )
                    if not track:
                        continue

                    # Collect embeddings for this track
                    for ref in track.get("frame_refs", []):
                        frame_id = ref["frame_id"]
                        det_idx = ref["det_idx"]

                        emb_row = embeddings_df[
                            (embeddings_df["frame_id"] == frame_id)
                            & (embeddings_df["det_idx"] == det_idx)
                        ]

                        if len(emb_row) > 0:
                            embedding = np.array(emb_row.iloc[0]["embedding"])
                            track_embeddings.append(embedding)

                # Compute centroid (mean of all embeddings in cluster)
                if track_embeddings:
                    centroid = np.mean(track_embeddings, axis=0)
                else:
                    # Fallback to zeros only if no embeddings found
                    logger.warning(
                        f"Cluster {cluster['cluster_id']}: no embeddings found, using zero centroid"
                    )
                    centroid = np.zeros(512)

                metadata = ClusterMetadata(
                    cluster_id=cluster["cluster_id"],
                    track_ids=cluster["track_ids"],
                    size=cluster["size"],
                    centroid=centroid,
                    variance=cluster["variance"],
                    silhouette_score=cluster["silhouette_score"],
                    quality_score=cluster["quality_score"],
                    is_lowconf=cluster["is_lowconf"],
                )
                cluster_metadata.append(metadata)

            # Generate suggestions
            suggester = MergeSuggester()
            suggestions = suggester.generate_suggestions(cluster_metadata)

            # Save suggestions
            if suggestions:
                assist_dir = self.harvest_dir / "assist"
                assist_dir.mkdir(parents=True, exist_ok=True)

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

                logger.info(f"Regenerated {len(suggestions)} merge suggestions")

        except Exception as e:
            logger.warning(f"Failed to regenerate suggestions: {e}")

    def assign_tracks_to_identity(
        self,
        track_ids: list[int],
        identity: str,
        source_cluster_id: int | None = None
    ) -> dict:
        """
        Reassign given tracks to identity (create target cluster if needed).

        Args:
            track_ids: List of track IDs to move
            identity: Target identity name
            source_cluster_id: Optional source cluster (if None, derive per-track)

        Returns:
            {
                'moved': N,
                'identity': identity,
                'source_clusters': [ids],
                'new_cluster_id': optional,
                'orphans': K
            }
        """
        if not track_ids or not identity:
            raise ValueError("track_ids and identity must be non-empty")

        logger.info(f"Assigning {len(track_ids)} tracks to '{identity}'")

        clusters_data = load_clusters(self.episode_id, self.data_root)
        if not clusters_data:
            raise ValueError(f"No clusters data found for {self.episode_id}")

        clusters = clusters_data.get("clusters", [])

        # Build track -> cluster map if source_cluster_id is None
        track_to_cluster = {}
        if source_cluster_id is None:
            for cluster in clusters:
                for tid in cluster.get('track_ids', []):
                    track_to_cluster[tid] = cluster['cluster_id']

        # Find or create target cluster
        target_cluster = next((c for c in clusters if c.get('name') == identity), None)

        if not target_cluster:
            # Create new cluster for identity
            max_cluster_id = max(c["cluster_id"] for c in clusters) if clusters else 0
            new_cluster_id = max_cluster_id + 1

            target_cluster = {
                "cluster_id": new_cluster_id,
                "size": 0,
                "track_ids": [],
                "name": identity,
                "assignment_confidence": 1.0,  # Locked
                "variance": 0.0,
                "silhouette_score": 0.0,
                "quality_score": 0.5,
                "is_lowconf": False,
            }
            clusters.append(target_cluster)
            logger.info(f"Created new cluster {new_cluster_id} for '{identity}'")

        # Move tracks
        moved = 0
        orphans = 0
        source_clusters = set()

        for track_id in track_ids:
            # Find source cluster
            if source_cluster_id is not None:
                src_id = source_cluster_id
            else:
                src_id = track_to_cluster.get(track_id)
                if src_id is None:
                    orphans += 1
                    logger.warning(f"Track {track_id} has no source cluster (orphan)")

            if src_id is not None:
                source_clusters.add(src_id)
                src_cluster = next((c for c in clusters if c['cluster_id'] == src_id), None)

                if src_cluster and track_id in src_cluster['track_ids']:
                    src_cluster['track_ids'].remove(track_id)
                    src_cluster['size'] = len(src_cluster['track_ids'])

            # Add to target
            if track_id not in target_cluster['track_ids']:
                target_cluster['track_ids'].append(track_id)
                moved += 1

        target_cluster['size'] = len(target_cluster['track_ids'])

        # Remove empty source clusters
        clusters = [c for c in clusters if c['size'] > 0]
        clusters_data['clusters'] = clusters
        clusters_data['total_clusters'] = len(clusters)

        # Save
        self._save_clusters_atomic(clusters_data)

        # Log constraints
        self._log_constraints(track_ids, identity, "assign_tracks")

        # Mark analytics dirty
        self._mark_analytics_dirty("track assignment")

        logger.info(f"Assigned {moved} tracks to '{identity}' (orphans: {orphans})")

        return {
            'moved': moved,
            'identity': identity,
            'source_clusters': list(source_clusters),
            'new_cluster_id': target_cluster['cluster_id'],
            'orphans': orphans
        }

    def split_frames_and_assign(
        self,
        track_id: int,
        frame_ids: list[int],
        identity: str
    ) -> dict:
        """
        Split specific frames from a track and assign them to an identity.

        Creates a new track with the specified frames and assigns it to the target identity.

        Args:
            track_id: Source track ID
            frame_ids: List of frame IDs to split out
            identity: Target identity name

        Returns:
            {
                'new_track_id': int,
                'identity': identity,
                'frames_moved': int,
                'target_cluster_id': int
            }
        """
        from app.lib.data import load_tracks

        logger.info(f"Splitting {len(frame_ids)} frames from track {track_id} to {identity}")

        # Load tracks
        tracks_data = load_tracks(self.episode_id, self.data_root)
        if not tracks_data:
            raise ValueError(f"No tracks data found for {self.episode_id}")

        tracks = tracks_data.get('tracks', [])
        source_track = next((t for t in tracks if t['track_id'] == track_id), None)
        if not source_track:
            raise ValueError(f"Track {track_id} not found")

        # Split frame_refs
        frame_ids_set = set(frame_ids)
        split_refs = [ref for ref in source_track.get('frame_refs', []) if ref['frame_id'] in frame_ids_set]
        remaining_refs = [ref for ref in source_track.get('frame_refs', []) if ref['frame_id'] not in frame_ids_set]

        if not split_refs:
            raise ValueError(f"No matching frames found in track {track_id}")

        # Create new track with split frames
        max_track_id = max(t['track_id'] for t in tracks) if tracks else 0
        new_track_id = max_track_id + 1

        new_track = {
            'track_id': new_track_id,
            'frame_refs': split_refs,
            'start_ms': min(ref.get('ts_ms', 0) for ref in split_refs) if split_refs else 0,
            'end_ms': max(ref.get('ts_ms', 0) for ref in split_refs) if split_refs else 0,
        }

        # Update source track
        source_track['frame_refs'] = remaining_refs
        if remaining_refs:
            source_track['start_ms'] = min(ref.get('ts_ms', 0) for ref in remaining_refs)
            source_track['end_ms'] = max(ref.get('ts_ms', 0) for ref in remaining_refs)

        # Add new track
        tracks.append(new_track)

        # Save tracks
        tracks_data['tracks'] = tracks
        tracks_data['total_tracks'] = len(tracks)
        self._save_tracks_atomic(tracks_data)

        # Now assign the new track to the identity
        result = self.assign_tracks_to_identity([new_track_id], identity, source_cluster_id=None)

        # Log constraints
        self._log_constraints([new_track_id], identity, "split_and_assign", source_track_id=track_id, frame_count=len(split_refs))

        # Mark analytics dirty
        self._mark_analytics_dirty("frames split and assigned")

        return {
            'new_track_id': new_track_id,
            'identity': identity,
            'frames_moved': len(split_refs),
            'target_cluster_id': result['new_cluster_id']
        }

    def assign_cluster_name(self, cluster_id: int, person_name: str) -> dict:
        """
        Backwards compatibility alias for assign_name().

        Args:
            cluster_id: Cluster ID
            person_name: Person name

        Returns:
            Result from assign_name()
        """
        return self.assign_name(cluster_id, person_name, lock=True)

    def _log_constraints(self, track_ids: list[int], identity: str, action: str, **kwargs):
        """Log ML/CL constraints to track_constraints.jsonl."""
        diagnostics_dir = self.harvest_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        constraints_path = diagnostics_dir / "track_constraints.jsonl"

        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'action': action,
            'track_ids': track_ids,
            'identity': identity,
            **kwargs
        }

        with open(constraints_path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def _mark_analytics_dirty(self, reason: str):
        """Mark analytics as dirty (needs rebuild)."""
        try:
            from app.lib.analytics_dirty import mark_analytics_dirty
            mark_analytics_dirty(self.episode_id, self.data_root, reason=reason)
        except Exception as e:
            logger.warning(f"Failed to mark analytics dirty: {e}")
