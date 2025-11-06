"""
Track re-identification and stitching across gaps.

Handles track fragmentation caused by occlusions, camera cuts, and detection gaps
by re-linking terminated tracks to new tracks based on appearance similarity.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TrackPrototype:
    """Representative embedding for a track."""

    track_id: int
    rep_vec: np.ndarray  # Mean embedding of top-K frames
    mean_conf: float  # Mean detection confidence
    mean_px: float  # Mean face size in pixels
    start_ms: int
    end_ms: int
    scene_ids: list[int]  # Scene segments this track appears in


class TrackReID:
    """
    Track re-identification using appearance similarity.

    Attempts to re-link terminated tracks to new tracks across temporal gaps
    using cosine similarity of representative embeddings.
    """

    def __init__(
        self,
        max_gap_ms: int = 2500,
        min_sim: float = 0.82,
        min_margin: float = 0.08,
        use_scene_bounds: bool = True,
        topk: int = 5,
        per_identity: dict[str, dict] = None,
    ):
        """
        Initialize track re-ID.

        Args:
            max_gap_ms: Maximum temporal gap to attempt re-linking
            min_sim: Minimum cosine similarity to accept match
            min_margin: Minimum margin over second-best match
            use_scene_bounds: Only link tracks within same/adjacent scenes
            topk: Number of top frames to use for representative embedding
            per_identity: Per-identity threshold overrides (e.g., {"PersonA": {"min_sim": 0.85, "min_margin": 0.10}})  # allowlist: example only
        """
        self.max_gap_ms = max_gap_ms
        self.min_sim = min_sim
        self.min_margin = min_margin
        self.use_scene_bounds = use_scene_bounds
        self.topk = topk
        self.per_identity = per_identity or {}

        self.stats = {
            "relink_attempts": 0,
            "relink_accepted": 0,
            "rejected_margin": 0,
            "rejected_similarity": 0,
            "track_relinked": 0,
            "frozen_skipped": 0,
            "reid_overrides_used": defaultdict(int),
            "reid_links_rejected_margin": defaultdict(int),
            "reid_links_rejected_similarity": defaultdict(int),
        }

    def compute_track_prototype(
        self,
        track: dict,
        embeddings_by_track: dict[int, list[tuple[np.ndarray, float, float]]],
    ) -> Optional[TrackPrototype]:
        """
        Compute representative embedding for a track.

        Args:
            track: Track dict with track_id, start_ms, end_ms, frame_refs
            embeddings_by_track: Map of track_id -> [(embedding, conf, face_px)]

        Returns:
            TrackPrototype or None if insufficient data
        """
        track_id = track["track_id"]
        track_embeddings = embeddings_by_track.get(track_id, [])

        if not track_embeddings:
            return None

        # Sort by quality (conf * face_px) and take top-K
        scored = [(emb, conf, px, conf * px) for emb, conf, px in track_embeddings]
        scored.sort(key=lambda x: x[3], reverse=True)
        top_samples = scored[: self.topk]

        if not top_samples:
            return None

        # Compute mean embedding
        embeddings = np.array([s[0] for s in top_samples])
        rep_vec = np.mean(embeddings, axis=0)

        # Normalize
        rep_vec = rep_vec / (np.linalg.norm(rep_vec) + 1e-8)

        # Compute mean stats
        mean_conf = float(np.mean([s[1] for s in top_samples]))
        mean_px = float(np.mean([s[2] for s in top_samples]))

        # Extract scene IDs (if available)
        scene_ids = track.get("scene_ids", [])

        return TrackPrototype(
            track_id=track_id,
            rep_vec=rep_vec,
            mean_conf=mean_conf,
            mean_px=mean_px,
            start_ms=track["start_ms"],
            end_ms=track["end_ms"],
            scene_ids=scene_ids,
        )

    @staticmethod
    def _resolve_identity(
        track: dict,
        track_id: int,
        identity_by_track: Optional[Dict[int, str]] = None,
    ) -> Optional[str]:
        """
        Resolve person identity for a track.

        Order of precedence:
        1. Explicit mapping supplied via identity_by_track (cluster-backed)
        2. Track-level annotation (`person_name` or `identity`)
        """
        if identity_by_track and track_id in identity_by_track:
            return identity_by_track[track_id]
        if "person_name" in track and track["person_name"]:
            return track["person_name"]
        if "identity" in track and track["identity"]:
            return track["identity"]
        return None

    def find_stitch_candidate(
        self,
        terminated_proto: TrackPrototype,
        new_tracks: list[dict],
        new_protos: dict[int, TrackPrototype],
        person_name: Optional[str] = None,
        identity_lookup: Optional[Dict[int, str]] = None,
    ) -> Optional[tuple[int, float]]:
        """
        Find best re-ID candidate for a terminated track.

        Args:
            terminated_proto: Prototype of terminated track
            new_tracks: List of new track dicts that started after termination
            new_protos: Map of track_id -> TrackPrototype for new tracks
            person_name: Optional identity name for per-identity thresholds

        Returns:
            (best_track_id, similarity) or None if no viable match
        """
        # Check if this identity is frozen (skip re-ID)
        if person_name and person_name in self.per_identity:
            identity_config = self.per_identity[person_name]
            if identity_config.get("freeze", False):
                logger.debug(f"Skipping re-ID for frozen identity: {person_name}")
                self.stats["frozen_skipped"] += 1
                return None

        self.stats["relink_attempts"] += 1

        # Get per-identity thresholds if configured
        min_sim = self.min_sim
        min_margin = self.min_margin
        if person_name and person_name in self.per_identity:
            identity_config = self.per_identity[person_name]
            min_sim = identity_config.get("min_sim", self.min_sim)
            min_margin = identity_config.get("min_margin", self.min_margin)
            self.stats["reid_overrides_used"][person_name] += 1

        candidates = []

        for new_track in new_tracks:
            new_track_id = new_track["track_id"]
            new_start_ms = new_track["start_ms"]

            if person_name and identity_lookup:
                candidate_identity = identity_lookup.get(new_track_id)
                if candidate_identity and candidate_identity != person_name:
                    continue

            # Check temporal gap
            gap_ms = new_start_ms - terminated_proto.end_ms
            if gap_ms < 0 or gap_ms > self.max_gap_ms:
                continue

            # Check scene bounds if enabled
            if self.use_scene_bounds and terminated_proto.scene_ids:
                new_proto = new_protos.get(new_track_id)
                if new_proto and new_proto.scene_ids:
                    # Allow same scene or adjacent scenes
                    last_scene = terminated_proto.scene_ids[-1] if terminated_proto.scene_ids else -1
                    first_new_scene = new_proto.scene_ids[0] if new_proto.scene_ids else -1

                    if abs(last_scene - first_new_scene) > 1:
                        continue

            # Compute similarity
            new_proto = new_protos.get(new_track_id)
            if not new_proto:
                continue

            sim = float(np.dot(terminated_proto.rep_vec, new_proto.rep_vec))
            candidates.append((new_track_id, sim, gap_ms))

        if not candidates:
            return None

        # Sort by similarity
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_id, best_sim, best_gap = candidates[0]

        # Check minimum similarity (use per-identity threshold if available)
        if best_sim < min_sim:
            self.stats["rejected_similarity"] += 1
            if person_name:
                self.stats["reid_links_rejected_similarity"][person_name] += 1
            if person_name:
                logger.debug(
                    f"Rejected relink for {person_name}: track {terminated_proto.track_id} -> {best_id} "
                    f"(sim={best_sim:.3f} < {min_sim:.3f})"
                )
            return None

        # Check margin over second-best (use per-identity threshold if available)
        if len(candidates) > 1:
            second_sim = candidates[1][1]
            margin = best_sim - second_sim

            if margin < min_margin:
                self.stats["rejected_margin"] += 1
                if person_name:
                    self.stats["reid_links_rejected_margin"][person_name] += 1
                logger.debug(
                    f"Rejected relink{f' for {person_name}' if person_name else ''}: "
                    f"track {terminated_proto.track_id} -> {best_id} "
                    f"(sim={best_sim:.3f}, margin={margin:.3f} < {min_margin:.3f})"
                )
                return None

        self.stats["relink_accepted"] += 1
        self.stats["track_relinked"] += 1

        logger.debug(
            f"Re-linked{f' {person_name}' if person_name else ''} track {terminated_proto.track_id} -> {best_id} "
            f"(sim={best_sim:.3f}, gap={best_gap}ms)"
        )

        return (best_id, best_sim)

    def stitch_tracks(
        self,
        tracks: list[dict],
        embeddings_by_track: dict[int, list[tuple[np.ndarray, float, float]]],
        cluster_assignments: Optional[dict[int, str]] = None,
        identity_by_track: Optional[Dict[int, str]] = None,
    ) -> tuple[list[dict], dict]:
        """
        Perform track stitching on a list of tracks.

        Args:
            tracks: List of track dicts
            embeddings_by_track: Map of track_id -> [(embedding, conf, face_px)]
            cluster_assignments: Optional map of track_id -> person_name for per-identity thresholds

        Returns:
            (updated_tracks, stitch_metadata)
        """
        logger.info(f"Starting track stitching on {len(tracks)} tracks")

        # Compute prototypes for all tracks
        protos = {}
        for track in tracks:
            proto = self.compute_track_prototype(track, embeddings_by_track)
            if proto:
                protos[track["track_id"]] = proto

        logger.info(f"Computed {len(protos)} track prototypes")

        # Sort tracks by end time
        sorted_tracks = sorted(tracks, key=lambda t: t["end_ms"])

        # Track linking metadata
        stitch_links = {}  # track_id -> (linked_to, score)

        # Process each track
        identity_lookup: Dict[int, str] = {}
        if identity_by_track:
            identity_lookup.update(identity_by_track)

        for i, track in enumerate(sorted_tracks):
            track_id = track["track_id"]
            proto = protos.get(track_id)

            if not proto:
                continue

            person_name = self._resolve_identity(track, track_id, identity_lookup)

            # Find candidates: tracks that start after this one ends
            candidate_tracks = [
                t
                for t in sorted_tracks[i + 1 :]
                if t["start_ms"] >= proto.end_ms
                and t["start_ms"] <= proto.end_ms + self.max_gap_ms
            ]

            if not candidate_tracks:
                continue

            # Find best match (with per-identity thresholds if person_name available)
            match = self.find_stitch_candidate(
                proto,
                candidate_tracks,
                protos,
                person_name,
                identity_lookup=identity_lookup,
            )

            if match:
                linked_to, score = match
                stitch_links[track_id] = (linked_to, score)

        # Update tracks with linking metadata
        updated_tracks = []
        for track in tracks:
            track_copy = dict(track)

            # Add linked_to if this track links forward
            if track["track_id"] in stitch_links:
                linked_to, score = stitch_links[track["track_id"]]
                track_copy["linked_to"] = linked_to
                track_copy["stitch_score"] = float(score)

            # Add linked_from if another track links to this one
            linked_from = [tid for tid, (to_id, _) in stitch_links.items() if to_id == track["track_id"]]
            if linked_from:
                track_copy["linked_from"] = linked_from

            updated_tracks.append(track_copy)

        logger.info(
            f"Track stitching complete: {self.stats['relink_accepted']}/{self.stats['relink_attempts']} links created"
        )

        return updated_tracks, {
            "stats": self.stats,
            "links": stitch_links,
        }

    def get_stats(self) -> dict:
        """Get stitching statistics."""
        stats_copy = dict(self.stats)
        stats_copy["reid_overrides_used"] = dict(stats_copy["reid_overrides_used"])
        stats_copy["reid_links_rejected_margin"] = dict(stats_copy["reid_links_rejected_margin"])
        stats_copy["reid_links_rejected_similarity"] = dict(
            stats_copy["reid_links_rejected_similarity"]
        )
        return stats_copy
