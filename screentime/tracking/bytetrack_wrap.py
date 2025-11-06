"""
ByteTrack-based face tracking.

Tracks faces across frames using IoU and embedding similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import lap
import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Face track across frames."""

    track_id: int
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    bboxes: list[list[int]] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    frame_ids: list[int] = field(default_factory=list)
    det_indices: list[int] = field(default_factory=list)
    state: str = "active"  # active, lost, terminated
    frames_since_update: int = 0
    labels: list[str] = field(default_factory=list)
    global_id: str = "UNK"

    @property
    def duration_ms(self) -> int:
        """Track duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def count(self) -> int:
        """Number of detections in track."""
        return len(self.frame_ids)

    def get_mean_embedding(self) -> np.ndarray:
        """Get mean embedding for track."""
        if not self.embeddings:
            return np.zeros(512)
        return np.mean(self.embeddings, axis=0)

    def update(
        self,
        bbox: list[int],
        embedding: np.ndarray,
        confidence: float,
        frame_id: int,
        det_idx: int,
        ts_ms: int,
    ) -> None:
        """Update track with new detection."""
        self.bboxes.append(bbox)
        self.embeddings.append(embedding)
        self.confidences.append(confidence)
        self.frame_ids.append(frame_id)
        self.det_indices.append(det_idx)
        self.end_frame = frame_id
        self.end_ms = ts_ms
        self.frames_since_update = 0
        self.state = "active"
        self.labels.append(self.global_id)


class ByteTracker:
    """ByteTrack-based face tracker."""

    def __init__(
        self,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        embedding_thresh: float = 0.6,
    ):
        """
        Initialize tracker.

        Args:
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Minimum similarity score for matching
            conf_thresh: Minimum detection confidence
            iou_thresh: IoU threshold for motion matching
            embedding_thresh: Cosine similarity threshold for appearance matching
        """
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.embedding_thresh = embedding_thresh

        self.tracks: list[Track] = []
        self.next_track_id = 1
        self.frame_count = 0

    def update(
        self,
        detections: list[dict],
        frame_id: int,
        ts_ms: int,
    ) -> list[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: List of detections with bbox, embedding, confidence, det_idx
            frame_id: Frame number
            ts_ms: Timestamp in milliseconds

        Returns:
            List of active tracks
        """
        self.frame_count += 1

        # Split detections by confidence
        high_conf_dets = [d for d in detections if d["confidence"] >= self.conf_thresh]
        low_conf_dets = [d for d in detections if d["confidence"] < self.conf_thresh]

        # Split tracks into active and lost
        active_tracks = [t for t in self.tracks if t.state == "active"]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]

        # First association: high-confidence detections with active tracks
        matched_tracks, unmatched_tracks, unmatched_dets = self._match(
            active_tracks, high_conf_dets, frame_id, ts_ms
        )

        # Second association: low-confidence detections with unmatched tracks
        if low_conf_dets and unmatched_tracks:
            matched_tracks2, unmatched_tracks2, unmatched_dets2 = self._match(
                unmatched_tracks, low_conf_dets, frame_id, ts_ms
            )
            matched_tracks.extend(matched_tracks2)
            unmatched_tracks = unmatched_tracks2
            unmatched_dets.extend(unmatched_dets2)

        # Third association: remaining detections with lost tracks
        remaining_dets = unmatched_dets
        if remaining_dets and lost_tracks:
            matched_tracks3, _, unmatched_dets3 = self._match(
                lost_tracks, remaining_dets, frame_id, ts_ms
            )
            matched_tracks.extend(matched_tracks3)
            unmatched_dets = unmatched_dets3

        # Mark unmatched tracks as lost
        for track in unmatched_tracks:
            track.frames_since_update += 1
            if track.frames_since_update <= self.track_buffer:
                track.state = "lost"
            else:
                track.state = "terminated"

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            new_track = Track(
                track_id=self.next_track_id,
                start_frame=frame_id,
                end_frame=frame_id,
                start_ms=ts_ms,
                end_ms=ts_ms,
            )
            new_track.update(
                bbox=det["bbox"],
                embedding=det["embedding"],
                confidence=det["confidence"],
                frame_id=frame_id,
                det_idx=det["det_idx"],
                ts_ms=ts_ms,
            )
            self.tracks.append(new_track)
            self.next_track_id += 1

        # Remove terminated tracks
        self.tracks = [t for t in self.tracks if t.state != "terminated"]

        # Return active tracks
        return [t for t in self.tracks if t.state == "active"]

    def _match(
        self,
        tracks: list[Track],
        detections: list[dict],
        frame_id: int,
        ts_ms: int,
    ) -> tuple[list[Track], list[Track], list[dict]]:
        """
        Match detections to tracks using IoU and embedding similarity.

        Returns:
            Tuple of (matched_tracks, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], tracks, detections

        # Compute cost matrix (lower is better)
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            last_bbox = track.bboxes[-1]
            track_embedding = track.get_mean_embedding()

            for j, det in enumerate(detections):
                det_bbox = det["bbox"]
                det_embedding = det["embedding"]

                # IoU similarity (motion)
                iou = self._compute_iou(last_bbox, det_bbox)

                # Embedding similarity (appearance)
                emb_sim = 1.0 - cosine(track_embedding, det_embedding)

                # Combined similarity (weighted average)
                similarity = 0.4 * iou + 0.6 * emb_sim

                # Convert to cost (1 - similarity)
                cost_matrix[i, j] = 1.0 - similarity

        # Solve assignment problem
        try:
            result = lap.lapjv(
                cost_matrix, extend_cost=True, cost_limit=1.0 - self.match_thresh
            )
            # Handle both lap 0.5.9 (2 returns) and 0.5.12+ (3 returns)
            if len(result) == 3:
                row_ind, col_ind, _ = result
            else:
                row_ind, col_ind = result
        except Exception as e:
            logger.warning(f"LAP solver failed: {e}, falling back to greedy matching")
            row_ind, col_ind = self._greedy_match(cost_matrix)

        # Collect matches and unmatched
        matched_tracks = []
        unmatched_track_indices = set(range(len(tracks)))
        unmatched_det_indices = set(range(len(detections)))

        for i, j in enumerate(col_ind):
            if j >= 0 and cost_matrix[i, j] < (1.0 - self.match_thresh):
                # Valid match
                track = tracks[i]
                det = detections[j]
                track.update(
                    bbox=det["bbox"],
                    embedding=det["embedding"],
                    confidence=det["confidence"],
                    frame_id=frame_id,
                    det_idx=det["det_idx"],
                    ts_ms=ts_ms,
                )
                matched_tracks.append(track)
                unmatched_track_indices.discard(i)
                unmatched_det_indices.discard(j)

        unmatched_tracks = [tracks[i] for i in unmatched_track_indices]
        unmatched_dets = [detections[j] for j in unmatched_det_indices]

        return matched_tracks, unmatched_tracks, unmatched_dets

    def _compute_iou(self, bbox1: list[int], bbox2: list[int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)
        inter_area = inter_width * inter_height

        # Union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def _greedy_match(self, cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Greedy matching fallback if LAP solver fails."""
        n_tracks, n_dets = cost_matrix.shape
        row_ind = np.arange(n_tracks)
        col_ind = np.full(n_tracks, -1, dtype=int)

        used_dets = set()
        for i in range(n_tracks):
            best_j = -1
            best_cost = float("inf")
            for j in range(n_dets):
                if j not in used_dets and cost_matrix[i, j] < best_cost:
                    best_j = j
                    best_cost = cost_matrix[i, j]

            if best_j >= 0:
                col_ind[i] = best_j
                used_dets.add(best_j)

        return row_ind, col_ind

    def get_all_tracks(self) -> list[Track]:
        """Get all tracks (active, lost, and terminated)."""
        return self.tracks

    def terminate_all_active_tracks(self) -> None:
        """Force termination of all active and lost tracks (e.g., at scene boundaries)."""
        for track in self.tracks:
            if track.state in ["active", "lost"]:
                track.state = "terminated"
        # Clear active/lost track lists
        self.active_tracks = []
        self.lost_tracks = []

    def get_stats(self) -> dict:
        """Get tracking statistics."""
        active_tracks = [t for t in self.tracks if t.state == "active"]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]
        terminated_tracks = [t for t in self.tracks if t.state == "terminated"]

        track_durations = [t.duration_ms for t in self.tracks if t.count > 1]
        track_counts = [t.count for t in self.tracks]

        return {
            "total_tracks": len(self.tracks),
            "active_tracks": len(active_tracks),
            "lost_tracks": len(lost_tracks),
            "terminated_tracks": len(terminated_tracks),
            "mean_track_duration_ms": np.mean(track_durations) if track_durations else 0,
            "mean_track_count": np.mean(track_counts) if track_counts else 0,
            "frames_processed": self.frame_count,
        }
