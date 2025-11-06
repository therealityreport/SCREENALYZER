"""
Interval-merge timeline algorithm.

Definitive screen-time calculation with co-appearance credit.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Merge threshold: detections <2s apart are fused
MERGE_THRESHOLD_MS = 2000


@dataclass
class Interval:
    """Screen-time interval for a person."""

    person_name: str
    start_ms: int
    end_ms: int
    source: str  # track_id or "merged"
    confidence: float
    frame_count: int = 0
    visible_fraction: float = 0.0

    @property
    def duration_ms(self) -> int:
        """Interval duration in milliseconds."""
        return self.end_ms - self.start_ms

    def overlaps(self, other: Interval, threshold_ms: int = MERGE_THRESHOLD_MS) -> bool:
        """Check if this interval overlaps or is within threshold of another."""
        # Check if gap between intervals is <= threshold
        if self.end_ms < other.start_ms:
            gap = other.start_ms - self.end_ms
            return gap <= threshold_ms
        elif other.end_ms < self.start_ms:
            gap = self.start_ms - other.end_ms
            return gap <= threshold_ms
        else:
            # Direct overlap
            return True


class TimelineBuilder:
    """Build screen-time timelines with interval merging."""

    def __init__(
        self,
        gap_merge_ms_base: int = 2500,
        gap_merge_ms_max: int = 3000,
        min_interval_quality: float = 0.60,
        conflict_guard_ms: int = 500,
        use_scene_bounds: bool = True,
        edge_epsilon_ms: int = 0,
        per_identity: dict[str, dict] = None,
    ):
        """
        Initialize timeline builder with adaptive merging.

        Args:
            gap_merge_ms_base: Base gap threshold for merging
            gap_merge_ms_max: Maximum gap threshold for high-quality intervals
            min_interval_quality: Minimum quality to use max gap
            conflict_guard_ms: Minimum gap to avoid cross-person conflicts
            use_scene_bounds: Use scene boundaries for merging decisions
            edge_epsilon_ms: Treat gaps ≤ this as continuous (for edge quantization, e.g., 120ms)
            per_identity: Per-identity threshold overrides (person_name -> {gap_merge_ms_max, min_interval_quality})
        """
        self.gap_merge_ms_base = gap_merge_ms_base
        self.gap_merge_ms_max = gap_merge_ms_max
        self.min_interval_quality = min_interval_quality
        self.conflict_guard_ms = conflict_guard_ms
        self.use_scene_bounds = use_scene_bounds
        self.edge_epsilon_ms = edge_epsilon_ms
        self.per_identity = per_identity or {}

        # Legacy support
        self.merge_threshold_ms = gap_merge_ms_base

        # Telemetry
        self.merge_stats = {
            "gaps_seen": 0,
            "gaps_merged": 0,
            "merge_conflicts_blocked": 0,
            "quality_bumps_applied": 0,
            "edge_epsilon_merged": 0,
        }
        self.merge_stats["merge_clamped"] = defaultdict(int)
        self.merge_stats["timeline_merges_suppressed_by_visibility"] = 0
        self._visibility_audit: list[dict[str, Any]] = []

    def build_timeline(
        self,
        clusters_data: dict,
        tracks_data: dict,
        cluster_assignments: dict[int, str],
    ) -> tuple[list[Interval], dict[str, dict]]:
        """
        Build screen-time timeline from clusters and tracks.

        Args:
            clusters_data: Clusters data from load_clusters()
            tracks_data: Tracks data from load_tracks()
            cluster_assignments: Map of cluster_id -> person_name

        Returns:
            Tuple of (intervals, totals_by_person)
        """
        # Step 1: Collect all intervals from assigned clusters
        raw_intervals = []

        clusters = clusters_data.get("clusters", [])
        tracks = tracks_data.get("tracks", [])

        # Track schema validation stats for diagnostics
        self.schema_stats = {
            "total_tracks": len(tracks),
            "tracks_with_start_ms": len([t for t in tracks if "start_ms" in t]),
            "tracks_reconstructed": 0,
            "tracks_skipped": 0,
        }

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            person_name = cluster_assignments.get(cluster_id)

            if not person_name or person_name == "SKIP":
                # Skip unassigned clusters and SKIP clusters
                continue

            track_ids = cluster["track_ids"]

            # Get per-identity filtering thresholds
            identity_config = self.per_identity.get(person_name, {})
            is_frozen = identity_config.get("freeze", False)

            # Only apply filters to non-frozen identities
            min_interval_frames = 0 if is_frozen else identity_config.get("min_interval_frames", 0)
            min_visible_frac = 0.0 if is_frozen else identity_config.get("min_visible_frac", 0.0)

            for track_id in track_ids:
                # Find track
                track = next((t for t in tracks if t["track_id"] == track_id), None)
                if not track:
                    continue

                # SCHEMA VALIDATION: Ensure track has required fields (start_ms, end_ms)
                # Some tracks created by Split operations may be missing these fields
                if "start_ms" not in track or "end_ms" not in track:
                    frame_refs = track.get("frame_refs", [])
                    if frame_refs:
                        # Reconstruct from frame_refs (which have ts_ms)
                        timestamps = [ref.get("ts_ms") for ref in frame_refs if "ts_ms" in ref]
                        if timestamps:
                            track["start_ms"] = min(timestamps)
                            track["end_ms"] = max(timestamps)
                            track["duration_ms"] = track["end_ms"] - track["start_ms"]
                            self.schema_stats["tracks_reconstructed"] += 1
                            logger.warning(
                                f"Track {track_id} missing start_ms/end_ms - reconstructed from frame_refs: "
                                f"{track['start_ms']}ms - {track['end_ms']}ms"
                            )
                        else:
                            # No ts_ms in frame_refs - skip this track
                            self.schema_stats["tracks_skipped"] += 1
                            logger.error(f"Track {track_id} missing start_ms/end_ms and no ts_ms in frame_refs - skipping")
                            continue
                    else:
                        # No frame_refs - skip empty track
                        self.schema_stats["tracks_skipped"] += 1
                        logger.error(f"Track {track_id} missing start_ms/end_ms and no frame_refs - skipping")
                        continue

                # Apply per-identity filters (skip for frozen identities)
                frame_refs = track.get("frame_refs", [])
                frame_count = int(track.get("count", len(frame_refs)))
                if frame_count == 0 and frame_refs:
                    frame_count = len(frame_refs)

                visible_conf_threshold = identity_config.get(
                    "visible_conf_threshold", identity_config.get("min_interval_quality", 0.7)
                )
                visible_frames = 0
                for ref in frame_refs:
                    if ref.get("confidence", 0.0) >= visible_conf_threshold:
                        visible_frames += 1

                visible_fraction = (visible_frames / frame_count) if frame_count else 0.0

                if not is_frozen:
                    # Filter 1: min_interval_frames
                    if min_interval_frames > 0 and frame_count < min_interval_frames:
                        self.merge_stats.setdefault("intervals_filtered_by_frame_count", 0)
                        self.merge_stats["intervals_filtered_by_frame_count"] += 1
                        continue

                    # Filter 2: min_visible_frac
                    if min_visible_frac > 0.0 and visible_fraction < min_visible_frac:
                        self.merge_stats.setdefault("intervals_filtered_by_visibility", 0)
                        self.merge_stats["intervals_filtered_by_visibility"] += 1
                        continue

                # Create interval from track
                interval = Interval(
                    person_name=person_name,
                    start_ms=track["start_ms"],
                    end_ms=track["end_ms"],
                    source=f"track_{track_id}",
                    confidence=track.get("mean_confidence", 0.0),
                    frame_count=frame_count,
                    visible_fraction=visible_fraction,
                )

                raw_intervals.append(interval)

        logger.info(f"Collected {len(raw_intervals)} raw intervals")

        # Step 2: Sort intervals by person and time
        raw_intervals.sort(key=lambda i: (i.person_name, i.start_ms))

        # Step 3: Merge intervals per person
        merged_intervals = []

        intervals_by_person = defaultdict(list)
        for interval in raw_intervals:
            intervals_by_person[interval.person_name].append(interval)

        for person_name, person_intervals in intervals_by_person.items():
            # Pass all raw intervals for conflict checking
            merged = self._merge_intervals(person_intervals, person_name, all_intervals=raw_intervals)
            merged_intervals.extend(merged)

        logger.info(f"Merged into {len(merged_intervals)} intervals")
        logger.info(
            f"Merge stats: {self.merge_stats['gaps_merged']}/{self.merge_stats['gaps_seen']} gaps merged, "
            f"{self.merge_stats['quality_bumps_applied']} quality bumps, "
            f"{self.merge_stats['merge_conflicts_blocked']} conflicts blocked"
        )

        # Step 4: Calculate totals
        totals_by_person = self._calculate_totals(merged_intervals)

        return merged_intervals, totals_by_person

    def _merge_intervals(self, intervals: list[Interval], person_name: str, all_intervals: list[Interval] = None) -> list[Interval]:
        """
        Merge overlapping/nearby intervals for a single person with adaptive thresholds.

        Args:
            intervals: List of intervals for one person (sorted by start_ms)
            person_name: Name of the person (for per-identity threshold lookup)
            all_intervals: All intervals across all persons (for conflict checking)

        Returns:
            List of merged intervals
        """
        if not intervals:
            return []

        # Get per-identity thresholds if configured
        identity_config = self.per_identity.get(person_name, {})
        is_frozen = identity_config.get("freeze", False)

        # Get threshold overrides
        gap_merge_base = identity_config.get("gap_merge_ms_base", self.gap_merge_ms_base)  # Per-identity base
        gap_merge_max = identity_config.get("gap_merge_ms_max", self.gap_merge_ms_max)
        gap_merge_lo_conf = identity_config.get("gap_merge_ms_lo_conf")  # New: low-conf cap
        gap_merge_hi_conf = identity_config.get("gap_merge_ms_hi_conf")  # New: high-conf cap
        min_quality = identity_config.get("min_interval_quality", self.min_interval_quality)
        edge_epsilon = identity_config.get("edge_epsilon_ms", self.edge_epsilon_ms)  # Per-identity edge epsilon
        min_visible_frac = None if is_frozen else identity_config.get("min_visible_frac")

        # Frozen identities: skip confidence-based caps (EILEEN hardening) but still do standard merge
        if is_frozen:
            gap_merge_lo_conf = None
            gap_merge_hi_conf = None
            self.merge_stats.setdefault("frozen_identities_standard_merge", 0)
            self.merge_stats["frozen_identities_standard_merge"] += 1

        merged = []
        current = intervals[0]

        for next_interval in intervals[1:]:
            # Calculate gap
            gap = next_interval.start_ms - current.end_ms
            self.merge_stats["gaps_seen"] += 1
            avg_conf = (current.confidence + next_interval.confidence) / 2

            if gap < 0:
                # Overlapping - always merge
                current = Interval(
                    person_name=current.person_name,
                    start_ms=current.start_ms,
                    end_ms=max(current.end_ms, next_interval.end_ms),
                    source="merged",
                    confidence=max(current.confidence, next_interval.confidence),
                )
                self.merge_stats["gaps_merged"] += 1
                continue

            # Edge epsilon: treat tiny gaps as continuous (for edge quantization)
            # Use per-identity edge_epsilon if configured
            if edge_epsilon > 0 and gap <= edge_epsilon:
                current = Interval(
                    person_name=current.person_name,
                    start_ms=current.start_ms,
                    end_ms=max(current.end_ms, next_interval.end_ms),
                    source="merged",
                    confidence=max(current.confidence, next_interval.confidence),
                )
                self.merge_stats["edge_epsilon_merged"] += 1
                self.merge_stats["gaps_merged"] += 1
                continue

            # Determine merge threshold (adaptive based on quality)
            # Use confidence-based caps if configured (EILEEN hardening)
            if gap_merge_lo_conf is not None and gap_merge_hi_conf is not None:
                if avg_conf >= min_quality:
                    merge_threshold = min(gap_merge_hi_conf, gap_merge_max)
                    self.merge_stats["quality_bumps_applied"] += 1
                else:
                    merge_threshold = min(gap_merge_lo_conf, gap_merge_max)
                    self.merge_stats.setdefault("lo_conf_cap_applied", 0)
                    self.merge_stats["lo_conf_cap_applied"] += 1
            else:
                # Standard adaptive logic (existing behavior)
                # Use per-identity base threshold
                merge_threshold = gap_merge_base

                # Quality-aware bump: if both intervals are high quality, allow longer gap
                if (
                    current.confidence >= min_quality
                    and next_interval.confidence >= min_quality
                ):
                    merge_threshold = gap_merge_max
                    self.merge_stats["quality_bumps_applied"] += 1

            # Check if gap is within threshold
            if gap <= merge_threshold:
                if (
                    min_visible_frac
                    and (
                        current.visible_fraction < min_visible_frac
                        or next_interval.visible_fraction < min_visible_frac
                    )
                ):
                    self.merge_stats["timeline_merges_suppressed_by_visibility"] += 1
                    self._visibility_audit.append(
                        {
                            "person_name": person_name,
                            "current_start_ms": current.start_ms,
                            "current_end_ms": current.end_ms,
                            "next_start_ms": next_interval.start_ms,
                            "next_end_ms": next_interval.end_ms,
                            "gap_ms": gap,
                            "avg_conf": avg_conf,
                            "current_visible_fraction": current.visible_fraction,
                            "next_visible_fraction": next_interval.visible_fraction,
                        }
                    )
                    merged.append(current)
                    current = next_interval
                    continue
                # Check for conflicts: is there another person in this gap?
                conflict = False
                if all_intervals and gap > self.conflict_guard_ms:
                    gap_start = current.end_ms
                    gap_end = next_interval.start_ms

                    for other_interval in all_intervals:
                        # Skip same person
                        if other_interval.person_name == current.person_name:
                            continue

                        # Check if other person appears in gap with sufficient confidence
                        if (
                            other_interval.confidence >= 0.7
                            and other_interval.start_ms < gap_end
                            and other_interval.end_ms > gap_start
                        ):
                            # Check overlap duration
                            overlap_start = max(gap_start, other_interval.start_ms)
                            overlap_end = min(gap_end, other_interval.end_ms)
                            overlap_duration = overlap_end - overlap_start

                            if overlap_duration >= self.conflict_guard_ms:
                                conflict = True
                                self.merge_stats["merge_conflicts_blocked"] += 1
                                break

                if not conflict:
                    # Merge: extend current interval
                    current = Interval(
                        person_name=current.person_name,
                        start_ms=current.start_ms,
                        end_ms=max(current.end_ms, next_interval.end_ms),
                        source="merged",
                        confidence=max(current.confidence, next_interval.confidence),
                    )
                    self.merge_stats["gaps_merged"] += 1
                else:
                    # Don't merge due to conflict
                    merged.append(current)
                    current = next_interval
            else:
                # Don't merge: gap too large
                self.merge_stats["merge_clamped"][person_name] += 1
                merged.append(current)
                current = next_interval

        # Append last interval
        merged.append(current)

        return merged

    def _calculate_totals(self, intervals: list[Interval]) -> dict[str, dict]:
        """
        Calculate total screen time per person.

        Args:
            intervals: List of merged intervals

        Returns:
            Dict of person_name -> totals dict
        """
        totals = defaultdict(
            lambda: {
                "total_ms": 0,
                "appearances": 0,
                "first_ms": None,
                "last_ms": None,
                "mean_confidence": 0.0,
                "confidence_sum": 0.0,
            }
        )

        for interval in intervals:
            person_name = interval.person_name
            totals[person_name]["total_ms"] += interval.duration_ms
            totals[person_name]["appearances"] += 1
            totals[person_name]["confidence_sum"] += interval.confidence

            if (
                totals[person_name]["first_ms"] is None
                or interval.start_ms < totals[person_name]["first_ms"]
            ):
                totals[person_name]["first_ms"] = interval.start_ms

            if (
                totals[person_name]["last_ms"] is None
                or interval.end_ms > totals[person_name]["last_ms"]
            ):
                totals[person_name]["last_ms"] = interval.end_ms

        # Calculate mean confidence
        for person_name, data in totals.items():
            if data["appearances"] > 0:
                data["mean_confidence"] = data["confidence_sum"] / data["appearances"]
            del data["confidence_sum"]  # Remove intermediate value

        return dict(totals)

    def detect_co_appearances(
        self, intervals: list[Interval], window_ms: int = 1000
    ) -> list[tuple[str, str, int, int]]:
        """
        Detect co-appearances (multiple people on screen simultaneously).

        Args:
            intervals: List of all intervals
            window_ms: Time window for co-appearance (default 1s)

        Returns:
            List of (person_a, person_b, start_ms, end_ms) tuples
        """
        co_appearances = []

        # Sort all intervals by start time
        sorted_intervals = sorted(intervals, key=lambda i: i.start_ms)

        # Find overlaps
        for i, interval_a in enumerate(sorted_intervals):
            for interval_b in sorted_intervals[i + 1 :]:
                # Stop if intervals are too far apart
                if interval_b.start_ms > interval_a.end_ms + window_ms:
                    break

                # Skip same person
                if interval_a.person_name == interval_b.person_name:
                    continue

                # Check for overlap
                overlap_start = max(interval_a.start_ms, interval_b.start_ms)
                overlap_end = min(interval_a.end_ms, interval_b.end_ms)

                if overlap_end > overlap_start:
                    co_appearances.append(
                        (
                            interval_a.person_name,
                            interval_b.person_name,
                            overlap_start,
                            overlap_end,
                        )
                    )

        return co_appearances

    def export_timeline_df(self, intervals: list[Interval]) -> pd.DataFrame:
        """
        Export timeline as DataFrame.

        Args:
            intervals: List of intervals

        Returns:
            DataFrame with columns: person_name, start_ms, end_ms, duration_ms, source, confidence
        """
        data = [
            {
                "person_name": i.person_name,
                "start_ms": i.start_ms,
                "end_ms": i.end_ms,
                "duration_ms": i.duration_ms,
                "source": i.source,
                "confidence": i.confidence,
                "frame_count": i.frame_count,
                "visible_fraction": round(i.visible_fraction, 3),
            }
            for i in intervals
        ]

        return pd.DataFrame(data)

    def export_totals_df(
        self, totals_by_person: dict[str, dict], episode_duration_ms: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Export totals as DataFrame.

        Args:
            totals_by_person: Totals dict from build_timeline()
            episode_duration_ms: Total episode duration for percentage calculation

        Returns:
            DataFrame with columns: person_name, total_ms, total_sec, appearances, percent, first_ms, last_ms
        """
        data = []

        for person_name, totals in totals_by_person.items():
            total_ms = totals["total_ms"]
            total_sec = total_ms / 1000.0

            row = {
                "person_name": person_name,
                "total_ms": total_ms,
                "total_sec": round(total_sec, 1),
                "appearances": totals["appearances"],
                "first_ms": totals["first_ms"],
                "last_ms": totals["last_ms"],
                "mean_confidence": round(totals["mean_confidence"], 3),
            }

            if episode_duration_ms:
                row["percent"] = round((total_ms / episode_duration_ms) * 100, 2)
            else:
                row["percent"] = None

            data.append(row)

        # Sort by total_ms descending
        df = pd.DataFrame(data)

        if len(df) > 0 and "total_ms" in df.columns:
            df = df.sort_values("total_ms", ascending=False)

        return df

    def get_visibility_audit(self) -> list[dict[str, Any]]:
        """Return visibility suppression audit entries."""
        return list(self._visibility_audit)

    def get_merge_stats(self) -> dict[str, Any]:
        """Return merge stats with JSON-serialisable structures."""
        stats_copy = dict(self.merge_stats)
        stats_copy["merge_clamped"] = dict(stats_copy["merge_clamped"])
        return stats_copy

    def get_schema_stats(self) -> dict[str, Any]:
        """Return schema validation stats for diagnostics."""
        return dict(self.schema_stats) if hasattr(self, 'schema_stats') else {}
