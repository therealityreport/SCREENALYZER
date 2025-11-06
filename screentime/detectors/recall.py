"""
High-recall detection pass for targeted windows.

Identifies gaps and runs relaxed detection only where needed to improve recall
without sacrificing global precision.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RecallWindow:
    """A targeted window for high-recall detection."""

    start_ms: int
    end_ms: int
    reason: str  # "gap", "low_track", "small_face"
    person_name: Optional[str] = None  # If gap is person-specific


class RecallWindowSelector:
    """
    Select targeted windows for high-recall detection pass.

    Analyzes gaps from previous run to identify where additional
    detections are needed.
    """

    def __init__(
        self,
        gap_threshold_ms: int = 3200,  # Max gap to consider
        expansion_ms: int = 300,  # Expand window by ±300ms
        low_track_threshold: int = 10,  # Persons with <10 tracks
        sparse_scan_interval_ms: int = 2500,  # Interval for sparse scanning
    ):
        """
        Initialize recall window selector.

        Args:
            gap_threshold_ms: Max gap duration to target
            expansion_ms: Expand windows by this amount
            low_track_threshold: Track count threshold for sparse scanning
            sparse_scan_interval_ms: Interval for sparse person scans
        """
        self.gap_threshold_ms = gap_threshold_ms
        self.expansion_ms = expansion_ms
        self.low_track_threshold = low_track_threshold
        self.sparse_scan_interval_ms = sparse_scan_interval_ms

    def select_windows(
        self,
        episode_id: str,
        data_root: Path,
        target_persons: Optional[list[str]] = None,
    ) -> list[RecallWindow]:
        """
        Select recall windows based on previous analytics.

        Args:
            episode_id: Episode ID
            data_root: Data root path
            target_persons: Optional list of persons to focus on

        Returns:
            List of recall windows
        """
        windows = []

        # Try to load previous timeline
        outputs_dir = data_root / "outputs" / episode_id
        timeline_path = outputs_dir / "timeline.csv"

        if not timeline_path.exists():
            logger.warning(f"No previous timeline found at {timeline_path}, using full episode scan")
            return self._full_episode_windows(episode_id, data_root)

        # Load timeline
        timeline_df = pd.read_csv(timeline_path)

        if len(timeline_df) == 0:
            logger.warning("Empty timeline, using full episode scan")
            return self._full_episode_windows(episode_id, data_root)

        # Get unique persons
        persons = timeline_df["person_name"].unique()

        if target_persons:
            persons = [p for p in persons if p in target_persons]

        logger.info(f"Analyzing gaps for {len(persons)} persons")

        # Analyze each person
        for person_name in persons:
            person_intervals = timeline_df[timeline_df["person_name"] == person_name].sort_values(
                "start_ms"
            )

            # Find gaps between intervals
            for i in range(len(person_intervals) - 1):
                curr_end = person_intervals.iloc[i]["end_ms"]
                next_start = person_intervals.iloc[i + 1]["start_ms"]

                gap = next_start - curr_end

                if 0 < gap <= self.gap_threshold_ms:
                    # Create recall window for this gap
                    window_start = max(0, curr_end - self.expansion_ms)
                    window_end = next_start + self.expansion_ms

                    windows.append(
                        RecallWindow(
                            start_ms=window_start,
                            end_ms=window_end,
                            reason=f"gap_{person_name}",
                            person_name=person_name,
                        )
                    )

                    logger.debug(
                        f"Gap window for {person_name}: {window_start}-{window_end}ms (gap={gap}ms)"
                    )

        # Check for low-track persons
        tracks_path = data_root / "harvest" / episode_id / "tracks.json"
        if tracks_path.exists():
            import json

            with open(tracks_path) as f:
                tracks_data = json.load(f)

            clusters_path = data_root / "harvest" / episode_id / "clusters.json"
            if clusters_path.exists():
                with open(clusters_path) as f:
                    clusters_data = json.load(f)

                # Count tracks per person
                for cluster in clusters_data.get("clusters", []):
                    person_name = cluster.get("name")
                    if not person_name:
                        continue

                    if target_persons and person_name not in target_persons:
                        continue

                    track_count = len(cluster.get("track_ids", []))

                    if track_count < self.low_track_threshold:
                        logger.info(
                            f"Low-track person {person_name} ({track_count} tracks), adding sparse scan"
                        )

                        # Add sparse scan windows for this person
                        # Get their rough time range
                        person_intervals = timeline_df[timeline_df["person_name"] == person_name]

                        if len(person_intervals) > 0:
                            first_ms = person_intervals["start_ms"].min()
                            last_ms = person_intervals["end_ms"].max()

                            # Create sparse windows
                            current_ms = first_ms
                            while current_ms < last_ms:
                                window_end = min(
                                    current_ms + self.sparse_scan_interval_ms, last_ms
                                )

                                windows.append(
                                    RecallWindow(
                                        start_ms=current_ms,
                                        end_ms=window_end,
                                        reason=f"low_track_{person_name}",
                                        person_name=person_name,
                                    )
                                )

                                current_ms += self.sparse_scan_interval_ms * 2  # Skip one interval

        # Merge overlapping windows
        windows = self._merge_windows(windows)

        logger.info(f"Selected {len(windows)} recall windows")

        return windows

    def _full_episode_windows(self, episode_id: str, data_root: Path) -> list[RecallWindow]:
        """
        Create windows covering full episode (fallback).

        Args:
            episode_id: Episode ID
            data_root: Data root path

        Returns:
            List of windows covering full episode
        """
        # Get episode duration from tracks
        tracks_path = data_root / "harvest" / episode_id / "tracks.json"

        if not tracks_path.exists():
            logger.error("Cannot determine episode duration, no tracks found")
            return []

        import json

        with open(tracks_path) as f:
            tracks_data = json.load(f)

        tracks = tracks_data.get("tracks", [])
        if not tracks:
            return []

        max_time = max(t["end_ms"] for t in tracks)

        # Create windows every 5 seconds
        windows = []
        window_size_ms = 5000

        current_ms = 0
        while current_ms < max_time:
            window_end = min(current_ms + window_size_ms, max_time)

            windows.append(
                RecallWindow(
                    start_ms=current_ms,
                    end_ms=window_end,
                    reason="full_scan",
                )
            )

            current_ms += window_size_ms

        logger.info(f"Created {len(windows)} full-episode windows")

        return windows

    def _merge_windows(self, windows: list[RecallWindow]) -> list[RecallWindow]:
        """
        Merge overlapping recall windows.

        Args:
            windows: List of windows

        Returns:
            List of merged windows
        """
        if not windows:
            return []

        # Sort by start time
        windows = sorted(windows, key=lambda w: w.start_ms)

        merged = []
        current = windows[0]

        for next_window in windows[1:]:
            # Check if overlapping
            if next_window.start_ms <= current.end_ms:
                # Merge
                current = RecallWindow(
                    start_ms=current.start_ms,
                    end_ms=max(current.end_ms, next_window.end_ms),
                    reason=f"{current.reason}+{next_window.reason}",
                    person_name=current.person_name or next_window.person_name,
                )
            else:
                # No overlap, save current and move to next
                merged.append(current)
                current = next_window

        # Append last
        merged.append(current)

        logger.info(f"Merged {len(windows)} windows into {len(merged)}")

        return merged
