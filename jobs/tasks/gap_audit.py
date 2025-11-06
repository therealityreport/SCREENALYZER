"""
Gap audit task - Analyze missing screen time windows for targeted densify.

Identifies gap windows and calculates coverage ratios to prioritize
which gaps to scan during densify.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def gap_audit_task(
    job_id: str,
    episode_id: str,
    target_identity: str,
    max_gap_ms: int = 10000,
    pad_ms: int = 800,
) -> dict:
    """
    Audit gap windows for a specific identity.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        target_identity: Person name to audit (e.g., "YOLANDA")
        max_gap_ms: Maximum gap size to consider
        pad_ms: Padding around each gap window

    Returns:
        Dict with gap audit results
    """
    logger.info(f"[{job_id}] Starting gap audit for {target_identity} in {episode_id}")

    # Setup paths
    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    outputs_dir = data_root / "outputs" / episode_id
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load timeline
    timeline_path = outputs_dir / "timeline.csv"
    timeline_df = pd.read_csv(timeline_path)

    # Load tracks to calculate frame-level coverage
    tracks_path = harvest_dir / "tracks.json"
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data.get("tracks", [])

    # Get video duration from tracks
    video_duration_ms = max(t["end_ms"] for t in tracks)

    logger.info(f"[{job_id}] Video duration: {video_duration_ms}ms")

    # Filter timeline for target identity
    identity_intervals = timeline_df[timeline_df["person_name"] == target_identity].copy()
    identity_intervals = identity_intervals.sort_values("start_ms").reset_index(drop=True)

    logger.info(f"[{job_id}] Found {len(identity_intervals)} intervals for {target_identity}")

    # Identify gaps between intervals
    gap_windows = []

    # Check for gap at beginning of video
    if len(identity_intervals) > 0:
        first_start = identity_intervals.iloc[0]["start_ms"]
        if first_start > max_gap_ms:
            gap_windows.append({
                "gap_type": "video_start",
                "gap_start_ms": 0,
                "gap_end_ms": first_start,
                "gap_duration_ms": first_start,
            })

    # Check gaps between intervals
    for i in range(len(identity_intervals) - 1):
        current_end = identity_intervals.iloc[i]["end_ms"]
        next_start = identity_intervals.iloc[i + 1]["start_ms"]
        gap_duration = next_start - current_end

        if 0 < gap_duration <= max_gap_ms:
            gap_windows.append({
                "gap_type": "inter_interval",
                "gap_start_ms": current_end,
                "gap_end_ms": next_start,
                "gap_duration_ms": gap_duration,
            })

    # Check for gap at end of video
    if len(identity_intervals) > 0:
        last_end = identity_intervals.iloc[-1]["end_ms"]
        remaining = video_duration_ms - last_end
        if 0 < remaining <= max_gap_ms:
            gap_windows.append({
                "gap_type": "video_end",
                "gap_start_ms": last_end,
                "gap_end_ms": video_duration_ms,
                "gap_duration_ms": remaining,
            })

    logger.info(f"[{job_id}] Identified {len(gap_windows)} candidate gap windows")

    # For each gap, add padding and calculate coverage
    audit_results = []

    for gap_idx, gap in enumerate(gap_windows):
        # Add padding
        padded_start = max(0, gap["gap_start_ms"] - pad_ms)
        padded_end = min(video_duration_ms, gap["gap_end_ms"] + pad_ms)
        padded_duration = padded_end - padded_start

        # Calculate existing coverage in this padded window
        # Look at all tracks that overlap with this window
        overlapping_tracks = []
        total_overlap_ms = 0

        for track in tracks:
            track_start = track["start_ms"]
            track_end = track["end_ms"]

            # Check if track overlaps with padded window
            if track_end > padded_start and track_start < padded_end:
                overlap_start = max(track_start, padded_start)
                overlap_end = min(track_end, padded_end)
                overlap_ms = overlap_end - overlap_start

                overlapping_tracks.append({
                    "track_id": track["track_id"],
                    "overlap_ms": overlap_ms,
                    "mean_confidence": track.get("mean_confidence", 0.0),
                })
                total_overlap_ms += overlap_ms

        coverage_ratio = total_overlap_ms / padded_duration if padded_duration > 0 else 0.0

        # Calculate avg face size and confidence from overlapping tracks
        if overlapping_tracks:
            avg_confidence = sum(t["mean_confidence"] for t in overlapping_tracks) / len(overlapping_tracks)
        else:
            avg_confidence = 0.0

        audit_entry = {
            "window_idx": int(gap_idx),
            "gap_type": str(gap["gap_type"]),
            "gap_start_ms": int(gap["gap_start_ms"]),
            "gap_end_ms": int(gap["gap_end_ms"]),
            "gap_duration_ms": int(gap["gap_duration_ms"]),
            "padded_start_ms": int(padded_start),
            "padded_end_ms": int(padded_end),
            "padded_duration_ms": int(padded_duration),
            "existing_coverage_ms": int(total_overlap_ms),
            "coverage_ratio": float(coverage_ratio),
            "overlapping_tracks_count": int(len(overlapping_tracks)),
            "avg_confidence": float(avg_confidence),
            "priority": "high" if coverage_ratio < 0.2 else "medium" if coverage_ratio < 0.5 else "low",
        }

        audit_results.append(audit_entry)

    # Sort by priority (high coverage ratio = low priority)
    audit_results.sort(key=lambda x: (x["priority"] == "low", x["coverage_ratio"]))

    # Calculate totals
    total_gap_duration = sum(w["gap_duration_ms"] for w in gap_windows)
    high_priority_count = sum(1 for r in audit_results if r["priority"] == "high")
    high_priority_duration = sum(r["gap_duration_ms"] for r in audit_results if r["priority"] == "high")

    # Save audit report
    audit_report = {
        "job_id": job_id,
        "episode_id": episode_id,
        "target_identity": target_identity,
        "video_duration_ms": int(video_duration_ms),
        "max_gap_ms": int(max_gap_ms),
        "pad_ms": int(pad_ms),
        "summary": {
            "total_gaps": int(len(gap_windows)),
            "total_gap_duration_ms": int(total_gap_duration),
            "high_priority_gaps": int(high_priority_count),
            "high_priority_duration_ms": int(high_priority_duration),
        },
        "gap_windows": audit_results,
    }

    output_path = reports_dir / f"{target_identity.lower()}_gap_audit.json"
    with open(output_path, "w") as f:
        json.dump(audit_report, f, indent=2)

    logger.info(f"[{job_id}] Gap audit saved to {output_path}")
    logger.info(f"[{job_id}] Summary: {high_priority_count} high-priority gaps totaling {high_priority_duration}ms")

    return audit_report
