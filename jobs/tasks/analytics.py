"""
Analytics task.

Generates screen-time analytics and exports.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from app.lib.data import load_clusters, load_tracks
from screentime.attribution.timeline import TimelineBuilder
from screentime.attribution.auto_caps import compute_auto_caps, save_auto_caps
from screentime.diagnostics.telemetry import telemetry, TelemetryEvent

logger = logging.getLogger(__name__)


def analytics_task(job_id: str, episode_id: str, cluster_assignments: dict[int, str]) -> dict:
    """
    Generate analytics and exports for episode.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        cluster_assignments: Map of cluster_id -> person_name

    Returns:
        Dict with analytics results
    """
    logger.info(f"[{job_id}] Starting analytics task for {episode_id}")

    start_time = time.time()

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup paths
    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    outputs_dir = data_root / "outputs" / episode_id
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ALWAYS rebuild from current state (no caching)
    logger.info(f"[{job_id}] === REBUILDING analytics from current clusters.json (no cache) ===")

    # Load data
    clusters_data = load_clusters(episode_id, data_root)
    tracks_data = load_tracks(episode_id, data_root)

    if not clusters_data or not tracks_data:
        raise ValueError(f"Missing clusters or tracks data for {episode_id}")

    logger.info(
        f"[{job_id}] Loaded {len(clusters_data.get('clusters', []))} clusters and {len(tracks_data.get('tracks', []))} tracks"
    )

    # Load suppression data and filter out suppressed tracks
    from app.lib.episode_status import load_suppress_data
    suppress_data = load_suppress_data(episode_id, data_root)
    deleted_tracks = set(suppress_data.get('deleted_tracks', []))
    deleted_clusters = set(suppress_data.get('deleted_clusters', []))

    if deleted_tracks or deleted_clusters:
        logger.info(f"[{job_id}] Filtering suppressed items: {len(deleted_tracks)} tracks, {len(deleted_clusters)} clusters")

        # Filter suppressed tracks
        tracks_before = len(tracks_data.get('tracks', []))
        tracks_data['tracks'] = [t for t in tracks_data.get('tracks', []) if t['track_id'] not in deleted_tracks]
        logger.info(f"[{job_id}] Filtered tracks: {tracks_before} → {len(tracks_data['tracks'])}")

        # Filter suppressed clusters
        clusters_before = len(clusters_data.get('clusters', []))
        clusters_data['clusters'] = [c for c in clusters_data.get('clusters', []) if c['cluster_id'] not in deleted_clusters]
        logger.info(f"[{job_id}] Filtered clusters: {clusters_before} → {len(clusters_data['clusters'])}")

    # ========================================
    # Compute Auto-Caps (Identity-Agnostic)
    # ========================================
    timeline_config = config.get("timeline", {})
    auto_caps_config = timeline_config.get("auto_caps", {})

    per_identity_overrides = {}

    if auto_caps_config.get("enabled", False):
        logger.info(f"[{job_id}] === Computing Auto-Caps (Identity-Agnostic) ===")

        try:
            # Compute auto-caps from episode data
            auto_caps_results = compute_auto_caps(
                episode_id=episode_id,
                data_root=data_root,
                config=config,
                tracks_data=tracks_data,
                clusters_data=clusters_data,
            )

            # Save auto-caps to diagnostics
            save_auto_caps(episode_id, data_root, auto_caps_results)

            # Convert auto_caps_results to per_identity format for TimelineBuilder
            # auto_caps_results: {identity_name: {auto_cap_ms, safe_gap_count, ...}}
            # per_identity format: {identity_name: {gap_merge_ms_max, ...}}
            for identity_name, caps_data in auto_caps_results.items():
                auto_cap_ms = caps_data.get("auto_cap_ms")
                if auto_cap_ms:
                    per_identity_overrides[identity_name] = {
                        "gap_merge_ms_max": auto_cap_ms
                    }

            logger.info(
                f"[{job_id}] Auto-caps computed for {len(per_identity_overrides)} identities"
            )
            for identity, caps in per_identity_overrides.items():
                logger.info(
                    f"[{job_id}]   {identity}: auto_cap_ms={caps['gap_merge_ms_max']}ms"
                )
        except Exception as e:
            logger.warning(f"[{job_id}] Auto-caps computation failed: {e}", exc_info=True)
            logger.warning(f"[{job_id}] Falling back to global defaults")
    else:
        logger.info(f"[{job_id}] Auto-caps disabled, using global defaults")

    # Merge auto-caps with any existing per_identity config (auto-caps take precedence)
    per_identity_final = timeline_config.get("per_identity", {}).copy()
    per_identity_final.update(per_identity_overrides)

    # Build timeline with adaptive gap-merge (using auto-caps if enabled)
    timeline_builder = TimelineBuilder(
        gap_merge_ms_base=timeline_config.get("gap_merge_ms_base", 2000),
        gap_merge_ms_max=timeline_config.get("gap_merge_ms_max", 2500),
        min_interval_quality=timeline_config.get("min_interval_quality", 0.60),
        conflict_guard_ms=timeline_config.get("conflict_guard_ms", 500),
        use_scene_bounds=timeline_config.get("use_scene_bounds", False),
        edge_epsilon_ms=timeline_config.get("edge_epsilon_ms", 150),
        per_identity=per_identity_final,  # Use computed auto-caps
    )
    intervals, totals_by_person = timeline_builder.build_timeline(
        clusters_data, tracks_data, cluster_assignments
    )
    merge_stats = timeline_builder.get_merge_stats()
    visibility_audit = timeline_builder.get_visibility_audit()

    logger.info(
        f"[{job_id}] Built timeline with {len(intervals)} intervals for {len(totals_by_person)} people"
    )

    # Calculate episode duration
    all_tracks = tracks_data.get("tracks", [])
    if all_tracks:
        episode_duration_ms = max(t["end_ms"] for t in all_tracks)
    else:
        episode_duration_ms = None

    # Export timeline
    timeline_df = timeline_builder.export_timeline_df(intervals)
    timeline_csv_path = outputs_dir / "timeline.csv"
    timeline_df.to_csv(timeline_csv_path, index=False)

    logger.info(f"[{job_id}] Saved timeline to {timeline_csv_path}")

    # Export totals
    totals_df = timeline_builder.export_totals_df(totals_by_person, episode_duration_ms)
    totals_csv_path = outputs_dir / "totals.csv"
    totals_df.to_csv(totals_csv_path, index=False)

    totals_parquet_path = outputs_dir / "totals.parquet"
    totals_df.to_parquet(totals_parquet_path, index=False)

    logger.info(f"[{job_id}] Saved totals to {totals_csv_path} and {totals_parquet_path}")

    # Generate Excel export
    excel_path = outputs_dir / "totals.xlsx"
    _generate_excel_export(
        excel_path,
        totals_df,
        timeline_df,
        episode_id,
        episode_duration_ms,
        clusters_data,
        tracks_data,
    )

    logger.info(f"[{job_id}] Saved Excel export to {excel_path}")

    # Calculate stats
    stage_time_ms = int((time.time() - start_time) * 1000)

    analytics_stats = {
        "episode_id": episode_id,
        "intervals_created": len(intervals),
        "people_detected": len(totals_by_person),
        "episode_duration_ms": episode_duration_ms,
        "stage_time_ms_analytics": stage_time_ms,
        "exports_generated": 4,  # timeline.csv, totals.csv, totals.parquet, totals.xlsx
        "merge_stats": merge_stats,
    }

    # Save analytics stats
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / "analytics_stats.json"
    with open(stats_path, "w") as f:
        json.dump(analytics_stats, f, indent=2)

    logger.info(f"[{job_id}] Analytics stats saved to {stats_path}")

    # Save visibility suppression audit (EILEEN hardening)
    audit_path = reports_dir / "eileen_merge_audit.json"
    with open(audit_path, "w") as f:
        json.dump(visibility_audit, f, indent=2)
    logger.info(f"[{job_id}] Visibility audit saved to {audit_path}")

    # Save analytics debug info (schema validation, interval reconstruction)
    schema_stats = timeline_builder.get_schema_stats()
    debug_info = {
        "episode_id": episode_id,
        "timestamp": datetime.utcnow().isoformat(),
        "schema_validation": schema_stats,
        "data_source": "tracks.json (full intervals, not samples)",
        "intervals_built_from": "track start_ms/end_ms",
        "schema_notes": {
            "tracks_reconstructed": "Tracks missing start_ms/end_ms had intervals reconstructed from frame_refs[].ts_ms",
            "tracks_skipped": "Tracks without start_ms/end_ms or ts_ms were skipped",
            "variable_fps": "Uses actual ts_ms timestamps (10fps baseline, 30fps densify)",
        },
        "suppression": {
            "deleted_tracks": len(deleted_tracks),
            "deleted_clusters": len(deleted_clusters),
        },
        "analytics_stats": {
            "total_intervals": len(intervals),
            "total_people": len(totals_by_person),
        }
    }
    debug_path = harvest_dir / "diagnostics" / "analytics_debug.json"
    with open(debug_path, "w") as f:
        json.dump(debug_info, f, indent=2)
    logger.info(f"[{job_id}] Analytics debug info saved to {debug_path}")

    # Telemetry
    telemetry.log(
        TelemetryEvent.JOB_STAGE_COMPLETE,
        metadata={
            "job_id": job_id,
            "stage": "analytics",
            "intervals_created": len(intervals),
            "stage_time_ms": stage_time_ms,
        },
    )

    # Clear analytics dirty flag on success
    from app.lib.analytics_dirty import clear_analytics_dirty
    clear_analytics_dirty(episode_id, data_root)
    logger.info(f"[{job_id}] Cleared analytics dirty flag - analytics are now fresh")

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "timeline_path": str(timeline_csv_path),
        "totals_path": str(totals_csv_path),
        "excel_path": str(excel_path),
        "stats": analytics_stats,
        "analytics_fresh": True,
        "suppressed_tracks_count": len(deleted_tracks),
        "suppressed_clusters_count": len(deleted_clusters),
    }


def _generate_excel_export(
    excel_path: Path,
    totals_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    episode_id: str,
    episode_duration_ms: int | None,
    clusters_data: dict,
    tracks_data: dict,
) -> None:
    """
    Generate Excel export with multiple sheets.

    Args:
        excel_path: Output Excel file path
        totals_df: Totals DataFrame
        timeline_df: Timeline DataFrame
        episode_id: Episode ID
        episode_duration_ms: Episode duration in ms
        clusters_data: Clusters data
        tracks_data: Tracks data
    """
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # Sheet 1: Summary
        totals_df.to_excel(writer, sheet_name="Summary", index=False)

        # Format millisecond columns as integers (not times)
        worksheet = writer.sheets["Summary"]
        for col_name in ["total_ms", "first_ms", "last_ms"]:
            if col_name in totals_df.columns:
                col_idx = list(totals_df.columns).index(col_name) + 1  # +1 for Excel 1-indexing
                col_letter = chr(64 + col_idx)  # Convert to letter (A, B, C...)
                for row_idx in range(2, len(totals_df) + 2):  # Start from row 2 (after header)
                    cell = worksheet[f"{col_letter}{row_idx}"]
                    cell.number_format = '0'  # Integer format

        # Sheet 2: Timeline
        timeline_df.to_excel(writer, sheet_name="Timeline", index=False)

        # Format millisecond columns as integers (not times)
        worksheet = writer.sheets["Timeline"]
        for col_name in ["start_ms", "end_ms", "duration_ms"]:
            if col_name in timeline_df.columns:
                col_idx = list(timeline_df.columns).index(col_name) + 1
                col_letter = chr(64 + col_idx)
                for row_idx in range(2, len(timeline_df) + 2):
                    cell = worksheet[f"{col_letter}{row_idx}"]
                    cell.number_format = '0'  # Integer format

        # Sheet 3: Metadata
        metadata = {
            "Episode ID": [episode_id],
            "Generated": [datetime.utcnow().isoformat()],
            "Episode Duration (ms)": [episode_duration_ms or "Unknown"],
            "Episode Duration (min)": [
                round(episode_duration_ms / 60000, 2) if episode_duration_ms else "Unknown"
            ],
            "Total Clusters": [len(clusters_data.get("clusters", []))],
            "Total Tracks": [len(tracks_data.get("tracks", []))],
            "People Detected": [len(totals_df)],
            "Total Intervals": [len(timeline_df)],
        }
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_excel(writer, sheet_name="Metadata", index=False)
