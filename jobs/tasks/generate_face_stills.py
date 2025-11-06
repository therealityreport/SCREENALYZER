"""Generate face-aware stills with FIQA scoring for an episode."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.stills.serfiq import create_serfiq_scorer
from screentime.stills.track_stills import (
    generate_track_still,
    load_config,
    write_manifest_entry,
    log_telemetry_event,
)
from screentime.utils import get_video_path

logger = logging.getLogger(__name__)


def generate_face_stills_task(
    episode_id: str,
    *,
    data_root: Path = Path("data"),
    config_path: Path = Path("configs/stills.yaml"),
    force: bool = False,
    resume: bool = True,
    progress_callback=None,
) -> dict:
    """
    Generate face-aware stills with SER-FIQ quality assessment.

    Args:
        episode_id: Episode identifier
        data_root: Root data directory
        config_path: Path to stills config YAML
        force: Regenerate even if stills exist (clears manifest)
        resume: Skip tracks that already have stills (default: True)
        progress_callback: Optional callback(done, total, message) for progress updates

    Returns:
        Stats dict with generation results
    """
    logger.info(f"Starting face stills generation for {episode_id}")
    start_time = time.time()

    # Load config
    config = load_config(config_path)

    # Setup paths
    harvest_dir = data_root / "harvest" / episode_id
    tracks_path = harvest_dir / "tracks.json"

    if not tracks_path.exists():
        logger.error(f"Tracks not found for {episode_id}: {tracks_path}")
        return {
            "episode_id": episode_id,
            "status": "error",
            "reason": "tracks_not_found",
            "tracks_path": str(tracks_path),
        }

    # Load tracks
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data.get("tracks", [])
    total_tracks = len(tracks)

    if total_tracks == 0:
        logger.warning(f"No tracks found for {episode_id}")
        return {
            "episode_id": episode_id,
            "status": "completed",
            "total_tracks": 0,
            "generated": 0,
            "skipped": 0,
            "failed": 0,
        }

    # Get video path
    video_path = get_video_path(episode_id, data_root)
    if not video_path or not video_path.exists():
        logger.error(f"Video not found for {episode_id}: {video_path}")
        return {
            "episode_id": episode_id,
            "status": "error",
            "reason": "video_not_found",
            "video_path": str(video_path) if video_path else None,
        }

    # Initialize embedder and SER-FIQ scorer
    embedder: Optional[ArcFaceEmbedder] = None
    serfiq_scorer = None
    fiqa_method = config.get("quality", {}).get("fiqa_method", "embedding_norm")

    if fiqa_method == "serfiq":
        logger.info("Initializing ArcFace embedder for SER-FIQ...")
        embedder = ArcFaceEmbedder(skip_redetect=True)
        logger.info(f"Embedder initialized with provider: {embedder.actual_provider}")

        logger.info("Initializing SER-FIQ scorer (5 stochastic passes)...")
        serfiq_scorer = create_serfiq_scorer(num_passes=5)
        logger.info("SER-FIQ scorer ready")
    else:
        logger.info(f"Using FIQA method: {fiqa_method} (no embedder needed)")

    # Setup output manifest
    manifest_rel = config["io"]["manifest"].format(episode_id=episode_id)
    manifest_path = data_root / manifest_rel

    if force and manifest_path.exists():
        logger.info(f"Force mode: clearing existing manifest at {manifest_path}")
        manifest_path.unlink()

    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing stills if not force mode
    existing_track_ids = set()
    if not force and resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        # Only skip if we have at least one of thumb_path or crop_path
                        if entry.get("thumb_path") or entry.get("crop_path"):
                            existing_track_ids.add(entry["track_id"])
                    except (json.JSONDecodeError, KeyError):
                        continue

    logger.info(
        f"Generating face stills for {total_tracks} tracks from {episode_id}"
    )
    logger.info(f"Video: {video_path}")
    logger.info(f"Already have stills for {len(existing_track_ids)} tracks (resume={resume})")

    # Generate stills
    generated = 0
    skipped = 0
    failed = 0
    errors = []

    for i, track in enumerate(tracks, 1):
        track_id = track["track_id"]

        # Skip if already exists (unless force mode)
        if track_id in existing_track_ids:
            skipped += 1
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{total_tracks} (skipped: {skipped})")
                if progress_callback:
                    progress_callback(
                        done=generated + skipped,
                        total=total_tracks,
                        message=f"Stills: {generated + skipped}/{total_tracks}"
                    )
            continue

        try:
            result = generate_track_still(
                track_id=track_id,
                track_dict=track,
                video_path=str(video_path),
                episode_id=episode_id,
                config=config,
                embedder=embedder,
                serfiq_scorer=serfiq_scorer,
            )

            if result:
                # Write to manifest
                write_manifest_entry(
                    manifest_path=manifest_path,
                    result=result,
                )
                generated += 1

                # Log progress and notify callback
                if i % 10 == 0 or progress_callback:
                    logger.info(
                        f"Progress: {i}/{total_tracks} "
                        f"(generated: {generated}, skipped: {skipped}, failed: {failed})"
                    )
                    if progress_callback:
                        progress_callback(
                            done=generated + skipped,
                            total=total_tracks,
                            message=f"Stills: {generated + skipped}/{total_tracks}"
                        )
            else:
                failed += 1
                errors.append({"track_id": track_id, "reason": "generation_failed"})
                logger.warning(f"Failed to generate still for track {track_id}")

        except Exception as exc:
            failed += 1
            errors.append({"track_id": track_id, "reason": str(exc)})
            logger.error(f"Error generating still for track {track_id}: {exc}")

    # Calculate stats
    elapsed_ms = int((time.time() - start_time) * 1000)
    coverage = (generated + skipped) / total_tracks if total_tracks > 0 else 0.0

    stats = {
        "episode_id": episode_id,
        "status": "completed",
        "total_tracks": total_tracks,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "coverage": round(coverage, 3),
        "elapsed_ms": elapsed_ms,
        "fiqa_method": fiqa_method,
        "manifest_path": str(manifest_path),
        "video_path": str(video_path),
        "completed_at": datetime.utcnow().isoformat(),
    }

    if errors:
        stats["errors"] = errors[:100]  # Limit error list size

    # Write stats file
    stats_dir = harvest_dir / "diagnostics" / "reports"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = stats_dir / "stills_stats.json"

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"Face stills generation complete for {episode_id}: "
        f"{generated} generated, {skipped} skipped, {failed} failed "
        f"({coverage*100:.1f}% coverage) in {elapsed_ms/1000:.1f}s"
    )

    # Log telemetry
    log_telemetry_event(
        "stills_generation_complete",
        {
            "episode_id": episode_id,
            "total_tracks": total_tracks,
            "generated": generated,
            "skipped": skipped,
            "failed": failed,
            "coverage": coverage,
            "elapsed_ms": elapsed_ms,
        },
    )

    return stats
