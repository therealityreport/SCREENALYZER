#!/usr/bin/env python3
"""
Generate face-aware track stills with FIQA scoring.

Usage:
    python scripts/generate_face_stills.py RHOBH_S05_E15_11052025
"""

import argparse
import json
import logging
from pathlib import Path

from screentime.episode_registry import episode_registry
from screentime.stills.track_stills import (
    generate_track_still,
    load_config,
    write_manifest_entry,
    log_telemetry_event,
)
from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.stills.serfiq import create_serfiq_scorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate face-aware track stills with FIQA scoring"
    )
    parser.add_argument("episode_id", help="Episode ID (e.g., RHOBH_S05_E15_11052025)")
    parser.add_argument("--config", default="configs/stills.yaml", help="Config file path")
    parser.add_argument("--data-root", default="data", help="Data root directory")
    parser.add_argument("--limit", type=int, help="Limit number of tracks (for testing)")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Resolve paths
    data_root = Path(args.data_root)
    harvest_dir = data_root / "harvest" / args.episode_id
    tracks_json_path = harvest_dir / "tracks.json"

    if not tracks_json_path.exists():
        logger.error(f"tracks.json not found: {tracks_json_path}")
        return 1

    # Get video path from episode registry
    episode_data = episode_registry.get_episode(args.episode_id)
    if not episode_data:
        logger.error(f"Episode {args.episode_id} not found in registry")
        return 1

    video_path = str(data_root / episode_data["video_path"])
    if not Path(video_path).exists():
        logger.error(f"Video file not found: {video_path}")
        return 1

    # Load tracks
    with open(tracks_json_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data.get("tracks", [])
    if not tracks:
        logger.error("No tracks found in tracks.json")
        return 1

    if args.limit:
        tracks = tracks[:args.limit]
        logger.info(f"Limited to {args.limit} tracks for testing")

    logger.info(f"Generating face stills for {len(tracks)} tracks from {args.episode_id}")
    logger.info(f"Video: {video_path}")

    # Initialize embedder and SER-FIQ scorer
    embedder = None
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

    # Clear existing manifest
    out_dir = Path(config["io"]["out_dir"].format(episode_id=args.episode_id))
    manifest_path = data_root / config["io"]["manifest"].format(episode_id=args.episode_id)
    if manifest_path.exists():
        manifest_path.unlink()
        logger.info(f"Cleared existing manifest: {manifest_path}")

    # Generate stills
    success_count = 0
    fallback_count = 0
    failed_count = 0

    telemetry_path = Path("logs") / "stills_events.jsonl"

    for track in tracks:
        track_id = track["track_id"]

        try:
            result = generate_track_still(
                track_id=track_id,
                track_dict=track,
                video_path=video_path,
                episode_id=args.episode_id,
                config=config,
                embedder=embedder,
                serfiq_scorer=serfiq_scorer,
            )

            if result:
                # Write to manifest
                write_manifest_entry(manifest_path, result)

                # Log telemetry
                if config.get("telemetry", {}).get("verbose", True):
                    log_telemetry_event(
                        telemetry_path,
                        track_id,
                        {
                            "timestamp": result.timestamp,
                            "frame_idx": result.source_frame_idx,
                            "scores": result.scores,
                            "fallback": result.fallback,
                            "bbox": result.bbox_used,
                        },
                    )

                success_count += 1
                if result.fallback:
                    fallback_count += 1

                if success_count % 10 == 0:
                    logger.info(f"Progress: {success_count}/{len(tracks)} stills generated")

            else:
                failed_count += 1
                logger.warning(f"Failed to generate still for track {track_id}")

        except Exception as e:
            failed_count += 1
            logger.error(f"Error generating still for track {track_id}: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Face Stills Generation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Total tracks:     {len(tracks)}")
    logger.info(f"Success:          {success_count} ({100.0*success_count/len(tracks):.1f}%)")
    logger.info(f"Fallbacks:        {fallback_count} ({100.0*fallback_count/max(success_count,1):.1f}% of success)")
    logger.info(f"Failed:           {failed_count}")
    logger.info(f"Manifest:         {manifest_path}")
    if config.get("telemetry", {}).get("verbose", True):
        logger.info(f"Telemetry:        {telemetry_path}")
    logger.info(f"{'='*60}")

    return 0


if __name__ == "__main__":
    exit(main())
