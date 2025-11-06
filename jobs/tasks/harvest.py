"""
Harvest task: Frame sampling and manifest generation.

This is a stub implementation for Phase 1.2. Full sampling logic will be added later.
Currently produces a minimal manifest for Phase 1.3 (detection) to run.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import yaml
from rq import get_current_job

from screentime.diagnostics.telemetry import telemetry, TelemetryEvent

logger = logging.getLogger(__name__)

# Checkpoint interval (30 seconds)
CHECKPOINT_INTERVAL_SEC = 30


def harvest_task(
    job_id: str, episode_id: str, video_path: str, resume_from: str | None = None
) -> dict:
    """
    Harvest frames from video and create manifest.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        video_path: Path to video file
        resume_from: Optional stage to resume from

    Returns:
        Dict with harvest results
    """
    logger.info(f"[{job_id}] Starting harvest task for {episode_id}")

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup paths
    harvest_dir = Path("data/harvest") / episode_id
    harvest_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = harvest_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint state
    checkpoint_file = checkpoint_dir / "job.json"
    last_checkpoint_time = time.time()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        # Get video metadata
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0

        logger.info(f"[{job_id}] Video: {frame_count} frames, {fps:.2f} FPS, {duration_sec:.1f}s")

        # Sampling parameters
        sampling_stride_ms = config["video"]["sampling_stride_ms"]
        frame_stride = int((sampling_stride_ms / 1000) * fps)
        frame_stride = max(1, frame_stride)

        # Sample frames
        manifest_data = []
        frame_idx = 0
        sampled_count = 0

        while True:
            ret = cap.grab()
            if not ret:
                break

            # Sample at stride
            if frame_idx % frame_stride == 0:
                # Read frame
                ret, frame = cap.retrieve()
                if not ret:
                    break

                ts_ms = int((frame_idx / fps) * 1000)

                # Add to manifest
                manifest_data.append(
                    {
                        "episode_id": episode_id,
                        "frame_id": frame_idx,
                        "ts_ms": ts_ms,
                        "sampled": True,
                    }
                )

                sampled_count += 1

            frame_idx += 1

            # Checkpoint every 5 minutes
            if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL_SEC:
                _save_checkpoint(
                    checkpoint_file,
                    {
                        "job_id": job_id,
                        "episode_id": episode_id,
                        "last_completed_stage": "harvest_partial",
                        "frames_processed": frame_idx,
                        "frames_sampled": sampled_count,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                last_checkpoint_time = time.time()

                # Update job progress - only for job workflow, not manual
                if job_id != "manual":
                    progress_pct = (frame_idx / frame_count) * 100
                    _update_job_progress(job_id, "harvest", progress_pct)

                # Telemetry
                telemetry.log(
                    TelemetryEvent.JOB_STARTED,
                    metadata={
                        "job_id": job_id,
                        "event": "checkpoint",
                        "progress_pct": (frame_idx / frame_count) * 100,
                    },
                )

        # Save manifest
        manifest_df = pd.DataFrame(manifest_data)
        manifest_path = harvest_dir / "manifest.parquet"
        manifest_df.to_parquet(manifest_path, index=False)

        # Calculate coverage
        coverage_pct = (sampled_count / frame_count * 100) if frame_count > 0 else 0

        logger.info(
            f"[{job_id}] Harvest complete: {sampled_count}/{frame_count} frames sampled ({coverage_pct:.1f}% coverage)"
        )

        # Save final checkpoint
        _save_checkpoint(
            checkpoint_file,
            {
                "job_id": job_id,
                "episode_id": episode_id,
                "last_completed_stage": "harvest",
                "frames_processed": frame_count,
                "frames_sampled": sampled_count,
                "coverage_pct": coverage_pct,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_STAGE_COMPLETE,
            metadata={
                "job_id": job_id,
                "stage": "harvest",
                "frames_total": frame_count,
                "frames_sampled": sampled_count,
                "sampling_coverage_pct": coverage_pct,
            },
        )

        # Enqueue next stage (detect_embed) - only for job workflow, not manual
        if job_id != "manual":
            from api.jobs import inference_queue
            from jobs.tasks.detect_embed import detect_embed_task

            inference_queue.enqueue(
                detect_embed_task,
                job_id=job_id,
                episode_id=episode_id,
                job_timeout="30m",
            )

        return {
            "job_id": job_id,
            "episode_id": episode_id,
            "frames_sampled": sampled_count,
            "manifest_path": str(manifest_path),
        }

    finally:
        cap.release()


def _save_checkpoint(checkpoint_file: Path, checkpoint_data: dict) -> None:
    """Save checkpoint to disk and Redis."""
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_file}")


def _update_job_progress(job_id: str, stage: str, progress_pct: float) -> None:
    """Update job progress in Redis."""
    from api.jobs import job_manager

    job_manager.update_job_progress(job_id, stage, progress_pct, message=f"Processing {stage}")
