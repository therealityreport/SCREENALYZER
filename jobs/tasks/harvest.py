"""
Harvest task: Frame sampling and manifest generation.

This is a stub implementation for Phase 1.2. Full sampling logic will be added later.
Currently produces a minimal manifest for Phase 1.3 (detection) to run.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
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

# Frame asset contract (UI + asset server)
FRAME_ASSET_SUBDIR = Path("frames/full")
FRAME_INDEX_FILENAME = "frames_index.json"
FRAME_EXT = ".jpg"
FRAME_PAD = 6
FRAME_JPEG_QUALITY = 92
FRAME_IMWRITE_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_JPEG_QUALITY]


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
        frames_dir = harvest_dir / FRAME_ASSET_SUBDIR
        frames_dir.mkdir(parents=True, exist_ok=True)
        _ensure_dir_mode(frames_dir.parent)
        _ensure_dir_mode(frames_dir)
        frame_index_entries: list[dict] = []

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

                rel_path = _ensure_frame_asset(
                    frame,
                    frames_dir,
                    harvest_dir,
                    frame_idx,
                )
                frame_index_entries.append(
                    {
                        "frame_id": int(frame_idx),
                        "ts_ms": ts_ms,
                        "path": rel_path,
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

        frame_index_path = harvest_dir / "frames" / FRAME_INDEX_FILENAME
        frame_index_payload = {
            "episode_id": episode_id,
            "asset_base": FRAME_ASSET_SUBDIR.as_posix(),
            "frame_ext": FRAME_EXT,
            "frame_pad": FRAME_PAD,
            "fps": round(float(fps), 6) if fps else None,
            "frame_count": frame_count,
            "sample_count": sampled_count,
            "generated_at": datetime.utcnow().isoformat(),
            "paths": frame_index_entries,
        }
        _write_json_atomic(frame_index_path, frame_index_payload)

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
            "frames_index_path": str(frame_index_path),
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


def _ensure_dir_mode(path: Path, mode: int = 0o755) -> None:
    """Best-effort chmod to ensure assets are world-readable."""
    try:
        os.chmod(path, mode)
    except PermissionError:
        logger.debug("chmod permission denied for %s", path)
    except OSError:
        logger.debug("chmod os error for %s", path)


def _ensure_frame_asset(
    frame,
    frames_dir: Path,
    harvest_dir: Path,
    frame_id: int,
) -> str:
    """
    Persist decoded frame as JPEG if needed and return relative path.

    Returns:
        Relative POSIX path to the saved frame asset.
    """
    # Lazy import to avoid hard dependency at module import time
    import numpy as np

    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy ndarray")

    filename = f"{frame_id:0{FRAME_PAD}d}{FRAME_EXT}"
    target_path = frames_dir / filename

    if target_path.exists() and target_path.stat().st_size > 0:
        return target_path.relative_to(harvest_dir).as_posix()

    success, encoded = cv2.imencode(FRAME_EXT, frame, FRAME_IMWRITE_PARAMS)
    if not success:
        raise RuntimeError(f"Failed to encode frame {frame_id} to JPEG")

    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(frames_dir), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            fh.write(encoded.tobytes())
        os.replace(tmp_name, target_path)
        os.chmod(target_path, 0o644)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    return target_path.relative_to(harvest_dir).as_posix()


def _write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON payload atomically with 0644 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_name, path)
        os.chmod(path, 0o644)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
