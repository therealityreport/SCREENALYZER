"""
Detection and embedding task.

Runs RetinaFace detection and ArcFace embedding on harvested frames.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.diagnostics.telemetry import telemetry, TelemetryEvent
from screentime.recognition.embed_arcface import ArcFaceEmbedder

logger = logging.getLogger(__name__)

# Checkpoint interval (30 seconds)
CHECKPOINT_INTERVAL_SEC = 30


def detect_embed_task(job_id: str, episode_id: str) -> dict:
    """
    Detect faces and generate embeddings.

    Args:
        job_id: Job ID
        episode_id: Episode ID

    Returns:
        Dict with detection results
    """
    from api.jobs import job_manager

    # Get canonical episode key for logging and registry updates
    episode_key = job_manager.normalize_episode_key(episode_id)

    logger.info(f"[DETECT] {episode_key} stage=start job_id={job_id}")

    # CRITICAL: Update envelope and registry at START to show progress in UI
    # Mark stage as running in envelope (for UI polling)
    if job_id != "manual":
        try:
            job_manager.update_stage_status(job_id, "detect", "running")
            logger.info(f"[DETECT] {episode_key} envelope stage=running")
        except Exception as e:
            logger.warning(f"[DETECT] {episode_key} Could not update envelope: {e}")

    # Mark detected=false in registry (will flip to true on success)
    try:
        job_manager.update_registry_state(episode_key, "detected", False)
        logger.info(f"[DETECT] {episode_key} registry detected=false")
    except Exception as e:
        logger.warning(f"[DETECT] {episode_key} Could not update registry: {e}")

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup paths
    harvest_dir = Path("data/harvest") / episode_id
    manifest_path = harvest_dir / "manifest.parquet"

    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")

    # Load manifest
    manifest_df = pd.read_parquet(manifest_path)
    logger.info(f"[{job_id}] Loaded manifest with {len(manifest_df)} frames")

    # Get video path
    if job_id == "manual":
        # Manual workflow: get video path from episode registry
        from screentime.episode_registry import episode_registry
        episode_data = episode_registry.get_episode(episode_id)
        if not episode_data:
            raise ValueError(f"Episode {episode_id} not found in registry")
        video_path = str(Path("data") / episode_data["video_path"])
    else:
        # Job workflow: get video path from job metadata (with self-healing)
        from api.jobs import job_manager

        # Try Redis first
        job_data = job_manager._get_job_metadata(job_id)

        if not job_data:
            # Self-heal: try loading job envelope from disk
            logger.warning(f"[{job_id}] Job metadata not in Redis, attempting self-heal from disk envelope")
            envelope = job_manager.load_job_envelope(job_id)

            if envelope:
                logger.info(f"[{job_id}] Recovered from disk envelope")
                video_path = envelope.get("video_path")
                if not video_path:
                    raise ValueError(f"Job envelope missing video_path for {job_id}")
            else:
                # Try episode registry
                logger.warning(f"[{job_id}] No disk envelope, attempting self-heal from episode registry")

                # Extract episode_id from job_id (e.g., prepare_RHOBH_S05_E03_11062025 -> RHOBH_S05_E03_11062025)
                episode_id_from_job = None
                if job_id.startswith("prepare_"):
                    episode_id_from_job = job_id.replace("prepare_", "")
                elif job_id.startswith("cluster_"):
                    episode_id_from_job = job_id.replace("cluster_", "")
                elif job_id.startswith("analytics_"):
                    episode_id_from_job = job_id.replace("analytics_", "")

                if episode_id_from_job:
                    episode_key = job_manager.normalize_episode_key(episode_id_from_job)
                    registry = job_manager.load_episode_registry(episode_key)

                    if registry:
                        video_path = registry.get("video_path")
                        logger.info(f"[{job_id}] Recovered video path from episode registry: {video_path}")

                        # Reconstruct envelope from registry
                        reconstructed_envelope = {
                            "job_id": job_id,
                            "episode_id": episode_id_from_job,
                            "episode_key": episode_key,
                            "video_path": video_path,
                            "mode": "prepare",
                            "created_at": time.time(),
                            "self_healed": True,
                            "registry_path": f"episodes/{episode_key}/state.json",
                            "stages": {
                                "detect": {"status": "running"},
                            },
                        }
                        job_manager.write_job_envelope(job_id, reconstructed_envelope)
                        logger.info(f"[{job_id}] Wrote reconstructed envelope from registry")
                    else:
                        # Last resort: try to reconstruct from episodes.json
                        logger.warning(f"[{job_id}] No episode registry, attempting self-heal from episodes.json")
                        from app.lib.registry import load_episodes_json
                        episodes_data = load_episodes_json()

                        # Find episode by matching job_id pattern (e.g., prepare_RHOBH_S05_E03_11062025)
                episode_match = None
                for ep in episodes_data.get("episodes", []):
                    ep_id = ep.get("episode_id", "")
                    if ep_id and ep_id in job_id:
                        episode_match = ep
                        break

                if episode_match:
                    video_path = episode_match.get("video_path")
                    logger.info(f"[{job_id}] Recovered video path from episodes.json: {video_path}")

                    # Reconstruct minimal envelope for future use
                    reconstructed_envelope = {
                        "job_id": job_id,
                        "episode_key": episode_match.get("episode_id"),
                        "video_path": video_path,
                        "mode": "prepare",
                        "created_at": time.time(),
                        "self_healed": True,
                        "stages": {
                            "detect": {"status": "running"},
                        },
                    }
                    job_manager.write_job_envelope(job_id, reconstructed_envelope)
                    logger.info(f"[{job_id}] Wrote reconstructed envelope")
        else:
            video_path = job_data["video_path"]

    # Final safety check - video_path MUST be set by now
    if not video_path:
        raise ValueError(f"ERR_EPISODE_NOT_REGISTERED: video_path could not be resolved for job {job_id}, episode {episode_id}")

    # Initialize detector and embedder
    min_face_px = config["video"]["min_face_px"]
    min_confidence = config["detection"]["min_confidence"]
    provider_order = config["detection"]["provider_order"]

    # Get embedding config
    embedding_cfg = config["embedding"]

    # Load models with timing
    model_start = time.time()
    logger.info(f"[DETECT] {episode_key} model=retinaface loading...")

    detector = RetinaFaceDetector(
        min_face_px=min_face_px,
        min_confidence=min_confidence,
        provider_order=provider_order,
    )

    detector_time = time.time() - model_start
    logger.info(f"[DETECT] {episode_key} model=retinaface loaded in {detector_time:.1f}s provider={detector.get_provider_info()}")

    embedder_start = time.time()
    logger.info(f"[DETECT] {episode_key} model=arcface loading...")

    embedder = ArcFaceEmbedder(
        provider_order=provider_order,
        skip_redetect=embedding_cfg.get("skip_redetect", False),
        align_priority=embedding_cfg.get("align_priority", "kps_then_bbox"),
        margin_scale=embedding_cfg.get("margin_scale", 1.25),
        min_chip_px=embedding_cfg.get("min_chip_px", 112),
        fallback_scales=embedding_cfg.get("fallback_scales", [1.0, 1.2, 1.4]),
    )

    embedder_time = time.time() - embedder_start
    logger.info(f"[DETECT] {episode_key} model=arcface loaded in {embedder_time:.1f}s provider={embedder.get_provider_info()}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        # Process frames
        embeddings_data = []
        detection_stats = defaultdict(int)
        confidence_values = []

        # Detailed embedding diagnostics
        embedding_diag = {
            "faces_seen": 0,
            "chips_attempted": 0,
            "embedded_ok": 0,
            "used_kps": 0,
            "used_bbox": 0,
            "chip_sizes": [],
            "tries_per_face": [],
        }

        last_checkpoint_time = time.time()
        processed_frames = 0

        for _, row in manifest_df.iterrows():
            frame_id = row["frame_id"]
            ts_ms = row["ts_ms"]

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret:
                logger.warning(f"[{job_id}] Failed to read frame {frame_id}")
                detection_stats["failed_reads"] += 1
                continue

            # Detect faces
            detections = detector.detect(frame)
            detection_stats["frames_processed"] += 1
            detection_stats["faces_detected"] += len(detections)

            if len(detections) == 0:
                detection_stats["frames_no_faces"] += 1

            # Generate embeddings for each detection
            for det_idx, det in enumerate(detections):
                bbox = det["bbox"]
                confidence = det["confidence"]
                kps = det.get("landmarks")  # Get keypoints if available
                confidence_values.append(confidence)

                embedding_diag["faces_seen"] += 1

                # Generate embedding using new method with diagnostics
                result = embedder.embed_from_detection(frame, bbox, kps=kps)

                # Track diagnostics
                embedding_diag["chips_attempted"] += 1
                embedding_diag["tries_per_face"].append(result.tries)

                if result.success:
                    embedding_diag["embedded_ok"] += 1
                    embedding_diag["chip_sizes"].append(result.chip_px)

                    if result.used_kps:
                        embedding_diag["used_kps"] += 1
                    else:
                        embedding_diag["used_bbox"] += 1

                    embeddings_data.append(
                        {
                            "episode_id": episode_id,
                            "frame_id": frame_id,
                            "ts_ms": ts_ms,
                            "det_idx": det_idx,
                            "bbox_x1": bbox[0],
                            "bbox_y1": bbox[1],
                            "bbox_x2": bbox[2],
                            "bbox_y2": bbox[3],
                            "confidence": confidence,
                            "face_size": det["face_size"],
                            "embedding": result.embedding.tolist(),
                        }
                    )
                    detection_stats["embeddings_computed"] += 1
                else:
                    detection_stats["embedding_failures"] += 1

            processed_frames += 1

            # Checkpoint every 5 minutes
            if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL_SEC:
                _save_checkpoint(
                    harvest_dir,
                    {
                        "job_id": job_id,
                        "episode_id": episode_id,
                        "last_completed_stage": "detect_embed_partial",
                        "frames_processed": processed_frames,
                        "faces_detected": detection_stats["faces_detected"],
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
                last_checkpoint_time = time.time()

                # Update job progress - only for job workflow, not manual
                progress_pct = (processed_frames / len(manifest_df)) * 100
                if job_id != "manual":
                    from api.jobs import job_manager
                    job_manager.update_job_progress(job_id, "detect_embed", progress_pct)

                # Telemetry
                telemetry.log(
                    TelemetryEvent.JOB_STARTED,
                    metadata={
                        "job_id": job_id,
                        "event": "checkpoint",
                        "progress_pct": progress_pct,
                    },
                )

        # Save embeddings
        embeddings_df = pd.DataFrame(embeddings_data)
        embeddings_path = harvest_dir / "embeddings.parquet"
        embeddings_df.to_parquet(embeddings_path, index=False)

        logger.info(f"[{job_id}] Saved {len(embeddings_df)} embeddings to {embeddings_path}")

        # Calculate confidence histogram
        if confidence_values:
            conf_hist, conf_bins = np.histogram(confidence_values, bins=10, range=(0.0, 1.0))
            detection_stats["confidence_histogram"] = {
                "bins": conf_bins.tolist(),
                "counts": conf_hist.tolist(),
            }

        # Calculate no-face warning
        no_face_pct = detection_stats["frames_no_faces"] / detection_stats["frames_processed"] * 100
        if no_face_pct > 20:
            logger.warning(f"[{job_id}] {no_face_pct:.1f}% of frames have no valid detections")
            detection_stats["warning"] = f"{no_face_pct:.1f}% frames have no faces"

        # Add provider info
        detection_stats["detector_provider"] = detector.get_provider_info()
        detection_stats["embedder_provider"] = embedder.get_provider_info()

        # Save stats
        reports_dir = harvest_dir / "diagnostics" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        stats_path = reports_dir / "det_stats.json"
        with open(stats_path, "w") as f:
            json.dump(dict(detection_stats), f, indent=2)

        logger.info(f"[{job_id}] Detection stats saved to {stats_path}")

        # Compute and save embedding diagnostics
        embedding_stats = {
            "faces_seen": embedding_diag["faces_seen"],
            "chips_accepted": embedding_diag["chips_attempted"],
            "embedded_ok": embedding_diag["embedded_ok"],
            "success_rate_pct": (
                100.0 * embedding_diag["embedded_ok"] / embedding_diag["faces_seen"]
                if embedding_diag["faces_seen"] > 0
                else 0.0
            ),
            "used_kps_pct": (
                100.0 * embedding_diag["used_kps"] / embedding_diag["embedded_ok"]
                if embedding_diag["embedded_ok"] > 0
                else 0.0
            ),
            "used_bbox_pct": (
                100.0 * embedding_diag["used_bbox"] / embedding_diag["embedded_ok"]
                if embedding_diag["embedded_ok"] > 0
                else 0.0
            ),
            "avg_chip_px": (
                np.mean(embedding_diag["chip_sizes"])
                if embedding_diag["chip_sizes"]
                else 0.0
            ),
            "avg_tries_per_face": (
                np.mean(embedding_diag["tries_per_face"])
                if embedding_diag["tries_per_face"]
                else 0.0
            ),
            "config": {
                "skip_redetect": embedding_cfg.get("skip_redetect", False),
                "min_chip_px": embedding_cfg.get("min_chip_px", 112),
                "margin_scale": embedding_cfg.get("margin_scale", 1.25),
                "align_priority": embedding_cfg.get("align_priority", "kps_then_bbox"),
            },
        }

        embedding_stats_path = harvest_dir / "diagnostics" / "embedding_stats.json"
        with open(embedding_stats_path, "w") as f:
            json.dump(embedding_stats, f, indent=2)

        logger.info(
            f"[{job_id}] Embedding diagnostics: {embedding_stats['embedded_ok']}/{embedding_stats['faces_seen']} "
            f"({embedding_stats['success_rate_pct']:.1f}%) → {embedding_stats_path}"
        )

        # Save final checkpoint
        _save_checkpoint(
            harvest_dir,
            {
                "job_id": job_id,
                "episode_id": episode_id,
                "last_completed_stage": "detect_embed",
                "frames_processed": processed_frames,
                "faces_detected": detection_stats["faces_detected"],
                "embeddings_computed": detection_stats["embeddings_computed"],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Telemetry
        telemetry.log(
            TelemetryEvent.DETECTION_COMPLETED,
            metadata={
                "job_id": job_id,
                "faces_detected": detection_stats["faces_detected"],
                "embeddings_computed": detection_stats["embeddings_computed"],
            },
        )

        telemetry.log(
            TelemetryEvent.JOB_STAGE_COMPLETE,
            metadata={
                "job_id": job_id,
                "stage": "detect_embed",
                "faces_detected": detection_stats["faces_detected"],
            },
        )

        # CRITICAL: Update envelope and registry on SUCCESS
        logger.info(f"[DETECT] {episode_key} stage=end status=ok frames={detection_stats['frames_processed']} faces={detection_stats['faces_detected']}")

        # Mark stage as complete in envelope
        if job_id != "manual":
            try:
                job_manager.update_stage_status(
                    job_id,
                    "detect",
                    "ok",
                    result={
                        "faces_detected": detection_stats["faces_detected"],
                        "embeddings_computed": detection_stats["embeddings_computed"],
                    },
                )
                logger.info(f"[DETECT] {episode_key} envelope stage=ok")
            except Exception as e:
                logger.error(f"[DETECT] {episode_key} Could not update envelope: {e}")

        # Mark detected=true in registry
        try:
            job_manager.update_registry_state(episode_key, "detected", True)
            logger.info(f"[DETECT] {episode_key} registry detected=true")
        except Exception as e:
            logger.error(f"[DETECT] {episode_key} ERR_REGISTRY_UPDATE_FAILED: {e}")
            raise ValueError(f"ERR_REGISTRY_UPDATE_FAILED: Could not update registry for {episode_key}: {e}")

        # Enqueue next stage (tracking) - only for job workflow, not manual
        if job_id != "manual":
            job_manager.tracking_queue.enqueue(
                "jobs.tasks.track.track_task",
                job_id=job_id,
                episode_id=episode_id,
                job_timeout="30m",
            )

            logger.info(f"[DETECT] {episode_key} enqueued tracking task")

        return {
            "job_id": job_id,
            "episode_id": episode_id,
            "episode_key": episode_key,
            "embeddings_path": str(embeddings_path),
            "stats_path": str(stats_path),
            "stats": dict(detection_stats),
        }

    except Exception as e:
        # CRITICAL: Update envelope and registry on FAILURE
        logger.error(f"[DETECT] {episode_key} stage=end status=error error={str(e)}")

        # Mark stage as error in envelope
        if job_id != "manual":
            try:
                job_manager.update_stage_status(job_id, "detect", "error", error=str(e))
                logger.info(f"[DETECT] {episode_key} envelope stage=error")
            except Exception as env_err:
                logger.error(f"[DETECT] {episode_key} Could not update envelope with error: {env_err}")

        # Re-raise to propagate error
        raise

    finally:
        cap.release()


def _save_checkpoint(harvest_dir: Path, checkpoint_data: dict) -> None:
    """Save checkpoint to disk."""
    checkpoint_dir = harvest_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "job.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_file}")
