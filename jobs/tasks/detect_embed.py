"""
Detection and embedding task.

Runs RetinaFace detection and ArcFace embedding on harvested frames.
"""

from __future__ import annotations

import json
import logging
import os
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

# Checkpoint interval (10 seconds) - faster heartbeat for better stall detection
CHECKPOINT_INTERVAL_SEC = 10


def detect_embed_task(job_id: str | None = None, episode_id: str | None = None, episode_key: str | None = None) -> dict:
    """
    Detect faces and generate embeddings.

    Args:
        job_id: Job ID (optional, will be auto-generated if not provided)
        episode_id: Episode ID (e.g., RHOBH_S05_E01_11062025) - provide either this or episode_key
        episode_key: Episode key (e.g., rhobh_s05_e01) - provide either this or episode_id

    Returns:
        Dict with detection results
    """
    from api.jobs import job_manager

    # CRITICAL: Log absolute paths and environment at startup for debugging
    RUNNING_FILE = Path(__file__).resolve()
    CWD = Path.cwd()
    BASE_DIR = RUNNING_FILE.parents[2]  # Go up from jobs/tasks/detect_embed.py to project root
    DATA_ROOT = BASE_DIR / "data"

    logger.info(f"[DETECT] ========== DETECT JOB STARTUP ==========")
    logger.info(f"[DETECT] file={RUNNING_FILE}")
    logger.info(f"[DETECT] cwd={CWD}")
    logger.info(f"[DETECT] base_dir={BASE_DIR}")
    logger.info(f"[DETECT] data_root={DATA_ROOT}")

    # CRITICAL: ID Resolution - accept either episode_id or episode_key, resolve the other
    if not episode_id and not episode_key:
        raise ValueError("ERR_MISSING_EPISODE_ID: Must provide either episode_id or episode_key")

    if episode_id and not episode_key:
        # Normalize episode_id to episode_key
        episode_key = job_manager.normalize_episode_key(episode_id)
        logger.info(f"[DETECT] Resolved episode_key={episode_key} from episode_id={episode_id}")
    elif episode_key and not episode_id:
        # Load episode_id from registry
        registry = job_manager.load_episode_registry(episode_key)
        if not registry:
            raise ValueError(f"ERR_EPISODE_NOT_FOUND: No registry found for episode_key={episode_key}")
        episode_id = registry.get("episode_id")
        if not episode_id:
            raise ValueError(f"ERR_MISSING_EPISODE_ID_IN_REGISTRY: Registry for {episode_key} missing episode_id field")
        logger.info(f"[DETECT] Resolved episode_id={episode_id} from episode_key={episode_key}")
    else:
        # Both provided - verify they match
        normalized_key = job_manager.normalize_episode_key(episode_id)
        if normalized_key != episode_key:
            logger.warning(f"[DETECT] episode_key mismatch: provided={episode_key}, normalized={normalized_key}, using normalized")
            episode_key = normalized_key

    # CRITICAL: Always normalize job_id to detect_{episode_id} for standalone jobs
    if not job_id or job_id == "manual":
        # Use unified job ID generator
        from episodes.runtime import generate_job_id
        job_id = generate_job_id("detect", episode_id)
        logger.info(f"[DETECT] Auto-generated job_id={job_id}")

    # Log resolved IDs
    logger.info(f"[DETECT] episode_id={episode_id}")
    logger.info(f"[DETECT] episode_key={episode_key}")
    logger.info(f"[DETECT] job_id={job_id}")
    logger.info(f"[DETECT] stage=start")

    # CRITICAL: Check for existing active job to prevent duplicates
    from episodes.runtime import get_active_job, check_job_stalled
    existing_job = get_active_job(episode_key, "detect", DATA_ROOT)
    if existing_job and existing_job != job_id:
        # Check if it's truly active or stalled
        if not check_job_stalled(existing_job, DATA_ROOT):
            raise ValueError(f"ERR_JOB_ALREADY_RUNNING: Detect job already running: {existing_job}. Use Resume or Cancel in UI.")
        else:
            logger.warning(f"[DETECT] {episode_key} Stalled job {existing_job} found, proceeding with new job {job_id}")

    # CRITICAL: Create job envelope with absolute paths
    # Ensure job envelope exists at absolute path
    job_dir = DATA_ROOT / "jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    meta_path = job_dir / "meta.json"

    logger.info(f"[DETECT] envelope_path={meta_path} (absolute)")

    if not meta_path.exists():
        # Create new envelope
        envelope = {
            "job_id": job_id,
            "episode_id": episode_id,
            "episode_key": episode_key,
            "mode": "detect",
            "created_at": datetime.utcnow().isoformat(),
            "stages": {"detect": {"status": "running"}},
            "registry_path": str(DATA_ROOT / "episodes" / episode_key / "state.json"),
        }
        with open(meta_path, "w") as f:
            json.dump(envelope, f, indent=2)
        logger.info(f"[DETECT] {episode_key} Created job envelope at {meta_path}")
    else:
        # Load existing envelope
        with open(meta_path) as f:
            envelope = json.load(f)
        logger.info(f"[DETECT] {episode_key} Loaded existing envelope from {meta_path}")

    # CRITICAL: Create lock file to prevent duplicate jobs
    lock_path = job_dir / ".lock"
    lock_data = {
        "job_id": job_id,
        "pid": os.getpid(),
        "started_at": datetime.utcnow().isoformat(),
    }
    with open(lock_path, "w") as f:
        json.dump(lock_data, f, indent=2)
    logger.info(f"[DETECT] {episode_key} Created lock file at {lock_path}")

    # Mark stage as running in envelope (for UI polling)
    try:
        job_manager.update_stage_status(job_id, "detect", "running")
        logger.info(f"[DETECT] {episode_key} envelope stage=running")
    except Exception as e:
        # If job_manager fails, update envelope directly
        logger.warning(f"[DETECT] {episode_key} job_manager failed, updating envelope directly: {e}")
        envelope["stages"]["detect"]["status"] = "running"
        envelope["stages"]["detect"]["updated_at"] = datetime.utcnow().isoformat()
        with open(meta_path, "w") as f:
            json.dump(envelope, f, indent=2)

    # Mark detected=false in registry (will flip to true on success)
    try:
        job_manager.update_registry_state(episode_key, "detected", False)
        logger.info(f"[DETECT] {episode_key} registry detected=false")
    except Exception as e:
        logger.warning(f"[DETECT] {episode_key} Could not update registry: {e}")

    # CRITICAL: Register active job in runtime for resume after refresh
    from episodes.runtime import set_active_job
    set_active_job(episode_key, "detect", job_id, DATA_ROOT)
    logger.info(f"[DETECT] {episode_key} Registered as active job: {job_id}")

    # Load config - use absolute path
    config_path = BASE_DIR / "configs" / "pipeline.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Setup paths - use episode_id for physical directories with absolute paths
    harvest_dir = DATA_ROOT / "harvest" / episode_id
    manifest_path = harvest_dir / "manifest.parquet"

    # Create detect artifact directory
    detect_dir = harvest_dir / "detect"
    detect_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[DETECT] artifacts_dir={detect_dir} (absolute)")

    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")

    # Load manifest
    manifest_df = pd.read_parquet(manifest_path)
    logger.info(f"[{job_id}] Loaded manifest with {len(manifest_df)} frames")

    # Get video path
    if job_id.startswith("detect_"):
        # Standalone detect job - get video path from episode registry
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

    # CRITICAL: Heartbeat during model init to prevent "stuck" detection
    # Write heartbeat before loading models
    from screentime.diagnostics.utils import emit_progress
    emit_progress(
        episode_id=episode_id,
        step="1. RetinaFace + ArcFace (Detect & Embed)",
        step_index=1,
        total_steps=5,
        status="running",
        message="Loading RetinaFace detection model...",
        pct=0.0,
    )
    logger.info(f"[DETECT] {episode_key} heartbeat=model_init_start")

    # Load models with timing and provider fallback
    model_start = time.time()
    logger.info(f"[DETECT] {episode_key} model=retinaface loading provider_order={provider_order}...")

    # Provider fallback: try in order, log chosen provider
    detector = None
    detector_error = None
    for i, provider in enumerate(provider_order):
        try:
            logger.info(f"[DETECT] {episode_key} Attempting RetinaFace with provider={provider}")
            detector = RetinaFaceDetector(
                min_face_px=min_face_px,
                min_confidence=min_confidence,
                provider_order=[provider],  # Try one at a time
            )
            detector_time = time.time() - model_start
            provider_info = detector.get_provider_info()
            logger.info(f"[DETECT] {episode_key} model=retinaface loaded in {detector_time:.1f}s provider={provider_info}")

            # Write heartbeat after successful detector load
            emit_progress(
                episode_id=episode_id,
                step="1. RetinaFace + ArcFace (Detect & Embed)",
                step_index=1,
                total_steps=5,
                status="running",
                message=f"RetinaFace loaded ({provider_info}), loading ArcFace...",
                pct=0.1,
            )
            logger.info(f"[DETECT] {episode_key} heartbeat=detector_loaded provider={provider_info}")
            break  # Success!
        except Exception as e:
            detector_error = e
            logger.warning(f"[DETECT] {episode_key} RetinaFace failed with provider={provider}: {e}")
            if i == len(provider_order) - 1:
                # Last provider failed
                logger.error(f"[DETECT] {episode_key} ERR_DETECT_INIT_TIMEOUT: All providers failed for RetinaFace")
                raise ValueError(f"ERR_DETECT_INIT_TIMEOUT: RetinaFace failed with all providers {provider_order}: {detector_error}")

    embedder_start = time.time()
    logger.info(f"[DETECT] {episode_key} model=arcface loading provider_order={provider_order}...")

    # Provider fallback for embedder
    embedder = None
    embedder_error = None
    for i, provider in enumerate(provider_order):
        try:
            logger.info(f"[DETECT] {episode_key} Attempting ArcFace with provider={provider}")
            embedder = ArcFaceEmbedder(
                provider_order=[provider],  # Try one at a time
                skip_redetect=embedding_cfg.get("skip_redetect", False),
                align_priority=embedding_cfg.get("align_priority", "kps_then_bbox"),
                margin_scale=embedding_cfg.get("margin_scale", 1.25),
                min_chip_px=embedding_cfg.get("min_chip_px", 112),
                fallback_scales=embedding_cfg.get("fallback_scales", [1.0, 1.2, 1.4]),
            )
            embedder_time = time.time() - embedder_start
            provider_info = embedder.get_provider_info()
            logger.info(f"[DETECT] {episode_key} model=arcface loaded in {embedder_time:.1f}s provider={provider_info}")

            # Write heartbeat after successful embedder load
            emit_progress(
                episode_id=episode_id,
                step="1. RetinaFace + ArcFace (Detect & Embed)",
                step_index=1,
                total_steps=5,
                status="running",
                message=f"Models loaded, processing frames...",
                pct=0.2,
            )
            logger.info(f"[DETECT] {episode_key} heartbeat=models_ready detector={detector.get_provider_info()} embedder={provider_info}")
            break  # Success!
        except Exception as e:
            embedder_error = e
            logger.warning(f"[DETECT] {episode_key} ArcFace failed with provider={provider}: {e}")
            if i == len(provider_order) - 1:
                # Last provider failed
                logger.error(f"[DETECT] {episode_key} ERR_DETECT_INIT_TIMEOUT: All providers failed for ArcFace")
                raise ValueError(f"ERR_DETECT_INIT_TIMEOUT: ArcFace failed with all providers {provider_order}: {embedder_error}")

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

            # Checkpoint every 30 seconds
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

                # CRITICAL: Update progress for ALL jobs (including standalone detect)
                total_frames = len(manifest_df)
                progress_pct = (processed_frames / total_frames) * 100

                # Update job manager progress (for prepare jobs)
                if not job_id.startswith("detect_"):
                    from api.jobs import job_manager
                    job_manager.update_job_progress(job_id, "detect_embed", progress_pct)

                # CRITICAL: Update envelope with frames_done/frames_total for ALL jobs
                try:
                    from api.jobs import job_manager
                    # Scale progress_pct to 0.2-0.9 range (models loaded at 0.2, final save at 0.9)
                    scaled_pct = 0.2 + (progress_pct / 100.0) * 0.7
                    job_manager.update_stage_status(
                        job_id,
                        "detect",
                        "running",
                        result={
                            "frames_done": processed_frames,
                            "frames_total": total_frames,
                            "faces_detected": detection_stats["faces_detected"],
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                    )

                    # CRITICAL: Update pipeline_state.json for UI polling
                    emit_progress(
                        episode_id=episode_id,
                        step="1. RetinaFace + ArcFace (Detect & Embed)",
                        step_index=1,
                        total_steps=5,
                        status="running",
                        message=f"Processing frames: {processed_frames}/{total_frames} ({progress_pct:.1f}%) • {detection_stats['faces_detected']} faces detected",
                        pct=scaled_pct,
                    )
                    logger.info(f"[DETECT] {episode_key} heartbeat=processing frames={processed_frames}/{total_frames} pct={progress_pct:.1f}% faces={detection_stats['faces_detected']}")
                except Exception as e:
                    logger.warning(f"[DETECT] {episode_key} Could not update progress: {e}")

                # Telemetry
                telemetry.log(
                    TelemetryEvent.JOB_STARTED,
                    metadata={
                        "job_id": job_id,
                        "event": "checkpoint",
                        "progress_pct": progress_pct,
                    },
                )

        # Save embeddings atomically to prevent race conditions with Track stage
        embeddings_df = pd.DataFrame(embeddings_data)
        embeddings_path = detect_dir / "embeddings.parquet"
        embeddings_tmp = detect_dir / "embeddings.parquet.tmp"

        # Write to temp file first
        embeddings_df.to_parquet(embeddings_tmp, index=False)

        # Fsync to ensure data is written to disk
        with open(embeddings_tmp, 'r+b') as f:
            f.flush()
            os.fsync(f.fileno())

        # Atomic rename
        os.replace(embeddings_tmp, embeddings_path)

        logger.info(f"[{job_id}] Saved {len(embeddings_df)} embeddings to {embeddings_path} (atomic write)")

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
        if True:  # Always update envelope for all job types
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

        # CRITICAL: Clear active job from runtime on success
        from episodes.runtime import clear_active_job
        clear_active_job(episode_key, "detect", DATA_ROOT)
        logger.info(f"[DETECT] {episode_key} Cleared active job from runtime")

        # Remove lock file on success
        lock_path = job_dir / ".lock"
        if lock_path.exists():
            lock_path.unlink()
            logger.info(f"[DETECT] {episode_key} Removed lock file")

        # CRITICAL: Add "done": true marker to meta.json for trivial polling
        try:
            with open(meta_path, "r") as f:
                envelope = json.load(f)
            envelope["done"] = True
            envelope["completed_at"] = datetime.utcnow().isoformat()
            with open(meta_path, "w") as f:
                json.dump(envelope, f, indent=2)
            logger.info(f"[DETECT] {episode_key} marked envelope as done")
        except Exception as e:
            logger.error(f"[DETECT] {episode_key} Could not mark envelope as done: {e}")

        # CRITICAL: Emit final progress to pipeline_state.json
        try:
            emit_progress(
                episode_id=episode_id,
                step="1. RetinaFace + ArcFace (Detect & Embed)",
                step_index=1,
                total_steps=5,
                status="ok",
                message=f"Complete! {detection_stats['faces_detected']} faces detected, {detection_stats['embeddings_computed']} embeddings computed",
                pct=1.0,
            )
            logger.info(f"[DETECT] {episode_key} pipeline_state=ok")
        except Exception as e:
            logger.warning(f"[DETECT] {episode_key} Could not emit final progress: {e}")

        # Enqueue next stage (tracking) - only for prepare jobs, not standalone detect
        if not job_id.startswith("detect_"):  # Only enqueue tracking for prepare jobs
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
        error_message = str(e)
        logger.error(f"[DETECT] {episode_key} stage=end status=error error={error_message}")

        # Mark stage as error in envelope
        if True:  # Always update envelope for all job types
            try:
                job_manager.update_stage_status(job_id, "detect", "error", error=error_message)
                logger.info(f"[DETECT] {episode_key} envelope stage=error")
            except Exception as env_err:
                logger.error(f"[DETECT] {episode_key} Could not update envelope with error: {env_err}")

        # CRITICAL: Clear active job on error
        from episodes.runtime import clear_active_job
        try:
            clear_active_job(episode_key, "detect", DATA_ROOT)
            logger.info(f"[DETECT] {episode_key} Cleared active job from runtime (error)")
        except Exception as clear_err:
            logger.warning(f"[DETECT] {episode_key} Could not clear active job: {clear_err}")

        # Remove lock file on error
        try:
            lock_path = job_dir / ".lock"
            if lock_path.exists():
                lock_path.unlink()
                logger.info(f"[DETECT] {episode_key} Removed lock file (error)")
        except Exception as lock_err:
            logger.warning(f"[DETECT] {episode_key} Could not remove lock file: {lock_err}")

        # CRITICAL: Emit error progress to pipeline_state.json
        try:
            emit_progress(
                episode_id=episode_id,
                step="1. RetinaFace + ArcFace (Detect & Embed)",
                step_index=1,
                total_steps=5,
                status="error",
                message=f"Error: {error_message[:200]}",  # Truncate long errors
                pct=0.0,
            )
            logger.info(f"[DETECT] {episode_key} pipeline_state=error")
        except Exception as emit_err:
            logger.warning(f"[DETECT] {episode_key} Could not emit error progress: {emit_err}")

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
