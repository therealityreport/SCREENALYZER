"""
Tracking task.

Builds face tracks across frames using ByteTrack.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from screenalyzer.reid.assigner import HysteresisAssigner, ReIdConfig
from screenalyzer.reid.faiss_index import FaceIndex
from screenalyzer.reid.metrics import compute_id_metrics, log_identity_event, _json_default
from screentime.diagnostics.telemetry import telemetry, TelemetryEvent
from screentime.episode_registry import episode_registry
from screentime.tracking.bytetrack_wrap import ByteTracker
from screentime.tracking.reid import TrackReID
from screentime.utils import canonical_show_slug

logger = logging.getLogger(__name__)

# Checkpoint interval (5 minutes)
CHECKPOINT_INTERVAL_SEC = 5 * 60


def _resolve_episode_keys(episode_id: str) -> tuple[str, str]:
    """Derive show and season identifiers for artifact placement."""
    info = episode_registry.get_episode(episode_id)
    if info:
        show_id = canonical_show_slug(info.get("show_id") or "unknown")
        season_id = (info.get("season_id") or "unknown").lower()
    else:
        tokens = episode_id.split("_")
        if len(tokens) >= 2:
            show_id = canonical_show_slug(tokens[0])
            season_id = tokens[1].lower()
        else:
            parts = episode_id.split("-")
            show_id = canonical_show_slug(parts[0])
            season_id = "unknown"
    return show_id or "unknown", season_id or "unknown"


def track_task(job_id: str, episode_id: str) -> dict:
    """
    Build face tracks across frames.

    Args:
        job_id: Job ID
        episode_id: Episode ID

    Returns:
        Dict with tracking results
    """
    logger.info(f"[{job_id}] Starting track task for {episode_id}")

    # Load configs
    bytetrack_config_path = Path("configs/bytetrack.yaml")
    with open(bytetrack_config_path) as f:
        bytetrack_config = yaml.safe_load(f)

    pipeline_config_path = Path("configs/pipeline.yaml")
    with open(pipeline_config_path) as f:
        pipeline_config = yaml.safe_load(f)

    track_config = bytetrack_config["track"]
    reid_config = pipeline_config.get("tracking", {}).get("reid", {})

    persistent_cfg_path = Path("config/reid.yaml")
    persistent_cfg = {}
    if persistent_cfg_path.exists():
        try:
            with open(persistent_cfg_path, "r", encoding="utf-8") as f:
                persistent_cfg = yaml.safe_load(f) or {}
        except yaml.YAMLError as exc:
            logger.warning("Failed to load persistent re-ID config %s: %s", persistent_cfg_path, exc)
            persistent_cfg = {}

    persistent_assigner = None
    face_index = None
    identity_log_path: Optional[Path] = None
    faces_path: Optional[Path] = None
    centroids_path: Optional[Path] = None
    id_map_path: Optional[Path] = None
    identity_write_events = False
    persistent_events: list[dict] = []
    last_assign_frame: dict[int, int] = {}
    prev_active_ids: set[int] = set()

    if persistent_cfg.get("enabled"):
        show_id, season_id = _resolve_episode_keys(episode_id)

        faiss_cfg = persistent_cfg.get("faiss", {})
        metric_key = faiss_cfg.get("type", "FlatIP")
        metric = "ip" if metric_key.lower() in {"flatip", "ip"} else "l2"
        normalize = faiss_cfg.get("normalize", True)

        face_index = FaceIndex(metric=metric, normalize=normalize)

        io_cfg = persistent_cfg.get("io", {})
        index_root = Path(io_cfg.get("index_dir", "data/indices"))
        bank_root = Path(io_cfg.get("bank_dir", "data/facebank"))

        episode_index_dir = index_root / show_id / season_id / episode_id
        episode_index_dir.mkdir(parents=True, exist_ok=True)

        faces_path = episode_index_dir / "faces.faiss"
        centroids_path = episode_index_dir / "centroids.json"
        id_map_path = episode_index_dir / "id_map.json"

        bank_embeddings = bank_root / show_id / season_id / "embeddings.npz"
        if bank_embeddings.exists():
            try:
                bank_data = np.load(bank_embeddings, allow_pickle=True)
                embeddings = bank_data.get("embeddings")
                labels = bank_data.get("labels")
                if embeddings is not None and labels is not None:
                    for label, emb in zip(labels, embeddings):
                        face_index.add(str(label), emb)
                else:
                    logger.info("Facebank embeddings file %s missing expected arrays", bank_embeddings)
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Failed to preload facebank embeddings from %s: %s", bank_embeddings, exc)
        else:
            logger.info("No curated facebank embeddings found at %s", bank_embeddings)

        if faces_path.exists() and centroids_path.exists():
            try:
                face_index.load(str(faces_path), str(centroids_path))
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.warning("Unable to load existing re-ID index for %s: %s", episode_index_dir, exc)

        thresholds = persistent_cfg.get("thresholds", {})
        stability = persistent_cfg.get("stability", {})
        centroid_cfg = persistent_cfg.get("centroid", {})

        defaults = ReIdConfig()
        assigner_config = ReIdConfig(
            tau_join=thresholds.get("tau_join", defaults.tau_join),
            tau_stay=thresholds.get("tau_stay", defaults.tau_stay),
            tau_spawn=thresholds.get("tau_spawn", defaults.tau_spawn),
            ema_window=stability.get("ema_window", defaults.ema_window),
            confirm_frames=stability.get("confirm_frames", defaults.confirm_frames),
            drop_frames=stability.get("drop_frames", defaults.drop_frames),
            ema_alpha=centroid_cfg.get("ema_alpha", defaults.ema_alpha),
            max_sigma=centroid_cfg.get("max_sigma", defaults.max_sigma),
        )

        persistent_assigner = HysteresisAssigner(face_index, assigner_config)

        if id_map_path.exists():
            try:
                with open(id_map_path, "r", encoding="utf-8") as f:
                    persistent_assigner.id_map.update(json.load(f))
            except json.JSONDecodeError:
                logger.warning("Failed to parse id_map.json at %s", id_map_path)

        logging_cfg = persistent_cfg.get("logging", {})
        identity_write_events = logging_cfg.get("write_events", False)
        if identity_write_events:
            identity_log_path = Path(logging_cfg.get("path", "logs/reid_events.jsonl"))


    # Setup paths
    harvest_dir = Path("data/harvest") / episode_id
    embeddings_path = harvest_dir / "embeddings.parquet"
    manifest_path = harvest_dir / "manifest.parquet"

    if not embeddings_path.exists():
        raise ValueError(f"Embeddings not found: {embeddings_path}")

    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")

    # Load embeddings and manifest
    embeddings_df = pd.read_parquet(embeddings_path)
    manifest_df = pd.read_parquet(manifest_path)

    logger.info(f"[{job_id}] Loaded {len(embeddings_df)} embeddings from {len(manifest_df)} frames")

    # Detect scene boundaries
    scene_boundary_frames = set()
    if reid_config.get("use_scene_bounds", False):
        from screentime.video.scene_detect import detect_scene_boundaries

        # Get video path
        video_path = None
        if job_id == "manual":
            from screentime.episode_registry import episode_registry
            episode_data = episode_registry.get_episode(episode_id)
            if episode_data:
                video_path = str(Path("data") / episode_data["video_path"])
        else:
            from api.jobs import job_manager
            job_data = job_manager._get_job_metadata(job_id)
            video_path = job_data.get("video_path")

        if video_path:
            try:
                boundaries = detect_scene_boundaries(
                    manifest_path,
                    video_path,
                    threshold=80.0,  # High threshold = hard cuts only (not motion)
                    min_scene_duration_ms=2000,  # Require 2s minimum scene
                )
                scene_boundary_frames = set(boundaries)
                logger.info(f"[{job_id}] Detected {len(scene_boundary_frames)} scene boundaries")
            except Exception as e:
                logger.warning(f"[{job_id}] Scene detection failed: {e}, continuing without it")

    # Initialize tracker
    tracker = ByteTracker(
        track_buffer=track_config["track_buffer"],
        match_thresh=track_config["match_thresh"],
        conf_thresh=track_config["conf_thresh"],
        iou_thresh=track_config["iou_thresh"],
    )

    # Get unique frame IDs in order
    frame_ids = sorted(embeddings_df["frame_id"].unique())

    # Process frames
    tracking_stats = defaultdict(int)
    track_switches = 0
    last_checkpoint_time = time.time()
    processed_frames = 0

    for frame_id in frame_ids:
        # Get all detections for this frame
        frame_dets = embeddings_df[embeddings_df["frame_id"] == frame_id]

        # Get timestamp from manifest
        frame_row = manifest_df[manifest_df["frame_id"] == frame_id].iloc[0]
        ts_ms = frame_row["ts_ms"]

        # Prepare detections for tracker
        detections = []
        for _, det in frame_dets.iterrows():
            bbox = [
                int(det["bbox_x1"]),
                int(det["bbox_y1"]),
                int(det["bbox_x2"]),
                int(det["bbox_y2"]),
            ]
            embedding = np.array(det["embedding"])
            confidence = det["confidence"]
            det_idx = det["det_idx"]

            detections.append(
                {
                    "bbox": bbox,
                    "embedding": embedding,
                    "confidence": confidence,
                    "det_idx": det_idx,
                }
            )

        # Update tracker (process detections first)
        active_tracks = tracker.update(detections, frame_id, ts_ms)
        tracking_stats["frames_processed"] += 1
        tracking_stats["detections_processed"] += len(detections)

        if persistent_assigner:
            current_active_ids = set()
            for track in active_tracks:
                current_active_ids.add(track.track_id)
                if not track.embeddings:
                    continue

                last_observed_frame = track.frame_ids[-1] if track.frame_ids else frame_id
                should_assign = False
                if track.count == 1:
                    should_assign = True
                else:
                    previous_frame = last_assign_frame.get(track.track_id)
                    if previous_frame is None or last_observed_frame - previous_frame >= persistent_assigner.cfg.ema_window:
                        should_assign = True

                if should_assign:
                    label, debug = persistent_assigner.assign(track.embeddings[-1], frame_id, str(track.track_id))
                    track.global_id = label
                    if track.labels:
                        track.labels[-1] = label
                    else:
                        track.labels.append(label)
                    last_assign_frame[track.track_id] = last_observed_frame

                    debug_event = dict(debug)
                    debug_event.setdefault("frame", frame_id)
                    debug_event["track_id"] = str(track.track_id)
                    debug_event.setdefault("episode_id", episode_id)

                    if "reason" in debug_event:
                        persistent_events.append(debug_event)
                        if identity_write_events and identity_log_path is not None:
                            log_identity_event(str(identity_log_path), debug_event)
                else:
                    if track.labels:
                        track.labels[-1] = track.global_id
                    else:
                        track.labels.append(track.global_id)

            ended_ids = prev_active_ids - current_active_ids
            for ended_id in ended_ids:
                persistent_assigner.end_track(str(ended_id))
                last_assign_frame.pop(ended_id, None)
            prev_active_ids = current_active_ids

        # Terminate all tracks AFTER update if we just crossed a scene boundary
        # This forces tracks to end at cuts, preventing cross-cut false matches
        frame_id_for_boundary = int(frame_row["frame_id"])
        if frame_id_for_boundary in scene_boundary_frames:
            tracker.terminate_all_active_tracks()
            tracking_stats["scene_boundaries_hit"] += 1

        processed_frames += 1

        # Checkpoint every 5 minutes
        if time.time() - last_checkpoint_time > CHECKPOINT_INTERVAL_SEC:
            _save_checkpoint(
                harvest_dir,
                {
                    "job_id": job_id,
                    "episode_id": episode_id,
                    "last_completed_stage": "track_partial",
                    "frames_processed": processed_frames,
                    "tracks_active": len(active_tracks),
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )
            last_checkpoint_time = time.time()

            # Update job progress
            # Update job progress - only for job workflow, not manual
            progress_pct = (processed_frames / len(frame_ids)) * 100
            if job_id != "manual":
                from api.jobs import job_manager
                job_manager.update_job_progress(job_id, "track", progress_pct)

                # CRITICAL: Emit progress with frames_done/frames_total for UI progress bars
                try:
                    from screentime.diagnostics.utils import emit_progress

                    # Update envelope with frames_done/frames_total
                    job_manager.update_stage_status(
                        job_id,
                        "track",
                        "running",
                        result={
                            "frames_done": processed_frames,
                            "frames_total": len(frame_ids),
                            "tracks_active": len(active_tracks),
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                    )

                    # Emit progress to pipeline_state.json
                    emit_progress(
                        episode_id=episode_id,
                        step="2. ByteTrack (Track Faces)",
                        step_index=2,
                        total_steps=4,
                        status="running",
                        message=f"Tracking frames: {processed_frames}/{len(frame_ids)} ({progress_pct:.1f}%) • {len(active_tracks)} active tracks",
                        pct=progress_pct / 100.0,
                    )
                    logger.info(f"[TRACK] {episode_id} heartbeat=tracking frames={processed_frames}/{len(frame_ids)} pct={progress_pct:.1f}% tracks={len(active_tracks)}")
                except Exception as e:
                    logger.warning(f"[TRACK] {episode_id} Could not update progress: {e}")

            # Telemetry
            telemetry.log(
                TelemetryEvent.JOB_STARTED,
                metadata={
                    "job_id": job_id,
                    "event": "checkpoint",
                    "progress_pct": progress_pct,
                },
            )

    if persistent_assigner and prev_active_ids:
        for remaining_id in prev_active_ids:
            persistent_assigner.end_track(str(remaining_id))
        prev_active_ids.clear()

    # Get all tracks
    all_tracks = tracker.get_all_tracks()
    tracking_stats["tracks_built"] = len(all_tracks)

    # Calculate track switch rate
    for track in all_tracks:
        if track.count > 1:
            # Count gaps in frame sequence
            frame_gaps = 0
            for i in range(1, len(track.frame_ids)):
                if track.frame_ids[i] - track.frame_ids[i - 1] > 1:
                    frame_gaps += 1
            track_switches += frame_gaps

    tracking_stats["track_switches"] = track_switches

    # Build tracks.json output (initial)
    tracks_data = []
    track_durations = []

    for track in all_tracks:
        if track.count == 0:
            continue

        track_duration_ms = track.duration_ms
        track_durations.append(track_duration_ms)

        # Calculate stitch score (ratio of actual detections to expected detections)
        expected_dets = (track.end_frame - track.start_frame) + 1
        stitch_score = track.count / expected_dets if expected_dets > 0 else 0.0

        tracks_data.append(
            {
                "track_id": track.track_id,
                "start_ms": track.start_ms,
                "end_ms": track.end_ms,
                "duration_ms": track_duration_ms,
                "count": track.count,
                "stitch_score": round(stitch_score, 3),
                "mean_confidence": round(float(np.mean(track.confidences)), 3),
                "global_id": getattr(track, "global_id", "UNK"),
                "frame_refs": [
                    {
                        "frame_id": int(fid),
                        "det_idx": int(didx),
                        "bbox": bbox,
                        "confidence": round(conf, 3),
                        "global_id": label,
                    }
                    for fid, didx, bbox, conf, label in zip(
                        track.frame_ids,
                        track.det_indices,
                        track.bboxes,
                        track.confidences,
                        track.labels or ["UNK"] * len(track.frame_ids),
                    )
                ],
            }
        )

    # Apply re-ID track stitching if enabled
    reid_stats = {}
    if reid_config.get("enabled", False):
        logger.info(f"[{job_id}] Applying track re-ID stitching")
        start_time = time.time()

        # Build embeddings_by_track: track_id -> [(embedding, conf, face_px)]
        embeddings_by_track = defaultdict(list)

        for track_dict in tracks_data:
            track_id = track_dict["track_id"]

            for frame_ref in track_dict["frame_refs"]:
                frame_id = frame_ref["frame_id"]
                det_idx = frame_ref["det_idx"]

                # Find embedding
                emb_row = embeddings_df[
                    (embeddings_df["frame_id"] == frame_id)
                    & (embeddings_df["det_idx"] == det_idx)
                ]

                if len(emb_row) > 0:
                    embedding = np.array(emb_row.iloc[0]["embedding"])
                    conf = frame_ref["confidence"]
                    bbox = frame_ref["bbox"]
                    face_px = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                    embeddings_by_track[track_id].append((embedding, conf, face_px))

        # Initialize re-ID tracker
        reid_tracker = TrackReID(
            max_gap_ms=reid_config.get("max_gap_ms", 2500),
            min_sim=reid_config.get("min_sim", 0.82),
            min_margin=reid_config.get("min_margin", 0.08),
            use_scene_bounds=reid_config.get("use_scene_bounds", True),
            topk=reid_config.get("topk", 5),
            per_identity=reid_config.get("per_identity", {}),
        )

        # Perform stitching (no cluster assignments on initial run, will use global defaults)
        tracks_data, stitch_metadata = reid_tracker.stitch_tracks(tracks_data, embeddings_by_track)

        reid_stats = reid_tracker.get_stats()
        reid_stats["stage_time_ms_reid"] = int((time.time() - start_time) * 1000)

        logger.info(
            f"[{job_id}] Re-ID stitching complete: "
            f"{reid_stats['relink_accepted']}/{reid_stats['relink_attempts']} links created"
        )

        tracking_stats.update(reid_stats)

    if persistent_assigner and face_index is not None:
        tracking_stats["persistent_reid_labels"] = len(face_index.labels)
        tracking_stats["persistent_reid_events"] = len(persistent_events)

        if persistent_events:
            with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp_file:
                for event in persistent_events:
                    tmp_file.write(json.dumps(event, default=_json_default) + "\n")
                tmp_metrics_path = tmp_file.name
            try:
                persistent_metrics = compute_id_metrics(tmp_metrics_path)
            finally:
                Path(tmp_metrics_path).unlink(missing_ok=True)
        else:
            persistent_metrics = {
                "events": 0,
                "id_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "switch_rate_per_min": 0.0,
                "reattach_latency_mean": None,
                "reattach_latency_p95": None,
            }

        tracking_stats["persistent_reid_metrics"] = persistent_metrics

    # Sort by track_id
    tracks_data.sort(key=lambda t: t["track_id"])

    # Save tracks.json (convert numpy types to Python types)
    tracks_path = harvest_dir / "tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(
            {
                "episode_id": episode_id,
                "total_tracks": len(tracks_data),
                "tracks": tracks_data,
            },
            f,
            indent=2,
            default=int,  # Convert numpy int64 to int
        )

    logger.info(f"[{job_id}] Saved {len(tracks_data)} tracks to {tracks_path}")

    # Calculate track duration histogram
    if track_durations:
        duration_hist, duration_bins = np.histogram(track_durations, bins=10)
        tracking_stats["track_duration_hist"] = {
            "bins": duration_bins.tolist(),
            "counts": duration_hist.tolist(),
        }

    # Add tracker stats
    tracker_stats = tracker.get_stats()
    tracking_stats.update(tracker_stats)

    if persistent_assigner and face_index is not None and faces_path and centroids_path:
        try:
            face_index.save(str(faces_path), str(centroids_path))
        except Exception as exc:  # pragma: no cover - persistence guard
            logger.error("Failed to save persistent re-ID index to %s: %s", faces_path, exc)
        if id_map_path:
            try:
                with open(id_map_path, "w", encoding="utf-8") as f:
                    json.dump(persistent_assigner.id_map, f, indent=2)
            except Exception as exc:  # pragma: no cover - persistence guard
                logger.error("Failed to write id_map.json at %s: %s", id_map_path, exc)
        tracking_stats["persistent_reid_index_path"] = str(faces_path)
        if id_map_path:
            tracking_stats["persistent_reid_id_map_path"] = str(id_map_path)

    # Save tracking stats
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / "track_stats.json"
    with open(stats_path, "w") as f:
        json.dump(dict(tracking_stats), f, indent=2)

    logger.info(f"[{job_id}] Tracking stats saved to {stats_path}")

    # Save final checkpoint
    _save_checkpoint(
        harvest_dir,
        {
            "job_id": job_id,
            "episode_id": episode_id,
            "last_completed_stage": "track",
            "frames_processed": processed_frames,
            "tracks_built": len(tracks_data),
            "timestamp": datetime.utcnow().isoformat(),
        },
    )

    # Telemetry
    telemetry.log(
        TelemetryEvent.JOB_STAGE_COMPLETE,
        metadata={
            "job_id": job_id,
            "stage": "track",
            "tracks_built": len(tracks_data),
            "track_switches": track_switches,
        },
    )

    # Enqueue next stage (cluster) - only for job workflow, not manual
    if job_id != "manual":
        from api.jobs import job_manager

        cluster_queue = job_manager.cluster_queue

        cluster_queue.enqueue(
            "jobs.tasks.cluster.cluster_task",
            job_id=job_id,
            episode_id=episode_id,
            job_timeout="30m",
        )

    logger.info(f"[{job_id}] Enqueued cluster task for {episode_id}")

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "tracks_path": str(tracks_path),
        "stats_path": str(stats_path),
        "stats": dict(tracking_stats),
    }


def _save_checkpoint(harvest_dir: Path, checkpoint_data: dict) -> None:
    """Save checkpoint to disk."""
    checkpoint_dir = harvest_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = checkpoint_dir / "job.json"
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    logger.info(f"Checkpoint saved: {checkpoint_file}")
