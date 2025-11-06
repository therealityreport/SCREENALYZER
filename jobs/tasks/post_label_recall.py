"""
Post-label identity-guided recall detection.

After clusters are labeled, this task:
1. Builds per-person embedding templates from labeled clusters
2. Identifies gap windows where each person is missing
3. Runs high-recall detection only in those windows
4. Verifies identity using embedding similarity to person templates
5. Creates new tracks from verified detections and integrates via re-ID

This targeted approach maintains precision while boosting recall for missed faces.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from screentime.diagnostics.telemetry import telemetry, TelemetryEvent

logger = logging.getLogger(__name__)


def post_label_recall_task(
    job_id: str,
    episode_id: str,
    cluster_assignments: dict[int, str],
) -> dict:
    """
    Run identity-guided recall detection on labeled episode.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        cluster_assignments: Map of cluster_id -> person_name

    Returns:
        Dict with recall results and statistics
    """
    logger.info(f"[{job_id}] Starting post-label recall for {episode_id}")

    start_time = time.time()

    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    recall_config = config.get("post_label_recall", {})
    if not recall_config.get("enabled", False):
        logger.info(f"[{job_id}] Post-label recall is disabled in config")
        return {
            "job_id": job_id,
            "episode_id": episode_id,
            "enabled": False,
            "stats": {},
        }

    # Setup paths
    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id
    outputs_dir = data_root / "outputs" / episode_id

    # Load existing data
    clusters_path = harvest_dir / "clusters.json"
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"
    timeline_path = outputs_dir / "timeline.csv"

    if not all(p.exists() for p in [clusters_path, tracks_path, embeddings_path, timeline_path]):
        raise ValueError(f"Missing required data files for {episode_id}")

    with open(clusters_path) as f:
        clusters_data = json.load(f)

    with open(tracks_path) as f:
        tracks_data = json.load(f)

    embeddings_df = pd.read_parquet(embeddings_path)
    timeline_df = pd.read_csv(timeline_path)

    logger.info(
        f"[{job_id}] Loaded {len(clusters_data['clusters'])} clusters, "
        f"{len(tracks_data['tracks'])} tracks, {len(embeddings_df)} embeddings"
    )

    # Filter cluster assignments by target_identities (skip frozen identities)
    target_identities = recall_config.get("target_identities", [])
    if target_identities:
        filtered_assignments = {
            cid: name
            for cid, name in cluster_assignments.items()
            if name in target_identities
        }
        logger.info(
            f"[{job_id}] Targeting recall for {len(filtered_assignments)} identities: "
            f"{', '.join(filtered_assignments.values())}"
        )
    else:
        filtered_assignments = cluster_assignments

    # Step 1: Build per-person embedding templates
    logger.info(f"[{job_id}] Building per-person embedding templates...")
    person_templates = _build_person_templates(
        clusters_data, tracks_data, embeddings_df, filtered_assignments
    )

    logger.info(f"[{job_id}] Built templates for {len(person_templates)} people")

    # Step 2: Identify gap windows for each person
    logger.info(f"[{job_id}] Identifying gap windows per person...")
    gap_windows = _identify_gap_windows(
        timeline_df,
        filtered_assignments,
        max_gap_ms=recall_config.get("max_gap_ms", 3200),
        window_pad_ms=recall_config.get("window_pad_ms", 300),
    )

    total_windows = sum(len(windows) for windows in gap_windows.values())
    logger.info(f"[{job_id}] Identified {total_windows} gap windows across {len(gap_windows)} people")

    # Step 3: Run high-recall detection in gap windows with identity verification
    logger.info(f"[{job_id}] Running identity-guided recall detection...")
    recall_detections = _run_identity_guided_recall(
        job_id,
        episode_id,
        gap_windows,
        person_templates,
        recall_config,
        harvest_dir,
    )

    logger.info(
        f"[{job_id}] Recall detection complete: "
        f"{sum(len(dets) for dets in recall_detections.values())} verified detections"
    )

    # Step 4: Create tracklets from recall detections
    logger.info(f"[{job_id}] Creating tracklets from recall detections...")
    new_tracks = _create_recall_tracklets(
        recall_detections,
        recall_config.get("track", {}).get("birth_min_frames", 3),
    )

    logger.info(f"[{job_id}] Created {len(new_tracks)} new recall tracklets")

    # Step 5: Integrate new tracks via re-ID and re-run analytics
    logger.info(f"[{job_id}] Integrating recall tracks and regenerating analytics...")
    _integrate_recall_tracks(
        job_id,
        episode_id,
        new_tracks,
        tracks_data,
        harvest_dir,
        outputs_dir,
        cluster_assignments,
        config,
    )

    # Calculate statistics
    stage_time_ms = int((time.time() - start_time) * 1000)

    recall_stats = {
        "episode_id": episode_id,
        "people_processed": len(gap_windows),
        "total_windows_scanned": total_windows,
        "recall_detections_verified": sum(len(dets) for dets in recall_detections.values()),
        "new_tracklets_created": len(new_tracks),
        "stage_time_ms": stage_time_ms,
    }

    # Save recall stats
    reports_dir = harvest_dir / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = reports_dir / "recall_stats.json"
    with open(stats_path, "w") as f:
        json.dump(recall_stats, f, indent=2)

    logger.info(f"[{job_id}] Recall stats saved to {stats_path}")

    # Telemetry
    telemetry.log(
        TelemetryEvent.JOB_STAGE_COMPLETE,
        metadata={
            "job_id": job_id,
            "stage": "post_label_recall",
            "recall_detections": recall_stats["recall_detections_verified"],
            "new_tracklets": recall_stats["new_tracklets_created"],
            "stage_time_ms": stage_time_ms,
        },
    )

    return {
        "job_id": job_id,
        "episode_id": episode_id,
        "enabled": True,
        "stats": recall_stats,
    }


def _build_person_templates(
    clusters_data: dict,
    tracks_data: dict,
    embeddings_df: pd.DataFrame,
    cluster_assignments: dict[int, str],
) -> dict[str, np.ndarray]:
    """
    Build per-person embedding templates from top-K high-quality crops.

    Args:
        clusters_data: Clusters data
        tracks_data: Tracks data
        embeddings_df: Embeddings DataFrame
        cluster_assignments: Map of cluster_id -> person_name

    Returns:
        Dict of person_name -> template_embedding (512-d vector)
    """
    person_templates = {}

    # Build track_id -> track map
    tracks_by_id = {t["track_id"]: t for t in tracks_data.get("tracks", [])}

    # Build track_id -> embeddings map via frame_refs
    track_embeddings = defaultdict(list)
    for track_id, track in tracks_by_id.items():
        for frame_ref in track.get("frame_refs", []):
            frame_id = frame_ref["frame_id"]
            det_idx = frame_ref["det_idx"]
            # Find matching embedding
            match = embeddings_df[
                (embeddings_df["frame_id"] == frame_id) &
                (embeddings_df["det_idx"] == det_idx)
            ]
            if len(match) > 0:
                row = match.iloc[0]
                embedding = np.array(row["embedding"])
                confidence = row["confidence"]
                face_px = (row["bbox_x2"] - row["bbox_x1"]) * (row["bbox_y2"] - row["bbox_y1"]) ** 0.5
                track_embeddings[track_id].append((embedding, confidence, face_px))

    # For each person, collect top-K embeddings from their clusters
    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        person_name = cluster_assignments.get(cluster_id)

        if not person_name:
            continue

        # Collect all embeddings from tracks in this cluster
        person_embeds = []
        for track_id in cluster["track_ids"]:
            person_embeds.extend(track_embeddings.get(track_id, []))

        if not person_embeds:
            continue

        # Sort by quality (confidence * face_size) and take top 10
        person_embeds.sort(key=lambda x: x[1] * x[2], reverse=True)
        top_k = person_embeds[:10]

        # Compute median embedding as template
        template = np.median([e[0] for e in top_k], axis=0)
        template = template / np.linalg.norm(template)  # Normalize

        person_templates[person_name] = template

        logger.debug(
            f"Built template for {person_name} from {len(top_k)} high-quality crops "
            f"(cluster {cluster_id}, {len(person_embeds)} total embeddings)"
        )

    return person_templates


def _identify_gap_windows(
    timeline_df: pd.DataFrame,
    cluster_assignments: dict[int, str],
    max_gap_ms: int = 3200,
    window_pad_ms: int = 300,
) -> dict[str, list[tuple[int, int]]]:
    """
    Identify inter-interval gaps for each person.

    Args:
        timeline_df: Timeline DataFrame with intervals
        cluster_assignments: Map of cluster_id -> person_name
        max_gap_ms: Maximum gap size to consider
        window_pad_ms: Padding to add around gaps

    Returns:
        Dict of person_name -> list of (start_ms, end_ms) gap windows
    """
    gap_windows = defaultdict(list)

    # Get unique people
    people = set(cluster_assignments.values())

    for person in people:
        # Get all intervals for this person, sorted by start time
        person_intervals = timeline_df[timeline_df["person_name"] == person].sort_values("start_ms")

        if len(person_intervals) < 2:
            continue

        # Find gaps between consecutive intervals
        for i in range(len(person_intervals) - 1):
            curr_end = person_intervals.iloc[i]["end_ms"]
            next_start = person_intervals.iloc[i + 1]["start_ms"]

            gap_ms = next_start - curr_end

            if 0 < gap_ms <= max_gap_ms:
                # Add window with padding
                window_start = max(0, curr_end - window_pad_ms)
                window_end = next_start + window_pad_ms

                gap_windows[person].append((window_start, window_end))

    return dict(gap_windows)


def _run_identity_guided_recall(
    job_id: str,
    episode_id: str,
    gap_windows: dict[str, list[tuple[int, int]]],
    person_templates: dict[str, np.ndarray],
    recall_config: dict,
    harvest_dir: Path,
) -> dict[str, list[dict]]:
    """
    Run high-recall detection in gap windows with identity verification.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        gap_windows: Gap windows per person
        person_templates: Person embedding templates
        recall_config: Recall configuration
        harvest_dir: Harvest directory

    Returns:
        Dict of person_name -> list of verified detections
    """
    import cv2
    from scipy.spatial.distance import cosine
    from screentime.detectors.face_retina import RetinaFaceDetector
    from screentime.recognition.embed_arcface import ArcFaceEmbedder
    from api.jobs import job_manager

    logger.info(f"[{job_id}] Running identity-guided recall detection")

    # Get video path
    job_data = job_manager._get_job_metadata(job_id)
    if not job_data:
        raise ValueError(f"Job metadata not found for {job_id}")
    video_path = job_data["video_path"]

    # Load manifest to get frame -> timestamp mapping
    manifest_path = harvest_dir / "manifest.parquet"
    manifest_df = pd.read_parquet(manifest_path)
    fps = 30  # Assume 30fps for sampling

    # Initialize detector and embedder with relaxed thresholds
    detection_config = recall_config.get("detection", {})
    min_confidence = detection_config.get("min_confidence", 0.60)
    min_face_px = detection_config.get("min_face_px", 50)
    provider_order = detection_config.get("provider_order", ["cuda", "cpu"])

    detector = RetinaFaceDetector(
        min_face_px=min_face_px,
        min_confidence=min_confidence,
        provider_order=provider_order,
    )
    embedder = ArcFaceEmbedder(provider_order=provider_order)

    logger.info(
        f"[{job_id}] Recall detection config: "
        f"min_confidence={min_confidence}, min_face_px={min_face_px}"
    )

    # Re-ID thresholds
    reid_config = recall_config.get("reid", {})
    min_sim = reid_config.get("min_sim", 0.82)
    min_margin = reid_config.get("min_margin", 0.08)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        recall_detections = defaultdict(list)

        # Process each person's gap windows
        for person_name, windows in gap_windows.items():
            if person_name not in person_templates:
                logger.warning(f"[{job_id}] No template for {person_name}, skipping recall")
                continue

            person_template = person_templates[person_name]

            logger.info(
                f"[{job_id}] Processing {len(windows)} gap windows for {person_name}"
            )

            for window_start_ms, window_end_ms in windows:
                # Sample frames at 15fps (every 2 frames at 30fps) for better coverage
                sample_interval_frames = 2
                start_frame = int(window_start_ms * fps / 1000)
                end_frame = int(window_end_ms * fps / 1000)

                for frame_id in range(start_frame, end_frame + 1, sample_interval_frames):
                    # Find timestamp for this frame
                    frame_row = manifest_df[manifest_df["frame_id"] == frame_id]
                    if len(frame_row) == 0:
                        continue
                    ts_ms = int(frame_row.iloc[0]["ts_ms"])

                    # Read frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    # Detect faces
                    detections = detector.detect(frame)

                    # Process each detection
                    for det_idx, det in enumerate(detections):
                        bbox = det["bbox"]
                        confidence = det["confidence"]

                        # Generate embedding
                        embedding = embedder.embed(frame, bbox)
                        if embedding is None:
                            continue

                        # Normalize embedding
                        embedding = embedding / np.linalg.norm(embedding)

                        # Compute similarity to target person
                        target_sim = 1.0 - cosine(embedding, person_template)

                        # Verify identity threshold
                        if target_sim < min_sim:
                            continue

                        # Compute similarity to all other people (margin check)
                        max_other_sim = 0.0
                        for other_person, other_template in person_templates.items():
                            if other_person == person_name:
                                continue
                            other_sim = 1.0 - cosine(embedding, other_template)
                            max_other_sim = max(max_other_sim, other_sim)

                        # Verify margin
                        if target_sim - max_other_sim < min_margin:
                            continue

                        # Verified detection!
                        recall_detections[person_name].append(
                            {
                                "frame_id": frame_id,
                                "ts_ms": ts_ms,
                                "det_idx": det_idx,
                                "bbox": bbox,
                                "confidence": confidence,
                                "embedding": embedding.tolist(),
                                "similarity": float(target_sim),
                                "margin": float(target_sim - max_other_sim),
                            }
                        )

            logger.info(
                f"[{job_id}] {person_name}: {len(recall_detections[person_name])} verified detections"
            )

        return dict(recall_detections)

    finally:
        cap.release()


def _create_recall_tracklets(
    recall_detections: dict[str, list[dict]],
    birth_min_frames: int = 3,
) -> list[dict]:
    """
    Create tracklets from recall detections (require N consecutive frames).

    Args:
        recall_detections: Verified detections per person
        birth_min_frames: Minimum consecutive frames to create track

    Returns:
        List of new track dicts
    """
    new_tracks = []
    track_id_counter = 10000  # Start at high number to avoid conflicts

    for person_name, detections in recall_detections.items():
        if not detections:
            continue

        # Sort by frame_id
        detections = sorted(detections, key=lambda d: d["frame_id"])

        # Group into consecutive sequences
        current_sequence = []

        for det in detections:
            if not current_sequence:
                current_sequence.append(det)
            else:
                # Check if consecutive (within 4 frames = ~133ms at 30fps)
                last_frame = current_sequence[-1]["frame_id"]
                if det["frame_id"] - last_frame <= 4:
                    current_sequence.append(det)
                else:
                    # End of sequence - create track if meets minimum
                    if len(current_sequence) >= birth_min_frames:
                        track = _create_track_from_detections(
                            current_sequence, track_id_counter, person_name
                        )
                        new_tracks.append(track)
                        track_id_counter += 1

                    # Start new sequence
                    current_sequence = [det]

        # Handle final sequence
        if len(current_sequence) >= birth_min_frames:
            track = _create_track_from_detections(
                current_sequence, track_id_counter, person_name
            )
            new_tracks.append(track)
            track_id_counter += 1

        logger.info(
            f"Created {len([t for t in new_tracks if t['person_name'] == person_name])} "
            f"recall tracklets for {person_name} from {len(detections)} detections"
        )

    return new_tracks


def _create_track_from_detections(
    detections: list[dict], track_id: int, person_name: str
) -> dict:
    """Create a track dict from a sequence of detections."""
    frame_refs = []
    for det in detections:
        frame_refs.append(
            {
                "frame_id": det["frame_id"],
                "det_idx": det["det_idx"],
                "bbox": det["bbox"],
                "confidence": det["confidence"],
            }
        )

    return {
        "track_id": track_id,
        "start_ms": detections[0]["ts_ms"],
        "end_ms": detections[-1]["ts_ms"],
        "frame_refs": frame_refs,
        "person_name": person_name,  # Pre-assigned from recall
        "source": "recall",  # Mark as recall-generated
    }


def _integrate_recall_tracks(
    job_id: str,
    episode_id: str,
    new_tracks: list[dict],
    existing_tracks_data: dict,
    harvest_dir: Path,
    outputs_dir: Path,
    cluster_assignments: dict[int, str],
    config: dict,
) -> None:
    """
    Integrate new recall tracks and regenerate analytics.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        new_tracks: New recall tracklets
        existing_tracks_data: Existing tracks data
        harvest_dir: Harvest directory
        outputs_dir: Outputs directory
        cluster_assignments: Cluster assignments
        config: Pipeline config
    """
    if not new_tracks:
        logger.info(f"[{job_id}] No new recall tracks to integrate")
        return

    logger.info(f"[{job_id}] Integrating {len(new_tracks)} recall tracks")

    # 1. Group recall tracks by person before removing person_name
    recall_tracks_by_person = defaultdict(list)
    for track in new_tracks:
        person_name = track.get("person_name")
        if person_name:
            recall_tracks_by_person[person_name].append(track["track_id"])

    # 2. Load clusters and add recall tracks to corresponding clusters
    clusters_path = harvest_dir / "clusters.json"
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    # Build person_name -> cluster map
    cluster_by_person = {}
    for cluster in clusters_data["clusters"]:
        cluster_id = cluster["cluster_id"]
        person_name = cluster_assignments.get(cluster_id)
        if person_name:
            cluster_by_person[person_name] = cluster

    # Add recall track_ids to their clusters
    total_added = 0
    for person_name, track_ids in recall_tracks_by_person.items():
        if person_name in cluster_by_person:
            cluster = cluster_by_person[person_name]
            cluster["track_ids"].extend(track_ids)
            total_added += len(track_ids)
            logger.info(
                f"[{job_id}] Added {len(track_ids)} recall tracks to {person_name}'s cluster"
            )

    # Save updated clusters
    with open(clusters_path, "w") as f:
        json.dump(clusters_data, f, indent=2)

    logger.info(f"[{job_id}] Updated clusters with {total_added} recall track assignments")

    # 3. Clean person_name/source from tracks and merge into tracks.json
    for track in new_tracks:
        track.pop("person_name", None)
        track.pop("source", None)

    all_tracks = existing_tracks_data["tracks"] + new_tracks
    updated_tracks_data = {
        "episode_id": episode_id,
        "total_tracks": len(all_tracks),
        "tracks": all_tracks,
    }

    tracks_path = harvest_dir / "tracks.json"
    with open(tracks_path, "w") as f:
        json.dump(updated_tracks_data, f, indent=2)

    logger.info(f"[{job_id}] Saved {len(all_tracks)} total tracks (including {len(new_tracks)} recall)")

    # 4. Re-run analytics with updated tracks and clusters
    from jobs.tasks.analytics import analytics_task

    logger.info(f"[{job_id}] Re-running analytics with recall tracks integrated")

    analytics_result = analytics_task(job_id, episode_id, cluster_assignments)

    logger.info(
        f"[{job_id}] Analytics complete: {analytics_result['stats'].get('intervals_created', 0)} intervals"
    )
