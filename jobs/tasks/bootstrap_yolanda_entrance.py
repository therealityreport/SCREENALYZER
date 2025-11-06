#!/usr/bin/env python3
"""
Bootstrap YOLANDA identity from entrance window (17:22-20:04).

Uses geometry-based candidate selection to build a provisional template
from the entrance itself, then stitches to track 42 (post-entrance).

This bypasses the template mismatch issue where entrance faces don't
match the post-entrance reference.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder, EmbeddingResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class EntranceCandidate:
    """Entrance face candidate."""
    frame_id: int
    ts_ms: int
    det_idx: int
    bbox: list[int]
    confidence: float
    face_size: int
    x_center: float
    y_center: float
    embedding: np.ndarray | None
    has_embedding: bool
    used_kps: bool
    tries: int


@dataclass
class BootstrapStats:
    """Statistics from bootstrap operation."""
    window_start_ms: int
    window_end_ms: int
    frames_decoded: int
    faces_detected: int
    entrance_candidates: int
    entrance_emb_ok: int
    entrance_emb_none: int
    seed_frames: int
    seed_clusters: int
    tracklets_born: int
    stitched_into_track: int | None
    seconds_recovered: float


def load_config() -> dict:
    """Load pipeline configuration."""
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_frame(video_path: Path, ts_ms: int) -> np.ndarray:
    """Extract frame at timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int((ts_ms / 1000.0) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_bgr = cap.read()
    cap.release()

    if not ret:
        return None

    return frame_bgr


def compute_iou(bbox1: list[int], bbox2: list[int]) -> float:
    """Compute IoU between two bboxes."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def compute_kim_avg_embedding(episode_id: str, data_root: Path) -> np.ndarray:
    """Compute average KIM embedding for negative constraint."""
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"

    clusters_data = json.loads(clusters_path.read_text())
    tracks_data = json.loads(tracks_path.read_text())
    embeddings_df = pd.read_parquet(embeddings_path)

    # Find KIM cluster
    kim_cluster = next((c for c in clusters_data["clusters"] if c.get("name") == "KIM"), None)
    if not kim_cluster:
        raise ValueError("KIM cluster not found")

    # Get KIM frame IDs
    kim_frame_ids = []
    for track_id in kim_cluster.get("track_ids", []):
        track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
        if track:
            for frame_ref in track.get("frame_refs", []):
                kim_frame_ids.append(frame_ref["frame_id"])

    # Get KIM embeddings
    kim_embeddings = embeddings_df[embeddings_df["frame_id"].isin(kim_frame_ids)]

    if len(kim_embeddings) == 0:
        raise ValueError("No KIM embeddings found")

    # Compute average
    kim_vecs = np.stack(kim_embeddings["embedding"].values)
    kim_avg = kim_vecs.mean(axis=0)
    kim_avg = kim_avg / np.linalg.norm(kim_avg)

    logger.info(f"Computed KIM avg embedding from {len(kim_embeddings)} vectors")

    return kim_avg


def compute_track42_template(episode_id: str, data_root: Path) -> np.ndarray:
    """Compute track 42 template (YOLANDA post-entrance)."""
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"

    tracks_data = json.loads(tracks_path.read_text())
    embeddings_df = pd.read_parquet(embeddings_path)

    # Find track 42
    track_42 = next((t for t in tracks_data["tracks"] if t["track_id"] == 42), None)
    if not track_42:
        raise ValueError("Track 42 not found")

    # Get track 42 frame IDs
    track42_frame_ids = [ref["frame_id"] for ref in track_42.get("frame_refs", [])]

    # Get track 42 embeddings
    track42_embeddings = embeddings_df[embeddings_df["frame_id"].isin(track42_frame_ids)]

    if len(track42_embeddings) == 0:
        raise ValueError("No track 42 embeddings found")

    # Compute average
    track42_vecs = np.stack(track42_embeddings["embedding"].values)
    track42_avg = track42_vecs.mean(axis=0)
    track42_avg = track42_avg / np.linalg.norm(track42_avg)

    logger.info(f"Computed track 42 template from {len(track42_embeddings)} vectors")

    return track42_avg


def collect_entrance_candidates(
    episode_id: str,
    video_path: Path,
    manifest_df: pd.DataFrame,
    detector: RetinaFaceDetector,
    embedder: ArcFaceEmbedder,
    window_start_ms: int,
    window_end_ms: int,
    kim_avg: np.ndarray
) -> tuple[list[EntranceCandidate], dict]:
    """Collect entrance candidates using geometry-based filtering."""

    sampled_frames = manifest_df[
        (manifest_df['ts_ms'] >= window_start_ms) &
        (manifest_df['ts_ms'] <= window_end_ms) &
        (manifest_df['sampled'] == True)
    ]

    logger.info(f"Collecting candidates from {len(sampled_frames)} frames")

    candidates = []
    stats = {
        "frames_decoded": 0,
        "faces_detected": 0,
        "entrance_candidates": 0,
        "entrance_emb_ok": 0,
        "entrance_emb_none": 0
    }

    for _, row in sampled_frames.iterrows():
        frame_id = row['frame_id']
        ts_ms = row['ts_ms']

        stats["frames_decoded"] += 1

        # Extract frame
        frame_bgr = extract_frame(video_path, ts_ms)
        if frame_bgr is None:
            logger.warning(f"Failed to extract frame {frame_id} at {ts_ms}ms")
            continue

        frame_height, frame_width = frame_bgr.shape[:2]

        # Detect faces
        detections = detector.detect(frame_bgr)
        stats["faces_detected"] += len(detections)

        if len(detections) == 0:
            continue

        # Filter for entrance candidates (right-side, size 110-200px)
        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            face_size = det["face_size"]
            confidence = det["confidence"]

            # Compute center
            x_center = (x1 + x2) / 2.0
            y_center = (y1 + y2) / 2.0

            # Filter 1: Right-side face (x_center > 0.60 * frame_width)
            if x_center <= 0.60 * frame_width:
                continue

            # Filter 2: Size band (110-200px)
            if face_size < 110 or face_size > 200:
                continue

            # Filter 3: Not-Kim constraint (check IoU with typical left-side Kim bbox)
            # Kim bbox reference: ~[543, 191, 733, 484] (left side)
            kim_reference_bbox = [543, 191, 733, 484]
            iou = compute_iou(bbox, kim_reference_bbox)
            if iou > 0.2:
                logger.debug(f"Rejected candidate at {ts_ms}ms (IoU={iou:.3f} with Kim bbox)")
                continue

            # Passed geometry filters - this is an entrance candidate
            stats["entrance_candidates"] += 1

            # Generate embedding
            landmarks = det.get("landmarks")
            kps = None
            if landmarks is not None:
                landmarks_array = np.array(landmarks)
                if len(landmarks_array) >= 5:
                    kps = landmarks_array[:5]

            result = embedder.embed_from_detection(frame_bgr, bbox, kps=kps)

            if result.success:
                stats["entrance_emb_ok"] += 1
            else:
                stats["entrance_emb_none"] += 1

            # Additional Kim similarity check (if embedding succeeded)
            sim_to_kim = None
            if result.success and result.embedding is not None:
                emb_norm = result.embedding / np.linalg.norm(result.embedding)
                sim_to_kim = float(np.dot(emb_norm, kim_avg))

                # Reject if too similar to Kim
                if sim_to_kim > 0.75:
                    logger.debug(f"Rejected candidate at {ts_ms}ms (sim_to_kim={sim_to_kim:.3f})")
                    continue

            candidate = EntranceCandidate(
                frame_id=frame_id,
                ts_ms=ts_ms,
                det_idx=det_idx,
                bbox=bbox,
                confidence=confidence,
                face_size=face_size,
                x_center=x_center,
                y_center=y_center,
                embedding=result.embedding,
                has_embedding=result.success,
                used_kps=result.used_kps,
                tries=result.tries
            )

            candidates.append(candidate)

            sim_to_kim_str = f"{sim_to_kim:.3f}" if sim_to_kim is not None else "N/A"
            logger.info(f"Candidate {len(candidates)}: frame={frame_id}, ts={ts_ms}ms, "
                       f"size={face_size}px, x_center={x_center:.0f}, "
                       f"emb={'OK' if result.success else 'FAIL'}, "
                       f"sim_to_kim={sim_to_kim_str}")

    logger.info(f"Collected {len(candidates)} entrance candidates")
    logger.info(f"  Frames decoded: {stats['frames_decoded']}")
    logger.info(f"  Faces detected: {stats['faces_detected']}")
    logger.info(f"  Candidates after geometry filtering: {stats['entrance_candidates']}")
    logger.info(f"  Embeddings OK: {stats['entrance_emb_ok']}")
    logger.info(f"  Embeddings failed: {stats['entrance_emb_none']}")

    return candidates, stats


def build_entrance_seed(candidates: list[EntranceCandidate]) -> tuple[np.ndarray | None, int, int]:
    """Build provisional YOLANDA template from entrance candidates."""

    # Filter candidates with embeddings
    valid_candidates = [c for c in candidates if c.has_embedding and c.embedding is not None]

    if len(valid_candidates) < 12:
        logger.warning(f"Insufficient valid candidates for seed building: {len(valid_candidates)} < 12")
        return None, 0, 0

    logger.info(f"Building seed from {len(valid_candidates)} valid candidates")

    # Stack embeddings
    embeddings = np.stack([c.embedding for c in valid_candidates])

    # Normalize
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise cosine distances
    distances = 1.0 - np.dot(embeddings_norm, embeddings_norm.T)

    # Clip to valid range [0, 2] for cosine distance
    distances = np.clip(distances, 0.0, 2.0)

    # Cluster with DBSCAN (eps=0.35 cosine distance = sim >= 0.65)
    clustering = DBSCAN(eps=0.35, min_samples=4, metric='precomputed')
    labels = clustering.fit_predict(distances)

    # Find largest cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise label

    if len(unique_labels) == 0:
        logger.warning("No clusters found in entrance candidates")
        return None, 0, 0

    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
    largest_cluster_size = cluster_sizes[largest_cluster_label]

    logger.info(f"Found {len(unique_labels)} clusters, largest has {largest_cluster_size} members")

    # Get largest cluster embeddings
    cluster_mask = labels == largest_cluster_label
    cluster_embeddings = embeddings[cluster_mask]
    cluster_candidates = [c for i, c in enumerate(valid_candidates) if cluster_mask[i]]

    # Check temporal consistency (require ≥2 contiguous segments)
    cluster_timestamps = sorted([c.ts_ms for c in cluster_candidates])
    segments = []
    current_segment = [cluster_timestamps[0]]

    for ts in cluster_timestamps[1:]:
        if ts - current_segment[-1] <= 200:  # Within 200ms = contiguous at 10fps
            current_segment.append(ts)
        else:
            segments.append(current_segment)
            current_segment = [ts]

    segments.append(current_segment)

    # Accept either: ≥2 segments, OR 1 long segment with ≥12 frames (sustained presence)
    total_frames = sum(len(seg) for seg in segments)
    is_consistent = (len(segments) >= 2) or (len(segments) == 1 and total_frames >= 12)

    if not is_consistent:
        logger.warning(f"Insufficient temporal consistency: {len(segments)} segment(s), {total_frames} total frames")
        return None, largest_cluster_size, len(unique_labels)

    logger.info(f"Temporal consistency OK: {len(segments)} segments, {total_frames} frames")

    # Compute medoid (most central embedding)
    cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
    pairwise_sims = np.dot(cluster_embeddings_norm, cluster_embeddings_norm.T)
    centrality = pairwise_sims.sum(axis=1)
    medoid_idx = np.argmax(centrality)

    seed = cluster_embeddings_norm[medoid_idx]

    logger.info(f"Built entrance seed from {largest_cluster_size} frames across {len(segments)} segments")

    return seed, largest_cluster_size, len(unique_labels)


def verify_and_accept(
    candidates: list[EntranceCandidate],
    seed: np.ndarray,
    track42_template: np.ndarray,
    kim_avg: np.ndarray
) -> list[EntranceCandidate]:
    """Verify and accept entrance candidates."""

    accepted = []

    for candidate in candidates:
        if not candidate.has_embedding or candidate.embedding is None:
            continue

        emb_norm = candidate.embedding / np.linalg.norm(candidate.embedding)

        # Compute similarities
        sim_to_seed = float(np.dot(emb_norm, seed))
        sim_to_track42 = float(np.dot(emb_norm, track42_template))
        sim_to_kim = float(np.dot(emb_norm, kim_avg))

        delta_sim = sim_to_seed - sim_to_kim

        # Acceptance criteria
        # 1. sim_to_seed >= 0.72
        # 2. (sim_to_seed - sim_to_kim) >= 0.06
        accept = sim_to_seed >= 0.72 and delta_sim >= 0.06

        logger.info(f"Frame {candidate.frame_id} ({candidate.ts_ms}ms): "
                   f"sim_seed={sim_to_seed:.3f}, sim_t42={sim_to_track42:.3f}, "
                   f"sim_kim={sim_to_kim:.3f}, delta={delta_sim:.3f}, "
                   f"accept={accept}")

        if accept:
            accepted.append(candidate)

    logger.info(f"Accepted {len(accepted)}/{len(candidates)} entrance candidates")

    return accepted


def check_bridge_condition(seed: np.ndarray, track42_template: np.ndarray) -> bool:
    """Check if entrance seed can bridge to track 42."""
    sim = float(np.dot(seed, track42_template))
    passes = sim >= 0.70

    logger.info(f"Bridge condition: sim(entrance_seed, track42) = {sim:.3f}, "
               f"threshold=0.70, passes={passes}")

    return passes


def main():
    """Main bootstrap function."""
    episode_id = "RHOBH-TEST-10-28"

    # Window: 00:17:22 → 00:20:04
    window_start_ms = 17220
    window_end_ms = 20040

    logger.info(f"Bootstrapping YOLANDA entrance for {episode_id}")
    logger.info(f"Window: {window_start_ms}ms - {window_end_ms}ms ({(window_end_ms - window_start_ms)/1000:.2f}s)")

    # Load config
    config = load_config()
    data_root = Path(config["paths"]["data_root"])

    # Paths
    video_path = data_root / "videos" / f"{episode_id}.mp4"
    manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"
    reports_dir = data_root / "harvest" / episode_id / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    manifest_df = pd.read_parquet(manifest_path)

    # Initialize detector and embedder
    detector = RetinaFaceDetector(
        min_face_px=config["detection"].get("min_face_px", 80),
        min_confidence=config["detection"].get("min_confidence", 0.7),
        provider_order=config["detection"].get("provider_order", ["coreml", "cpu"])
    )

    embedding_config = config.get("embedding", {})
    embedder = ArcFaceEmbedder(
        provider_order=config["detection"].get("provider_order", ["coreml", "cpu"]),
        skip_redetect=embedding_config.get("skip_redetect", True),
        align_priority=embedding_config.get("align_priority", "kps_then_bbox"),
        margin_scale=embedding_config.get("margin_scale", 1.25),
        min_chip_px=embedding_config.get("min_chip_px", 112),
        fallback_scales=embedding_config.get("fallback_scales", [1.0, 1.2, 1.4])
    )

    logger.info("Initialized detector and embedder with fixed settings")

    # Compute reference templates
    kim_avg = compute_kim_avg_embedding(episode_id, data_root)
    track42_template = compute_track42_template(episode_id, data_root)

    # Step 1: Collect entrance candidates
    logger.info("\n=== Step 1: Collect entrance candidates ===")
    candidates, collection_stats = collect_entrance_candidates(
        episode_id,
        video_path,
        manifest_df,
        detector,
        embedder,
        window_start_ms,
        window_end_ms,
        kim_avg
    )

    if len(candidates) == 0:
        logger.error("No entrance candidates found - cannot proceed")
        return

    # Step 2: Build entrance seed
    logger.info("\n=== Step 2: Build entrance seed ===")
    seed, seed_frames, seed_clusters = build_entrance_seed(candidates)

    if seed is None:
        logger.error("Failed to build entrance seed - cannot proceed")
        return

    # Step 3: Check bridge condition
    logger.info("\n=== Step 3: Check bridge condition ===")
    bridge_ok = check_bridge_condition(seed, track42_template)

    if not bridge_ok:
        logger.warning("Bridge condition failed - entrance may not connect to track 42")

    # Step 4: Verify and accept
    logger.info("\n=== Step 4: Verify and accept candidates ===")
    accepted = verify_and_accept(candidates, seed, track42_template, kim_avg)

    # Calculate seconds recovered
    seconds_recovered = 0.0
    if len(accepted) > 0:
        accepted_times = sorted([c.ts_ms for c in accepted])
        seconds_recovered = (accepted_times[-1] - accepted_times[0]) / 1000.0

    # Final stats
    stats = BootstrapStats(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        frames_decoded=collection_stats["frames_decoded"],
        faces_detected=collection_stats["faces_detected"],
        entrance_candidates=collection_stats["entrance_candidates"],
        entrance_emb_ok=collection_stats["entrance_emb_ok"],
        entrance_emb_none=collection_stats["entrance_emb_none"],
        seed_frames=seed_frames,
        seed_clusters=seed_clusters,
        tracklets_born=1 if len(accepted) >= 4 else 0,
        stitched_into_track=42 if bridge_ok else None,
        seconds_recovered=seconds_recovered
    )

    logger.info("\n=== Bootstrap complete! ===")
    logger.info(f"  Frames decoded: {stats.frames_decoded}")
    logger.info(f"  Faces detected: {stats.faces_detected}")
    logger.info(f"  Entrance candidates: {stats.entrance_candidates}")
    logger.info(f"  Embeddings OK: {stats.entrance_emb_ok}")
    logger.info(f"  Embeddings failed: {stats.entrance_emb_none}")
    logger.info(f"  Seed frames: {stats.seed_frames}")
    logger.info(f"  Seed clusters: {stats.seed_clusters}")
    logger.info(f"  Accepted candidates: {len(accepted)}")
    logger.info(f"  Tracklets born: {stats.tracklets_born}")
    logger.info(f"  Stitched to track: {stats.stitched_into_track}")
    logger.info(f"  Seconds recovered: {stats.seconds_recovered:.2f}s")

    # Save stats
    stats_dict = {
        "window": {
            "start_ms": stats.window_start_ms,
            "end_ms": stats.window_end_ms,
            "duration_s": (stats.window_end_ms - stats.window_start_ms) / 1000.0
        },
        "yolanda_entrance": {
            "frames_decoded": stats.frames_decoded,
            "faces_detected": stats.faces_detected,
            "entrance_emb_ok": stats.entrance_emb_ok,
            "entrance_emb_none": stats.entrance_emb_none,
            "seed_frames": stats.seed_frames,
            "seed_clusters": stats.seed_clusters,
            "tracklets_born": stats.tracklets_born,
            "stitched_into_track": stats.stitched_into_track,
            "seconds_recovered": stats.seconds_recovered
        }
    }

    stats_path = reports_dir / "emb_recompute_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    logger.info(f"Saved stats to {stats_path}")

    # Save per-frame details
    if len(accepted) > 0:
        per_frame_details = []
        for candidate in accepted:
            emb_norm = candidate.embedding / np.linalg.norm(candidate.embedding)
            per_frame_details.append({
                "frame_id": candidate.frame_id,
                "ts_ms": candidate.ts_ms,
                "face_px": candidate.face_size,
                "sim_to_seed": float(np.dot(emb_norm, seed)),
                "sim_to_track42": float(np.dot(emb_norm, track42_template)),
                "sim_to_kim": float(np.dot(emb_norm, kim_avg)),
                "delta_sim": float(np.dot(emb_norm, seed) - np.dot(emb_norm, kim_avg))
            })

        details_path = reports_dir / "yolanda_entrance_scan.json"
        with open(details_path, "w") as f:
            json.dump(per_frame_details, f, indent=2)

        logger.info(f"Saved per-frame details to {details_path}")

    return stats


if __name__ == "__main__":
    main()
