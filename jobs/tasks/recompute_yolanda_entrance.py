#!/usr/bin/env python3
"""
Recompute embeddings for YOLANDA entrance window (17:22-20:04).

This script fixes the embedding dropout bug by using the updated embedder
that bypasses re-detection and aligns directly from detector outputs.

Timeline: 00:17:22 → 00:20:04 (17,220ms - 20,040ms)
Expected: ~2.8s duration, ~28 frames at 10fps baseline
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

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder, EmbeddingResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class RecomputeStats:
    """Statistics from recompute operation."""
    window_start_ms: int
    window_end_ms: int
    frames_processed: int
    faces_detected: int
    emb_ok: int
    emb_none: int
    used_kps: int
    fallback_used: int
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


def compute_yolanda_avg_embedding(episode_id: str, data_root: Path) -> np.ndarray:
    """Compute average YOLANDA embedding from existing tracks."""
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"

    clusters_data = json.loads(clusters_path.read_text())
    tracks_data = json.loads(tracks_path.read_text())
    embeddings_df = pd.read_parquet(embeddings_path)

    # Find YOLANDA cluster
    yolanda_cluster = next((c for c in clusters_data["clusters"] if c.get("name") == "YOLANDA"), None)
    if not yolanda_cluster:
        raise ValueError("YOLANDA cluster not found")

    # Get YOLANDA frame IDs
    yolanda_frame_ids = []
    for track_id in yolanda_cluster.get("track_ids", []):
        track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
        if track:
            for frame_ref in track.get("frame_refs", []):
                yolanda_frame_ids.append(frame_ref["frame_id"])

    # Get YOLANDA embeddings
    yolanda_embeddings = embeddings_df[embeddings_df["frame_id"].isin(yolanda_frame_ids)]

    if len(yolanda_embeddings) == 0:
        raise ValueError("No YOLANDA embeddings found")

    # Compute average
    yolanda_vecs = np.stack(yolanda_embeddings["embedding"].values)
    yolanda_avg = yolanda_vecs.mean(axis=0)
    yolanda_avg = yolanda_avg / np.linalg.norm(yolanda_avg)

    logger.info(f"Computed YOLANDA avg embedding from {len(yolanda_embeddings)} vectors across {len(yolanda_frame_ids)} frames")

    return yolanda_avg


def verify_identity(embedding: np.ndarray, yolanda_avg: np.ndarray, threshold: float = 0.86) -> tuple[bool, float]:
    """Verify if embedding matches YOLANDA."""
    if embedding is None:
        return False, 0.0

    embedding_norm = embedding / np.linalg.norm(embedding)
    similarity = float(np.dot(embedding_norm, yolanda_avg))

    return similarity >= threshold, similarity


def main():
    """Main recompute function."""
    episode_id = "RHOBH-TEST-10-28"

    # Window: 00:17:22 → 00:20:04 (convert from MM:SS:MS to milliseconds)
    # 00:17:22 = 17.367s = 17,367ms (approximately 17,220ms based on user's earlier data)
    # 00:20:04 = 20.067s = 20,067ms (approximately 20,040ms)
    window_start_ms = 17220
    window_end_ms = 20040

    logger.info(f"Recomputing embeddings for {episode_id}")
    logger.info(f"Window: {window_start_ms}ms - {window_end_ms}ms ({(window_end_ms - window_start_ms)/1000:.2f}s)")

    # Load config
    config = load_config()
    data_root = Path(config["paths"]["data_root"])

    # Paths
    video_path = data_root / "videos" / f"{episode_id}.mp4"
    manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"
    reports_dir = data_root / "harvest" / episode_id / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest to find sampled frames
    manifest_df = pd.read_parquet(manifest_path)
    sampled_frames = manifest_df[
        (manifest_df['ts_ms'] >= window_start_ms) &
        (manifest_df['ts_ms'] <= window_end_ms) &
        (manifest_df['sampled'] == True)
    ]

    logger.info(f"Found {len(sampled_frames)} sampled frames in window")

    # Initialize detector and embedder with new settings
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

    logger.info("Initialized detector and embedder with new settings")
    logger.info(f"  skip_redetect: {embedding_config.get('skip_redetect', True)}")
    logger.info(f"  align_priority: {embedding_config.get('align_priority', 'kps_then_bbox')}")

    # Get YOLANDA average embedding for verification
    yolanda_avg = compute_yolanda_avg_embedding(episode_id, data_root)

    # Statistics
    stats = RecomputeStats(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        frames_processed=0,
        faces_detected=0,
        emb_ok=0,
        emb_none=0,
        used_kps=0,
        fallback_used=0,
        seconds_recovered=0.0
    )

    # Load existing embeddings
    existing_embeddings = pd.read_parquet(embeddings_path)
    logger.info(f"Loaded {len(existing_embeddings)} existing embeddings")

    # Process each frame
    new_embeddings = []
    yolanda_detections = []

    for _, row in sampled_frames.iterrows():
        frame_id = row['frame_id']
        ts_ms = row['ts_ms']

        stats.frames_processed += 1

        # Extract frame
        frame_bgr = extract_frame(video_path, ts_ms)
        if frame_bgr is None:
            logger.warning(f"Failed to extract frame {frame_id} at {ts_ms}ms")
            continue

        # Detect faces
        detections = detector.detect(frame_bgr)
        stats.faces_detected += len(detections)

        if len(detections) == 0:
            continue

        logger.info(f"Frame {frame_id} ({ts_ms}ms): {len(detections)} faces detected")

        # Generate embeddings for each face
        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            landmarks = det.get("landmarks")
            confidence = det["confidence"]
            face_size = det["face_size"]

            # Convert landmarks to kps format if available
            kps = None
            if landmarks is not None:
                # InsightFace returns 106 landmarks, we need first 5 points
                landmarks_array = np.array(landmarks)
                if len(landmarks_array) >= 5:
                    kps = landmarks_array[:5]

            # Generate embedding using new method
            result = embedder.embed_from_detection(frame_bgr, bbox, kps=kps)

            # Update stats
            if result.success:
                stats.emb_ok += 1
                if result.used_kps:
                    stats.used_kps += 1
                if result.tries > 1:
                    stats.fallback_used += 1
            else:
                stats.emb_none += 1

            # Verify if this is YOLANDA
            is_yolanda, similarity = verify_identity(result.embedding, yolanda_avg, threshold=0.86)

            logger.info(f"  Face {det_idx+1}: size={face_size}px, conf={confidence:.3f}, "
                       f"emb={'OK' if result.success else 'FAIL'}, "
                       f"tries={result.tries}, kps={result.used_kps}, "
                       f"similarity={similarity:.3f}, yolanda={is_yolanda}")

            # Only keep YOLANDA faces
            if is_yolanda and result.success:
                yolanda_detections.append({
                    "frame_id": frame_id,
                    "ts_ms": ts_ms,
                    "det_idx": det_idx,
                    "bbox": bbox,
                    "confidence": confidence,
                    "face_size": face_size,
                    "embedding": result.embedding,
                    "similarity": similarity,
                    "used_kps": result.used_kps,
                    "tries": result.tries
                })

                # Add to new embeddings list
                new_embeddings.append({
                    "episode_id": episode_id,
                    "frame_id": frame_id,
                    "ts_ms": ts_ms,
                    "det_idx": det_idx,
                    "bbox_x1": bbox[0],
                    "bbox_y1": bbox[1],
                    "bbox_x2": bbox[2],
                    "bbox_y3": bbox[3],
                    "confidence": confidence,
                    "face_size": face_size,
                    "embedding": result.embedding,
                    "has_embedding": True,
                    "used_kps": result.used_kps,
                    "tries": result.tries
                })

    # Calculate seconds recovered
    if len(yolanda_detections) > 0:
        # Group by consecutive frames
        yolanda_times = sorted([d["ts_ms"] for d in yolanda_detections])
        stats.seconds_recovered = (yolanda_times[-1] - yolanda_times[0]) / 1000.0

    logger.info(f"\nRecompute complete!")
    logger.info(f"  Frames processed: {stats.frames_processed}")
    logger.info(f"  Faces detected: {stats.faces_detected}")
    logger.info(f"  Embeddings OK: {stats.emb_ok}")
    logger.info(f"  Embeddings failed: {stats.emb_none}")
    logger.info(f"  Used keypoints: {stats.used_kps}")
    logger.info(f"  Used fallback scales: {stats.fallback_used}")
    logger.info(f"  YOLANDA faces found: {len(yolanda_detections)}")
    logger.info(f"  Seconds recovered: {stats.seconds_recovered:.2f}s")

    # Save new embeddings (append to existing)
    if len(new_embeddings) > 0:
        new_embeddings_df = pd.DataFrame(new_embeddings)

        # Remove any existing embeddings in this window (to avoid duplicates)
        existing_embeddings = existing_embeddings[
            (existing_embeddings["ts_ms"] < window_start_ms) |
            (existing_embeddings["ts_ms"] > window_end_ms)
        ]

        # Append new embeddings
        updated_embeddings = pd.concat([existing_embeddings, new_embeddings_df], ignore_index=True)
        updated_embeddings = updated_embeddings.sort_values(["ts_ms", "det_idx"])

        # Save
        updated_embeddings.to_parquet(embeddings_path, index=False)
        logger.info(f"Saved {len(new_embeddings)} new YOLANDA embeddings to {embeddings_path}")

    # Save stats
    stats_dict = {
        "window": {
            "start_ms": stats.window_start_ms,
            "end_ms": stats.window_end_ms,
            "duration_s": (stats.window_end_ms - stats.window_start_ms) / 1000.0
        },
        "recompute": {
            "frames_processed": stats.frames_processed,
            "faces_detected": stats.faces_detected,
            "emb_ok": stats.emb_ok,
            "emb_none": stats.emb_none,
            "used_kps": stats.used_kps,
            "fallback_used": stats.fallback_used
        },
        "yolanda": {
            "faces_found": len(yolanda_detections),
            "seconds_recovered": stats.seconds_recovered
        }
    }

    stats_path = reports_dir / "emb_recompute_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    logger.info(f"Saved stats to {stats_path}")

    return stats


if __name__ == "__main__":
    main()
