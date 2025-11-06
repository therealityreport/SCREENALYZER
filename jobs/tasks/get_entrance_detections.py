#!/usr/bin/env python3
"""
Extract actual YOLANDA detection bboxes from entrance window.

Re-runs detection on entrance window and saves YOLANDA-specific detections
with verified embeddings.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder

# Import from entrance_recovery module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from entrance_recovery import compute_identity_prototypes

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_config():
    """Load pipeline configuration."""
    with open("configs/pipeline.yaml") as f:
        return yaml.safe_load(f)


def main():
    """Extract YOLANDA detections from entrance window."""
    episode_id = "RHOBH-TEST-10-28"
    config = load_config()
    data_root = Path(config["paths"]["data_root"])

    entrance_start = 17220
    entrance_end = 19916

    logger.info(f"Extracting YOLANDA detections from {entrance_start}-{entrance_end}ms")

    # Load video
    video_path = data_root / "videos" / f"{episode_id}.mp4"
    manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    manifest_df = pd.read_parquet(manifest_path)

    # Get frames in entrance window
    entrance_frames = manifest_df[
        (manifest_df['ts_ms'] >= entrance_start) &
        (manifest_df['ts_ms'] <= entrance_end)
    ].copy()

    logger.info(f"Processing {len(entrance_frames)} frames")

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

    # Get YOLANDA prototype using entrance_recovery module
    prototypes = compute_identity_prototypes("YOLANDA", episode_id, data_root)
    yolanda_prototype = prototypes["YOLANDA"]
    logger.info(f"Loaded YOLANDA prototype")

    # Process frames
    yolanda_detections = []

    for _, row in entrance_frames.iterrows():
        frame_id = int(row['frame_id'])
        ts_ms = int(row['ts_ms'])

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            continue

        # Detect faces
        detections = detector.detect(frame)

        # Process each detection
        for det_idx, det in enumerate(detections):
            bbox = det['bbox']
            kps = det.get('kps')
            conf = det.get('score', det.get('confidence', 0.9))

            # Generate embedding
            result = embedder.embed_from_detection(frame, bbox, kps)

            if not result.success:
                continue

            # Compare to YOLANDA prototype
            sim = float(np.dot(result.embedding, yolanda_prototype))

            # Accept if similarity >= 0.72 (same threshold as entrance recovery)
            if sim >= 0.72:
                yolanda_detections.append({
                    'frame_id': frame_id,
                    'ts_ms': ts_ms,
                    'det_idx': det_idx,
                    'bbox': [int(x) for x in bbox],
                    'confidence': float(conf),
                    'similarity': sim
                })

    cap.release()

    logger.info(f"Found {len(yolanda_detections)} YOLANDA detections")

    # Save detections
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "reports" / "entrance_detections.json"
    with open(output_path, 'w') as f:
        json.dump({
            'episode_id': episode_id,
            'window': {'start_ms': entrance_start, 'end_ms': entrance_end},
            'detections': yolanda_detections
        }, f, indent=2)

    logger.info(f"Saved detections to {output_path}")

    # Show sample detections
    logger.info("\nSample detections:")
    for det in yolanda_detections[:5]:
        logger.info(f"  Frame {det['frame_id']} @ {det['ts_ms']}ms: bbox={det['bbox']}, sim={det['similarity']:.3f}")


if __name__ == "__main__":
    main()
