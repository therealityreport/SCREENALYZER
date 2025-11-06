#!/usr/bin/env python3
"""
Simplified Detector A/B - Run SCRFD in parallel to RetinaFace.

Generates harvest__scrfd/ using the same frames as existing harvest/.
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

# Import both detectors
from screentime.detectors.face_scrfd import SCRFDDetector
from screentime.detectors.face_retina import RetinaFaceDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run SCRFD detector on same frames as RetinaFace."""
    episode_id = "RHOBH-TEST-10-28"

    # Load config
    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    video_path = data_root / "videos" / f"{episode_id}.mp4"

    # Load existing manifest (from RetinaFace run)
    retina_manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"

    if not retina_manifest_path.exists():
        logger.error(f"RetinaFace manifest not found: {retina_manifest_path}")
        logger.error("Run RetinaFace baseline first!")
        return

    retina_manifest = pd.read_parquet(retina_manifest_path)
    logger.info(f"Loaded RetinaFace manifest: {len(retina_manifest)} frames")

    # Create SCRFD harvest directory
    scrfd_harvest_dir = data_root / "harvest__scrfd" / episode_id
    scrfd_harvest_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"SCRFD harvest directory: {scrfd_harvest_dir}")

    # Initialize SCRFD detector
    logger.info("Initializing SCRFD detector...")
    detection_config = config.get("detection", {})

    scrfd_detector = SCRFDDetector(
        min_face_px=detection_config.get("min_face_px", 72),
        min_confidence=detection_config.get("min_confidence", 0.70),
        provider_order=detection_config.get("provider_order", ["coreml", "cpu"]),
        model_name="scrfd_10g_bnkps"
    )

    logger.info(f"SCRFD initialized with provider: {scrfd_detector.actual_provider}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path.name}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Total frames: {total_frames}")
    logger.info(f"  Frames to process: {len(retina_manifest)}")

    # Process frames
    logger.info("")
    logger.info("="*80)
    logger.info("PROCESSING FRAMES WITH SCRFD")
    logger.info("="*80)

    start_time = time.time()
    detections_data = []
    frames_processed = 0
    total_faces = 0

    for idx, row in retina_manifest.iterrows():
        frame_id = int(row['frame_id'])
        ts_ms = int(row['ts_ms'])

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            logger.warning(f"Failed to read frame {frame_id}")
            continue

        # Detect with SCRFD
        detections = scrfd_detector.detect(frame)

        # Store detections
        for det_idx, det in enumerate(detections):
            detections_data.append({
                'frame_id': frame_id,
                'ts_ms': ts_ms,
                'det_idx': det_idx,
                'bbox': det.bbox,
                'confidence': det.confidence,
                'face_size': det.face_size,
                'kps': det.kps.tolist() if det.kps is not None else None
            })
            total_faces += 1

        frames_processed += 1

        if frames_processed % 100 == 0:
            elapsed = time.time() - start_time
            fps_proc = frames_processed / elapsed
            logger.info(f"Processed {frames_processed}/{len(retina_manifest)} frames ({fps_proc:.1f} fps) - {total_faces} faces detected")

    cap.release()

    elapsed = time.time() - start_time
    logger.info("")
    logger.info("="*80)
    logger.info("SCRFD DETECTION COMPLETE")
    logger.info("="*80)
    logger.info(f"Frames processed: {frames_processed}")
    logger.info(f"Total faces detected: {total_faces}")
    logger.info(f"Time: {elapsed:.1f}s ({frames_processed/elapsed:.1f} fps)")
    logger.info(f"Avg faces/frame: {total_faces/frames_processed:.2f}")

    # Save SCRFD detections
    logger.info("")
    logger.info("Saving SCRFD detections...")

    # Save as JSON (will convert to parquet later in full pipeline)
    detections_path = scrfd_harvest_dir / "detections_scrfd.json"
    with open(detections_path, 'w') as f:
        json.dump({
            'episode_id': episode_id,
            'detector': 'scrfd',
            'model': 'scrfd_10g_bnkps',
            'frames_processed': frames_processed,
            'total_detections': total_faces,
            'runtime_seconds': elapsed,
            'fps': frames_processed / elapsed,
            'detections': detections_data
        }, f, indent=2)

    logger.info(f"✓ Saved SCRFD detections to: {detections_path}")

    # Copy manifest for SCRFD branch
    scrfd_manifest_path = scrfd_harvest_dir / "manifest.parquet"
    retina_manifest.to_parquet(scrfd_manifest_path)
    logger.info(f"✓ Copied manifest to: {scrfd_manifest_path}")

    logger.info("")
    logger.info("="*80)
    logger.info("NEXT STEPS")
    logger.info("="*80)
    logger.info("1. Run ByteTrack on SCRFD detections")
    logger.info("2. Generate embeddings for SCRFD tracks")
    logger.info("3. Transfer labels from RetinaFace")
    logger.info("4. Run entrance recovery + analytics")
    logger.info("5. Compare final results")


if __name__ == "__main__":
    main()
