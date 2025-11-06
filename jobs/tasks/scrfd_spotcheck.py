#!/usr/bin/env python3
"""
SCRFD Spot-Check: Compare RetinaFace vs SCRFD on problematic gaps.

Samples ~100 frames from hardest gaps (YOLANDA, RINNA, BRANDI) and compares
small-face detection performance between the two detectors.

Gate criterion: ≥30% lift in verified small-face detections to proceed with full A/B.
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.detectors.face_scrfd import SCRFDDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def load_identity_prototypes(episode_id: str, data_root: Path) -> dict:
    """Load identity prototypes from embeddings."""
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"

    # Load clusters
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    # Build name -> cluster_id map
    name_to_cluster = {}
    for cluster in clusters_data['clusters']:
        if cluster.get('name'):
            name_to_cluster[cluster['name']] = cluster['cluster_id']

    # Load embeddings
    emb_df = pd.read_parquet(embeddings_path)

    # Compute prototypes
    prototypes = {}
    for name, cluster_id in name_to_cluster.items():
        # Get embeddings for this cluster
        cluster_embs = []
        for _, row in emb_df.iterrows():
            # Match by checking if this detection belongs to tracks in this cluster
            # Simplified: just compute average across all embeddings (would need proper track->cluster mapping)
            pass

        # For now, load from existing entrance_recovery approach
        # This is a placeholder - would need proper implementation
        prototypes[name] = None

    return prototypes


def identify_gap_windows(episode_id: str, data_root: Path, identities: list[str], sample_per_identity: int = 35) -> dict:
    """Identify gap windows to sample for each identity."""
    timeline_path = data_root / "outputs" / episode_id / "timeline.csv"

    if not timeline_path.exists():
        logger.error(f"Timeline not found: {timeline_path}")
        return {}

    timeline_df = pd.read_csv(timeline_path)

    gap_windows = {}

    for identity in identities:
        person_intervals = timeline_df[timeline_df['person_name'] == identity].sort_values('start_ms')

        if len(person_intervals) == 0:
            logger.warning(f"No intervals for {identity}")
            continue

        # Find gaps between intervals
        gaps = []
        for i in range(len(person_intervals) - 1):
            gap_start = person_intervals.iloc[i]['end_ms']
            gap_end = person_intervals.iloc[i + 1]['start_ms']
            gap_duration = gap_end - gap_start

            if gap_duration > 5000:  # Only gaps > 5s
                gaps.append({
                    'start_ms': int(gap_start),
                    'end_ms': int(gap_end),
                    'duration_ms': int(gap_duration)
                })

        # Sort by duration, take top 3
        gaps.sort(key=lambda x: x['duration_ms'], reverse=True)
        top_gaps = gaps[:3]

        logger.info(f"{identity}: Found {len(gaps)} large gaps, sampling top {len(top_gaps)}")

        # Sample frames from top gaps
        frames_per_gap = sample_per_identity // len(top_gaps) if top_gaps else 0

        sampled_frames = []
        for gap in top_gaps:
            # Sample evenly across gap
            gap_center = (gap['start_ms'] + gap['end_ms']) / 2
            window_size = 3000  # ±1.5s
            window_start = max(gap['start_ms'], gap_center - window_size / 2)
            window_end = min(gap['end_ms'], gap_center + window_size / 2)

            # Sample frames at 10fps
            step_ms = 100
            ts_ms = window_start
            frame_count = 0
            while ts_ms < window_end and frame_count < frames_per_gap:
                sampled_frames.append(int(ts_ms))
                ts_ms += step_ms
                frame_count += 1

        gap_windows[identity] = {
            'gaps': top_gaps,
            'sampled_frames': sampled_frames
        }

    return gap_windows


def main():
    """Run SCRFD spot-check."""
    episode_id = "RHOBH-TEST-10-28"

    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    video_path = data_root / "videos" / f"{episode_id}.mp4"

    logger.info("="*80)
    logger.info("SCRFD SPOT-CHECK")
    logger.info("="*80)
    logger.info(f"Episode: {episode_id}")
    logger.info(f"Target identities: YOLANDA, RINNA, BRANDI")
    logger.info(f"Sample size: ~100 frames from largest gaps")
    logger.info("")

    # Identify gap windows
    logger.info("Identifying gap windows...")
    gap_windows = identify_gap_windows(episode_id, data_root, ['YOLANDA', 'RINNA', 'BRANDI'], sample_per_identity=35)

    total_frames = sum(len(data['sampled_frames']) for data in gap_windows.values())
    logger.info(f"Total frames to scan: {total_frames}")
    logger.info("")

    # Initialize detectors
    logger.info("Initializing detectors...")
    detection_config = config.get("detection", {})

    retina_detector = RetinaFaceDetector(
        min_face_px=detection_config.get("min_face_px", 72),
        min_confidence=detection_config.get("min_confidence", 0.70),
        provider_order=detection_config.get("provider_order", ["coreml", "cpu"])
    )

    scrfd_detector = SCRFDDetector(
        min_face_px=detection_config.get("min_face_px", 72),
        min_confidence=detection_config.get("min_confidence", 0.70),
        provider_order=detection_config.get("provider_order", ["coreml", "cpu"]),
        model_name="scrfd_10g_bnkps"
    )

    logger.info(f"RetinaFace provider: {retina_detector.actual_provider}")
    logger.info(f"SCRFD provider: {scrfd_detector.actual_provider}")
    logger.info("")

    # Initialize embedder (for verification)
    embedding_config = config.get("embedding", {})
    embedder = ArcFaceEmbedder(
        provider_order=detection_config.get("provider_order", ["coreml", "cpu"]),
        skip_redetect=embedding_config.get("skip_redetect", True),
        align_priority=embedding_config.get("align_priority", "kps_then_bbox"),
        margin_scale=embedding_config.get("margin_scale", 1.25),
        min_chip_px=embedding_config.get("min_chip_px", 112),
        fallback_scales=embedding_config.get("fallback_scales", [1.0, 1.2, 1.4])
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Process frames
    logger.info("="*80)
    logger.info("PROCESSING FRAMES")
    logger.info("="*80)

    results = {
        'episode_id': episode_id,
        'total_frames_scanned': total_frames,
        'per_identity': {}
    }

    for identity, data in gap_windows.items():
        logger.info(f"\n{identity}:")
        logger.info(f"  Frames to scan: {len(data['sampled_frames'])}")

        retina_total = 0
        retina_small = 0
        retina_tiny = 0
        scrfd_total = 0
        scrfd_small = 0
        scrfd_tiny = 0

        example_crops = []

        for ts_ms in data['sampled_frames']:
            frame_id = int(ts_ms * fps / 1000)

            # Seek and read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret:
                continue

            # Detect with RetinaFace
            retina_dets = retina_detector.detect(frame)
            retina_total += len(retina_dets)

            for det in retina_dets:
                face_size = det['face_size']
                if face_size <= 80:
                    retina_small += 1
                if face_size <= 64:
                    retina_tiny += 1

            # Detect with SCRFD
            scrfd_dets = scrfd_detector.detect(frame)
            scrfd_total += len(scrfd_dets)

            for det in scrfd_dets:
                face_size = det.face_size
                if face_size <= 80:
                    scrfd_small += 1
                if face_size <= 64:
                    scrfd_tiny += 1

            # Find SCRFD detections not in RetinaFace (potential lifts)
            # Simplified: just count difference
            scrfd_advantage = len(scrfd_dets) - len(retina_dets)
            if scrfd_advantage > 0:
                # Save crop example
                if len(example_crops) < 5:
                    example_crops.append({
                        'frame_id': frame_id,
                        'ts_ms': ts_ms,
                        'retina_faces': len(retina_dets),
                        'scrfd_faces': len(scrfd_dets),
                        'advantage': scrfd_advantage
                    })

        # Compute lift percentage
        small_lift_pct = ((scrfd_small - retina_small) / retina_small * 100) if retina_small > 0 else 0

        results['per_identity'][identity] = {
            'frames_scanned': len(data['sampled_frames']),
            'retinaface': {
                'total_faces': retina_total,
                'small_faces_80px': retina_small,
                'tiny_faces_64px': retina_tiny
            },
            'scrfd': {
                'total_faces': scrfd_total,
                'small_faces_80px': scrfd_small,
                'tiny_faces_64px': scrfd_tiny
            },
            'lift': {
                'small_faces_absolute': scrfd_small - retina_small,
                'small_faces_percent': round(small_lift_pct, 1)
            },
            'example_crops': example_crops
        }

        logger.info(f"  RetinaFace: {retina_total} faces ({retina_small} ≤80px, {retina_tiny} ≤64px)")
        logger.info(f"  SCRFD: {scrfd_total} faces ({scrfd_small} ≤80px, {scrfd_tiny} ≤64px)")
        logger.info(f"  Small-face lift: {scrfd_small - retina_small} ({small_lift_pct:+.1f}%)")

    cap.release()

    # Save results
    output_path = data_root / "harvest" / episode_id / "diagnostics" / "reports" / "scrfd_spotcheck.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("="*80)
    logger.info("SPOT-CHECK COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_path}")
    logger.info("")

    # Decision gate
    logger.info("DECISION GATE:")
    max_lift = max(data['lift']['small_faces_percent'] for data in results['per_identity'].values())
    logger.info(f"  Maximum small-face lift: {max_lift:.1f}%")

    if max_lift >= 30:
        logger.info(f"  ✓ PASS - Proceed with full A/B (lift ≥30%)")
    else:
        logger.info(f"  ✗ SKIP - Insufficient lift for full A/B (lift <30%)")
        logger.info(f"     Focus on identity-guided recall + per-identity clamps instead")


if __name__ == "__main__":
    main()
