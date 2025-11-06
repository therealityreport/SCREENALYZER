#!/usr/bin/env python3
"""
Diagnose Track 307 bbox/thumbnail issue.

Checks if bboxes are correctly positioned and generates overlay visualization.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Diagnose Track 307."""
    episode_id = "RHOBH-TEST-10-28"

    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    harvest_dir = data_root / "harvest" / episode_id

    tracks_path = harvest_dir / "tracks.json"
    video_path = data_root / "videos" / f"{episode_id}.mp4"

    # Load tracks
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    # Find Track 307
    track_307 = None
    for track in tracks_data['tracks']:
        if track['track_id'] == 307:
            track_307 = track
            break

    if track_307 is None:
        logger.error("Track 307 not found")
        return

    logger.info(f"Track 307 found:")
    logger.info(f"  Time range: {track_307['start_ms']}-{track_307['end_ms']}ms")
    logger.info(f"  Frame refs: {len(track_307['frame_refs'])}")

    if len(track_307['frame_refs']) == 0:
        logger.error("No frame_refs in Track 307")
        return

    # Check first 5 frame refs
    logger.info(f"\nFirst 5 frame refs:")
    for i, ref in enumerate(track_307['frame_refs'][:5]):
        logger.info(f"  {i}: frame_id={ref['frame_id']}, bbox={ref['bbox']}, ts={ref.get('ts_ms', 'N/A')}ms")

    # Open video and check a sample frame
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    # Get video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    logger.info(f"\nVideo properties:")
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps}")

    # Sample middle frame
    sample_ref = track_307['frame_refs'][len(track_307['frame_refs']) // 2]
    frame_id = sample_ref['frame_id']
    bbox = sample_ref['bbox']

    logger.info(f"\nSample frame {frame_id}:")
    logger.info(f"  Bbox: {bbox}")

    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()

    if not ret:
        logger.error(f"Failed to read frame {frame_id}")
        cap.release()
        return

    # Check bbox validity
    x1, y1, x2, y2 = bbox
    logger.info(f"  Bbox coords: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    logger.info(f"  Bbox size: {x2-x1}x{y2-y1}")

    # Validate bbox is within frame
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        logger.warning(f"  ⚠️  Bbox extends outside frame! Frame: {width}x{height}")

    if x1 >= x2 or y1 >= y2:
        logger.error(f"  ❌ Invalid bbox coordinates!")

    # Check if bbox captures YOLANDA (right side) or KIM (left/center)
    center_x = (x1 + x2) / 2
    relative_x = center_x / width

    logger.info(f"  Bbox center: ({center_x:.0f}, {(y1+y2)/2:.0f})")
    logger.info(f"  Relative X position: {relative_x:.2f} (0=left, 0.5=center, 1=right)")

    if relative_x < 0.5:
        logger.warning(f"  ⚠️  Bbox is on LEFT side of frame (likely KIM)")
    elif relative_x > 0.7:
        logger.info(f"  ✓ Bbox is on RIGHT side of frame (likely YOLANDA)")
    else:
        logger.warning(f"  ⚠️  Bbox is in CENTER (ambiguous)")

    # Draw bbox and save overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(overlay, f"Track 307 Frame {frame_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(overlay, f"Bbox: {bbox}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save overlay
    output_dir = data_root / "harvest" / episode_id / "diagnostics" / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"track307_frame{frame_id}.jpg"

    cv2.imwrite(str(output_path), overlay)
    logger.info(f"\n✓ Saved overlay to: {output_path}")

    # Also save the crop
    crop = frame[y1:y2, x1:x2]
    crop_path = output_dir / f"track307_frame{frame_id}_crop.jpg"
    cv2.imwrite(str(crop_path), crop)
    logger.info(f"✓ Saved crop to: {crop_path}")

    cap.release()

    logger.info(f"\n{'='*80}")
    logger.info(f"DIAGNOSIS COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Review the overlay image to verify bbox correctness:")
    logger.info(f"  {output_path}")


if __name__ == "__main__":
    main()
