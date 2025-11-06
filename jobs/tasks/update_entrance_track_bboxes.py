#!/usr/bin/env python3
"""
Update Track 307 with real YOLANDA detection bounding boxes.
"""

import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Update Track 307 with real detection bboxes."""
    episode_id = "RHOBH-TEST-10-28"

    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    harvest_dir = data_root / "harvest" / episode_id

    tracks_path = harvest_dir / "tracks.json"
    detections_path = harvest_dir / "diagnostics" / "reports" / "entrance_detections.json"

    # Load entrance detections
    with open(detections_path) as f:
        entrance_data = json.load(f)

    detections = entrance_data['detections']
    logger.info(f"Loaded {len(detections)} YOLANDA detections from entrance recovery")

    # Load tracks
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    # Find Track 307
    track_307 = None
    track_307_idx = None

    for idx, track in enumerate(tracks_data['tracks']):
        if track['track_id'] == 307:
            track_307 = track
            track_307_idx = idx
            break

    if track_307 is None:
        logger.error("Track 307 not found")
        return

    logger.info(f"Found Track 307 at index {track_307_idx}")
    logger.info(f"Current frame_refs: {len(track_307['frame_refs'])}")

    # Build new frame_refs from actual detections
    new_frame_refs = []

    for det in detections:
        frame_ref = {
            "frame_id": det['frame_id'],
            "det_idx": det['det_idx'],
            "bbox": det['bbox'],
            "confidence": det['confidence'],
            "ts_ms": det['ts_ms']
        }
        new_frame_refs.append(frame_ref)

    # Update Track 307
    track_307['frame_refs'] = new_frame_refs
    track_307['count'] = len(new_frame_refs)

    if len(new_frame_refs) > 0:
        track_307['start_ms'] = min(ref['ts_ms'] for ref in new_frame_refs)
        track_307['end_ms'] = max(ref['ts_ms'] for ref in new_frame_refs)
        track_307['duration_ms'] = track_307['end_ms'] - track_307['start_ms']

    # Save updated tracks
    with open(tracks_path, 'w') as f:
        json.dump(tracks_data, f, indent=2)

    logger.info("")
    logger.info("="*80)
    logger.info("TRACK 307 UPDATED WITH REAL YOLANDA BBOXES")
    logger.info("="*80)
    logger.info(f"Frame refs: {len(new_frame_refs)}")
    logger.info(f"Time range: {track_307['start_ms']}-{track_307['end_ms']}ms")
    logger.info(f"Duration: {track_307['duration_ms']/1000:.2f}s")
    logger.info("")
    logger.info("Sample bboxes:")
    for ref in new_frame_refs[:5]:
        logger.info(f"  Frame {ref['frame_id']} @ {ref['ts_ms']}ms: {ref['bbox']}")
    logger.info("")
    logger.info("Refresh Streamlit to see correct YOLANDA thumbnails")
    logger.info("="*80)


if __name__ == "__main__":
    main()
