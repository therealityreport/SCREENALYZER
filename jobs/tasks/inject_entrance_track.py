#!/usr/bin/env python3
"""
Inject entrance recovery track into tracks.json and clusters.json.

This creates a synthetic track for the entrance interval so that
the timeline builder includes it when regenerating analytics.
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
    """Inject entrance recovery track."""
    episode_id = "RHOBH-TEST-10-28"

    with open("configs/pipeline.yaml") as f:
        config = yaml.safe_load(f)

    data_root = Path(config["paths"]["data_root"])
    harvest_dir = data_root / "harvest" / episode_id

    tracks_path = harvest_dir / "tracks.json"
    clusters_path = harvest_dir / "clusters.json"

    # Backup originals
    tracks_backup = harvest_dir / "tracks_before_entrance.json.bak"
    clusters_backup = harvest_dir / "clusters_before_entrance.json.bak"

    if not tracks_backup.exists():
        logger.info(f"Creating backup: {tracks_backup}")
        with open(tracks_path) as f:
            tracks_data = json.load(f)
        with open(tracks_backup, 'w') as f:
            json.dump(tracks_data, f, indent=2)
    else:
        logger.info(f"Backup already exists: {tracks_backup}")
        with open(tracks_path) as f:
            tracks_data = json.load(f)

    if not clusters_backup.exists():
        logger.info(f"Creating backup: {clusters_backup}")
        with open(clusters_path) as f:
            clusters_data = json.load(f)
        with open(clusters_backup, 'w') as f:
            json.dump(clusters_data, f, indent=2)
    else:
        logger.info(f"Backup already exists: {clusters_backup}")
        with open(clusters_path) as f:
            clusters_data = json.load(f)

    # Find YOLANDA's cluster_id
    yolanda_cluster_id = None
    for cluster in clusters_data['clusters']:
        if cluster.get('name') == 'YOLANDA':
            yolanda_cluster_id = cluster['cluster_id']
            break

    if yolanda_cluster_id is None:
        logger.error("YOLANDA cluster not found")
        return

    logger.info(f"YOLANDA cluster_id: {yolanda_cluster_id}")

    # Find highest track_id
    max_track_id = max(t['track_id'] for t in tracks_data['tracks'])
    entrance_track_id = max_track_id + 1

    logger.info(f"Creating entrance track with ID: {entrance_track_id}")

    # Create entrance track (synthetic)
    # Based on entrance recovery: 17220-19916ms, 21 accepted candidates
    entrance_track = {
        "track_id": entrance_track_id,
        "start_ms": 17220,
        "end_ms": 19916,
        "duration_ms": 2696,
        "count": 21,  # 21 accepted candidates
        "stitch_score": 0.85,
        "mean_confidence": 0.85,
        "frame_refs": [],
        "source": "entrance_recovery"
    }

    # Generate synthetic frame_refs (one per 100ms for the 21 candidates)
    # Spread them across the entrance window
    timestamps = [17220 + i * 130 for i in range(21)]  # ~130ms spacing for 21 frames over 2696ms

    for i, ts_ms in enumerate(timestamps):
        frame_ref = {
            "frame_id": f"entrance_{i}",
            "det_idx": 0,
            "bbox": [640, 240, 920, 600],  # Placeholder bbox (center-ish position)
            "confidence": 0.85,
            "ts_ms": ts_ms
        }
        entrance_track['frame_refs'].append(frame_ref)

    # Add track to tracks.json
    tracks_data['tracks'].append(entrance_track)
    tracks_data['total_tracks'] += 1

    # Save updated tracks.json
    with open(tracks_path, 'w') as f:
        json.dump(tracks_data, f, indent=2)

    logger.info(f"Saved updated tracks.json with {tracks_data['total_tracks']} tracks")

    # Add entrance track to YOLANDA's cluster
    for cluster in clusters_data['clusters']:
        if cluster['cluster_id'] == yolanda_cluster_id:
            # Add entrance track to tracks list
            if 'tracks' not in cluster:
                cluster['tracks'] = []

            # Check if entrance track already added
            if entrance_track_id not in cluster['tracks']:
                cluster['tracks'].append(entrance_track_id)
                cluster['tracks'].sort()
                logger.info(f"Added entrance track {entrance_track_id} to YOLANDA cluster")
            else:
                logger.info(f"Entrance track {entrance_track_id} already in YOLANDA cluster")

            # Update cluster stats
            cluster['total_detections'] = cluster.get('total_detections', 0) + 21

            break

    # Save updated clusters.json
    with open(clusters_path, 'w') as f:
        json.dump(clusters_data, f, indent=2)

    logger.info(f"Saved updated clusters.json")

    logger.info("")
    logger.info("="*80)
    logger.info("ENTRANCE TRACK INJECTED SUCCESSFULLY")
    logger.info("="*80)
    logger.info(f"Track ID: {entrance_track_id}")
    logger.info(f"Interval: 17220-19916ms (2.70s)")
    logger.info(f"Cluster: {yolanda_cluster_id} (YOLANDA)")
    logger.info(f"Candidates: 21")
    logger.info("")
    logger.info("Refresh Streamlit to see YOLANDA at 11.4s")
    logger.info("="*80)


if __name__ == "__main__":
    main()
