"""
Data loading helpers for Review UI (Phase 1.6).

Read-only functions to load tracks, clusters, and suggestions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd


def load_tracks(episode_id: str, data_root: Path = Path("data")) -> Optional[dict]:
    """
    Load tracks.json for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Tracks data dict or None if not found
    """
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    if not tracks_path.exists():
        return None

    with open(tracks_path) as f:
        return json.load(f)


def load_clusters(episode_id: str, data_root: Path = Path("data")) -> Optional[dict]:
    """
    Load clusters.json for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Clusters data dict or None if not found
    """
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    if not clusters_path.exists():
        return None

    with open(clusters_path) as f:
        return json.load(f)


def load_merge_suggestions(
    episode_id: str, data_root: Path = Path("data")
) -> Optional[pd.DataFrame]:
    """
    Load merge suggestions for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Suggestions DataFrame or None if not found
    """
    suggestions_path = data_root / "harvest" / episode_id / "assist" / "merge_suggestions.parquet"
    if not suggestions_path.exists():
        return None

    return pd.read_parquet(suggestions_path)


def load_lowconf_queue(episode_id: str, data_root: Path = Path("data")) -> Optional[pd.DataFrame]:
    """
    Load low-confidence queue for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Low-conf queue DataFrame or None if not found
    """
    lowconf_path = data_root / "harvest" / episode_id / "assist" / "lowconf_queue.parquet"
    if not lowconf_path.exists():
        return None

    return pd.read_parquet(lowconf_path)


def load_cluster_metrics(episode_id: str, data_root: Path = Path("data")) -> Optional[pd.DataFrame]:
    """
    Load cluster-level confidence metrics for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        DataFrame of cluster metrics or None if not found
    """
    metrics_path = data_root / "harvest" / episode_id / "diagnostics" / "cluster_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    if not data:
        return pd.DataFrame(columns=["cluster_id"])

    return pd.DataFrame(data)


def load_person_metrics(episode_id: str, data_root: Path = Path("data")) -> Optional[pd.DataFrame]:
    """
    Load person-level confidence metrics for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        DataFrame of person metrics or None if not found
    """
    metrics_path = data_root / "harvest" / episode_id / "diagnostics" / "person_metrics.json"
    if not metrics_path.exists():
        return None

    with open(metrics_path) as f:
        data = json.load(f)

    if not data:
        return pd.DataFrame(columns=["person"])

    return pd.DataFrame(data)


def load_embeddings(episode_id: str, data_root: Path = Path("data")) -> Optional[pd.DataFrame]:
    """
    Load embeddings for episode.

    Prefers picked_samples.parquet (face-only, top-K per track) for gallery display.
    Falls back to embeddings.parquet for backward compatibility.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Embeddings DataFrame or None if not found
    """
    # Try picked_samples first (face-only gallery feed)
    picked_samples_path = data_root / "harvest" / episode_id / "picked_samples.parquet"
    if picked_samples_path.exists():
        return pd.read_parquet(picked_samples_path)

    # Fallback to all embeddings (backward compatibility)
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"
    if not embeddings_path.exists():
        return None

    return pd.read_parquet(embeddings_path)


def get_cluster_by_id(clusters_data: dict, cluster_id: int) -> Optional[dict]:
    """
    Get cluster by ID.

    Args:
        clusters_data: Clusters data from load_clusters()
        cluster_id: Cluster ID

    Returns:
        Cluster dict or None if not found
    """
    for cluster in clusters_data.get("clusters", []):
        if cluster["cluster_id"] == cluster_id:
            return cluster
    return None


def get_track_by_id(tracks_data: dict, track_id: int) -> Optional[dict]:
    """
    Get track by ID.

    Args:
        tracks_data: Tracks data from load_tracks()
        track_id: Track ID

    Returns:
        Track dict or None if not found
    """
    for track in tracks_data.get("tracks", []):
        if track["track_id"] == track_id:
            return track
    return None


def get_frames_for_track(track: dict, embeddings_df: pd.DataFrame) -> list[dict]:
    """
    Get frame data for track.

    Args:
        track: Track dict from get_track_by_id()
        embeddings_df: Embeddings DataFrame from load_embeddings()

    Returns:
        List of frame dicts with frame_id, det_idx, bbox, confidence, embedding
    """
    frames = []
    for ref in track["frame_refs"]:
        frame_id = ref["frame_id"]
        det_idx = ref["det_idx"]

        # Find embedding
        emb_row = embeddings_df[
            (embeddings_df["frame_id"] == frame_id) & (embeddings_df["det_idx"] == det_idx)
        ]

        if len(emb_row) > 0:
            row = emb_row.iloc[0]
            frames.append(
                {
                    "frame_id": int(row["frame_id"]),
                    "det_idx": int(row["det_idx"]),
                    "bbox": [
                        int(row["bbox_x1"]),
                        int(row["bbox_y1"]),
                        int(row["bbox_x2"]),
                        int(row["bbox_y2"]),
                    ],
                    "confidence": float(row["confidence"]),
                    "embedding": row["embedding"],
                }
            )

    return frames


def get_episode_summary(episode_id: str, data_root: Path = Path("data")) -> dict:
    """
    Get summary statistics for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Dict with summary stats
    """
    harvest_dir = data_root / "harvest" / episode_id
    reports_dir = harvest_dir / "diagnostics" / "reports"

    # Load stats files
    det_stats = {}
    det_stats_path = reports_dir / "det_stats.json"
    if det_stats_path.exists():
        with open(det_stats_path) as f:
            det_stats = json.load(f)

    track_stats = {}
    track_stats_path = reports_dir / "track_stats.json"
    if track_stats_path.exists():
        with open(track_stats_path) as f:
            track_stats = json.load(f)

    cluster_stats = {}
    cluster_stats_path = reports_dir / "cluster_stats.json"
    if cluster_stats_path.exists():
        with open(cluster_stats_path) as f:
            cluster_stats = json.load(f)

    # Combine summary
    return {
        "episode_id": episode_id,
        "detection": {
            "frames_processed": det_stats.get("frames_processed", 0),
            "faces_detected": det_stats.get("faces_detected", 0),
            "embeddings_computed": det_stats.get("embeddings_computed", 0),
        },
        "tracking": {
            "tracks_built": track_stats.get("tracks_built", 0),
            "track_switches": track_stats.get("track_switches", 0),
        },
        "clustering": {
            "clusters_built": cluster_stats.get("clusters_built", 0),
            "noise_tracks": cluster_stats.get("noise_tracks", 0),
            "suggestions_enqueued": cluster_stats.get("suggestions_enqueued", 0),
            "lowconf_enqueued": cluster_stats.get("lowconf_enqueued", 0),
        },
    }


def load_assets_thumbnail(
    cluster_id: Optional[int] = None, person_name: Optional[str] = None
) -> Optional[Path]:
    """
    Load thumbnail from assets/ based on mapping.

    Checks in order:
    1. cluster:<cluster_id>
    2. person_name:<UPPERCASE>
    3. alias:<lowercase>

    Args:
        cluster_id: Cluster ID to lookup
        person_name: Person name to lookup (case-insensitive)

    Returns:
        Path to asset thumbnail or None if not found
    """
    assets_root = Path("assets")
    thumbnails_map_path = assets_root / "thumbnails_map.json"

    if not thumbnails_map_path.exists():
        return None

    try:
        with open(thumbnails_map_path) as f:
            thumbnails_map = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    # Check 1: cluster:<cluster_id>
    if cluster_id is not None:
        key = f"cluster:{cluster_id}"
        if key in thumbnails_map:
            thumbnail_path = assets_root / thumbnails_map[key]
            if thumbnail_path.exists():
                return thumbnail_path

    # Check 2: person_name:<UPPERCASE>
    if person_name:
        key = f"person_name:{person_name.upper()}"
        if key in thumbnails_map:
            thumbnail_path = assets_root / thumbnails_map[key]
            if thumbnail_path.exists():
                return thumbnail_path

        # Check 3: alias:<lowercase>
        key = f"alias:{person_name.lower()}"
        if key in thumbnails_map:
            thumbnail_path = assets_root / thumbnails_map[key]
            if thumbnail_path.exists():
                return thumbnail_path

    return None
