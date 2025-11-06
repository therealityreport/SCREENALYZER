"""
Enhanced Episode Status with Faces counts, Constraints, and Suppression tracking.
"""

import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd


def get_enhanced_episode_status(episode_id: str, data_root: Path) -> Dict[str, Any]:
    """
    Get comprehensive episode status including faces, constraints, and suppressions.

    Returns dict with:
    - faces_total: raw embeddings count
    - faces_used: picked_samples count (Top-K)
    - tracks: number of tracks
    - clusters: number of clusters
    - suggestions: merge suggestions count
    - constraints_ml: must-link count
    - constraints_cl: cannot-link count
    - suppressed_tracks: deleted tracks count
    - suppressed_clusters: deleted clusters count
    """
    harvest_dir = data_root / "harvest" / episode_id
    diagnostics_dir = harvest_dir / "diagnostics"

    status = {
        "episode_id": episode_id,
        "faces_total": 0,
        "faces_used": 0,
        "tracks": 0,
        "clusters": 0,
        "suggestions": 0,
        "constraints_ml": 0,
        "constraints_cl": 0,
        "suppressed_tracks": 0,
        "suppressed_clusters": 0
    }

    # Faces total (embeddings.parquet)
    embeddings_path = harvest_dir / "embeddings.parquet"
    if embeddings_path.exists():
        try:
            df = pd.read_parquet(embeddings_path)
            status["faces_total"] = len(df)
        except Exception:
            pass

    # Faces used (picked_samples.parquet - Top-K for clustering)
    picked_samples_path = harvest_dir / "picked_samples.parquet"
    if picked_samples_path.exists():
        try:
            df = pd.read_parquet(picked_samples_path)
            status["faces_used"] = len(df)
        except Exception:
            pass

    # Tracks
    tracks_path = harvest_dir / "tracks.json"
    if tracks_path.exists():
        try:
            with open(tracks_path) as f:
                tracks_data = json.load(f)
                status["tracks"] = len(tracks_data.get("tracks", []))
        except Exception:
            pass

    # Clusters
    clusters_path = harvest_dir / "clusters.json"
    if clusters_path.exists():
        try:
            with open(clusters_path) as f:
                clusters_data = json.load(f)
                status["clusters"] = len(clusters_data.get("clusters", []))
        except Exception:
            pass

    # Merge suggestions
    suggestions_path = harvest_dir / "assist" / "merge_suggestions.parquet"
    if suggestions_path.exists():
        try:
            df = pd.read_parquet(suggestions_path)
            status["suggestions"] = len(df)
        except Exception:
            pass

    # Constraints (from diagnostics/constraints.json)
    constraints_path = diagnostics_dir / "constraints.json"
    if constraints_path.exists():
        try:
            with open(constraints_path) as f:
                constraints_data = json.load(f)
                extraction = constraints_data.get("extraction", {})
                status["constraints_ml"] = extraction.get("must_link_count", 0)
                status["constraints_cl"] = extraction.get("cannot_link_count", 0)
        except Exception:
            pass

    # Suppressed items (from diagnostics/suppress.json)
    suppress_path = diagnostics_dir / "suppress.json"
    if suppress_path.exists():
        try:
            with open(suppress_path) as f:
                suppress_data = json.load(f)
                status["suppressed_tracks"] = len(suppress_data.get("deleted_tracks", []))
                status["suppressed_clusters"] = len(suppress_data.get("deleted_clusters", []))
        except Exception:
            pass

    return status


def save_episode_status(episode_id: str, data_root: Path):
    """Save current episode status snapshot to diagnostics/episode_status.json."""
    status = get_enhanced_episode_status(episode_id, data_root)

    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    status_path = diagnostics_dir / "episode_status.json"
    temp_path = status_path.with_suffix('.json.tmp')

    with open(temp_path, 'w') as f:
        json.dump(status, f, indent=2)

    temp_path.rename(status_path)


def load_suppress_data(episode_id: str, data_root: Path) -> Dict[str, Any]:
    """Load suppression data (deleted tracks/clusters)."""
    suppress_path = data_root / "harvest" / episode_id / "diagnostics" / "suppress.json"
    if not suppress_path.exists():
        return {
            "show_id": "rhobh",
            "season_id": "s05",
            "episode_id": episode_id,
            "deleted_tracks": [],
            "deleted_clusters": []
        }

    try:
        with open(suppress_path) as f:
            return json.load(f)
    except Exception:
        return {
            "show_id": "rhobh",
            "season_id": "s05",
            "episode_id": episode_id,
            "deleted_tracks": [],
            "deleted_clusters": []
        }


def save_suppress_data(episode_id: str, data_root: Path, suppress_data: Dict[str, Any]):
    """Save suppression data (atomic write)."""
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Deduplicate to prevent same items being added multiple times
    if 'deleted_tracks' in suppress_data:
        suppress_data['deleted_tracks'] = list(set(suppress_data['deleted_tracks']))
    if 'deleted_clusters' in suppress_data:
        suppress_data['deleted_clusters'] = list(set(suppress_data['deleted_clusters']))

    suppress_path = diagnostics_dir / "suppress.json"
    temp_path = suppress_path.with_suffix('.json.tmp')

    with open(temp_path, 'w') as f:
        json.dump(suppress_data, f, indent=2)

    temp_path.rename(suppress_path)
