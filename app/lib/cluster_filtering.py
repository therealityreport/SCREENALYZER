"""
Cluster filtering and grouping utilities for All Faces view.

Handles suppression filtering and identity grouping without modifying source data.
"""

from pathlib import Path
from typing import List, Dict


def build_visible_clusters(
    clusters_data: dict,
    tracks_data: dict,
    episode_id: str,
    data_root: Path,
    group_by_identity: bool = False
) -> List[Dict]:
    """
    Build filtered and optionally grouped cluster list for display.

    Args:
        clusters_data: Raw clusters from clusters.json
        tracks_data: Tracks data
        episode_id: Episode ID
        data_root: Data root path
        group_by_identity: If True, collapse same-name clusters into one row

    Returns:
        List of cluster dicts (filtered and optionally grouped)
    """
    from app.lib.episode_status import load_suppress_data

    # Load suppression data
    suppress_data = load_suppress_data(episode_id, data_root)
    deleted_tracks = set(suppress_data.get('deleted_tracks', []))
    deleted_clusters = set(suppress_data.get('deleted_clusters', []))

    # Filter clusters
    visible_clusters = []

    for cluster in clusters_data.get('clusters', []):
        cluster_id = cluster['cluster_id']

        # Skip if cluster itself is deleted
        if cluster_id in deleted_clusters:
            continue

        # Filter track_ids to remove suppressed tracks
        track_ids = cluster.get('track_ids', [])
        filtered_track_ids = [tid for tid in track_ids if tid not in deleted_tracks]

        # Skip cluster if no tracks remain
        if not filtered_track_ids:
            continue

        # Create filtered cluster
        filtered_cluster = cluster.copy()
        filtered_cluster['track_ids'] = filtered_track_ids
        filtered_cluster['size'] = len(filtered_track_ids)

        visible_clusters.append(filtered_cluster)

    # Apply grouping if requested
    if group_by_identity:
        return _group_by_identity(visible_clusters)
    else:
        return visible_clusters


def _group_by_identity(clusters: List[Dict]) -> List[Dict]:
    """
    Group clusters by identity name into virtual rows.

    Args:
        clusters: List of filtered clusters

    Returns:
        List of grouped cluster dicts
    """
    # Group by name
    by_name = {}
    for cluster in clusters:
        name = cluster.get('name', 'Unknown')
        if name not in by_name:
            by_name[name] = []
        by_name[name].append(cluster)

    # Create grouped rows
    grouped = []

    for name, cluster_list in by_name.items():
        if len(cluster_list) == 1:
            # Single cluster - keep as-is but mark ungrouped
            single = cluster_list[0].copy()
            single['is_grouped'] = False
            single['cluster_ids'] = [single['cluster_id']]
            single['num_clusters'] = 1
            grouped.append(single)
        else:
            # Multiple clusters - create virtual group
            all_track_ids = []
            all_cluster_ids = []
            total_quality = 0.0
            all_locked = True  # Track if all clusters are locked

            for c in cluster_list:
                all_track_ids.extend(c.get('track_ids', []))
                all_cluster_ids.append(c['cluster_id'])
                total_quality += c.get('quality_score', 0.0)

                # Check if this cluster is locked (conf=1.0)
                if c.get('assignment_confidence', 0.0) != 1.0:
                    all_locked = False

            # Sort cluster IDs ascending
            all_cluster_ids.sort()

            # Create virtual grouped cluster
            grouped_cluster = {
                'cluster_id': all_cluster_ids[0],  # Use first ID for reference
                'cluster_ids': all_cluster_ids,  # Store all cluster IDs (sorted)
                'name': name,
                'track_ids': all_track_ids,
                'size': len(all_track_ids),
                'quality_score': total_quality / len(cluster_list),
                'is_grouped': True,
                'num_clusters': len(cluster_list),
                'assignment_confidence': 1.0 if all_locked else 0.0,
                'all_locked': all_locked  # Flag for lock pill display
            }

            grouped.append(grouped_cluster)

    # Sort: named first (by name), then Unknown
    grouped.sort(key=lambda c: (c.get('name') == 'Unknown', c.get('name', 'Unknown')))

    return grouped


def format_cluster_ids(ids: List[int], max_list: int = 6) -> str:
    """
    Format cluster IDs for display.

    Args:
        ids: List of cluster IDs (should be sorted)
        max_list: Maximum number of IDs to show before truncating

    Returns:
        Formatted string like "Cluster 7" or "Clusters 2, 7" or "Clusters 1, 2, 3 +5 more"
    """
    ids = sorted(ids)

    if len(ids) == 1:
        return f"Cluster {ids[0]}"

    if len(ids) <= max_list:
        return "Clusters " + ", ".join(str(i) for i in ids)

    head = ", ".join(str(i) for i in ids[:max_list])
    return f"Clusters {head} +{len(ids)-max_list} more"
