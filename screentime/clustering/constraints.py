"""
Constraint extraction and enforcement for clustering.

Extracts must-link (ML) and cannot-link (CL) constraints from manual assignments
and enforces them during re-clustering.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConstraintSet:
    """Container for clustering constraints."""
    must_link: List[Tuple[int, int]]  # Pairs of track IDs that must be in same cluster
    cannot_link: List[Tuple[int, int]]  # Pairs of track IDs that cannot be in same cluster
    ml_components: List[Set[int]]  # Connected components from must-link

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'must_link_count': len(self.must_link),
            'cannot_link_count': len(self.cannot_link),
            'ml_components_count': len(self.ml_components),
            'ml_component_sizes': [len(comp) for comp in self.ml_components]
        }


class UnionFind:
    """Union-Find data structure for computing connected components."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def make_set(self, x):
        """Create a new set containing x."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        """Find the root of the set containing x."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

    def get_components(self) -> List[Set[int]]:
        """Get all connected components as sets."""
        components = {}
        for x in self.parent:
            root = self.find(x)
            if root not in components:
                components[root] = set()
            components[root].add(x)
        return list(components.values())


def extract_constraints_from_clusters(clusters_data: dict, audit_log_path: Path = None) -> ConstraintSet:
    """
    Extract must-link and cannot-link constraints from cluster assignments.

    Args:
        clusters_data: Loaded clusters.json data
        audit_log_path: Optional path to audit log for explicit cannot-link pairs

    Returns:
        ConstraintSet with ML and CL pairs
    """
    must_link = []
    cannot_link = []

    # Track assignments: map track_id -> identity_name
    track_to_identity = {}

    # Extract must-link from manual assignments (same identity)
    if 'clusters' in clusters_data:
        for cluster in clusters_data['clusters']:
            identity = cluster.get('name')
            if not identity or identity == 'Unknown':
                continue

            # Check if this is a manual assignment (conf==1.0 or missing conf after manual edit)
            conf = cluster.get('assignment_confidence')
            is_manual = (conf is None) or (conf == 1.0)

            if is_manual:
                track_ids = cluster.get('track_ids', [])

                # Record identity for CL extraction
                for track_id in track_ids:
                    track_to_identity[track_id] = identity

                # All pairs within same cluster are must-link
                for i, tid1 in enumerate(track_ids):
                    for tid2 in track_ids[i+1:]:
                        must_link.append((min(tid1, tid2), max(tid1, tid2)))

    # Extract cannot-link from conflicting assignments (different identities)
    identities = {}
    for track_id, identity in track_to_identity.items():
        if identity not in identities:
            identities[identity] = []
        identities[identity].append(track_id)

    # Tracks in different identities cannot be linked
    identity_list = list(identities.keys())
    for i, id1 in enumerate(identity_list):
        for id2 in identity_list[i+1:]:
            tracks1 = identities[id1]
            tracks2 = identities[id2]

            # Add CL for all cross-identity pairs
            for tid1 in tracks1:
                for tid2 in tracks2:
                    cannot_link.append((min(tid1, tid2), max(tid1, tid2)))

    # TODO: Load explicit CL from audit log (e.g., "Not Same" clicks)
    # if audit_log_path and audit_log_path.exists():
    #     with open(audit_log_path) as f:
    #         for line in f:
    #             entry = json.loads(line)
    #             if entry.get('op') == 'not_same':
    #                 cluster_a_tracks = entry.get('cluster_a_tracks', [])
    #                 cluster_b_tracks = entry.get('cluster_b_tracks', [])
    #                 for tid1 in cluster_a_tracks:
    #                     for tid2 in cluster_b_tracks:
    #                         cannot_link.append((min(tid1, tid2), max(tid1, tid2)))

    # Remove duplicates
    must_link = list(set(must_link))
    cannot_link = list(set(cannot_link))

    # Compute ML connected components using Union-Find
    uf = UnionFind()
    for tid1, tid2 in must_link:
        uf.union(tid1, tid2)

    ml_components = uf.get_components()

    logger.info(f"Extracted constraints: {len(must_link)} ML pairs, {len(cannot_link)} CL pairs, {len(ml_components)} ML components")

    return ConstraintSet(
        must_link=must_link,
        cannot_link=cannot_link,
        ml_components=ml_components
    )


def enforce_constraints_post_clustering(
    clusters: List[dict],
    constraints: ConstraintSet
) -> Tuple[List[dict], Dict]:
    """
    Enforce constraints on clustering results via post-processing repair.

    Strategy:
    - ML: Merge clusters if they contain tracks from same ML component
    - CL: Split clusters if they contain CL violations (keep majority, split minority to Unknown)

    Args:
        clusters: List of cluster dicts with track_ids
        constraints: ConstraintSet with ML/CL pairs

    Returns:
        Tuple of (repaired_clusters, diagnostics)
    """
    diagnostics = {
        'ml_violations_repaired': 0,
        'cl_violations_repaired': 0,
        'clusters_split': 0,
        'clusters_merged': 0
    }

    # Build CL lookup for fast checking
    cl_set = set(constraints.cannot_link)

    # Build ML component lookup: track_id -> component_id
    track_to_ml_comp = {}
    for comp_id, component in enumerate(constraints.ml_components):
        for track_id in component:
            track_to_ml_comp[track_id] = comp_id

    # Step 1: Repair CL violations (split clusters)
    repaired_clusters = []
    next_cluster_id = max([c.get('cluster_id', 0) for c in clusters], default=-1) + 1

    for cluster in clusters:
        track_ids = cluster.get('track_ids', [])

        # Check for CL violations
        violations = []
        for i, tid1 in enumerate(track_ids):
            for tid2 in track_ids[i+1:]:
                pair = (min(tid1, tid2), max(tid1, tid2))
                if pair in cl_set:
                    violations.append((tid1, tid2))

        if not violations:
            # No violations - keep cluster as-is
            repaired_clusters.append(cluster)
        else:
            # Has violations - split cluster
            # Strategy: keep majority identity, move violators to Unknown
            diagnostics['cl_violations_repaired'] += len(violations)
            diagnostics['clusters_split'] += 1

            # Find which tracks to keep (majority identity)
            identity = cluster.get('name', 'Unknown')
            if identity != 'Unknown':
                # Keep all tracks with this identity
                keep_tracks = track_ids.copy()
            else:
                # Keep majority, split rest
                # Simple heuristic: remove tracks involved in violations
                violator_tracks = set()
                for tid1, tid2 in violations:
                    violator_tracks.add(tid1)
                    violator_tracks.add(tid2)

                keep_tracks = [tid for tid in track_ids if tid not in violator_tracks]
                split_tracks = [tid for tid in track_ids if tid in violator_tracks]

                if split_tracks:
                    # Create new Unknown cluster for violators
                    split_cluster = cluster.copy()
                    split_cluster['cluster_id'] = next_cluster_id
                    next_cluster_id += 1
                    split_cluster['track_ids'] = split_tracks
                    split_cluster['size'] = len(split_tracks)
                    split_cluster['name'] = 'Unknown'
                    split_cluster['assignment_confidence'] = 0.0
                    split_cluster['assignment_margin'] = 0.0
                    repaired_clusters.append(split_cluster)

            # Update original cluster with kept tracks
            cluster['track_ids'] = keep_tracks
            cluster['size'] = len(keep_tracks)
            if len(keep_tracks) > 0:
                repaired_clusters.append(cluster)

    # Step 2: Merge clusters with same ML component
    # Build cluster_id -> tracks mapping
    cluster_id_to_tracks = {}
    cluster_id_to_cluster = {}
    for cluster in repaired_clusters:
        cluster_id = cluster['cluster_id']
        cluster_id_to_tracks[cluster_id] = set(cluster.get('track_ids', []))
        cluster_id_to_cluster[cluster_id] = cluster

    # Group clusters by ML component
    ml_comp_to_clusters = {}
    for cluster_id, track_ids in cluster_id_to_tracks.items():
        # Find ML component for this cluster (any track in cluster)
        if not track_ids:
            continue

        # Get ML component of first track (all tracks in cluster should share same component due to ML)
        first_track = next(iter(track_ids))
        ml_comp = track_to_ml_comp.get(first_track)

        if ml_comp is not None:
            if ml_comp not in ml_comp_to_clusters:
                ml_comp_to_clusters[ml_comp] = []
            ml_comp_to_clusters[ml_comp].append(cluster_id)

    # Merge clusters in same ML component
    merged_clusters = []
    merged_cluster_ids = set()

    for ml_comp, cluster_ids in ml_comp_to_clusters.items():
        if len(cluster_ids) < 2:
            continue  # Only one cluster in this ML component

        # Merge all clusters in this ML component
        logger.info(f"Merging {len(cluster_ids)} clusters in ML component {ml_comp}")

        # Pick primary cluster (first one with assigned name, or first overall)
        primary_cluster = None
        for cid in cluster_ids:
            cluster = cluster_id_to_cluster[cid]
            if cluster.get('name') and cluster.get('name') != 'Unknown':
                primary_cluster = cluster
                break

        if primary_cluster is None:
            primary_cluster = cluster_id_to_cluster[cluster_ids[0]]

        # Merge track_ids from all clusters
        all_track_ids = []
        for cid in cluster_ids:
            all_track_ids.extend(cluster_id_to_cluster[cid].get('track_ids', []))
            merged_cluster_ids.add(cid)

        # Update primary cluster with merged tracks
        primary_cluster['track_ids'] = list(set(all_track_ids))  # Remove duplicates
        primary_cluster['size'] = len(primary_cluster['track_ids'])

        # Recompute centroid if available
        if 'centroid' in primary_cluster:
            # Average centroids from all merged clusters (simple approach)
            centroids = []
            for cid in cluster_ids:
                centroid = cluster_id_to_cluster[cid].get('centroid')
                if centroid:
                    import numpy as np
                    centroids.append(np.array(centroid, dtype=np.float32))

            if centroids:
                merged_centroid = np.mean(centroids, axis=0)
                merged_centroid = merged_centroid / np.linalg.norm(merged_centroid)  # Normalize
                primary_cluster['centroid'] = merged_centroid.tolist()

        merged_clusters.append(primary_cluster)
        diagnostics['clusters_merged'] += len(cluster_ids) - 1

    # Add non-merged clusters
    for cluster in repaired_clusters:
        if cluster['cluster_id'] not in merged_cluster_ids:
            merged_clusters.append(cluster)

    logger.info(f"Constraint enforcement: {diagnostics['cl_violations_repaired']} CL violations repaired, {diagnostics['clusters_split']} clusters split, {diagnostics['clusters_merged']} clusters merged")

    return merged_clusters, diagnostics


def save_constraint_diagnostics(episode_id: str, data_root: Path, constraints: ConstraintSet, enforcement_diag: dict, consolidations: dict = None):
    """Save constraint diagnostics to JSON."""
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    output = {
        'episode_id': episode_id,
        'extraction': constraints.to_dict(),
        'enforcement': enforcement_diag
    }

    # Add consolidations if provided
    if consolidations:
        output['same_name_consolidations'] = consolidations

    output_path = diagnostics_dir / "constraints.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Saved constraint diagnostics to {output_path}")


def save_track_level_constraints(episode_id: str, data_root: Path, constraint_info: dict):
    """
    Save track-level constraints from manual splitting.

    Args:
        episode_id: Episode ID
        data_root: Data root path
        constraint_info: Dict with must_link_a, must_link_b, cannot_link lists
    """
    constraints_dir = data_root / "harvest" / episode_id / "diagnostics"
    constraints_dir.mkdir(parents=True, exist_ok=True)

    # Append to constraints log
    constraints_log_path = constraints_dir / "track_constraints.jsonl"

    with open(constraints_log_path, 'a') as f:
        entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'show_id': constraint_info.get('show_id', 'rhobh'),
            'season_id': constraint_info.get('season_id', 's05'),
            'episode_id': episode_id,
            'must_link_a_count': len(constraint_info.get('must_link_a', [])),
            'must_link_b_count': len(constraint_info.get('must_link_b', [])),
            'must_link_moved_count': len(constraint_info.get('must_link_moved', [])),
            'cannot_link_count': len(constraint_info.get('cannot_link', [])),
            'constraints': constraint_info
        }
        f.write(json.dumps(entry) + '\n')

    ml_count = len(constraint_info.get('must_link_moved', [])) or (len(constraint_info.get('must_link_a', [])) + len(constraint_info.get('must_link_b', [])))
    cl_count = len(constraint_info.get('cannot_link', []))
    logger.info(f"Saved track-level constraints: ML={ml_count}, CL={cl_count}")


def consolidate_same_name_clusters(
    clusters_data: dict,
    existing_constraints: ConstraintSet,
    min_similarity: float = 0.75
) -> Tuple[ConstraintSet, dict]:
    """
    Add ML edges between tracks in clusters sharing the same name.

    When multiple clusters have the same identity name with assignment_confidence == 1.0,
    this function adds must-link constraints between all tracks in those clusters,
    causing them to merge during clustering.

    Args:
        clusters_data: Current clusters with names and track_ids
        existing_constraints: Existing ML/CL constraints
        min_similarity: Min centroid similarity to consolidate (optional guard)

    Returns:
        Updated ConstraintSet with consolidation ML edges
        Dict with consolidation stats: {"KIM": 2, "KYLE": 1, ...}
    """
    import numpy as np

    # Group clusters by name (only manual assignments, conf=1.0)
    name_to_clusters = {}
    for cluster in clusters_data.get('clusters', []):
        name = cluster.get('name')
        conf = cluster.get('assignment_confidence', 0.0)

        if name and name != 'Unknown' and conf == 1.0:
            if name not in name_to_clusters:
                name_to_clusters[name] = []
            name_to_clusters[name].append(cluster)

    # Build CL set for fast lookup
    cl_set = set(existing_constraints.cannot_link)

    # Consolidation stats
    consolidations = {}
    new_ml_pairs = list(existing_constraints.must_link)  # Start with existing

    for name, clusters in name_to_clusters.items():
        if len(clusters) < 2:
            continue  # Only one cluster with this name

        logger.info(f"Consolidating {len(clusters)} clusters for identity: {name}")

        # Optional: Check centroid similarity between clusters
        if min_similarity > 0:
            # Compute centroids for each cluster (if available)
            centroids = []
            for cluster in clusters:
                centroid = cluster.get('centroid')
                if centroid:
                    centroids.append(np.array(centroid, dtype=np.float32))
                else:
                    centroids.append(None)

            # Check pairwise similarity
            should_consolidate = True
            for i in range(len(centroids)):
                for j in range(i + 1, len(centroids)):
                    if centroids[i] is not None and centroids[j] is not None:
                        sim = np.dot(centroids[i], centroids[j])
                        if sim < min_similarity:
                            logger.warning(f"Skipping consolidation for {name}: centroid similarity {sim:.3f} < {min_similarity}")
                            should_consolidate = False
                            break
                if not should_consolidate:
                    break

            if not should_consolidate:
                continue

        # Collect all track IDs across these clusters
        all_track_ids = []
        for cluster in clusters:
            all_track_ids.extend(cluster.get('track_ids', []))

        # Add ML edges between all pairs (within this identity)
        # Guard: skip if CL exists between any pair
        added = 0
        for i in range(len(all_track_ids)):
            for j in range(i + 1, len(all_track_ids)):
                tid_a, tid_b = all_track_ids[i], all_track_ids[j]
                pair = (min(tid_a, tid_b), max(tid_a, tid_b))

                # Guard: respect CL
                if pair in cl_set:
                    logger.warning(f"Skipping consolidation for {name}: CL exists between tracks {tid_a} and {tid_b}")
                    continue

                # Add ML edge
                if pair not in new_ml_pairs:
                    new_ml_pairs.append(pair)
                    added += 1

        consolidations[name] = len(clusters)
        logger.info(f"Added {added} ML edges for {name} consolidation")

    # Recompute ML components
    uf = UnionFind()
    for tid_a, tid_b in new_ml_pairs:
        uf.union(tid_a, tid_b)
    ml_components = uf.get_components()

    updated_constraints = ConstraintSet(
        must_link=new_ml_pairs,
        cannot_link=list(existing_constraints.cannot_link),
        ml_components=ml_components
    )

    return updated_constraints, consolidations
