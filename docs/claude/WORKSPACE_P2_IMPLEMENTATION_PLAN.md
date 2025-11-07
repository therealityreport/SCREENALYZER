# Workspace Phase 3 P2 - Implementation Plan

**Date**: 2025-11-06
**Status**: Planning → Implementation
**Goal**: Full Faces + Clusters UI with refine operations and dirty tracking

## Overview

Phase 3 P2 builds on the stable P1 foundation (auto-extraction, detect/embed lifecycle) to deliver the complete Workspace experience with:

1. **Cast vs Other Faces** separation
2. **Pairwise Review** for merge candidates
3. **Unassigned clusters** view
4. **Refine Clusters** operation (recenter, eject, merge)
5. **Dirty cluster tracking** for incremental analytics

## Current State (Already Implemented)

From reviewing the codebase:

✅ **Faces Tab** ([app/workspace/faces.py](../../app/workspace/faces.py))
- Person tiles from facebank
- Click to navigate to Clusters filtered by person
- Integration with cluster metrics

✅ **Clusters Tab** ([app/workspace/clusters.py](../../app/workspace/clusters.py))
- "All Clusters" view with filtering
- "Low-Confidence Clusters" view (already exists!)
- Filter by identity, min tracks, min confidence
- Cluster cards with track strips
- Drill-down to cluster detail

✅ **Infrastructure**
- WorkspaceMutator API for mutations
- Cluster/track/person metrics dataframes
- Atomic writes with temp files
- Registry/envelope contract

## What Needs to be Added

### Phase 2A: UI Restructuring (High Priority)

#### 1. Faces Tab - Cast vs Other Sections

**Current**: Single grid of all facebank persons
**Target**: Two sections:

```
┌─────────────────────────────────────────┐
│ 👥 Cast Faces                           │
├─────────────────────────────────────────┤
│ [Person Cards for assigned/confirmed]   │
│ - Show cluster count, confidence        │
│ - Click → view clusters for that person │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 🤷 Other Faces (Excluded from Analytics)│
├─────────────────────────────────────────┤
│ [Cluster Cards for unassigned/uncertain]│
│ - Show as "Unknown", "Unassigned"       │
│ - Excluded from analytics computation   │
└─────────────────────────────────────────┘
```

**Implementation**:
- Split `merged_identities` into cast (has assignments) vs other (unassigned)
- Render two collapsible sections
- Add "Move to Cast" / "Move to Other" actions

**File**: `app/workspace/faces.py`

#### 2. Clusters Tab - Add Unassigned Sub-view

**Current**: "All Clusters", "Low-Confidence Clusters"
**Target**: Add "Unassigned" tab

```python
all_tab, pairwise_tab, low_tab, unassigned_tab = st.tabs([
    "All Clusters",
    "Pairwise Review",  # New!
    "Low-Confidence",
    "Unassigned"        # New!
])
```

**Unassigned Logic**:
```python
unassigned_df = clusters_df[
    (clusters_df["name"].isna()) |
    (clusters_df["name"] == "Unknown") |
    (clusters_df["assignment_conf"] < 0.3)
]
```

**File**: `app/workspace/clusters.py`

### Phase 2B: Pairwise Review (Medium Priority)

#### 3. Pairwise Review Algorithm

**Goal**: Surface cluster pairs that are likely duplicates for manual merge decision

**Candidate Selection Logic**:

```python
def find_pairwise_candidates(
    clusters: list[dict],
    embeddings_df: pd.DataFrame,
    max_centroid_distance: float = 0.35,
    min_silhouette_improvement: float = 0.05
) -> list[tuple[int, int, dict]]:
    """
    Find cluster pairs that are merge candidates.

    Returns:
        List of (cluster_id_1, cluster_id_2, metrics) tuples
    """
    candidates = []

    # Compute centroids for all clusters
    centroids = {}
    for cluster in clusters:
        cluster_id = cluster["cluster_id"]
        tracks = cluster.get("tracks", [])
        if not tracks:
            continue

        # Get embeddings for all tracks in cluster
        track_ids = [t["track_id"] for t in tracks]
        cluster_embeddings = embeddings_df[
            embeddings_df["track_id"].isin(track_ids)
        ]["embedding"].values

        if len(cluster_embeddings) == 0:
            continue

        # Compute centroid
        centroid = np.mean(cluster_embeddings, axis=0)
        centroids[cluster_id] = centroid

    # Find pairs with small centroid distance
    cluster_ids = list(centroids.keys())
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            c1, c2 = cluster_ids[i], cluster_ids[j]

            # Cosine distance
            dist = 1 - cosine_similarity(
                centroids[c1].reshape(1, -1),
                centroids[c2].reshape(1, -1)
            )[0, 0]

            if dist < max_centroid_distance:
                # Check silhouette score improvement
                # (merge clusters and compute new silhouette)
                improvement = estimate_silhouette_improvement(
                    clusters[c1], clusters[c2], embeddings_df
                )

                if improvement > min_silhouette_improvement:
                    candidates.append((c1, c2, {
                        "centroid_distance": dist,
                        "silhouette_improvement": improvement,
                        "cluster_1_size": len(clusters[c1]["tracks"]),
                        "cluster_2_size": len(clusters[c2]["tracks"])
                    }))

    # Sort by silhouette improvement (best candidates first)
    candidates.sort(key=lambda x: x[2]["silhouette_improvement"], reverse=True)

    return candidates
```

**UI**:

```
┌─────────────────────────────────────────┐
│ Pairwise Review: Merge Candidates       │
├─────────────────────────────────────────┤
│ Showing 5 of 12 candidates              │
│                                         │
│ ┌──────────────┐  ┌──────────────┐     │
│ │ Cluster 42   │  │ Cluster 87   │     │
│ │ [Track strip]│  │ [Track strip]│     │
│ │ 15 tracks    │  │ 12 tracks    │     │
│ └──────────────┘  └──────────────┘     │
│                                         │
│ Centroid Distance: 0.28                 │
│ Silhouette Improvement: +0.12           │
│                                         │
│ [Merge These] [Keep Separate] [Skip]   │
├─────────────────────────────────────────┤
│ ... next candidate ...                  │
└─────────────────────────────────────────┘
```

**File**: `app/workspace/pairwise.py` (new)

### Phase 2C: Refine Clusters Operation (High Priority)

#### 4. Refine Clusters Button & Algorithm

**Location**: Faces tab header, Clusters tab header

**Algorithm Steps**:

```python
def refine_clusters(
    episode_id: str,
    data_root: Path,
    min_track_similarity: float = 0.35,
    min_merge_similarity: float = 0.35,
    min_silhouette_improvement: float = 0.05
) -> dict:
    """
    Refine clusters by recomputing centroids, ejecting outliers, and merging duplicates.

    Steps:
    1. Recompute centroids post-edits
    2. Eject tracks with similarity < threshold → move to Unassigned
    3. Merge clusters with centroid distance < threshold if silhouette improves

    Returns:
        {
            "clusters_updated": int,
            "tracks_ejected": int,
            "clusters_merged": int,
            "merge_pairs": list[tuple[int, int]],
            "dirty_clusters": list[int]
        }
    """
    # Load current clusters and embeddings
    clusters = load_clusters(episode_id, data_root)
    embeddings_df = load_embeddings(episode_id, data_root)

    results = {
        "clusters_updated": 0,
        "tracks_ejected": 0,
        "clusters_merged": 0,
        "merge_pairs": [],
        "dirty_clusters": set()
    }

    # Step 1: Recompute centroids
    centroids = {}
    for cluster in clusters["clusters"]:
        cluster_id = cluster["cluster_id"]
        tracks = cluster.get("tracks", [])

        if not tracks:
            continue

        track_ids = [t["track_id"] for t in tracks]
        cluster_embs = embeddings_df[
            embeddings_df["track_id"].isin(track_ids)
        ]["embedding"].values

        if len(cluster_embs) == 0:
            continue

        centroid = np.mean(cluster_embs, axis=0)
        centroids[cluster_id] = centroid

    # Step 2: Eject outliers
    ejected_tracks = []
    for cluster in clusters["clusters"]:
        cluster_id = cluster["cluster_id"]

        if cluster_id not in centroids:
            continue

        centroid = centroids[cluster_id]
        original_tracks = cluster.get("tracks", []).copy()
        kept_tracks = []

        for track in original_tracks:
            track_id = track["track_id"]

            # Get track embedding
            track_emb = embeddings_df[
                embeddings_df["track_id"] == track_id
            ]["embedding"].values

            if len(track_emb) == 0:
                kept_tracks.append(track)
                continue

            # Compute cosine similarity to centroid
            similarity = cosine_similarity(
                track_emb.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0]

            if similarity >= min_track_similarity:
                kept_tracks.append(track)
            else:
                ejected_tracks.append({
                    "track_id": track_id,
                    "from_cluster": cluster_id,
                    "similarity": similarity
                })
                results["tracks_ejected"] += 1

        # Update cluster with kept tracks
        if len(kept_tracks) != len(original_tracks):
            cluster["tracks"] = kept_tracks
            results["dirty_clusters"].add(cluster_id)
            results["clusters_updated"] += 1

    # Step 3: Merge similar clusters
    merge_candidates = find_pairwise_candidates(
        clusters["clusters"],
        embeddings_df,
        max_centroid_distance=min_merge_similarity,
        min_silhouette_improvement=min_silhouette_improvement
    )

    # Auto-merge top candidates (with user approval in UI)
    # For now, just return candidates for manual review
    results["merge_pairs"] = [(c1, c2) for c1, c2, _ in merge_candidates[:10]]

    # Save updated clusters
    if results["dirty_clusters"]:
        # Mark dirty clusters in metadata
        for cluster in clusters["clusters"]:
            if cluster["cluster_id"] in results["dirty_clusters"]:
                cluster["dirty"] = True
                cluster["last_modified"] = datetime.utcnow().isoformat()

        # Save atomically
        save_clusters(clusters, episode_id, data_root)

    # Save ejected tracks to Unassigned cluster
    if ejected_tracks:
        create_or_update_unassigned_cluster(
            ejected_tracks, episode_id, data_root
        )

    results["dirty_clusters"] = list(results["dirty_clusters"])

    return results
```

**UI**:

```
┌─────────────────────────────────────────┐
│ [✨ Refine Clusters]                    │
└─────────────────────────────────────────┘

After clicking:

✅ Refined 12 clusters
   • Ejected 45 outlier tracks → Unassigned
   • Found 3 merge candidates (see Pairwise Review)
   • Marked 12 clusters as dirty for re-analysis
```

**Files**:
- `app/workspace/refine.py` (new)
- `jobs/tasks/refine_clusters.py` (backend task)

### Phase 2D: Dirty Tracking & Incremental Analytics (High Priority)

#### 5. Dirty Cluster Tracking

**Cluster Metadata**:

```python
{
    "cluster_id": 42,
    "name": "Lisa Rinna",
    "tracks": [...],
    "dirty": true,              # New field
    "last_modified": "2025-11-06T19:30:00Z",
    "dirty_reason": "tracks_ejected"  # or "manual_edit", "merge"
}
```

**Analytics Integration**:

```python
def analytics_task_incremental(
    episode_id: str,
    cluster_assignments: dict[int, str],
    only_dirty: bool = True
) -> dict:
    """
    Run analytics on dirty clusters only.

    Steps:
    1. Load all clusters
    2. Filter to dirty=true if only_dirty
    3. Exclude clusters with name="Unknown" or in Other Faces
    4. Recompute intervals for dirty clusters
    5. Merge with existing analytics (preserve clean clusters)
    6. Update totals
    7. Mark clusters as clean (dirty=false)
    """
    # Load clusters
    clusters_data = load_clusters(episode_id)

    if only_dirty:
        dirty_clusters = [
            c for c in clusters_data["clusters"]
            if c.get("dirty", False)
        ]
        logger.info(f"Re-analyzing {len(dirty_clusters)} dirty clusters")
    else:
        dirty_clusters = clusters_data["clusters"]

    # Exclude Other Faces / Unassigned
    analyzable_clusters = [
        c for c in dirty_clusters
        if c.get("name") not in (None, "Unknown", "Unassigned")
        and c.get("assignment_conf", 0) >= 0.3
    ]

    # Compute intervals for dirty clusters
    new_intervals = compute_intervals(analyzable_clusters)

    # Load existing analytics
    existing_analytics = load_analytics(episode_id)

    # Merge: remove intervals from dirty clusters, add new intervals
    dirty_cluster_ids = {c["cluster_id"] for c in dirty_clusters}
    clean_intervals = [
        interval for interval in existing_analytics.get("intervals", [])
        if interval["cluster_id"] not in dirty_cluster_ids
    ]

    all_intervals = clean_intervals + new_intervals

    # Recompute totals
    totals = compute_totals(all_intervals)

    # Save analytics
    analytics_output = {
        "episode_id": episode_id,
        "intervals": all_intervals,
        "totals": totals,
        "last_updated": datetime.utcnow().isoformat(),
        "dirty_clusters_recomputed": len(dirty_clusters)
    }

    save_analytics(analytics_output, episode_id)

    # Mark clusters as clean
    for cluster in clusters_data["clusters"]:
        if cluster["cluster_id"] in dirty_cluster_ids:
            cluster["dirty"] = False
            cluster["last_analyzed"] = datetime.utcnow().isoformat()

    save_clusters(clusters_data, episode_id)

    return {
        "status": "ok",
        "intervals_created": len(all_intervals),
        "dirty_clusters_recomputed": len(dirty_clusters),
        "totals": totals
    }
```

**Analytics Page Button**:

```python
# In app/pages/4_📊_Analytics.py
if is_dirty:
    st.warning("⚠️ Some clusters have been modified since last analysis")
    st.info("Click below to re-analyze only dirty clusters (faster)")

if st.button("5. Screenalyzer Analytics (Re-Analyze Dirty Only)"):
    result = analytics_task_incremental(
        episode_id=selected_episode,
        cluster_assignments=cluster_assignments,
        only_dirty=True
    )

    st.success(f"✅ Re-analyzed {result['dirty_clusters_recomputed']} dirty clusters")
    st.info(f"Generated {result['intervals_created']} total intervals")
```

## Implementation Phases

### Phase 1: UI Restructuring (2-3 hours)
- [ ] Add Cast vs Other sections to Faces tab
- [ ] Add Unassigned sub-view to Clusters tab
- [ ] Update Clusters tab to use 4-tab layout

### Phase 2: Refine Clusters (3-4 hours)
- [ ] Implement centroid recomputation
- [ ] Implement outlier ejection logic
- [ ] Implement Unassigned cluster creation
- [ ] Add Refine button to Faces and Clusters headers
- [ ] Add dirty flag to cluster metadata

### Phase 3: Pairwise Review (3-4 hours)
- [ ] Implement pairwise candidate detection
- [ ] Implement silhouette score estimation
- [ ] Create Pairwise Review UI
- [ ] Add merge confirmation flow

### Phase 4: Incremental Analytics (2-3 hours)
- [ ] Update analytics_task for incremental mode
- [ ] Add dirty tracking to cluster mutations
- [ ] Update Analytics page for dirty-only re-analysis
- [ ] Add dirty status indicator

## Files to Create/Modify

### New Files
- `app/workspace/refine.py` - Refine clusters UI and helpers
- `app/workspace/pairwise.py` - Pairwise review UI
- `jobs/tasks/refine_clusters.py` - Backend refine task

### Modified Files
- `app/workspace/faces.py` - Add Cast vs Other sections
- `app/workspace/clusters.py` - Add Unassigned and Pairwise tabs
- `jobs/tasks/analytics.py` - Add incremental mode
- `app/pages/4_📊_Analytics.py` - Update button for dirty-only
- `app/lib/mutator_api.py` - Add dirty tracking helpers

## Acceptance Criteria

- [ ] Faces tab shows Cast Faces and Other Faces sections
- [ ] Clicking person in Cast Faces opens their clusters
- [ ] Clusters tab has All / Pairwise / Low-Confidence / Unassigned views
- [ ] Pairwise Review shows merge candidates with metrics
- [ ] Refine Clusters button runs and returns summary
- [ ] Ejected tracks appear in Unassigned
- [ ] Dirty clusters are marked in metadata
- [ ] Analytics page re-analyzes only dirty clusters
- [ ] Other Faces and Unassigned excluded from analytics
- [ ] All operations use registry paths (no guessed paths)

## Testing Plan

1. **Upload → Detect → Track → Cluster**
   - Verify clusters appear in Clusters tab
   - Verify persons appear in Faces tab

2. **Refine Clusters**
   - Click Refine button
   - Verify outliers moved to Unassigned
   - Verify dirty flags set
   - Verify summary toast

3. **Pairwise Review**
   - Open Pairwise Review tab
   - Verify candidates shown
   - Merge a pair
   - Verify clusters merged

4. **Incremental Analytics**
   - Edit a cluster (reassign tracks)
   - Verify cluster marked dirty
   - Run Analytics (dirty only)
   - Verify only dirty cluster re-analyzed

5. **Other Faces**
   - Move a cluster to Other Faces
   - Run Analytics
   - Verify cluster excluded from totals

## Next Steps

1. Start with Phase 1 (UI restructuring) - foundation for the rest
2. Implement Phase 2 (Refine Clusters) - highest value feature
3. Add Phase 4 (Incremental Analytics) - enables efficient workflows
4. Finish with Phase 3 (Pairwise Review) - nice-to-have polish

## Notes

- Keep existing functionality intact during additions
- All new operations must use atomic writes (temp → rename)
- Use typed errors: ERR_CLUSTER_NOT_FOUND, ERR_REFINE_FAILED
- Log all operations with [REFINE], [PAIRWISE], [ANALYTICS] prefixes
- Document all new algorithms in code comments
