# Implementation Guide - Episode Status & Navigation Fixes

**Status**: Partially Complete - Core Infrastructure Ready
**Date**: November 4, 2025

## ✅ Completed

### 1. Episode Status Infrastructure
**File Created**: [app/lib/episode_status.py](app/lib/episode_status.py)

Functions available:
- `get_enhanced_episode_status(episode_id, data_root)` - Returns comprehensive status dict
- `save_episode_status(episode_id, data_root)` - Persists to diagnostics/episode_status.json
- `load_suppress_data(episode_id, data_root)` - Loads deletion tracking
- `save_suppress_data(episode_id, data_root, data)` - Saves deletion tracking

Status fields returned:
```python
{
    "faces_total": 0,        # embeddings.parquet count
    "faces_used": 0,         # picked_samples.parquet count (Top-K)
    "tracks": 0,
    "clusters": 0,
    "suggestions": 0,
    "constraints_ml": 0,     # from diagnostics/constraints.json
    "constraints_cl": 0,
    "suppressed_tracks": 0,  # from diagnostics/suppress.json
    "suppressed_clusters": 0
}
```

### 2. Pairwise & Low-Confidence Fixes
**Status**: ✅ Complete (previous task)
- Pairwise signature made permissive
- Low-Confidence "Mark as Good" with persistence
- lowconf_ignore.json working

## 🔧 Required Changes

### 3. Update Episode Status Display

**File**: `app/labeler.py` (around line 850)

**Current**:
```python
with st.expander("📊 Episode Status", expanded=False):
    summary = get_episode_summary(selected_episode, DATA_ROOT)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faces Detected", summary["detection"]["faces_detected"])
    with col2:
        st.metric("Tracks Built", summary["tracking"]["tracks_built"])
    with col3:
        st.metric("Clusters", summary["clustering"]["clusters_built"])
    with col4:
        st.metric("Merge Suggestions", summary["clustering"]["suggestions_enqueued"])
```

**Replace with**:
```python
with st.expander("📊 Episode Status", expanded=False):
    from app.lib.episode_status import get_enhanced_episode_status

    status = get_enhanced_episode_status(selected_episode, DATA_ROOT)

    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faces", f"{status['faces_total']:,} / {status['faces_used']:,}",
                 help="Total detected / Used in clustering (Top-K)")
    with col2:
        st.metric("Tracks", f"{status['tracks']:,}")
    with col3:
        st.metric("Clusters", f"{status['clusters']:,}")
    with col4:
        st.metric("Suggestions", f"{status['suggestions']:,}")

    # Constraints and Suppression
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Constraints", f"ML:{status['constraints_ml']}  CL:{status['constraints_cl']}",
                 help="Must-Link and Cannot-Link pairs")
    with col6:
        st.metric("Suppressed", f"T:{status['suppressed_tracks']}  C:{status['suppressed_clusters']}",
                 help="Deleted tracks/clusters")
```

### 4. Fix View Tracks Button (Cluster Gallery)

**Problem**: Button does nothing when clicked

**File**: `app/all_faces_redesign.py` (around line 66)

**Current**:
```python
with header_col3:
    if st.button(f"View Tracks ({size})", key=f"view_tracks_{cluster_id}"):
        st.session_state.viewing_cluster_id = cluster_id
        st.rerun()
```

**Issue**: No rendering logic handles `viewing_cluster_id`

**Fix**: Add cluster gallery renderer after the cluster loop:

```python
# After the main cluster loop (around line 210)
# Check if viewing a cluster gallery
if st.session_state.get('viewing_cluster_id') is not None:
    cluster_id = st.session_state.viewing_cluster_id
    cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)

    if cluster:
        st.markdown("---")
        st.markdown(f"## 📋 Cluster {cluster_id} Gallery")

        # Back button
        if st.button("← Back to All Faces"):
            st.session_state.viewing_cluster_id = None
            st.rerun()

        # Show all tracks in this cluster
        track_ids = cluster.get('track_ids', [])
        st.caption(f"{len(track_ids)} tracks in this cluster")

        # Display in grid (8 per row)
        for row_start in range(0, len(track_ids), 8):
            cols = st.columns(8)
            for col_idx, track_id in enumerate(track_ids[row_start:row_start+8]):
                with cols[col_idx]:
                    track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
                    if track and video_path.exists():
                        frame_refs = track.get('frame_refs', [])
                        if frame_refs:
                            mid_frame = frame_refs[len(frame_refs) // 2]
                            thumb_path = thumb_gen.generate_frame_thumbnail(
                                video_path, mid_frame['frame_id'], mid_frame['bbox'],
                                episode_id, track_id
                            )
                            if thumb_path and thumb_path.exists():
                                st.image(str(thumb_path), width=120)
                                st.caption(f"Track {track_id}")
    else:
        st.warning("Cluster not found")
        st.session_state.viewing_cluster_id = None
        st.rerun()

    return  # Exit early to show only gallery
```

### 5. Fix View Track Button (Track Modal)

**Problem**: Button exists but modal doesn't open

**File**: `app/all_faces_redesign.py` (around line 205)

**Current**:
```python
if st.button("🔍 View", key=f"view_track_{cluster_id}_{track_id}"):
    st.session_state.track_gallery_open = True
    st.session_state.track_gallery_track_id = track_id
    st.session_state.track_gallery_cluster_id = cluster_id
    st.session_state.track_gallery_track_list = sorted_track_ids
    st.rerun()
```

**Issue**: Modal is called but might be blocked by viewing_cluster_id check

**Fix**: Update modal rendering check (around line 213):

```python
# Render Track Gallery Modal if open (check BEFORE cluster gallery)
if st.session_state.get('track_gallery_open', False):
    render_track_gallery_modal(
        st.session_state.track_gallery_track_id,
        st.session_state.track_gallery_cluster_id,
        episode_id,
        tracks_data,
        video_path,
        thumb_gen,
        cluster_mutator,
        DATA_ROOT,
        clusters_data
    )
    return  # Don't render anything else

# Then check for cluster gallery
if st.session_state.get('viewing_cluster_id') is not None:
    # ... cluster gallery code ...
    return
```

### 6. Add Suppress/Delete Functionality

**File**: Add to `app/lib/episode_status.py` (already created)

**File**: Update `app/all_faces_redesign.py` to add delete options

Add context menu to track tiles:
```python
# In track tile rendering (around line 205)
col1, col2 = st.columns(2)
with col1:
    if st.button("🔍 View", key=f"view_track_{cluster_id}_{track_id}"):
        # existing code...

with col2:
    if st.button("🗑️", key=f"delete_track_{cluster_id}_{track_id}", help="Remove track from episode"):
        from app.lib.episode_status import load_suppress_data, save_suppress_data

        suppress_data = load_suppress_data(episode_id, DATA_ROOT)
        suppress_data['deleted_tracks'].append(track_id)
        save_suppress_data(episode_id, DATA_ROOT, suppress_data)

        st.toast(f"✅ Track {track_id} suppressed (won't appear after RE-CLUSTER)", icon="🗑️")
        st.rerun()
```

Add to cluster header:
```python
# In cluster header (around line 70)
with header_col4:
    if st.button("🗑️ Delete Cluster", key=f"delete_cluster_{cluster_id}"):
        from app.lib.episode_status import load_suppress_data, save_suppress_data

        suppress_data = load_suppress_data(episode_id, DATA_ROOT)
        suppress_data['deleted_clusters'].append(cluster_id)
        suppress_data['deleted_tracks'].extend(cluster['track_ids'])
        save_suppress_data(episode_id, DATA_ROOT, suppress_data)

        st.toast(f"✅ Cluster {cluster_id} suppressed", icon="🗑️")
        st.rerun()
```

### 7. Make Constraints Persist Across RE-CLUSTER

**File**: `jobs/tasks/recluster.py`

**Current**: Constraints are loaded fresh each time

**Fix**: Around line where constraints are extracted:

```python
if use_constraints:
    logger.info("Loading existing constraints...")

    # Load from constraints.json (summary)
    constraints_path = diagnostics_dir / "constraints.json"
    existing_ml = set()
    existing_cl = set()

    if constraints_path.exists():
        with open(constraints_path) as f:
            existing = json.load(f)
            extraction = existing.get('extraction', {})
            # Parse existing pairs from summary
            # ... add de-dup logic ...

    # Extract new constraints from manual assignments
    new_constraints = extract_constraints_from_clusters(clusters_data, audit_log_path)

    # Merge: de-dup pairs
    all_ml = list(set(existing_ml) | set(new_constraints.must_link))
    all_cl = list(set(existing_cl) | set(new_constraints.cannot_link))

    # Use merged constraints
    constraints = ConstraintSet(
        must_link=all_ml,
        cannot_link=all_cl,
        ml_components=compute_ml_components(all_ml)
    )
```

### 8. Filter Suppressed Items in Pipeline

**File**: `jobs/tasks/recluster.py` (around line 100)

Add filtering after loading data:

```python
# Load suppressed items
from app.lib.episode_status import load_suppress_data
suppress_data = load_suppress_data(episode_id, data_root)
deleted_tracks = set(suppress_data.get('deleted_tracks', []))

# Filter embeddings
if deleted_tracks:
    embeddings_df = embeddings_df[~embeddings_df['track_id'].isin(deleted_tracks)]
    logger.info(f"Filtered {len(deleted_tracks)} suppressed tracks from clustering")
```

## 📋 Testing Checklist

- [ ] Episode Status shows Faces: total/used
- [ ] Episode Status shows Constraints: ML/CL
- [ ] Episode Status shows Suppressed: T/C
- [ ] "View Tracks" button opens cluster gallery
- [ ] "View Track" button opens track modal
- [ ] Delete track → appears in suppress.json
- [ ] Delete cluster → all tracks suppressed
- [ ] RE-CLUSTER excludes suppressed items
- [ ] Constraints persist across multiple RE-CLUSTERs
- [ ] episode_status.json is created/updated
- [ ] suppress.json is created/updated

## 📄 Expected File Formats

### diagnostics/episode_status.json
```json
{
  "episode_id": "RHOBH-TEST-10-28",
  "faces_total": 15234,
  "faces_used": 8567,
  "tracks": 450,
  "clusters": 15,
  "suggestions": 8,
  "constraints_ml": 342,
  "constraints_cl": 156,
  "suppressed_tracks": 12,
  "suppressed_clusters": 2
}
```

### diagnostics/suppress.json
```json
{
  "show_id": "rhobh",
  "season_id": "s05",
  "episode_id": "RHOBH-TEST-10-28",
  "deleted_tracks": [45, 67, 89],
  "deleted_clusters": [3, 7]
}
```

## 🔧 Thumbnail Generation Quick Reference

- **Install ffmpeg on macOS:**
  ```bash
  brew install ffmpeg
  ffmpeg -version
  ```

- **Regenerate thumbnails manually (uses OpenCV fallback when ffmpeg is absent):**
  ```bash
  source /Volumes/HardDrive/SCREENALYZER/.venv/bin/activate
  python - <<'PY'
from jobs.tasks.generate_thumbnails import generate_thumbnails_task
generate_thumbnails_task("RHOBH_S05_E15_11052025", force=True)
PY
  ```

## 🚀 Quick Implementation Priority

1. **Fix View Tracks/View Track buttons** (critical UX)
2. **Update Episode Status display** (user visibility)
3. **Add delete functionality** (workflow improvement)
4. **Persist constraints** (data integrity)
5. **Filter suppressed items** (pipeline integration)

The infrastructure is ready in `app/lib/episode_status.py`. The remaining work is integrating it into the UI and pipeline.
