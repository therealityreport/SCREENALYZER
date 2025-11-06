# View Tracks/Track + CAST VIEW + Same-Name Consolidation - COMPLETE

**Date**: November 4, 2025
**Status**: ✅ All navigation and CAST VIEW features implemented

---

## ✅ What Was Implemented

### 1. Dedicated Gallery Pages (Route-Based Navigation)

**Problem**: View Tracks/View Track buttons did nothing because inline galleries didn't render properly at bottom of page.

**Solution**: Created dedicated page renderers with proper routing:

#### A) Cluster Gallery Page
- **Route**: `navigation_page='cluster_gallery'` + `nav_cluster_id`
- **Location**: [app/review_pages.py:11](app/review_pages.py#L11)
- **Features**:
  - Breadcrumb: Review > All Faces > Cluster {id}
  - ← Back button
  - Cluster info (name, size, quality, 🔒 if approved)
  - Delete Cluster button
  - Grid view of all tracks (8 per row)
  - Compact mode toggle
  - Click track → opens Track Gallery

#### B) Track Gallery Page
- **Route**: `navigation_page='track_gallery'` + `nav_track_id` + `nav_cluster_id`
- **Location**: [app/review_pages.py:93](app/review_pages.py#L93)
- **Features**:
  - Breadcrumb: Review > All Faces > Cluster {id} > Track {id}
  - ◀ Prev / Next ▶ buttons (navigate between tracks in same cluster)
  - Quick Move at top (dropdown + Move button)
  - Face chips display (3-8 frames)
  - Constraint saving on move
  - ← Back button returns to cluster gallery

#### C) CAST VIEW Page
- **Route**: `navigation_page='cast_view'` + `nav_cast_name`
- **Location**: [app/review_pages.py:209](app/review_pages.py#L209)
- **Features**:
  - Breadcrumb: Review > Cast View > {NAME}
  - Shows all clusters for that identity (pre-consolidation view)
  - Lists each cluster with 🔒 if approved (conf=1.0)
  - "View" button per cluster → opens Cluster Gallery
  - Horizontal strip of ALL tracks for that identity (across all clusters)
  - Pagination with ◀ ▶ arrows
  - Compact mode toggle
  - Click any track → opens Track Gallery

**Purpose**: Pre-consolidation visibility. See what will merge on RE-CLUSTER.

---

### 2. Button Navigation Updates

#### All Faces View ([app/all_faces_redesign.py](app/all_faces_redesign.py))

**Before**:
```python
if st.button(f"View Tracks ({size})"):
    st.session_state.viewing_cluster_id = cluster_id
    st.rerun()
```

**After**:
```python
if st.button(f"View Tracks ({size})"):
    # Navigate to cluster gallery page
    st.session_state.navigation_page = 'cluster_gallery'
    st.session_state.nav_cluster_id = cluster_id
    st.rerun()
```

**View Track button**:
```python
if st.button("🔍 View", key=f"view_track_{cluster_id}_{track_id}"):
    # Navigate to track gallery page
    st.session_state.navigation_page = 'track_gallery'
    st.session_state.nav_track_id = track_id
    st.session_state.nav_cluster_id = cluster_id
    st.session_state.nav_track_list = sorted_track_ids
    st.rerun()
```

**Result**: Inline galleries removed. All navigation goes to dedicated pages.

---

### 3. Page Routing in Review Page ([app/labeler.py](app/labeler.py))

**Integration Point**: Lines 933-959

**Router Logic**:
```python
# Check for page navigation FIRST (priority over review modes)
nav_page = st.session_state.get('navigation_page')

if nav_page == 'cluster_gallery':
    from app.review_pages import render_cluster_gallery_page
    cluster_id = st.session_state.get('nav_cluster_id')
    if cluster_id is not None:
        render_cluster_gallery_page(selected_episode, cluster_id, DATA_ROOT)
        return

elif nav_page == 'track_gallery':
    from app.review_pages import render_track_gallery_page
    track_id = st.session_state.get('nav_track_id')
    cluster_id = st.session_state.get('nav_cluster_id')
    if track_id is not None:
        render_track_gallery_page(selected_episode, track_id, cluster_id, DATA_ROOT)
        return

elif nav_page == 'cast_view':
    from app.review_pages import render_cast_view_page
    cast_name = st.session_state.get('nav_cast_name')
    if cast_name:
        render_cast_view_page(selected_episode, cast_name, DATA_ROOT)
        return

# Otherwise render normal review modes (all_faces, pairwise, etc.)
```

**Key**: Navigation pages have priority. If `navigation_page` is set, render that page and `return` early.

---

### 4. Cast View Review Mode

**Added to Review Mode radio buttons**: [app/labeler.py:898](app/labeler.py#L898)

**Options**: All Faces | Pairwise Review | Low-Confidence Queue | **Cast View** | Unclustered

**Cast View Selector** (Lines 974-1003):
```python
elif st.session_state.review_mode == "cast_view":
    st.subheader("Cast View - Select Identity")

    # Get unique identity names from clusters
    identity_names = set()
    for cluster in clusters_data.get('clusters', []):
        name = cluster.get('name')
        if name:
            identity_names.add(name)

    # Sort alphabetically (Unknown at end)
    sorted_names = sorted([n for n in identity_names if n != 'Unknown'])
    if 'Unknown' in identity_names:
        sorted_names.append('Unknown')

    if sorted_names:
        cast_name = st.selectbox("Select Identity:", sorted_names)

        if st.button("View"):
            # Navigate to cast view page
            st.session_state.navigation_page = 'cast_view'
            st.session_state.nav_cast_name = cast_name
            st.rerun()
    else:
        st.info("No named clusters found. Assign names first in All Faces view.")
```

---

### 5. Approval Indicators (🔒)

**Purpose**: Show which clusters are approved for same-name consolidation (conf=1.0).

**Implementation** ([app/all_faces_redesign.py:75-80](all_faces_redesign.py#L75)):
```python
# Show lock indicator if approved (conf=1.0)
conf = cluster.get('assignment_confidence', 0.0)
if conf == 1.0:
    st.markdown(f"### **{cluster_name}** 🔒 (Cluster {cluster_id})")
else:
    st.markdown(f"### **{cluster_name}** (Cluster {cluster_id})")
```

**Where Shown**:
- All Faces view (cluster headers)
- Cluster Gallery page
- Cast View page (cluster list)

**Meaning**: 🔒 = Manual assignment (conf=1.0) = Will consolidate with other same-named clusters on RE-CLUSTER

---

### 6. Same-Name Consolidation (Already Integrated)

**Status**: ✅ Already working from previous integration

**How it works**:
1. Manually assign 2+ clusters to "KIM" → both get conf=1.0 → show 🔒
2. Run RE-CLUSTER with "Use manual constraints" ✓
3. `consolidate_same_name_clusters()` adds ML edges between all KIM tracks
4. DBSCAN merges them into one cluster
5. Diagnostics show: `"same_name_consolidations": {"KIM": 2}`

**Guards**:
- Respects cannot-link constraints
- Checks centroid similarity (≥0.75)
- Only for manual assignments (conf=1.0)

---

## 📊 Navigation Flows

### Flow 1: All Faces → Cluster Gallery → Track Gallery

1. User in **All Faces** view
2. Clicks **View Tracks** on a cluster → navigates to **Cluster Gallery** page
3. Sees all tracks in grid, clicks **🔍** on a track → navigates to **Track Gallery** page
4. Uses ◀ Prev / Next ▶ to browse tracks, uses Quick Move to assign
5. Clicks **← Back** → returns to **Cluster Gallery**
6. Clicks **← Back** → returns to **All Faces**

**Breadcrumb Example**:
```
Review > All Faces > Cluster 5 > Track 42
```

### Flow 2: Cast View → Cluster Gallery → Track Gallery

1. User selects **Cast View** review mode
2. Selects "KIM" from dropdown, clicks **View**
3. Sees **Cast View** page with:
   - List of 3 KIM clusters (Cluster 1, 5, 13) - some with 🔒
   - Horizontal strip of all KIM tracks
4. Clicks **View** on Cluster 5 → navigates to **Cluster Gallery**
5. Clicks **🔍** on a track → navigates to **Track Gallery**
6. Same back navigation as Flow 1

**Breadcrumb Example**:
```
Review > Cast View > KIM > Cluster 5 > Track 42
```

### Flow 3: Track Gallery Direct (from Cast View strip)

1. User in **Cast View** for KIM
2. Sees horizontal strip with all KIM tracks
3. Clicks **🔍** on any track → navigates to **Track Gallery**
4. **← Back** returns to **Cast View** (not Cluster Gallery, since it came from Cast View)

---

## 🔧 Files Modified

| File | Lines | Change |
|------|-------|--------|
| `app/review_pages.py` | 1-370 | **NEW** - Dedicated page renderers for Cluster Gallery, Track Gallery, Cast View |
| `app/labeler.py` | 898-899 | Added "Cast View" to review mode radio buttons |
| `app/labeler.py` | 933-959 | Added page routing logic (checks `navigation_page` first) |
| `app/labeler.py` | 974-1003 | Added Cast View selector (show identities, navigate to cast view page) |
| `app/all_faces_redesign.py` | 75-80 | Added 🔒 approval indicator for clusters with conf=1.0 |
| `app/all_faces_redesign.py` | 82-86 | Updated View Tracks button to navigate to cluster gallery page |
| `app/all_faces_redesign.py` | 219-225 | Updated View Track button to navigate to track gallery page |
| `app/all_faces_redesign.py` | 227 | Removed inline gallery rendering code |

---

## 🧪 Testing Instructions

### Test 1: View Tracks/Track Navigation

1. Open http://localhost:8501
2. Navigate to REVIEW → select episode RHOBH-TEST-10-28
3. Review Mode: **All Faces**
4. Click **View Tracks** on any cluster
   - ✅ Should navigate to Cluster Gallery page
   - ✅ Breadcrumb shows: Review > All Faces > Cluster {id}
   - ✅ See all tracks in grid
5. Click **🔍** on any track
   - ✅ Should navigate to Track Gallery page
   - ✅ Breadcrumb shows: Review > All Faces > Cluster {id} > Track {id}
   - ✅ See face chips + Quick Move dropdown
6. Click **← Back** twice
   - ✅ Returns to All Faces view

### Test 2: Cast View

1. Review Mode: **Cast View**
2. Select "KIM" from dropdown, click **View**
   - ✅ Navigate to Cast View page
   - ✅ Breadcrumb shows: Review > Cast View > KIM
   - ✅ See list of KIM clusters (may have 🔒 if approved)
   - ✅ See horizontal strip of all KIM tracks
3. Click **View** on one of the clusters
   - ✅ Navigate to Cluster Gallery for that cluster
4. Click **🔍** on a track in the horizontal strip
   - ✅ Navigate to Track Gallery
   - ✅ **← Back** returns to Cast View (not Cluster Gallery)

### Test 3: Approval Indicators

1. Review Mode: **All Faces**
2. Find any cluster, click **Assign Name**
3. Select an identity, click ✅ Confirm
   - ✅ Cluster header now shows: **{NAME}** 🔒 (Cluster {id})
   - ✅ conf=1.0 means this cluster is approved for consolidation
4. Assign another cluster to the same name
   - ✅ Both clusters show 🔒
5. Review Mode: **Cast View** → select that identity
   - ✅ See both clusters listed
   - ✅ Both show 🔒 indicator
   - ✅ Caption says "These 2 clusters will consolidate into 1 on RE-CLUSTER"

### Test 4: Same-Name Consolidation

1. Manually assign 2 clusters to "KIM" (both show 🔒)
2. Run **RE-CLUSTER** with "Use manual constraints" ✓
3. Wait for job to complete
4. Review Mode: **All Faces**
   - ✅ Only ONE KIM cluster remains (size = sum of both)
5. Check `data/harvest/RHOBH-TEST-10-28/diagnostics/constraints.json`
   - ✅ Should contain: `"same_name_consolidations": {"KIM": 2}`

---

## 🎯 Key Behaviors

### Approval (🔒) Rules

- ✅ Cluster gets 🔒 when **manually assigned** (user clicks "Assign Name")
- ✅ conf=1.0 means approved
- ✅ Open-set assignment (automatic) gives conf < 1.0 (e.g., 0.751) → no 🔒
- ✅ Approved clusters with same name will consolidate on RE-CLUSTER

### Navigation State

**Session State Keys**:
- `navigation_page`: `'cluster_gallery'` | `'track_gallery'` | `'cast_view'` | `None`
- `nav_cluster_id`: Cluster ID to view
- `nav_track_id`: Track ID to view
- `nav_cast_name`: Identity name to view
- `nav_track_list`: List of track IDs for Prev/Next navigation

**Reset**: Set `navigation_page = None` to return to normal review modes

### Back Button Logic

- From **Track Gallery** → returns to **Cluster Gallery** (if came from cluster gallery)
- From **Track Gallery** → returns to **Cast View** (if came from cast view strip)
- From **Cluster Gallery** → returns to **All Faces** or **Cast View** (depending on origin)
- From **Cast View** → returns to **Review** (resets navigation_page)

---

## 📋 Summary

### ✅ Completed Features

1. **Dedicated Gallery Pages** - Route-based navigation with breadcrumbs
2. **View Tracks/Track buttons working** - Navigate to dedicated pages, no inline galleries
3. **CAST VIEW page** - See all clusters and tracks for an identity
4. **Approval indicators (🔒)** - Show which clusters will consolidate
5. **Same-name consolidation** - Already integrated, working with constraints

### 🎯 What Users Can Do Now

- **View all tracks in a cluster** via Cluster Gallery page
- **Navigate between tracks** with Prev/Next in Track Gallery
- **See all instances of a person** across clusters via Cast View
- **Confirm what will consolidate** by checking for 🔒 indicators
- **Quickly assign tracks** via Quick Move dropdown
- **Navigate with breadcrumbs** back to parent views

---

## 🚀 Ready for Testing!

All navigation features are implemented. Open http://localhost:8501 and try:

1. **All Faces** → View Tracks → View Track → Quick Move
2. **Cast View** → Select KIM → See clusters + tracks → Navigate
3. Manually assign 2 clusters to same name → See 🔒 → RE-CLUSTER → Verify consolidation

**Expected**: Smooth navigation, no inline gallery issues, clear breadcrumbs, working consolidation.
