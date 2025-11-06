# Implementation Status - Episode Pipeline

**Date**: 2025-11-05
**Status**: Core Features Complete, UI Polish Remaining

---

## ✅ **COMPLETED - Core Infrastructure**

### 1. **Confidence Scoring System** (COMPLETE)
- **Module**: `screentime/clustering/confidence_scoring.py`
- **Integration**: `jobs/tasks/recluster.py` lines 376-408
- **Features**:
  - Frame-level confidence: `id_sim(frame_embedding, identity_prototypes)`
  - Track metrics: `track_conf_p25` (robust 25th percentile), mean, min, n_frames_low
  - Saved to `clusters.json` with full `track_metrics` array
  - Tracks sorted by p25 DESC (highest confidence first)
  - Updates `picked_samples.parquet` with `frame_conf` column

### 2. **Same-Name Cluster Consolidation** (COMPLETE)
- **File**: `screentime/clustering/constraints.py` lines 269-351
- **Features**:
  - Physical cluster merging via ML component detection
  - Triggered by `assignment_confidence == 1.0` (lock signal)
  - Guards: cannot-link wins, centroid sim ≥ 0.75
  - Post-merge: one physical cluster per approved identity
  - Logs consolidations to `diagnostics/constraints.json`

### 3. **Analytics Dirty Flag System** (COMPLETE)
- **Module**: `app/lib/analytics_dirty.py`
- **Features**:
  - `mark_analytics_dirty()` - sets flag on edits/re-cluster
  - `clear_analytics_dirty()` - clears after successful analytics rebuild
  - `get_analytics_freshness()` - returns timestamp-based freshness status
  - Integration: recluster marks dirty, analytics task clears dirty

### 4. **Analytics Always Rebuilds** (COMPLETE)
- **File**: `jobs/tasks/analytics.py` lines 53-84, 234-249
- **Features**:
  - Explicitly logs "REBUILDING from current clusters.json (no cache)"
  - Loads `suppress.json` and filters deleted tracks/clusters
  - Clears dirty flag on success
  - Returns suppression counts in result

### 5. **All Faces View - Complete Redesign** (COMPLETE)
- **File**: `app/all_faces_redesign.py`
- **Module**: `app/lib/cluster_filtering.py`
- **Features**:
  - ✅ **Suppression filtering**: Filters deleted_tracks/deleted_clusters live
  - ✅ **Group by identity toggle**: Collapses same-name clusters into virtual rows
  - ✅ **DELETE works**: Writes suppress.json + marks analytics dirty → row disappears
  - ✅ **Lock on assign**: Sets `assignment_confidence=1.0` on name assignment
  - ✅ **Grouped row handling**: Navigate to Cast View, disable DELETE/Assign
  - ✅ **Confidence badges**: Shows p25 on track tiles

### 6. **Frame Selection Optimization** (COMPLETE)
- **File**: `app/review_pages.py`
- **Features**:
  - Pagination: 20-24 frames per page
  - Form-based selection: No rerun until "✓ Update Selection" clicked
  - Fixes lag issues in Track Gallery, Cluster Gallery, Cast View
  - Split and Delete frame-level operations functional

### 7. **Stale Banner Fix** (COMPLETE)
- **File**: `app/labeler.py` lines 836-881
- **Features**:
  - Queries actual RQ job status from Redis
  - Auto-removes finished/failed/not-found jobs
  - Banner clears automatically when job completes

---

## 🚧 **REMAINING - UI Polish** (Priority Order)

### A. Confidence Badges in Remaining Views
**Status**: Partial (done in All Faces, needs Cluster Gallery + Cast View)

**What's Needed**:
1. Add p25 badge to Cluster Gallery track rows
2. Add p25 badge to Cast View track rows
3. Add sort dropdown (p25/mean/min) + "low confidence first" toggle

**Files to Update**:
- `app/review_pages.py` - Cluster Gallery rendering
- `app/review_pages.py` - Cast View rendering

### B. Cast View Dense Browser
**Status**: Partial (has row-per-track, needs 75×75 + full frames)

**What's Needed**:
1. Show ALL frames (not paginated) at 75×75 size
2. Add compact 60×60 toggle
3. Row header chip: `Track X · p25=0.88 · frames=42 · low(<0.55)=2`
4. Click tile → Track Gallery detail

**Files to Update**:
- `app/review_pages.py` - `render_cast_view_page()`

### C. Unclustered Faces Live Updates
**Status**: Not Started

**What's Needed**:
1. Create `build_visible_unclustered()` filter function
2. Apply suppression filter (same as All Faces)
3. Rebuild datasource after Assign/Move/Split/Delete/Suppress
4. Rebuild after Analyze completes

**Files to Update**:
- Create `app/lib/unclustered_filtering.py` (mirror cluster_filtering)
- Update unclustered view in `app/labeler.py`

### D. Analytics Freshness Chip
**Status**: Not Started

**What's Needed**:
1. Episode Status: Show "Analytics: Fresh ✓ (timestamp)" or "Analytics: Stale"
2. Check `diagnostics/needs_analytics.flag` existence
3. Show "Rebuild Analytics" button when stale
4. Auto-hide banner after Analyze completes

**Files to Update**:
- `app/labeler.py` - Episode Status panel (around line 858-900)

### E. Actions Counters in Episode Status
**Status**: Not Started

**What's Needed**:
1. Read `diagnostics/contamination_audit.json` → Precision splits count
2. Read `diagnostics/recall_stats.json` → Recall windows, secs_recovered
3. Display: `Precision splits:<X> · Recall windows:<Y> · secs_recovered:<Z.s>`
4. Deep links:
   - Unknown count → `/review/cast?cast=Unknown`
   - Low confidence → filtered list (p25 ascending)
   - Suppressed → restore modal

**Files to Update**:
- `app/labeler.py` - Episode Status panel
- `app/lib/episode_status.py` - Add actions parsing helper

---

## 📊 **Current System State**

### **What Works End-to-End**:
1. RE-CLUSTER (constrained) → Scores confidence + consolidates same-name clusters
2. DELETE cluster → Suppresses + marks analytics dirty → row disappears
3. Assign Name → Sets conf=1.0 (🔒) + marks analytics dirty
4. Group by identity → Shows one row per person
5. Analytics → Always rebuilds from current state + filters suppression
6. Frame selection → Fast (form-based, paginated)

### **What's Ready for Validation**:
- Core pipeline: re-cluster → score → consolidate → suppress → analyze
- All Faces: delete, assign, group, filter
- Confidence metrics: frame_conf, track_conf_p25/mean/min/n_low
- Dirty flag system: marks/clears automatically

### **What Needs UI Polish Before Final Validation**:
- Confidence badges in Cluster Gallery/Cast View
- Cast View 75×75 dense layout
- Unclustered live updates
- Analytics freshness chip
- Actions counters

---

## 🎯 **Validation Checklist**

### Pre-Validation Setup:
- [x] Confidence scoring integrated
- [x] Same-name consolidation working
- [x] Suppression filtering active
- [x] DELETE button functional
- [x] Analytics dirty flag system
- [x] Lock (conf=1.0) on assign
- [ ] All views show p25 badges (partial)
- [ ] Cast View shows 75×75 frames (partial)
- [ ] Unclustered filters live (not started)
- [ ] Analytics freshness chip (not started)

### Validation Run Steps:
1. RE-CLUSTER (constrained) with manual constraints ON
2. Verify one cluster per approved identity (check consolidations)
3. DELETE two background clusters → verify suppression
4. Assign Name → verify lock (conf=1.0)
5. Toggle "Group by identity" → verify one row per person
6. Analyze → verify dirty flag cleared, timestamps newer
7. Check artifacts (see below)

### Expected Artifacts:
```
data/harvest/<ep>/clusters.json              # track_metrics with p25/mean/min/n_low
data/harvest/<ep>/picked_samples.parquet     # frame_conf column
data/harvest/<ep>/diagnostics/constraints.json  # same_name_consolidations
data/harvest/<ep>/diagnostics/suppress.json  # deleted_clusters, deleted_tracks
data/harvest/<ep>/diagnostics/episode_status.json  # updated stats
data/harvest/<ep>/diagnostics/contamination_audit.json  # precision splits
data/harvest/<ep>/diagnostics/recall_stats.json  # recall windows
data/outputs/<ep>/timeline.csv               # newer than clusters.json
data/outputs/<ep>/totals.csv                 # newer than clusters.json
```

---

## 🚀 **Recommended Path Forward**

### Option 1: Validate Core Now (Recommended)
**Rationale**: Core pipeline is complete and functional. UI polish is cosmetic.

**Steps**:
1. Run validation with current state
2. Generate artifacts
3. Verify consolidation, suppression, confidence scoring
4. Add UI polish in follow-up session

### Option 2: Complete UI First
**Rationale**: Ship everything in one validation pass.

**Steps**:
1. Add remaining badges (2-3 hours)
2. Fix Cast View layout (1 hour)
3. Add Unclustered filtering (1 hour)
4. Add freshness chip + Actions (2 hours)
5. Then run full validation

---

## 📝 **Implementation Notes**

### Key Design Decisions:
- **Identity-agnostic**: No per-person overrides, global thresholds only
- **Suppression-aware**: Never modify clusters.json for deletes, use suppress.json
- **Lock signal**: `assignment_confidence=1.0` triggers consolidation
- **Dirty flag**: Marks when analytics need rebuild, clears on success
- **Form-based selection**: Prevents lag from checkbox reruns

### File Organization:
- Core logic: `screentime/clustering/*.py`
- UI helpers: `app/lib/*.py`
- Views: `app/*.py`
- Tasks: `jobs/tasks/*.py`

### Next Session Priorities:
1. Complete confidence badge rollout
2. Fix Cast View 75×75 layout
3. Add Unclustered live filtering
4. Add Analytics freshness chip
5. Run final validation with full artifacts

---

**Last Updated**: 2025-11-05 01:40 UTC
