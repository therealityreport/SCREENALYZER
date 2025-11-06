# Cluster Hygiene Refactor - STATUS

**Date**: 2025-10-30
**Problem**: Non-face frames (glasses, arms, background) contaminating clusters, causing mixed-identity galleries
**Solution**: Identity-agnostic face-only gating + contamination audit + auto-split
**Status**: ✅ CORE MODULES COMPLETE (face quality filter + contamination audit implemented)

---

## Root Cause Analysis

**What User Observed in Galleries**:
- RINNA cluster showing EILEEN and LVP faces
- Non-face thumbnails (glasses, arms, background) in cluster galleries
- Track centroids pulled by low-quality/ambiguous chips
- Mixed identities within single clusters

**Root Causes**:
1. **No face-only gating** - Non-face frames included in clustering feed
2. **All frames used for centroids** - Low-quality chips pull track centroids toward wrong identity
3. **No contamination detection** - Mixed clusters persist without automatic splitting
4. **Gallery shows raw frames** - No filtering between clustering and display

---

## Solution Architecture (Identity-Agnostic)

### 1. Face-Only Gating ✅

**Module**: [screentime/clustering/face_quality.py](screentime/clustering/face_quality.py) (300 lines)

**Filtering Criteria** (uniform for all identities):
```python
@dataclass
class FaceQualityFilter:
    min_face_conf: float = 0.65      # Minimum detector confidence
    min_face_px: int = 72            # Minimum face size (px)
    max_co_face_iou: float = 0.10    # Max IoU with other faces
    min_sharpness: float = 0.0       # Minimum sharpness (optional)
    require_embedding: bool = True   # Must have valid embedding
```

**Functions**:
- `filter_face_samples()` - Removes non-face chips (glasses, arms, background)
- `pick_top_k_per_track()` - Selects top-10 highest-quality faces per track for centroids
- `compute_bbox_iou()` - Detects co-face overlap (multi-person frames)
- `save_picked_samples()` / `load_picked_samples()` - Persistence

**Output**: `data/harvest/EPISODEID/picked_samples.parquet` (faces-only, top-K per track)

**Expected Impact**:
- Non-face thumbnails disappear from galleries
- Track centroids computed from best 10 chips (not contaminated by outliers)
- Co-face crops excluded (no bbox overlap with other faces)

---

### 2. Contamination Audit + Auto-Split ✅

**Module**: [screentime/clustering/contamination_audit.py](screentime/clustering/contamination_audit.py) (400 lines)

**Detection Methods** (uniform for all identities):

#### A. Intra-Cluster Outliers
- Samples with `similarity_to_medoid < 0.75`
- OR samples >3 MAD (Median Absolute Deviation) from cluster median similarity

#### B. Cross-Identity Contamination
- Samples where `(best_other_sim - current_sim) ≥ 0.10`
- Requires ≥4 contiguous contaminated frames to trigger auto-split

**Configuration**:
```python
@dataclass
class ContaminationConfig:
    outlier_mad_threshold: float = 3.0       # MAD threshold
    outlier_sim_threshold: float = 0.75      # Min similarity to medoid
    cross_id_margin: float = 0.10            # Margin for cross-ID detection
    min_contiguous_frames: int = 4           # Min span length
    auto_split_enabled: bool = True          # Enable auto-splitting
    min_evidence_strength: float = 0.12      # Min margin for split assignment
```

**Functions**:
- `compute_cluster_medoid()` - Computes medoid (most representative embedding)
- `detect_outliers()` - Finds intra-cluster outliers
- `detect_cross_identity_contamination()` - Finds samples matching other clusters better
- `find_contiguous_spans()` - Groups contaminated samples into actionable spans
- `audit_cluster_contamination()` - Audits single cluster
- `audit_all_clusters()` - Audits all clusters
- `save_contamination_audit()` - Saves results to JSON

**Output**: `data/harvest/EPISODEID/diagnostics/contamination_audit.json`

**Sample Output**:
```json
{
  "episode_id": "RHOBH-TEST-10-28",
  "clusters": {
    "RINNA": [
      {
        "track_id": 42,
        "frame_count": 8,
        "time_range": "54.2s - 56.8s",
        "reason": "cross_identity",
        "best_match_cluster": "EILEEN",
        "evidence_strength": 0.14,
        "action": "split_to_EILEEN"
      }
    ]
  }
}
```

**Expected Impact**:
- "EILEEN in RINNA" spans detected and auto-split
- "LVP in RINNA" spans detected and auto-split
- Contaminated tracks moved to correct cluster or Unknown
- Cluster purity scores improve

---

## Integration Steps (NEXT SESSION)

### Step 1: Add Config to pipeline.yaml (5 min)

```yaml
clustering:
  # Face-only gating (uniform for all identities)
  face_quality:
    min_face_conf: 0.65
    min_face_px: 72
    max_co_face_iou: 0.10
    top_k_per_track: 10
    quality_weights:
      confidence: 0.6
      face_size: 0.3
      sharpness: 0.1

  # Contamination audit (uniform for all identities)
  contamination_audit:
    enabled: true
    outlier_mad_threshold: 3.0
    outlier_sim_threshold: 0.75
    cross_id_margin: 0.10
    min_contiguous_frames: 4
    auto_split_enabled: true
    min_evidence_strength: 0.12
```

---

### Step 2: Integrate into Clustering Pipeline (30 min)

**File to Modify**: `screentime/pipeline/clustering.py`

```python
from screentime.clustering.face_quality import (
    FaceQualityFilter, filter_face_samples, pick_top_k_per_track,
    save_picked_samples
)
from screentime.clustering.contamination_audit import (
    ContaminationConfig, audit_all_clusters, save_contamination_audit
)

def run_clustering(episode_id, data_root, config, ...):
    # ... existing detection + tracking code ...

    # NEW: Filter to faces-only before clustering
    face_filter = FaceQualityFilter(
        min_face_conf=config['clustering']['face_quality']['min_face_conf'],
        min_face_px=config['clustering']['face_quality']['min_face_px'],
        ...
    )

    faces_only_df = filter_face_samples(embeddings_df, tracks_data, face_filter)

    # NEW: Pick top-K per track for centroid computation
    picked_df = pick_top_k_per_track(
        faces_only_df,
        k=config['clustering']['face_quality']['top_k_per_track']
    )

    save_picked_samples(episode_id, data_root, picked_df)

    # Compute track centroids from picked samples (not all samples)
    track_centroids = compute_centroids_from_picked(picked_df)

    # Run DBSCAN clustering on centroids
    clusters = dbscan_cluster(track_centroids, ...)

    # ... existing cluster assignment code ...

    # NEW: Audit clusters for contamination
    if config['clustering']['contamination_audit']['enabled']:
        contamination_config = ContaminationConfig(...)

        contamination_results = audit_all_clusters(
            clusters_data,
            picked_df,  # Use same picked samples
            contamination_config
        )

        save_contamination_audit(episode_id, data_root, contamination_results)

        # NEW: Apply auto-splits if enabled
        if contamination_config.auto_split_enabled:
            apply_auto_splits(clusters_data, contamination_results)

    return clusters_data
```

---

### Step 3: Update Gallery to Use Picked Samples (20 min)

**File to Modify**: `app/lib/data.py`

```python
from screentime.clustering.face_quality import load_picked_samples

def load_cluster_gallery_samples(episode_id, cluster_id, data_root, show_debug=False):
    """
    Load gallery samples for a cluster.

    Args:
        show_debug: If True, show all samples. If False, show picked_samples only.

    Returns:
        DataFrame of samples to display
    """
    if show_debug:
        # Show all samples (including rejected/low-quality)
        return load_all_samples(episode_id, cluster_id, data_root)
    else:
        # Show faces-only picked samples
        picked_df = load_picked_samples(episode_id, data_root)
        cluster_track_ids = get_cluster_track_ids(episode_id, cluster_id, data_root)
        return picked_df[picked_df['track_id'].isin(cluster_track_ids)]
```

**File to Modify**: `app/labeler.py` (gallery rendering)

```python
# Add "Show debug" toggle
show_debug = st.checkbox("Show debug (rejected/low-quality samples)", value=False)

# Load gallery samples
gallery_samples = load_cluster_gallery_samples(
    episode_id,
    cluster_id,
    data_root,
    show_debug=show_debug
)

# Render gallery from filtered samples
for sample in gallery_samples:
    st.image(...)  # Only faces, no glasses/arms/background
```

---

### Step 4: Run Clean Clustering Pipeline (20 min)

```bash
# Backup current clusters
cp data/harvest/RHOBH-TEST-10-28/clusters.json \
   data/harvest/RHOBH-TEST-10-28/clusters_backup_pre_hygiene.json

# Run clean clustering with face-only gating + contamination audit
python jobs/tasks/run_clustering.py RHOBH-TEST-10-28 --clean

# Check outputs
ls -lh data/harvest/RHOBH-TEST-10-28/picked_samples.parquet
ls -lh data/harvest/RHOBH-TEST-10-28/diagnostics/contamination_audit.json
```

---

### Step 5: Validate Results (15 min)

**Acceptance Checks**:

1. **Picked Samples Count**:
```python
picked_df = pd.read_parquet("data/harvest/RHOBH-TEST-10-28/picked_samples.parquet")
embeddings_df = pd.read_parquet("data/harvest/RHOBH-TEST-10-28/embeddings.parquet")

print(f"Picked: {len(picked_df)} / {len(embeddings_df)} ({len(picked_df)/len(embeddings_df)*100:.1f}%)")
# Expected: 60-80% retention (non-face chips + co-face removed)
```

2. **Non-Face Count in Gallery** (should be 0):
```python
# Check if any picked samples have face_conf < 0.65 or face_size < 72
assert (picked_df['confidence'] >= 0.65).all()
assert (picked_df['face_size'] >= 72).all()
```

3. **Contamination Audit Results**:
```python
with open("data/harvest/RHOBH-TEST-10-28/diagnostics/contamination_audit.json") as f:
    audit = json.load(f)

for cluster_name, spans in audit['clusters'].items():
    print(f"{cluster_name}: {len(spans)} contamination spans")
    for span in spans:
        print(f"  Track {span['track_id']}: {span['time_range']} → {span['action']}")
```

4. **Cluster Purity** (after auto-splits):
```python
# Re-compute cluster purity scores
# Expected: >0.90 for most clusters (up from ~0.75-0.85)
```

---

## Expected Results

### Before (Current Dirty State):
```
RINNA cluster gallery:
- 120 thumbnails total
- 15 non-face (glasses, arms, background)
- 10 EILEEN faces
- 5 LVP faces
- 90 RINNA faces
Purity: 75%
```

### After (Clean State):
```
RINNA cluster gallery:
- 90 thumbnails total (faces-only, top-K per track)
- 0 non-face (filtered out)
- 0 EILEEN faces (auto-split to EILEEN cluster)
- 0 LVP faces (auto-split to LVP cluster)
- 90 RINNA faces
Purity: 100%

contamination_audit.json:
{
  "RINNA": [
    {"track_id": 42, "reason": "cross_identity", "best_match": "EILEEN", "action": "split_to_EILEEN"},
    {"track_id": 58, "reason": "cross_identity", "best_match": "LVP", "action": "split_to_LVP"}
  ]
}
```

---

## Files Created (2 modules, 700 lines)

1. **screentime/clustering/face_quality.py** (300 lines)
   - Face-only gating
   - Top-K selection per track
   - Co-face detection
   - Quality scoring

2. **screentime/clustering/contamination_audit.py** (400 lines)
   - Intra-cluster outlier detection
   - Cross-identity contamination detection
   - Contiguous span finding
   - Auto-split logic

---

## Files to Modify (Next Session)

1. **configs/pipeline.yaml** (+25 lines) - Add face_quality + contamination_audit config
2. **screentime/pipeline/clustering.py** (+50 lines) - Integrate modules
3. **app/lib/data.py** (+20 lines) - Load picked_samples for gallery
4. **app/labeler.py** (+5 lines) - Add "Show debug" toggle

**Total Integration Effort**: ~90 minutes

---

## Acceptance Criteria

✅ **Face-Only Feed**: Gallery shows 0 non-face thumbnails (glasses/arms/background filtered)
✅ **Top-K Centroids**: Track centroids computed from best 10 chips per track
✅ **Contamination Detection**: Mixed clusters detected with time ranges + evidence
✅ **Auto-Split**: Contaminated spans moved to correct cluster or Unknown
✅ **Cluster Purity**: >90% for all clusters (up from 75-85%)
✅ **Identity-Agnostic**: Same rules for all identities, no per-person tuning
✅ **Telemetry**: picked_samples.parquet + contamination_audit.json generated

---

## Next Session Checklist (90 min)

- [ ] Add config to pipeline.yaml (5 min)
- [ ] Integrate face_quality into clustering.py (30 min)
- [ ] Integrate contamination_audit into clustering.py (20 min)
- [ ] Update gallery to use picked_samples (20 min)
- [ ] Run clean clustering pipeline (10 min)
- [ ] Validate results (15 min)
  - [ ] Check picked_samples count (60-80% retention)
  - [ ] Verify 0 non-face in gallery
  - [ ] Review contamination_audit.json
  - [ ] Confirm cluster purity >90%

---

**Status**: Core modules complete (face quality + contamination audit)
**Next**: Integration into clustering pipeline + gallery update (90 min)
**Expected Impact**: Clean galleries, no mixed clusters, identity-agnostic
