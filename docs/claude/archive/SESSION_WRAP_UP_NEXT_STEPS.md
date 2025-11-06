# Session Wrap-Up & Next Steps

**Date**: October 30, 2025
**Total Session Time**: ~5-6 hours
**Status**: Infrastructure complete, Show/Season UI ready to implement

---

## ✅ Session Accomplishments

### Infrastructure Built (1,560 lines)

**1. Cluster Hygiene Pipeline** ✅
- Face-only filtering: 77.6% retention (272 non-face chips removed)
- Top-K selection per track (10 best samples for centroids)
- Gallery integration: Uses `picked_samples.parquet` automatically
- Files: `face_quality.py` (300 lines), `contamination_audit.py` (400 lines)

**2. Auto-Caps (Identity-Agnostic)** ✅
- Computes P80 safe gaps per identity
- Formula: `clamp(P80 × 1.2, 1200ms, 2500ms)`
- All identities capped at 1200ms minimum (short episode)
- File: `auto_caps.py` (150 lines)

**3. Purity-Driven Clustering** ✅
- Quality sweep: `silhouette - 0.75 * impurity`
- Evaluates 10-15 eps candidates
- Chooses optimal eps without manual tuning
- File: `purity_driven_eps.py` (400 lines)

**4. CI Guard** ✅
- Prevents per-person hardcoding regression
- File: `no-hardcoded-identities.sh` (80 lines)

**5. Configuration Updates** ✅
- Identity-agnostic thresholds in `pipeline.yaml`
- No per-person tuning, all global defaults

---

## 🎯 Current Status

### What Works
- ✅ Face-only filtering operational
- ✅ Auto-caps computes merge limits
- ✅ Purity-driven clustering finds optimal eps
- ✅ All infrastructure identity-agnostic
- ✅ Totals validated: 128.1s = 102.5s runtime + 25.6s overlaps

### What's Missing
- ❌ Kim/Kyle still merged in same cluster (sisters with similar features)
- ❌ LVP scattered across clusters
- ❌ Facebank empty (no seed faces for prototype-anchored splitting)

**Root Cause**: Unsupervised clustering can't distinguish Kim from Kyle (similarity ~0.92-0.95)

**Solution**: Prototype-anchored splitting using labeled seed faces

---

## 🚀 Next Session: Show/Season + Cast Images UI

### Implementation Plan (9 hours total)

#### **Phase 1: Database Schema** (1 hour)

**Create Tables**:
```sql
-- shows table
CREATE TABLE shows (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE,  -- "RHOBH"
    display_name VARCHAR,  -- "Real Housewives of Beverly Hills"
    created_at TIMESTAMP
);

-- seasons table
CREATE TABLE seasons (
    id INTEGER PRIMARY KEY,
    show_id INTEGER FOREIGN KEY REFERENCES shows(id),
    season_number INTEGER,
    label VARCHAR,  -- "Season 5" or "S05"
    created_at TIMESTAMP,
    UNIQUE(show_id, season_number)
);

-- cast_members table
CREATE TABLE cast_members (
    id INTEGER PRIMARY KEY,
    season_id INTEGER FOREIGN KEY REFERENCES seasons(id),
    name VARCHAR,  -- "KIM"
    seed_count INTEGER DEFAULT 0,
    created_at TIMESTAMP,
    UNIQUE(season_id, name)
);

-- Update episodes table
ALTER TABLE episodes ADD COLUMN season_id INTEGER FOREIGN KEY REFERENCES seasons(id);
```

**Files to Create**:
- `api/models.py` - SQLAlchemy models
- `api/database.py` - Database connection setup
- `migrations/001_add_show_season.py` - Migration script

---

#### **Phase 2: Upload Page Updates** (2 hours)

**File**: `app/pages/1_Upload.py`

**New UI Flow**:
```
1. Select Show (dropdown + "Create New")
   ↓
2. Select Season (dropdown + "Create New")
   ↓
3. Upload Video → Saved to data/videos/{show}/{season}/{episode}.mp4
```

**File Paths**:
```
data/
  videos/RHOBH/S05/E01-PilotParty.mp4
  facebank/RHOBH/S05/KIM/seed_001.jpg
  harvest/RHOBH/S05/E01-PilotParty/tracks.json
  outputs/RHOBH/S05/E01-PilotParty/timeline.csv
```

---

#### **Phase 3: Cast Images Page** (3 hours)

**File**: `app/pages/5_Cast_Images.py` (new)

**UI Flow**:
```
1. Select Show → Season
   ↓
2. Add Cast Member
   - Name: KIM
   - Upload 8-12 face photos
   ↓
3. Quality Checks (identity-agnostic):
   - Confidence ≥ 0.65
   - Face size ≥ 48px
   - Single face only
   - No blur
   ↓
4. Save to data/facebank/{show}/{season}/{cast}/
   - seed_001.jpg, seed_002.jpg, ...
   - seeds.json (metadata with embeddings)
   ↓
5. Show Facebank Gallery
   - Thumbnail grid per cast member
   - Badge: "Ready" when ≥5 seeds
```

**Features**:
- Drag & drop multiple images
- Real-time quality feedback
- Reject non-face uploads with toast explaining why
- Counter showing valid seeds per cast

---

#### **Phase 4: Multi-Prototype Bank** (1.5 hours)

**File**: `screentime/clustering/prototype_bank.py`

**Algorithm** (simplified for v1):
```python
def build_multi_prototype_bank_from_seeds(show, season):
    """
    Build prototype bank from uploaded seed images.

    For each cast member:
    1. Load seeds from data/facebank/{show}/{season}/{cast}/seeds.json
    2. Extract embeddings
    3. Use all seeds as prototypes (no pose/scale clustering for v1)

    Returns:
        {
            'KIM': [emb1, emb2, emb3, ...],  # 8-12 seed embeddings
            'KYLE': [emb1, emb2, ...],
            ...
        }
    """
```

**Future Enhancement**: Cluster by pose (frontal/3-4/profile) and scale (small/medium/large)

---

#### **Phase 5: Re-Harvest Integration** (1.5 hours)

**File**: `app/pages/5_Cast_Images.py` (add button)

**UI**:
```
Select Episode → Click "Re-Harvest & Analyze" button
   ↓
Runs full pipeline:
1. Harvest (10fps baseline, 30fps recall)
2. Face-only filtering + Top-K
3. Purity-driven clustering
4. Build prototype bank from uploaded seeds
5. Detect confusion (Kim vs Kyle ambiguous)
6. Prototype-anchored split
7. Open-set assignment (Unknown if margin < 0.08)
8. Analytics export
```

**Integration**: Update `jobs/tasks/cluster.py` to check for facebank and use prototype-anchored splitting when available.

---

## 📊 Expected Results

### Before (Current State)
```
Cluster 0: [KIM + KYLE mixed] - 250 tracks
  Gallery: Shows both Kim and Kyle faces ❌

Cluster 1: [RINNA] - 211 tracks
Cluster 2: [EILEEN] - 20 tracks
...
```

### After (With Show/Season + Cast Images)
```
Cluster 0: [KIM] - 130 tracks
  Gallery: Only Kim faces ✅

Cluster 1: [KYLE] - 120 tracks
  Gallery: Only Kyle faces ✅

Cluster 2: [RINNA] - 211 tracks
Cluster 3: [EILEEN] - 20 tracks
...
```

**Diagnostics Generated**:
- `confusion_matrix.json` - Shows Kim/Kyle ambiguity detected
- `anchored_split_audit.json` - Shows split decision and validation
- `id_assign_audit.jsonl` - Shows "Unknown" for ambiguous, assignments for confident
- `multi_prototypes.json` - Prototype bank built from uploaded seeds

---

## 🗺️ Complete User Workflow

### Step 1: Setup (Upload Page)
1. Open http://localhost:8501 → Upload page
2. Click "Create New Show" → Enter "RHOBH" / "Real Housewives of Beverly Hills"
3. Click "Create New Season" → Enter "5" / "Season 5"
4. Upload video "E01-PilotParty.mp4"

**Result**: Video saved to `data/videos/RHOBH/S05/E01-PilotParty.mp4`

---

### Step 2: Seed Facebank (Cast Images Page)
1. Navigate to "Cast Images" page
2. Select "RHOBH" → "Season 5"
3. Add cast members:
   - Name: "KIM" → Upload 10 clear face photos
   - Name: "KYLE" → Upload 10 clear face photos
   - Name: "RINNA" → Upload 10 clear face photos
   - Name: "EILEEN" → Upload 10 clear face photos
   - Name: "BRANDI" → Upload 10 clear face photos
   - Name: "YOLANDA" → Upload 10 clear face photos
   - Name: "LVP" → Upload 10 clear face photos
4. See "Ready ✅" badges when each has ≥5 valid seeds

**Result**: Facebank populated at `data/facebank/RHOBH/S05/{CAST}/`

---

### Step 3: Re-Harvest (Cast Images Page)
1. Stay on "Cast Images" page
2. Select episode "E01-PilotParty" from dropdown
3. Click "Re-Harvest & Analyze" button
4. Wait ~5-10 minutes for pipeline to complete

**Pipeline Execution**:
```
Harvest (10fps) → Face-only filter → Top-K selection →
Purity-driven clustering → Build prototype bank →
Detect confusion → Anchored split → Open-set assignment →
Analytics export
```

---

### Step 4: Validate Results (Clusters Page)
1. Navigate to "Clusters" page
2. Inspect galleries:
   - ✅ Each cluster shows single person
   - ✅ Kim ≠ Kyle (properly separated)
   - ✅ LVP not scattered (in own cluster or Unknown)
   - ✅ No non-face thumbnails

---

### Step 5: Review Analytics (Analytics Page)
1. Navigate to "Analytics" page
2. Check header:
   - Show: RHOBH
   - Season: 5
   - Detector: RetinaFace
   - Baseline: 10fps
   - Recall: 30fps (windows only)
3. Review delta_table.csv:
   - Accuracy improved (more identities ≤4.5s error)
   - No overrides, identity-agnostic
4. Validate:
   - Totals ≤ runtime
   - Overlaps = co-appearances (not duplicates)

---

## 📁 Files to Create (Next Session)

### Database & API
- `api/models.py` - Show/Season/CastMember models (150 lines)
- `api/database.py` - SQLAlchemy setup (50 lines)
- `migrations/001_add_show_season.py` - Migration script (100 lines)

### UI Pages
- `app/pages/1_Upload.py` - Update with Show/Season flow (+200 lines)
- `app/pages/5_Cast_Images.py` - New page for seed upload (400 lines)

### Clustering Modules
- `screentime/clustering/prototype_bank.py` - Build bank from seeds (200 lines)
- `screentime/clustering/confusion_detector.py` - Detect ambiguous clusters (150 lines)
- `screentime/clustering/anchored_split.py` - Split using prototypes (300 lines)
- `screentime/clustering/open_set_assign.py` - Assign with Unknown option (100 lines)

### Integration
- `jobs/tasks/cluster.py` - Update to use prototype-anchored split (+150 lines)

**Total**: ~1,800 lines of new code across 11 files

---

## ⏱️ Time Budget (Next Session)

| Phase | Task | Time |
|-------|------|------|
| 1 | Database schema + migration | 1 hr |
| 2 | Upload page updates | 2 hrs |
| 3 | Cast Images page | 3 hrs |
| 4 | Prototype bank builder | 1.5 hrs |
| 5 | Re-Harvest integration | 1.5 hrs |
| **Total** | | **9 hrs** |

**Recommendation**: Break into 3 sessions of 3 hours each:
- **Session 1**: Database + Upload page (phases 1-2)
- **Session 2**: Cast Images page (phase 3)
- **Session 3**: Prototype bank + Integration (phases 4-5)

---

## 🔑 Key Design Decisions

### Identity-Agnostic Thresholds

**All thresholds are global** (apply to everyone):

| Threshold | Value | Purpose |
|-----------|-------|---------|
| `min_face_conf` | 0.65 | Face quality filter |
| `min_face_px` | 48 | Minimum seed face size |
| `confusion_margin` | 0.10 | Detect ambiguous clusters |
| `tau_min` | 0.60 | Minimum similarity for assignment |
| `delta_open` | 0.08 | Margin for open-set (Unknown) |
| `min_split_duration` | 600ms | Each side of split must have ≥600ms |
| `min_split_chips` | 6 | Each side of split must have ≥6 samples |

**No Kim-specific or Kyle-specific tuning**

---

### File Path Structure

**Hierarchical Organization**:
```
data/
  videos/{show}/{season}/{episode}.mp4
  facebank/{show}/{season}/{cast}/
  harvest/{show}/{season}/{episode}/
  outputs/{show}/{season}/{episode}/
```

**Benefits**:
- Easy to find all episodes for a show/season
- Facebank scoped to season (cast changes between seasons)
- Clear separation between shows
- Backward compatible (null season_id for legacy episodes)

---

## 📊 Current vs Future State

### Current (After This Session)
- ✅ Cluster hygiene working
- ✅ Auto-caps working (limited by short episode)
- ✅ Purity-driven clustering working
- ⚠️ Kim/Kyle still merged (no facebank)
- ⚠️ LVP scattered (no facebank)

### Future (After Next Session)
- ✅ Show/Season organization
- ✅ Cast Images UI for seeding facebank
- ✅ Prototype-anchored splitting
- ✅ Kim/Kyle properly separated
- ✅ LVP in own cluster or Unknown
- ✅ Scalable to all shows/seasons/episodes

---

## 🎬 Starting Next Session

**To Resume**:

1. Review this document
2. Review [SHOW_SEASON_IMPLEMENTATION_PLAN.md](SHOW_SEASON_IMPLEMENTATION_PLAN.md)
3. Start with Phase 1 (Database schema)
4. Test each phase before moving to next
5. Use Streamlit to validate UI at each step

**Quick Start Command**:
```bash
# Start Streamlit for testing
source .venv/bin/activate
streamlit run app/labeler.py --server.port 8501
```

---

## 📚 Documentation Created This Session

1. [SESSION_STATUS_2025_10_30_FINAL.md](SESSION_STATUS_2025_10_30_FINAL.md) - Original session summary
2. [PURITY_DRIVEN_CLUSTERING_IMPLEMENTATION.md](PURITY_DRIVEN_CLUSTERING_IMPLEMENTATION.md) - Purity-driven eps details
3. [PROTOTYPE_ANCHORED_SPLIT_ROADMAP.md](PROTOTYPE_ANCHORED_SPLIT_ROADMAP.md) - Anchored split approach
4. [FINAL_SESSION_STATUS_KIM_KYLE_SOLUTION.md](FINAL_SESSION_STATUS_KIM_KYLE_SOLUTION.md) - Complete analysis
5. **[SHOW_SEASON_IMPLEMENTATION_PLAN.md](SHOW_SEASON_IMPLEMENTATION_PLAN.md)** - Detailed implementation plan
6. **[SESSION_WRAP_UP_NEXT_STEPS.md](SESSION_WRAP_UP_NEXT_STEPS.md)** - This document

---

## 💡 Summary

**This Session**: Built complete infrastructure for identity-agnostic clustering
**Next Session**: Build UI workflow to seed facebank and enable prototype-anchored splitting
**End Goal**: Clean galleries with Kim ≠ Kyle, scalable to all episodes

**Total Investment**:
- This session: 5-6 hours (infrastructure)
- Next session: 9 hours (UI + integration)
- **Total**: 14-15 hours for complete, production-ready solution

**Status**: ✅ Infrastructure complete, ready for UI implementation
