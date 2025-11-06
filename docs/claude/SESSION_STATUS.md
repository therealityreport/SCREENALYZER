# Cast Images Page - Show/Season Integration Complete

**Status**: ✅ Complete
**Date**: November 4, 2025
**Phases Completed**: 1, 2, + Cast Images Fix

---

## What Was Done

### Phase 1: Multi-Format Image Processing ✅
- Created `screentime/image_utils.py` with `ImageNormalizer` and `ImageDeduplicator`
- Supports: `.jpg`, `.jpeg`, `.png`, `.avif`, `.webp`, `.heic` → PNG
- Normalization: RGB, 8-bit, EXIF-aware orientation, strip EXIF
- Hash-based deduplication (SHA-256) + optional cosine similarity
- Dependencies installed: `pillow-heif==0.22.0`, `pillow-avif-plugin==1.5.2`

### Phase 2: Show/Season Registry ✅
- Created `screentime/models/show_season.py` - JSON-based registry (no database)
- Atomic writes (tmp → move) with schema validation
- Path helpers for facebank, videos, harvest, outputs
- Initialized RHOBH Season 5 with canonical IDs:
  - `show_id: rhobh`
  - `show_name: Real Housewives of Beverly Hills`
  - `season_id: s05`
  - `season_number: 5`
  - `season_label: Season 5`
- Registry saved to: `configs/shows_seasons.json`

### Cast Images Page Rewrite ✅
**File**: `app/labeler.py` - `render_cast_images_page()`

**New Features**:

1. **Show/Season Selectors** (top of page):
   - Show dropdown (defaults to RHOBH if available)
   - Season dropdown scoped to selected show (defaults to S05 if available)
   - Path display: `data/facebank/rhobh/s05/`

2. **Current Cast Display**:
   - Metrics showing cast name + seed count
   - Status indicators:
     - ✅ ≥3 seeds (ready)
     - ⚠️ 1-2 seeds (needs more)
     - ❌ 0 seeds (not ready)

3. **Multi-Format Upload**:
   - Accepts: `.jpg`, `.jpeg`, `.png`, `.webp`, `.avif`, `.heic`
   - Multiple file upload (3-5 recommended per cast)
   - Auto-converts to PNG using Phase 1 normalizer

4. **Face Validation** (identity-agnostic):
   - Detects faces using RetinaFace
   - Rejects if no face detected
   - Rejects if >1 face (suggests single-face crops)
   - Min confidence: 0.65
   - Min face height: 64px (rejects)
   - Warns if 64-72px (acceptable but small)

5. **Processing & Storage**:
   - Normalizes each image → `seed_NNN.png`
   - Saves to: `data/facebank/<show>/<season>/<cast>/seed_*.png`
   - Metadata sidecar: `seeds_metadata.json` with quality metrics
   - Updates registry with seed counts

6. **Results Display**:
   - Success message with valid seed count
   - Expandable rejected list (filename + reason)
   - Gallery view per cast (up to 12 seeds shown)

---

## Directory Structure Created

```
configs/
  shows_seasons.json          # Registry

data/
  facebank/
    rhobh/
      s05/
        <CAST_NAME>/
          seed_001.png
          seed_002.png
          ...
          seeds_metadata.json
  videos/
    rhobh/
      s05/
```

---

## Registry Contents

```json
{
  "version": "1.0",
  "shows": [
    {
      "show_id": "rhobh",
      "show_name": "Real Housewives of Beverly Hills",
      "seasons": [
        {
          "season_id": "s05",
          "season_label": "Season 5",
          "season_number": 5,
          "cast": [],
          "episodes": [],
          "created_at": "2025-11-04T21:22:28.339692Z"
        }
      ],
      "created_at": "2025-11-04T21:22:28.337797Z"
    }
  ]
}
```

---

## Cast Images Page Flow

### User Experience:

1. **Navigate to Cast Images page**
2. **Select Show/Season**:
   - Show: "Real Housewives of Beverly Hills" (defaults if rhobh exists)
   - Season: "Season 5" (defaults if s05 exists)
3. **View Current Cast** (initially empty)
4. **Add Cast Member**:
   - Enter name: `KIM`
   - Upload 3-5 images (jpg/png/heic/webp/avif)
   - Click "Process & Add to Season"
5. **Processing**:
   - Each image normalized to PNG
   - Face detected and validated
   - Quality checks applied
   - Progress bar shows status
6. **Results**:
   - "✅ Added 3 valid seeds for KIM"
   - Rejected images listed with reasons
   - Cast counter updates: "KIM: 3 seeds ⚠️"
7. **Gallery View**:
   - Expand "KIM (3 seeds)" to see thumbnails
   - All seeds displayed (up to 12)

---

## Validation Rules (Identity-Agnostic)

**Face Detection**:
- Must detect exactly 1 face
- Confidence ≥ 0.65
- Face height ≥ 64px (hard limit)
- Face height < 72px → warning (but accepted)

**Multi-Face Handling**:
- Rejects with message: "N faces detected - use single face images"
- User should crop to single face before upload

**Format Handling**:
- All formats auto-convert to PNG
- Original format tracked in metadata
- HEIC (iPhone photos) supported

---

## Code Changes

### Files Created:
1. `screentime/image_utils.py` - Image normalization pipeline
2. `screentime/models/show_season.py` - Registry system
3. `screentime/models/__init__.py` - Models package
4. `init_rhobh_s05.py` - Registry initialization script
5. `test_phase1_image_normalization.py` - Test suite for Phase 1

### Files Modified:
1. `app/labeler.py`:
   - Added `import json`
   - Completely rewrote `render_cast_images_page()` function (lines 405-671)
2. `requirements.txt`:
   - Added `pillow-heif==0.22.0`
   - Added `pillow-avif-plugin==1.5.2`

### Lines Changed:
- `app/labeler.py`: ~267 lines rewritten (render_cast_images_page function)

---

## Acceptance Criteria ✅

✅ **Show/Season selectors**: Present on Cast Images page, defaulting to RHOBH → Season 5
✅ **Multi-format upload**: Accepts jpg/png/webp/avif/heic, auto-converts to PNG
✅ **Face validation**: min_face_px=64, conf≥0.65, single-face required
✅ **Storage path**: `data/facebank/rhobh/s05/<CAST>/seed_*.png`
✅ **Registry updates**: Cast members tracked with seed counts
✅ **Counters**: Display valid seed count per cast with status indicators
✅ **Quality bars**: Status indicators (✅/⚠️/❌) show readiness
✅ **Gallery view**: Shows up to 12 seeds per cast in expandable panels

---

## Testing Instructions

### Manual Test Flow:

1. **Start Streamlit**:
   ```bash
   source .venv/bin/activate
   export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
   streamlit run app/labeler.py
   ```

2. **Navigate to Cast Images**:
   - Click "🎭 Cast Images" in sidebar

3. **Verify UI**:
   - Show selector should show "Real Housewives of Beverly Hills"
   - Season selector should show "Season 5"
   - Path display should show: `data/facebank/rhobh/s05/`

4. **Add Test Cast**:
   - Enter name: `TEST_CAST`
   - Upload 3-5 test images (any format)
   - Click "Process & Add to Season"

5. **Expected Behavior**:
   - Progress bar shows processing
   - Valid images → success message
   - No-face images → rejected list
   - Multi-face images → rejected list
   - Registry updates automatically
   - Gallery expander shows thumbnails

### Automated Test:

```bash
# Initialize RHOBH S05
python init_rhobh_s05.py

# Verify registry
cat configs/shows_seasons.json

# Check directories created
ls -la data/facebank/rhobh/s05/
ls -la data/videos/rhobh/s05/
```

---

## Next Steps

**Phases Remaining**:
- ✅ Phase 1: Multi-format image processing
- ✅ Phase 2: Show/Season registry
- ✅ Cast Images page fix
- ⏳ Phase 5: Season bank system (multi_prototypes.json)
- ⏳ Phase 6: Unknown/Not-in-Cast discovery panel
- ⏳ Phase 7: Re-Harvest & Analyze pipeline integration
- ⏳ Phase 8: Config updates
- ⏳ Phase 9: End-to-end testing with RHOBH S05

**Immediate Next**:
- User will test Cast Images page and upload seeds for RHOBH S05 cast
- Screenshot checkpoint requested (Show/Season selectors + seed upload)
- Then proceed to Phase 5 (season bank builder)

---

## Notes

1. **Identity-Agnostic**: No hardcoded cast names, thresholds are global
2. **Seeds Optional**: Pipeline must work with or without seeds (open-set fallback)
3. **3-5 Seeds Target**: Changed from original 8-12 per user request
4. **Canonical IDs Locked**: rhobh/s05 IDs are fixed, do not change
5. **Atomic Registry Writes**: tmp → move pattern prevents corruption
6. **Schema Validation**: Registry validates structure on load

---

**Ready for User Testing**: Cast Images page is fully functional and ready for seed uploads!

---

# Review UX Redesign - Complete

**Status**: ✅ Complete
**Date**: November 4, 2025

## Summary
- Review UX: All Faces single-row + Track Gallery modal + Split Cluster (ML/CL + constrained re-cluster)

## Features Implemented

### All Faces View
- Horizontal single-row layout per cluster with pagination
- Compact mode toggle (125×166 vs 150×200 tiles)
- Arrow navigation (◀/▶ buttons) for paging through tracks
- Confidence tooltips on hover (Track <id> · conf=<score>)
- 6-8 tracks per row depending on mode
- Sorted by confidence (descending)

### Track Gallery Modal
- Quick Move action at top for fast reassignment
- Next/Previous track navigation without closing modal
- 3-8 face chip gallery for selected track
- Season cast dropdown with "Add a Cast Member..." option
- Creates new cast member directory and assigns immediately
- Saves track-level constraints with show_id/season_id/episode_id

### Low-Confidence Split Cluster
- Multi-select tracks with checkboxes (8 per row)
- Bulk actions: Select All, Clear Selection
- Season cast dropdown for target identity
- Constraint count display: ML:+N, CL:+M after assignment
- Inline Re-Cluster button with constrained mode
- Stays in split view and refreshes after re-cluster

### Diagnostics
- track_constraints.jsonl includes show_id/season_id/episode_id for auditability
- Constraint counts saved with must_link_moved and cannot_link pairs
- Integration with existing constrained RE-CLUSTER workflow

## Files Modified
- `app/all_faces_redesign.py`: Row layout, pagination, compact mode, navigation
- `app/cluster_split.py`: Multi-select, constraint display, inline re-cluster
- `screentime/clustering/constraints.py`: Added IDs to constraint log entries

