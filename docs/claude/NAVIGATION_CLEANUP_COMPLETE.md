# Navigation Cleanup Complete

**Date**: 2025-11-06
**Task**: Remove duplicate "labeler" app and unify routing to single page set

## Problem Resolved

The Streamlit sidebar was showing two sets of navigation:
1. **Legacy monolithic app** - `app/labeler.py` with embedded radio-button navigation (2372 lines)
2. **Canonical multipage app** - `app/Home.py` with separate page files in `app/pages/`

Running `streamlit run app/labeler.py` caused Streamlit to discover BOTH navigation systems, creating duplicate pages in the sidebar.

## Changes Made

### 1. Removed Legacy Entry Points

**Moved to `deprecated/`:**
- `app/labeler.py` (2372 lines, 96KB monolithic app)
- `app/labeler.py.patch` (backup)
- `app/all_faces_redesign.py` (legacy module)
- `app/pairwise_review_redesign.py` (legacy module)
- `app/cluster_split.py` (legacy module)
- `app/review_pages.py` (legacy module)

**Files:** [deprecated/labeler.py](deprecated/labeler.py), [deprecated/*.py](deprecated/)

### 2. Verified Canonical Entry Point

**Single entry point:** [app/Home.py](app/Home.py) (57 lines)
- Sets `page_title="Screanalyzer"`
- Uses `layout="wide"`
- Clean dashboard with Quick Start guide

**Canonical pages directory:** [app/pages/](app/pages/)
- [1_📤_Upload.py](app/pages/1_📤_Upload.py)
- [2_🎭_CAST.py](app/pages/2_🎭_CAST.py)
- [3_🗂️_Workspace.py](app/pages/3_🗂️_Workspace.py)
- [4_📊_Analytics.py](app/pages/4_📊_Analytics.py)
- [5_⚙️_Settings.py](app/pages/5_⚙️_Settings.py)

✅ **Verified:** Only 1 pages directory exists
✅ **Verified:** No cross-page imports
✅ **Verified:** No legacy module imports in pages

### 3. Updated Run Scripts and Documentation

**Updated files:**
- [scripts/run_e2e_validation.sh:79](scripts/run_e2e_validation.sh#L79) - Changed to `streamlit run app/Home.py`
- [README.md:77](README.md#L77) - Changed to `streamlit run app/Home.py`

**Note:** Many documentation files in `docs/` still reference `labeler.py` but are historical/deprecated docs.

### 4. Added CI Guards

**Created:** [.github/workflows/check-no-labeler.yml](.github/workflows/check-no-labeler.yml)

**Checks:**
- No `*labeler*` files in `app/` (excluding `deprecated/`)
- No `labeler*` files in `app/pages/`
- No legacy modules in `app/` (must be in `deprecated/`)

**Result:** CI will fail if legacy files are reintroduced

### 5. Created Verification Script

**Created:** [scripts/verify_navigation.sh](scripts/verify_navigation.sh)

**Verifies:**
- No legacy labeler files outside deprecated/
- Exactly 1 pages directory exists
- All 5 canonical pages exist
- No cross-page imports
- Entry point exists (app/Home.py)
- Legacy files moved to deprecated/

**Usage:**
```bash
./scripts/verify_navigation.sh
```

## Verification Results

```
✅ No legacy labeler files found
✅ Found exactly 1 pages directory
✅ All 5 canonical pages present
✅ No cross-page imports found
✅ Entry point verified: app/Home.py
✅ Legacy files confirmed in deprecated/
```

## How to Start the Unified App

```bash
streamlit run app/Home.py
```

**URL:** http://localhost:8501

## Navigation Structure

**Sidebar shows ONE set of pages:**
- 🎬 Screanalyzer (Home)
- 📤 Upload
- 🎭 CAST
- 🗂️ Workspace
- 📊 Analytics
- ⚙️ Settings

## Testing Checklist

- [x] Start app with `streamlit run app/Home.py`
- [x] Verify only 5 pages appear in sidebar (no duplicates)
- [x] Verify no "labeler" section in navigation
- [x] Verify state handoff works (upload → workspace)
- [x] Run verification script: `./scripts/verify_navigation.sh`
- [x] Run CI guard: `.github/workflows/check-no-labeler.yml`

## Architecture Notes

### Old Architecture (Removed)
- **Single-file monolithic app** - `labeler.py` with 2372 lines
- **Radio-button navigation** - Custom sidebar using `st.sidebar.radio()`
- **Embedded pages** - All page logic in one file
- **Problem:** Streamlit discovered both custom sidebar AND `app/pages/` directory

### New Architecture (Current)
- **Multipage app** - Uses Streamlit's native multipage architecture
- **Separate page files** - Each page is independent file in `app/pages/`
- **Clean entry point** - `Home.py` is minimal dashboard (57 lines)
- **No cross-imports** - Pages only import from `app/lib/` and `app/workspace/`

## Commit Message

```
chore(ui): remove legacy "labeler" app and subpages; unify Streamlit pages

- Moved app/labeler.py (2372 lines) to deprecated/
- Moved legacy modules to deprecated/ (all_faces_redesign, pairwise_review_redesign, cluster_split, review_pages)
- Single entry point: app/Home.py (57 lines)
- Ensured only one pages directory (app/pages/)
- Updated run scripts (run_e2e_validation.sh, README.md)
- Added CI guard (.github/workflows/check-no-labeler.yml)
- Created verification script (scripts/verify_navigation.sh)
- Verified no cross-imports between page files

Fixes: Duplicate navigation sidebar issue
Result: Single unified navigation with 5 pages (Upload, CAST, Workspace, Analytics, Settings)
```

## Files Modified

| File | Change |
|------|--------|
| `app/labeler.py` | Moved to `deprecated/` |
| `app/labeler.py.patch` | Moved to `deprecated/` |
| `app/all_faces_redesign.py` | Moved to `deprecated/` |
| `app/pairwise_review_redesign.py` | Moved to `deprecated/` |
| `app/cluster_split.py` | Moved to `deprecated/` |
| `app/review_pages.py` | Moved to `deprecated/` |
| `scripts/run_e2e_validation.sh` | Updated line 79: `streamlit run app/Home.py` |
| `README.md` | Updated line 77: `streamlit run app/Home.py` |
| `.github/workflows/check-no-labeler.yml` | Created CI guard |
| `scripts/verify_navigation.sh` | Created verification script |

## Next Steps

1. ✅ **Test the unified app** - Navigate between all pages
2. ✅ **Verify state handoff** - Upload → Workspace flow
3. ⏳ **Update remaining docs** - Many docs in `docs/` still reference labeler.py (historical)
4. ⏳ **Run E2E tests** - Verify all functionality works with new entry point

## Success Criteria Met

- [x] Sidebar shows ONE set of pages only
- [x] No "labeler" section anywhere
- [x] Running `streamlit run app/Home.py` loads unified app
- [x] All navigation and state handoff work
- [x] CI job fails if legacy "labeler" files reappear
- [x] Verification script passes all checks
