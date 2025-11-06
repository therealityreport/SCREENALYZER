# Phase 1 Revised Status - Streamlit Keys (Partial)

**Status**: ⚠️ PARTIAL FIX - Image keys not supported in Streamlit 1.38.0
**Time**: 35 minutes total
**Blocker**: `st.image(key=...)` requires Streamlit 1.40+, current version is 1.38.0

---

## What Changed

### ✅ Kept: Timestamp Improvement
**Line 1175**: Better ts_ms handling
```python
ts_ms = frame_ref.get("ts_ms", track.get("start_ms", 0) + (frame_idx * 100))
```
Uses actual `ts_ms` from frame_refs when available (entrance/densify tracks), with fallback for baseline tracks.

### ❌ Reverted: Image Keys
Attempted to add `key=` to all `st.image()` calls, but Streamlit 1.38.0 doesn't support this parameter (added in 1.40+).

**Error**: `TypeError: ImageMixin.image() got an unexpected keyword argument 'key'`

**Fix**: Removed all image keys (lines 655, 887, 1180, 1368)

---

## Root Cause Analysis

DuplicateWidgetID errors are NOT from images - they're from **buttons/selectboxes** in loops that already have proper `wkey()` calls.

**Actual Issue**: The existing `wkey()` calls use correct context but may have collisions during rapid re-renders or when multiple tracks share frame_ids.

**Better Solution** (if needed):
1. Upgrade to Streamlit 1.40+ to enable image keys
2. OR: Add unique counters to button keys (already mostly done)
3. OR: Use container keys instead of widget keys

---

## Current State

| File | Lines Modified | Status |
|------|---------------|--------|
| app/labeler.py | 1175 | ✅ ts_ms improvement kept |
| app/labeler.py | 655, 887, 1180, 1368 | ❌ Image keys reverted |

**Net Change**: +1 improvement (timestamp), -4 failed attempts (image keys)

---

## Acceptance

| Criterion | Status |
|-----------|--------|
| No DuplicateWidgetID errors | ⏳ Unknown (requires user testing) |
| Uniform 160×160 tiles | ✅ Already implemented (width=160) |
| Per-image trash works | ✅ Already implemented with wkey() |
| Syntax valid | ✅ PASS |

---

## Recommendation

**Defer Streamlit key fixes** to future session with Streamlit upgrade.

**Current Priority**: Phases 3-6 (identity-guided recall, multi-proto bank, analytics) have higher ROI for accuracy improvements.

---

**Phase 1 Status**: Partial (timestamp improvement kept, image keys deferred)
**Time Spent**: 35 minutes
**Ready to Proceed**: Phase 3 (Identity-Guided Recall)
