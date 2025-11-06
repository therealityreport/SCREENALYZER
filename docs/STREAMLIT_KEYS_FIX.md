# Streamlit Widget Key Fix - DuplicateWidgetID Resolution

**Issue**: Multiple widgets in "View All Tracks" gallery generate identical keys, causing `DuplicateWidgetID` errors
**Root Cause**: Insufficient uniqueness in key generation when iterating over tiles
**Solution**: Use globally unique seed for all widgets

---

## The Fix

### 1. Helper Function (Already Exists in `app/labeler.py:39`)

```python
def wkey(*parts) -> str:
    """
    Generate unique widget key from parts.

    Ensures widget keys are unique across all UI contexts by combining
    all identifying information (episode, cluster, track, frame, action, etc.)
    """
    return "w_" + "_".join(str(p) for p in parts)
```

✅ This function already exists and is correctly implemented.

---

## 2. Apply to All Widgets in Gallery

### Current Problem Areas

**File**: `app/labeler.py` (lines ~1150-1300 in "View All Tracks" section)

#### ❌ Before (Causes Duplicates):
```python
# Multiple tracks can have same frame_id, causing collisions
key = f"delete_frame_c{cluster_id}_t{track_id}_f{frame_id}"
st.button("🗑", key=key)
```

#### ✅ After (Globally Unique):
```python
# Include ALL identifying information
seed = (episode_id, cluster_id, track_id, frame_id, ts_ms, tile_index)

# Image widget
st.image(img, key=wkey("img", *seed), use_column_width=False)

# Trash button (per-image delete)
if st.button("🗑", key=wkey("del", *seed)):
    cluster_mutator.delete_frame_from_track(track_id, frame_id)
    # Rebuild prototype, update suggestions, write audit
    st.rerun()

# Move button
if st.button("Move", key=wkey("move", *seed)):
    # ... move logic

# Split button
if st.button("Split", key=wkey("split", *seed)):
    # ... split logic
```

---

## 3. Implementation Checklist

### Find and Replace All Gallery Widgets

**Location**: `app/labeler.py` → `render_review_page()` → "View All Tracks" section

□ **Images** (line ~1180):
```python
# OLD: No key or insufficient key
st.image(img)

# NEW: Unique key
st.image(img, key=wkey("img", episode_id, cluster_id, track_id, frame_id, ts_ms, idx))
```

□ **Trash Buttons** (line ~1190):
```python
# OLD: Duplicate keys possible
key = f"delete_frame_c{cluster_id}_t{track_id}_f{frame_id}"

# NEW: Include tile index
key = wkey("del", episode_id, cluster_id, track_id, frame_id, ts_ms, tile_idx)
```

□ **Move Buttons** (line ~1240):
```python
# OLD
key = f"move_track_c{cluster_id}_t{track_id}"

# NEW: Include episode and timestamp
key = wkey("move", episode_id, cluster_id, track_id, ts_ms)
```

□ **Split Buttons** (line ~1260):
```python
# OLD
key = f"split_track_c{cluster_id}_t{track_id}"

# NEW
key = wkey("split", episode_id, cluster_id, track_id, ts_ms)
```

□ **Radio Buttons** (if any in loops):
```python
# NEW
st.radio("Options", options, key=wkey("radio", episode_id, cluster_id, track_id, option_type))
```

□ **Sliders** (if any in loops):
```python
# NEW
st.slider("Threshold", 0.0, 1.0, key=wkey("slider", episode_id, cluster_id, param_name))
```

---

## 4. Gallery Tile Standards

### CSS (add to Streamlit config or inline):

```css
.thumb {
    width: 160px;
    height: 160px;
    object-fit: cover;
    border: 1px solid #ddd;
    border-radius: 4px;
}

.thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, 160px);
    gap: 8px;
}

.thumb-trash {
    position: absolute;
    bottom: 4px;
    right: 4px;
    background: rgba(255, 0, 0, 0.8);
    color: white;
    border-radius: 50%;
    width: 24px;
    height: 24px;
    padding: 0;
}
```

### Streamlit Implementation:

```python
# Uniform 160x160 tiles
for idx, frame_ref in enumerate(track_frames):
    col = cols[idx % 6]  # 6 columns
    with col:
        # Load and resize image
        img = load_frame_crop(video_path, frame_ref)
        img_resized = img.resize((160, 160), Image.LANCZOS)

        # Display with unique key
        seed = (episode_id, cluster_id, track_id, frame_ref['frame_id'], frame_ref['ts_ms'], idx)
        st.image(img_resized, key=wkey("img", *seed), use_column_width=False)

        # Trash overlay (per-image delete)
        if st.button("🗑", key=wkey("del", *seed), type="secondary"):
            # Delete this single frame
            cluster_mutator.delete_frame_from_track(track_id, frame_ref['frame_id'])

            # Rebuild track prototype
            cluster_mutator.rebuild_track_prototype(track_id)

            # Update suggestions
            cluster_mutator.update_merge_suggestions(cluster_id)

            # Write audit entry
            state_mgr.record_action("delete_frame", {
                "track_id": track_id,
                "frame_id": frame_ref['frame_id'],
                "cluster_id": cluster_id
            })

            st.success(f"Deleted frame {frame_ref['frame_id']}")
            st.rerun()
```

---

## 5. Testing Checklist

After applying fixes:

□ Navigate to "View All Tracks"
□ Verify NO `DuplicateWidgetID` errors in console
□ Verify all tiles render at 160x160 uniformly
□ Click trash on multiple tiles - each deletes only that frame
□ Verify prototype rebuild after frame delete
□ Check audit log has delete_frame entries
□ Verify gallery doesn't jitter or resize unexpectedly

---

## 6. Common Pitfalls

**❌ Using frame_id alone**:
```python
key = f"img_{frame_id}"  # BAD - frame_id can repeat across tracks
```

**❌ Missing tile index**:
```python
key = wkey("img", track_id, frame_id)  # BAD - same frame can appear multiple times in same track
```

**✅ Include everything**:
```python
key = wkey("img", episode_id, cluster_id, track_id, frame_id, ts_ms, tile_idx)  # GOOD
```

---

## Files to Modify

1. **app/labeler.py** (primary):
   - Line ~1180: Image widgets
   - Line ~1190: Trash buttons
   - Line ~1240: Move buttons
   - Line ~1260: Split buttons
   - Any radios/sliders in loops

2. **app/lib/cluster_mutations.py** (if needed):
   - Ensure `delete_frame_from_track()` exists
   - Ensure `rebuild_track_prototype()` exists

3. **app/lib/review_state.py** (if needed):
   - Ensure `record_action()` logs to audit

---

## Acceptance Criteria

✅ NO `DuplicateWidgetID` errors in Streamlit logs
✅ All gallery tiles uniform 160×160 px
✅ Per-image trash deletes single frame
✅ Prototype rebuild triggered after delete
✅ Audit entry written for each delete
✅ No gallery jitter/resize on interaction

---

**Status**: Pattern documented, ready to apply in next session
**ETA**: 30 minutes to apply across all widgets
**Files**: `app/labeler.py` (lines 1150-1300 primarily)
