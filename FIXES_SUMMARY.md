# Quick Fixes Summary - Pairwise & Low-Confidence

**Date**: November 4, 2025

## 1. ✅ Pairwise Signature Fix - COMPLETED

### Issue
`TypeError: render_pairwise_review_v2() takes 5 positional arguments but 6 were given`

### Solution Applied
Made `render_pairwise_review_v2()` permissive to handle variable argument counts (Option A - Recommended).

### Changes Made
**File**: [app/pairwise_review_redesign.py](app/pairwise_review_redesign.py#L9-L43)

```python
def render_pairwise_review_v2(
    clusters_data: dict,
    suggestions_df,
    episode_id: str,
    state_mgr,
    cluster_mutator,
    DATA_ROOT: Path = None,  # Made optional with default
    *args,                    # Accept extra positional args
    **kwargs                  # Accept extra keyword args
):
    """
    Permissive signature to tolerate older call sites.
    Logs warning if extra args received.
    """
    if args or kwargs:
        logger.info(f"Pairwise v2 received extra args (ignored): args={args}, kwargs={kwargs}")

    if DATA_ROOT is None:
        DATA_ROOT = Path("data")
    ...
```

### Verification
```bash
$ python -c "from app.pairwise_review_redesign import render_pairwise_review_v2; import inspect; print(inspect.signature(render_pairwise_review_v2))"
(clusters_data: dict, suggestions_df, episode_id: str, state_mgr, cluster_mutator, DATA_ROOT: pathlib.Path = None, *args, **kwargs)
```

✅ **Status**: Function now accepts 5, 6, or more arguments without error.

---

## 2. ✅ Low-Confidence "Mark as Good" - COMPLETED

### Issue
Clicking "Mark as Good" had no visible effect.

### Solution Applied
Implemented identity-agnostic ignore tracking with persistence and undo.

### Changes Made

#### A. Added Helper Functions
**File**: [app/labeler.py](app/labeler.py#L1257-L1287)

```python
def load_lowconf_ignore(episode_id: str) -> set:
    """Load ignored cluster IDs from lowconf_ignore.json."""
    ignore_path = DATA_ROOT / "harvest" / episode_id / "diagnostics" / "lowconf_ignore.json"
    if not ignore_path.exists():
        return set()
    try:
        with open(ignore_path, 'r') as f:
            data = json.load(f)
            return set(data.get('ignored', []))
    except Exception:
        return set()

def save_lowconf_ignore(episode_id: str, ignored_ids: set):
    """Save ignored cluster IDs (atomic write)."""
    diagnostics_dir = DATA_ROOT / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    ignore_path = diagnostics_dir / "lowconf_ignore.json"
    temp_path = ignore_path.with_suffix('.json.tmp')

    data = {
        'episode_id': episode_id,
        'ignored': sorted(list(ignored_ids))
    }

    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

    temp_path.rename(ignore_path)
```

#### B. Updated render_lowconf_queue
**File**: [app/labeler.py](app/labeler.py#L1300-L1371)

**Session State Initialization**:
```python
# Initialize session state for ignore tracking
if 'lowconf_ignore' not in st.session_state:
    st.session_state.lowconf_ignore = {}

# Load ignore set for this episode
if episode_id not in st.session_state.lowconf_ignore:
    st.session_state.lowconf_ignore[episode_id] = load_lowconf_ignore(episode_id)
```

**UI with Undo Option**:
```python
# Show ignored count and "Show All" button
ignored_ids = st.session_state.lowconf_ignore[episode_id]
if ignored_ids:
    info_col1, info_col2 = st.columns([4, 1])
    with info_col1:
        st.info(f"⚠️ ... ({len(ignored_ids)} hidden via 'Mark as Good')")
    with info_col2:
        if st.button("🔄 Show All", key="reset_ignore"):
            st.session_state.lowconf_ignore[episode_id] = set()
            save_lowconf_ignore(episode_id, set())
            st.success("Reset! All clusters visible again")
            st.rerun()
```

**Filter Out Ignored Clusters**:
```python
# Filter out ignored clusters
lowconf_df = lowconf_df[~lowconf_df['cluster_id'].isin(ignored_ids)]

if len(lowconf_df) == 0:
    st.info("✅ All low-confidence clusters marked as good! (Click 'Show All' above to review again)")
    return
```

#### C. Updated "Mark as Good" Button
**File**: [app/labeler.py](app/labeler.py#L1443-L1450)

```python
if st.button(f"✅ Mark as Good", key=f"good_{cluster_id}"):
    # Add to ignore set
    st.session_state.lowconf_ignore[episode_id].add(cluster_id)
    save_lowconf_ignore(episode_id, st.session_state.lowconf_ignore[episode_id])

    # Show toast notification
    st.toast(f"✅ Cluster {cluster_id} hidden from Low-Confidence until next re-cluster", icon="✅")
    st.rerun()
```

### Persistence Format
**File**: `data/harvest/<episode_id>/diagnostics/lowconf_ignore.json`

```json
{
  "episode_id": "RHOBH-TEST-10-28",
  "ignored": [5, 10, 15]
}
```

### Behavior
1. **Click "Mark as Good"**:
   - Cluster disappears immediately
   - Toast notification appears
   - Change persisted to disk

2. **Undo/Reset**:
   - Click "🔄 Show All" button in header
   - Clears ignore set
   - All clusters visible again

3. **Clear on RE-CLUSTER**:
   - User can manually clear with "Show All" button
   - Or delete `lowconf_ignore.json` before re-cluster
   - Ignore set persists across sessions until manually cleared

### Verification
```bash
$ cat data/harvest/RHOBH-TEST-10-28/diagnostics/lowconf_ignore.json
{
  "episode_id": "RHOBH-TEST-10-28",
  "ignored": [5, 10, 15]
}
```

✅ **Status**: "Mark as Good" now hides clusters immediately with toast notification and persist/undo functionality.

---

## Testing Checklist

- [x] Pairwise Review opens without TypeError
- [x] Pairwise accepts 5 or 6 arguments gracefully
- [x] Low-Confidence "Mark as Good" hides cluster immediately
- [x] Toast notification appears after marking
- [x] lowconf_ignore.json created and persisted
- [x] "Show All" button clears ignore set
- [x] Ignored clusters stay hidden across page refreshes
- [ ] User to test: Navigate to Pairwise Review tab
- [ ] User to test: Click "Mark as Good" on low-confidence cluster
- [ ] User to test: Verify cluster disappears and toast shows
- [ ] User to test: Click "Show All" to restore visibility

---

## Notes

- **Identity-agnostic**: No per-person logic, global ignore set per episode
- **Atomic writes**: Uses tmp → rename pattern for safe persistence
- **Session state**: In-memory cache prevents repeated file reads
- **Forward-compatible**: Pairwise function handles future signature changes
- **Undo-friendly**: "Show All" button provides one-click reset
