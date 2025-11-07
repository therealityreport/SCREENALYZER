# Live UI Polling + Atomic Embeddings Handoff + Structured Prereq Errors

**Date:** 2025-11-07
**Branch:** `fix/ui-live-polling-embeddings-race-2025-11-07`
**Commit:** 95cd271

## Overview

This document describes three critical fixes to the SCREENALYZER pipeline:

1. **UI Auto-Refresh Not Working** - Progress bars and logs stayed frozen until manual "Resume" button click
2. **Embeddings Race Condition** - Track stage failed with "Embeddings not found" immediately after Detect reported success
3. **Cluster Auto-Run Generic Errors** - Showed "Unknown error" instead of specific error codes like `ERR_EMBEDDINGS_MISSING`

## Problem 1: UI Auto-Refresh Not Working

### Root Cause

The Streamlit `st_autorefresh` component was only being called when `not use_status_panel` (line 1078 in original code), but the status panel template existed, which prevented auto-refresh from triggering. Additionally, the refresh logic was buried deep in the page rendering flow, so it never executed early enough to take effect.

**Key Issue:** `st_autorefresh` must be called at the **very top of `main()`** before any other widgets are rendered, otherwise Streamlit won't trigger the page refresh.

### Symptoms Before Fix

- Progress % and face count stayed frozen at 0% until clicking "Resume Detect"
- Logs panel didn't update during job execution
- Status panel showed "⚠️ Live snapshot unavailable. Retrying…" indefinitely
- Users had to manually click "Resume Detect" to see any progress updates

### Changes Made

**File: `app/pages/3_🗂️_Workspace.py`**

1. **Moved `st_autorefresh` to top of `main()` (lines 455-462)**
   ```python
   # CRITICAL: Auto-refresh MUST be called at the top of main() to trigger page refreshes
   # This runs BEFORE any other widgets to ensure it takes effect
   active_jobs = st.session_state.get("active_polling_jobs", {})
   if active_jobs:
       # Use 2-second interval for consistent real-time updates
       st_autorefresh(interval=2000, key="workspace_auto_refresh")
       # Show visible heartbeat indicator
       st.caption(f"🔄 Live polling active ({len(active_jobs)} job{'s' if len(active_jobs) > 1 else ''}) • Auto-refreshing every 2s")
   ```

2. **Auto-attach to detect jobs on page load (lines 596-603)**
   - No "Resume Detect" button needed - automatically starts polling when detect job is active
   - Seeds `active_polling_jobs`, `active_job`, and `last_run_ts` in session state

3. **Fallback polling when pipeline_state shows detect running (lines 917-930)**
   - If no active jobs found but `detected=false` OR `pipeline_state.status=="running"`, auto-attach to polling
   - Handles edge case where job envelope doesn't exist but pipeline is actually running

4. **Removed old conditional auto-refresh logic (line 1084-1085)**
   - Deleted the `if auto_refresh_needed and not use_status_panel: schedule_auto_refresh(ep)` logic
   - No longer needed since refresh is now handled at top of main()

5. **Removed redundant `schedule_auto_refresh(ep_trigger)` call (line 1289)**
   - Was in the "trigger prepare" handler but no longer needed

### Behavior After Fix

- Page automatically refreshes every 2 seconds when any job is running
- Visible heartbeat indicator at top: `🔄 Live polling active (1 job) • Auto-refreshing every 2s`
- Progress bars update in real-time showing frames processed, faces detected, etc.
- Logs append continuously without any user action
- Works immediately on page load - no "Resume" button needed
- Gracefully stops auto-refresh when job completes

## Problem 2: Embeddings Race Condition

### Root Cause

The Detect stage wrote `embeddings.parquet` directly without `fsync()`, so the file metadata was updated before the actual data was flushed to disk. The Track stage could then read the file path, see it exists, but get partial/empty data because the write wasn't yet durable.

**Timeline of Race:**
1. Detect writes embeddings.parquet (buffered in OS cache)
2. Detect reports "1537 embeddings computed" (SUCCESS)
3. Track starts immediately, reads embeddings.parquet
4. Track gets empty/partial file because fsync hasn't completed
5. Track fails with "Embeddings not found"

### Symptoms Before Fix

```
[detect_e892c] 1537 embeddings computed
[track_e892c] ERROR: Embeddings not found: data/harvest/RHOBH_S05_E02_11072025/embeddings.parquet
```

Even though Detect reported success, Track immediately failed because the file wasn't fully written to disk.

### Changes Made

**File: `jobs/tasks/detect_embed.py` (lines 549-565)**

```python
# Save embeddings atomically to prevent race conditions with Track stage
embeddings_df = pd.DataFrame(embeddings_data)
embeddings_path = detect_dir / "embeddings.parquet"
embeddings_tmp = detect_dir / "embeddings.parquet.tmp"

# Write to temp file first
embeddings_df.to_parquet(embeddings_tmp, index=False)

# Fsync to ensure data is written to disk
with open(embeddings_tmp, 'r+b') as f:
    f.flush()
    os.fsync(f.fileno())

# Atomic rename
os.replace(embeddings_tmp, embeddings_path)

logger.info(f"[{job_id}] Saved {len(embeddings_df)} embeddings to {embeddings_path} (atomic write)")
```

**File: `jobs/tasks/track.py` (lines 180-202)**

```python
# Wait for embeddings file with bounded retry (prevent race condition with Detect)
max_retries = 50  # 50 × 100ms = 5 seconds total
retry_delay = 0.1  # 100ms
embeddings_found = False

for attempt in range(max_retries):
    if embeddings_path.exists() and embeddings_path.stat().st_size > 0:
        embeddings_found = True
        logger.info(f"[{job_id}] Embeddings file found: {embeddings_path} (size={embeddings_path.stat().st_size} bytes)")
        break
    elif attempt < max_retries - 1:
        logger.debug(f"[{job_id}] Waiting for embeddings file (attempt {attempt + 1}/{max_retries})...")
        time.sleep(retry_delay)

if not embeddings_found:
    # Enhanced error with debugging info
    cwd = Path.cwd()
    dir_listing = list(harvest_dir.glob("*")) if harvest_dir.exists() else []
    raise ValueError(
        f"ERR_EMBEDDINGS_MISSING: Embeddings file not found or empty after {max_retries * retry_delay}s wait. "
        f"Expected: {embeddings_path}, CWD: {cwd}, "
        f"harvest_dir contents: {[f.name for f in dir_listing]}"
    )
```

### Behavior After Fix

- Detect writes to `embeddings.parquet.tmp` first
- Calls `fsync()` to force OS to flush data to disk
- Atomically renames `.tmp` → `embeddings.parquet` (this is atomic at filesystem level)
- Track waits up to 5 seconds (50 × 100ms) for file to exist with non-zero size
- If file still doesn't exist after 5s, raises `ERR_EMBEDDINGS_MISSING` with debug info
- Race condition eliminated - Track will always see complete file or wait for it

## Problem 3: Cluster Auto-Run Generic Errors

### Root Cause

When Cluster auto-run of prerequisites failed (e.g., Track stage failed due to missing embeddings), the orchestration layer caught the exception but only showed "Unknown error" instead of extracting and propagating the specific error code like `ERR_EMBEDDINGS_MISSING`.

### Symptoms Before Fix

```
Cluster (Auto-run prerequisites): Auto-run of prerequisites failed: Unknown error
```

User couldn't tell what actually went wrong - was it embeddings missing? Track failure? Cluster failure?

### Changes Made

**File: `jobs/tasks/orchestrate.py` (lines 486-523)**

```python
if prep_result.get("status") != "ok":
    # Extract detailed error information
    error_detail = prep_result.get('error', 'Unknown error')
    error_code = "ERR_PREREQ_FAILED"

    # Try to extract error code from error message
    if "ERR_" in str(error_detail):
        # Extract error code (e.g., ERR_EMBEDDINGS_MISSING)
        import re
        match = re.search(r'(ERR_[A-Z_]+)', str(error_detail))
        if match:
            error_code = match.group(1)

    # Include full result for debugging
    error_msg = f"{error_code}: Auto-run of prerequisites failed: {error_detail}"
    logger.error(f"[{job_id}] {error_msg}")
    logger.error(f"[{job_id}] Full prep_result: {prep_result}")

    # Emit error progress with detailed message
    emit_progress(
        episode_id=episode_id,
        step="Cluster (Auto-run prerequisites)",
        step_index=1,
        total_steps=4,
        status="error",
        message=f"{error_code}: {error_detail[:150]}",
        pct=0.0,
    )

    return {
        "episode_id": episode_id,
        "job_id": job_id,
        "status": "error",
        "error": error_msg,
        "error_code": error_code,
        "error_detail": error_detail,
        "prep_result": prep_result,  # Include full result for debugging
    }
```

### Behavior After Fix

- Extracts error codes using regex pattern matching for `ERR_*` strings
- Returns structured error dict with `error_code`, `error_detail`, and full `prep_result`
- Emits progress messages with specific error codes: `ERR_EMBEDDINGS_MISSING`, `ERR_PREREQ_FAILED`, etc.
- Logs full exception context for debugging
- Users see exactly what failed instead of generic "Unknown error"

## Polling Architecture Overview

### Session State Keys

- **`active_polling_jobs`** - Dict of active jobs by stage: `{"detect": "detect_RHOBH_S05_E02", "full": "full_RHOBH_S05_E02"}`
- **`active_job`** - Current primary job ID for progress tracking
- **`last_run_ts`** - ISO timestamp when job was started (for log filtering)
- **`last_run_start_ts`** - Unix timestamp for elapsed time calculation
- **`first_heartbeat_seen`** - Boolean flag to track if progress has started
- **`progress_state`** - Dict with `{stage, pct, message}` for rendering progress bars

### Auto-Attach Flow

```
Page Load
   ↓
Check for active detect job (episodes.runtime.get_active_job)
   ↓
If found and not already polling:
   - Set active_polling_jobs["detect"] = job_id
   - Set active_job = job_id
   - Seed last_run_ts with current UTC timestamp
   - Log: "[UI] Auto-attached to detect job"
   ↓
st_autorefresh triggers every 2s (at top of main())
   ↓
On each refresh:
   - Read pipeline_state.json
   - Update progress_state in session
   - Render progress bars with latest pct/message
   - Append new logs to workspace_debug.log
   ↓
When job completes (status="done"):
   - Clear active_polling_jobs
   - Clear active_job
   - Stop auto-refresh (no active jobs = no refresh)
```

### Fallback Polling Logic

If `active_polling_jobs` is empty BUT:
- `artifact_status.detected == False` (detection not complete)
- OR `pipeline_state.status == "running"` and `"detect" in current_step`

Then auto-attach to polling anyway:
```python
st.session_state["active_polling_jobs"] = {"detect": f"detect_{current_ep}"}
```

This handles edge cases where the job envelope doesn't exist but the pipeline is actually running.

## Test/Verification Checklist

### UI Auto-Refresh
- [ ] Load Workspace page with active detect job - should see heartbeat indicator immediately
- [ ] Click "Full Pipeline" button - progress bars should update every 2s without any user action
- [ ] Logs should append continuously during job execution
- [ ] Progress % and face count should increment in real-time
- [ ] When job completes, heartbeat indicator should disappear
- [ ] No "Live snapshot unavailable" warnings

### Embeddings Race Condition
- [ ] Run Full Pipeline on a fresh episode
- [ ] Track stage should never fail with "Embeddings not found"
- [ ] Check logs for "Saved {N} embeddings to {path} (atomic write)"
- [ ] Track should log "Embeddings file found: {path} (size={bytes} bytes)"
- [ ] If Track does fail, error should be `ERR_EMBEDDINGS_MISSING` with debug info

### Cluster Auto-Run Error Propagation
- [ ] Trigger Cluster button on episode with missing embeddings
- [ ] Error message should show specific code: `ERR_EMBEDDINGS_MISSING: ...`
- [ ] NOT generic "Unknown error"
- [ ] Check logs for "Full prep_result: {...}" debug output

### Regression Tests
- [ ] All pipeline stages still work end-to-end
- [ ] Episode selector doesn't break on episode change
- [ ] Resume/Cancel detect buttons still work
- [ ] Stills generation error recovery still works
- [ ] Analytics button still works after clustering

## Configuration Toggles

### Current Hardcoded Values
- `st_autorefresh` interval: **2000ms** (2 seconds)
- Embeddings wait total: **5000ms** (5 seconds = 50 retries × 100ms)
- Embeddings wait backoff: **100ms** between retries

### Suggested Future Config (`configs/ui.yaml`)
```yaml
ui:
  poll_interval_ms: 2000  # Auto-refresh interval for live progress
  heartbeat_visible: true  # Show "Live polling active" caption

jobs:
  embeddings_wait_ms_total: 5000  # Total time to wait for embeddings file
  embeddings_wait_backoff: 100    # Delay between retry attempts (ms)
```

## Rollback Steps

If this patch causes issues, rollback with:

```bash
git checkout main
git branch -D fix/ui-live-polling-embeddings-race-2025-11-07

# Restart Streamlit
pkill -f "streamlit run"
.venv/bin/streamlit run app/Home.py --server.port=8888 --server.address=localhost
```

**Expected behavior after rollback:**
- UI will require manual "Resume Detect" button to see progress
- Embeddings race condition may occasionally occur (Track fails with "Embeddings not found")
- Cluster auto-run will show "Unknown error" instead of specific codes

## Related Issues & PRs

- Original issue: "Fix detect path still requires Resume to update UI"
- Related: "Fix stale UI polling + Cluster auto-run crash + missing embeddings race"
- Commit: 95cd271

## Author Notes

**Key Insight:** Streamlit's `st_autorefresh` component must be called **before any other widgets** in the page rendering flow. Placing it deep in conditional logic (like "only if not using status panel") prevents it from ever executing early enough to trigger the refresh.

**Testing Tip:** To verify auto-refresh is working, add a timestamp to the heartbeat caption:
```python
st.caption(f"🔄 Live polling active • {datetime.now().strftime('%H:%M:%S')}")
```

The timestamp should increment every 2 seconds when a job is running.

**Performance:** The 2-second polling interval is a good balance between responsiveness and server load. For large episodes (>10k frames), consider increasing to 3-5 seconds to reduce I/O overhead.
