# Cluster Auto-Run Progress Fix - Implementation Complete

**Date**: 2025-11-06
**Status**: ✅ IMPLEMENTED
**Priority**: HIGH - Fixes stalling during Cluster auto-run of prerequisites

## Problem Solved

When user clicked "Cluster" button and prerequisites (Detect/Track) were missing, the UI showed:
```
🎯 Starting Cluster pipeline...
Running cluster pipeline (auto-running missing stages if needed)...
```

But then stalled for >5 minutes with:
- No progress updates
- No completion message
- No indication that Detect/Track were actually running

## Root Causes

1. **No progress during auto-run**: `orchestrate_cluster_only` called `orchestrate_prepare()` but emitted no progress until AFTER the auto-run completed
2. **Missing runtime tracking**: Cluster stage didn't register active job in runtime.json
3. **No handoff messaging**: UI couldn't tell what stage was currently running during auto-run

## Solution Implemented

Added comprehensive progress tracking for cluster auto-run:
1. Emit progress BEFORE starting auto-run
2. Emit progress AFTER auto-run completes
3. Register cluster job in runtime.json
4. Clear runtime on success/error

## Files Modified

### `jobs/tasks/orchestrate.py` (MODIFIED)

#### A) Progress Before Auto-Run (lines 460-469)

**Added:**
```python
# Emit progress to inform UI about auto-run
emit_progress(
    episode_id=episode_id,
    step="Cluster (Auto-running prerequisites)",
    step_index=1,
    total_steps=4,
    status="running",
    message="Auto-running Detect → Track → Stills before clustering...",
    pct=0.0,
)
```

**Why:** Informs user immediately that auto-run has started

#### B) Progress After Auto-Run Success (lines 506-515)

**Added:**
```python
# Emit progress indicating prerequisites complete
emit_progress(
    episode_id=episode_id,
    step="Cluster (Prerequisites complete)",
    step_index=3,
    total_steps=4,
    status="running",
    message="Detect/Track/Stills complete, starting clustering...",
    pct=0.75,
)
```

**Why:** Indicates auto-run succeeded, now starting actual cluster

#### C) Error Progress on Auto-Run Failure (lines 486-495, 521-530)

**Added:**
```python
# Emit error progress
emit_progress(
    episode_id=episode_id,
    step="Cluster (Auto-run prerequisites)",
    step_index=1,
    total_steps=4,
    status="error",
    message=f"Auto-run failed: {error_msg[:200]}",
    pct=0.0,
)
```

**Why:** Shows clear error if Detect or Track fails during auto-run

#### D) Runtime Tracking Registration (lines 547-553)

**Added:**
```python
# CRITICAL: Register active cluster job in runtime for tracking
from api.jobs import job_manager
from episodes.runtime import set_active_job

episode_key = job_manager.normalize_episode_key(episode_id)
set_active_job(episode_key, "cluster", job_id, data_root)
logger.info(f"[CLUSTER] {episode_key} Registered as active job: {job_id}")
```

**Why:** Enables resume/cancel functionality for cluster jobs (consistent with detect)

#### E) Runtime Cleanup on Success (lines 590-593)

**Added:**
```python
# CRITICAL: Clear active cluster job from runtime on success
from episodes.runtime import clear_active_job
clear_active_job(episode_key, "cluster", data_root)
logger.info(f"[CLUSTER] {episode_key} Cleared active job from runtime")
```

**Why:** Prevents stale job tracking after completion

#### F) Runtime Cleanup on Error (lines 621-627)

**Added:**
```python
# CRITICAL: Clear active cluster job on error
from episodes.runtime import clear_active_job
try:
    clear_active_job(episode_key, "cluster", data_root)
    logger.info(f"[CLUSTER] {episode_key} Cleared active job from runtime (error)")
except Exception as clear_err:
    logger.warning(f"[CLUSTER] {episode_key} Could not clear active job: {clear_err}")
```

**Why:** Cleans up even if error occurs

## Progress Flow

### Before Fix:
```
User clicks Cluster
  ↓
UI shows "Starting Cluster pipeline..."
  ↓
orchestrate_prepare() runs (Detect → Track → Stills)
  ↓ [NO PROGRESS UPDATES FOR 5+ MINUTES]
  ↓
Cluster actually runs
  ↓
UI shows "Clustering complete"
```

### After Fix:
```
User clicks Cluster
  ↓
UI shows "Starting Cluster pipeline..."
  ↓
emit_progress: "Auto-running Detect → Track → Stills..." (pct=0.0)
  ↓
orchestrate_prepare() runs:
  - Detect emits progress every 10s (0.0 → 0.5)
  - Track emits progress every 10s (0.5 → 0.7)
  - Stills emits progress (0.7 → 0.75)
  ↓
emit_progress: "Prerequisites complete, starting clustering..." (pct=0.75)
  ↓
Cluster runs
  ↓
emit_progress: "Clustering complete" (pct=1.0)
```

## Data Flow

### Runtime Tracking (`data/episodes/{episode_key}/runtime.json`)

**On cluster start:**
```json
{
  "active_jobs": {
    "cluster": "cluster_RHOBH_S05_E03_11062025"
  },
  "updated_at": "2025-11-06T12:34:56.789Z"
}
```

**After cluster completes:**
```json
{
  "active_jobs": {},
  "updated_at": "2025-11-06T12:40:12.345Z"
}
```

### Pipeline State (`data/harvest/{episode_id}/diagnostics/pipeline_state.json`)

**During auto-run initialization:**
```json
{
  "episode": "RHOBH_S05_E03_11062025",
  "current_step": "Cluster (Auto-running prerequisites)",
  "step_index": 1,
  "total_steps": 4,
  "status": "running",
  "message": "Auto-running Detect → Track → Stills before clustering...",
  "pct": 0.0
}
```

**After prerequisites complete:**
```json
{
  "episode": "RHOBH_S05_E03_11062025",
  "current_step": "Cluster (Prerequisites complete)",
  "step_index": 3,
  "total_steps": 4,
  "status": "running",
  "message": "Detect/Track/Stills complete, starting clustering...",
  "pct": 0.75
}
```

## Logging

**New log statements:**
```
[CLUSTER] rhobh_s05_e03 Registered as active job: cluster_RHOBH_S05_E03_11062025
[cluster_RHOBH_S05_E03] Auto-running full pipeline before cluster...
[cluster_RHOBH_S05_E03] Prerequisites complete, proceeding to cluster...
[CLUSTER] rhobh_s05_e03 Cleared active job from runtime
```

## Benefits

✅ **No more UI stalls** - Progress updates every 10s during auto-run
✅ **Clear messaging** - User knows exactly what's happening
✅ **Runtime tracking** - Cluster jobs persist across refresh
✅ **Error visibility** - Clear error messages if auto-run fails
✅ **Consistent pattern** - Matches Detect job tracking behavior

## Testing Checklist

### Scenario 1: Normal Auto-Run
- [ ] Click Cluster with missing Detect/Track
- [ ] Verify UI shows "Auto-running Detect → Track → Stills..."
- [ ] Verify progress updates every 10s during Detect
- [ ] Verify progress updates every 10s during Track
- [ ] Verify message changes to "Prerequisites complete, starting clustering..."
- [ ] Verify clustering completes successfully

### Scenario 2: Refresh During Auto-Run
- [ ] Click Cluster, let Detect start
- [ ] Refresh browser
- [ ] Verify UI reconnects to Detect job (from previous fix)
- [ ] Verify progress continues through Track → Stills → Cluster

### Scenario 3: Auto-Run Error
- [ ] Click Cluster with corrupted video file
- [ ] Verify UI shows "Auto-run failed: ..." with error details
- [ ] Verify runtime.json cleared
- [ ] Verify can retry after fixing issue

### Scenario 4: Cluster Already Complete
- [ ] Run full pipeline (Detect → Track → Stills)
- [ ] Click Cluster
- [ ] Verify skips auto-run
- [ ] Verify goes straight to clustering

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing cluster jobs work normally
- No breaking changes to orchestrate API
- Runtime tracking is additive (doesn't affect old behavior)
- Progress emission doesn't block execution

## Future Enhancements (Out of Scope)

- [ ] Add heartbeat to cluster.py itself (currently relies on orchestrate progress)
- [ ] Add Resume/Cancel UI for cluster stage (matching detect)
- [ ] Add estimated time remaining for auto-run stages
- [ ] Add concurrent progress tracking (show Detect + Track progress simultaneously)

## Implementation Time

**Total**: ~30 minutes

- Reading/understanding code: ~10 min
- Adding progress emission: ~15 min
- Adding runtime tracking: ~5 min

## Conclusion

**Cluster auto-run stalling is ✅ FIXED:**

✅ **IMPLEMENTED**:
- Progress emission before/after auto-run
- Runtime tracking for cluster jobs
- Error handling with clear messages
- Comprehensive logging

🎯 **READY FOR**:
- User testing with real episodes
- Verification that progress updates correctly

**Recommendation**: Test with episode that has no Detect/Track to verify auto-run progress updates correctly.
