# Detect/Embed Resume & Stall Recovery - Implementation Plan

**Date**: 2025-01-07
**Status**: 📋 PLAN READY FOR REVIEW
**Priority**: HIGH - Fixes critical UX issue with detect stalls after refresh

## Problem Statement

Users click "Detect/Embed", progress reaches mid-bar, then appears frozen. After browser refresh, the UI shows the same frozen progress bar and never completes. The job is actually running but the UI lost track of it.

## Root Causes

1. **UI loses job_id on refresh** - No persistence of active job identity
2. **Insufficient heartbeat frequency** - 30s interval too slow to detect stalls
3. **No duplicate job prevention** - Can accidentally start multiple detect jobs
4. **No recovery mechanism** - Users can't resume or cancel stalled jobs
5. **Session state loss** - episode_key/episode_id mismatch after refresh

## Solution Overview

Implement a comprehensive job persistence and recovery system:

1. **Runtime tracking** - Persist active job_id per episode in `runtime.json`
2. **Faster heartbeats** - 10s interval instead of 30s
3. **Lock files** - Prevent duplicate jobs
4. **Resume/Cancel UI** - User controls for recovery
5. **Automatic reattachment** - UI reconnects to existing jobs on load

## Implementation Plan

### Phase 1: Runtime Tracking Module

**File**: `episodes/runtime.py` ✅ CREATED

Functions:
- `set_active_job(episode_key, stage, job_id)` - Mark job as active
- `get_active_job(episode_key, stage)` - Get active job_id
- `clear_active_job(episode_key, stage)` - Clear on completion/cancel
- `check_job_stalled(job_id)` - Detect if heartbeat >30s old

**Storage**: `data/episodes/{episode_key}/runtime.json`
```json
{
  "active_jobs": {
    "detect": "detect_RHOBH_S05_E03_20250107"
  },
  "updated_at": "2025-01-07T12:34:56.789Z"
}
```

### Phase 2: Detect Worker Updates

**File**: `jobs/tasks/detect_embed.py`

**Changes needed**:

1. **Reduce checkpoint interval** (line 29):
   ```python
   CHECKPOINT_INTERVAL_SEC = 10  # Changed from 30
   ```

2. **Add runtime tracking on start** (after line 138):
   ```python
   # Register active job in runtime
   from episodes.runtime import set_active_job
   set_active_job(episode_key, "detect", job_id, DATA_ROOT)
   logger.info(f"[DETECT] {episode_key} Registered as active job: {job_id}")
   ```

3. **Create lock file** (after envelope creation, ~line 115):
   ```python
   # Create lock file to prevent duplicates
   lock_path = job_dir / ".lock"
   lock_data = {
       "job_id": job_id,
       "pid": os.getpid(),
       "started_at": datetime.utcnow().isoformat(),
   }
   with open(lock_path, "w") as f:
       json.dump(lock_data, f, indent=2)
   logger.info(f"[DETECT] {episode_key} Created lock file")
   ```

4. **Check for existing job** (before starting, ~line 95):
   ```python
   # Check if another detect job is active
   from episodes.runtime import get_active_job, check_job_stalled
   existing_job = get_active_job(episode_key, "detect", DATA_ROOT)
   if existing_job and existing_job != job_id:
       # Check if it's truly active or stalled
       if not check_job_stalled(existing_job, DATA_ROOT):
           raise ValueError(f"Detect job already running: {existing_job}. Use Resume or Cancel in UI.")
       else:
           logger.warning(f"[DETECT] {episode_key} Stalled job {existing_job} found, proceeding with new job")
   ```

5. **Clear runtime on completion** (in success block, ~line 656):
   ```python
   # Clear active job from runtime
   from episodes.runtime import clear_active_job
   clear_active_job(episode_key, "detect", DATA_ROOT)
   logger.info(f"[DETECT] {episode_key} Cleared active job from runtime")

   # Remove lock file
   lock_path = job_dir / ".lock"
   if lock_path.exists():
       lock_path.unlink()
   ```

6. **Clear runtime on error** (in except block, ~line 605):
   ```python
   # Clear active job on error
   from episodes.runtime import clear_active_job
   clear_active_job(episode_key, "detect", DATA_ROOT)
   ```

7. **Add heartbeat sequence numbering** (in checkpoint block, ~line 441):
   ```python
   # Add sequence number to track heartbeats
   heartbeat_seq = processed_frames  # Use frame count as seq
   logger.info(f"[DETECT] {episode_key} hb seq={heartbeat_seq} frames={processed_frames}/{total_frames} pct={progress_pct:.1f}% faces={detection_stats['faces_detected']}")
   ```

### Phase 3: Workspace UI Updates

**File**: `app/pages/3_🗂️_Workspace.py`

**Changes needed**:

1. **Check for active jobs on load** (in Workspace main function, after episode selection):
   ```python
   # Check for active detect job
   from episodes.runtime import get_active_job, check_job_stalled
   active_detect_job = get_active_job(episode_key, "detect", DATA_ROOT)
   detect_is_stalled = False

   if active_detect_job:
       detect_is_stalled = check_job_stalled(active_detect_job, DATA_ROOT)
       if detect_is_stalled:
           st.warning(f"⚠️ Detect job appears stalled (no heartbeat >30s). Use Resume or Cancel.")
       else:
           st.info(f"ℹ️ Detect job active: {active_detect_job}")
   ```

2. **Update Detect button logic** (in header_cols[0] block):
   ```python
   # Disable detect button if job is active and not stalled
   detect_disabled = (active_detect_job and not detect_is_stalled) or not extraction_ready

   if active_detect_job and not detect_is_stalled:
       detect_help = f"Detect job already running: {active_detect_job}"
   elif active_detect_job and detect_is_stalled:
       detect_help = "Detect job stalled. Use Resume or Cancel buttons below."
   else:
       detect_help = STAGE_HELP.get("detect", "Run detection and embedding")
   ```

3. **Add Resume/Cancel buttons** (after Detect button):
   ```python
   # Show Resume/Cancel controls if job is active
   if active_detect_job:
       col1, col2 = st.columns(2)

       with col1:
           if st.button("🔄 Resume Detect", key="resume_detect", use_container_width=True):
               st.session_state["_resume_detect"] = active_detect_job
               st.rerun()

       with col2:
           if st.button("❌ Cancel Detect", key="cancel_detect", use_container_width=True):
               st.session_state["_cancel_detect"] = active_detect_job
               st.rerun()
   ```

4. **Add Resume handler** (after button handlers):
   ```python
   # Handle Resume
   if st.session_state.pop("_resume_detect", None):
       st.info("🔄 **Resuming Detect job...**")
       st.write(f"Reattaching to job: {active_detect_job}")

       # Just reattach polling - job is already running
       # The progress polling code will pick it up automatically
       st.success("✅ Resumed! Polling for progress...")
       time.sleep(1)
       st.rerun()
   ```

5. **Add Cancel handler** (after Resume handler):
   ```python
   # Handle Cancel
   if st.session_state.pop("_cancel_detect", None):
       st.warning("❌ **Canceling Detect job...**")

       from episodes.runtime import clear_active_job
       from api.jobs import job_manager

       try:
           # Mark job as canceled in envelope
           job_manager.update_stage_status(active_detect_job, "detect", "canceled")

           # Clear from runtime
           clear_active_job(episode_key, "detect", DATA_ROOT)

           # Remove lock file if exists
           lock_path = DATA_ROOT / "jobs" / active_detect_job / ".lock"
           if lock_path.exists():
               lock_path.unlink()

           st.success("✅ Detect job canceled. You can start a new run.")
           st.rerun()
       except Exception as e:
           st.error(f"Failed to cancel: {e}")
   ```

6. **Update progress polling** (in existing polling code):
   ```python
   # Use active_detect_job if available, otherwise use current job_id
   poll_job_id = active_detect_job or current_job_id

   # Poll envelope
   envelope_path = DATA_ROOT / "jobs" / poll_job_id / "meta.json"
   if envelope_path.exists():
       with open(envelope_path) as f:
           envelope = json.load(f)

       # Check for stall
       stages = envelope.get("stages", {})
       detect_stage = stages.get("detect", {})
       if detect_stage.get("status") == "running":
           result = detect_stage.get("result", {})
           updated_at = result.get("updated_at")

           if updated_at:
               from datetime import datetime, timezone
               last_update = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
               now = datetime.now(timezone.utc)
               age_seconds = (now - last_update).total_seconds()

               if age_seconds > 30:
                   st.warning(f"⚠️ No heartbeat for {age_seconds:.0f}s. Job may be stalled.")
   ```

### Phase 4: Testing Checklist

**Scenario 1: Normal operation**
- [ ] Start Detect
- [ ] Verify runtime.json created with job_id
- [ ] Verify lock file created
- [ ] Verify heartbeats every 10s in envelope
- [ ] Verify completion clears runtime and lock

**Scenario 2: Refresh mid-run**
- [ ] Start Detect, let it run to 50%
- [ ] Refresh browser
- [ ] Verify UI shows "Detect job active" message
- [ ] Verify Detect button disabled
- [ ] Verify progress continues to update
- [ ] Verify completion works normally

**Scenario 3: Stall detection**
- [ ] Start Detect
- [ ] Kill worker process (simulate crash)
- [ ] Wait 35 seconds
- [ ] Verify UI shows "appears stalled" warning
- [ ] Verify Resume/Cancel buttons appear

**Scenario 4: Resume**
- [ ] Create stalled job (Scenario 3)
- [ ] Click Resume
- [ ] Verify UI reattaches to job
- [ ] Restart worker manually
- [ ] Verify progress resumes

**Scenario 5: Cancel**
- [ ] Create stalled job (Scenario 3)
- [ ] Click Cancel
- [ ] Verify runtime cleared
- [ ] Verify lock file removed
- [ ] Verify envelope marked as canceled
- [ ] Verify can start new Detect job

**Scenario 6: Duplicate prevention**
- [ ] Start Detect
- [ ] Try to start another Detect (e.g., new tab)
- [ ] Verify error: "Detect job already running"
- [ ] Verify no duplicate job created

**Scenario 7: Error handling**
- [ ] Start Detect with intentional error (e.g., bad video path)
- [ ] Verify runtime cleared on error
- [ ] Verify lock file removed
- [ ] Verify can retry

## Benefits

✅ **No more lost jobs** - UI always reconnects to running jobs
✅ **Faster stall detection** - 10s heartbeats vs 30s
✅ **User control** - Resume/Cancel buttons for recovery
✅ **Duplicate prevention** - Lock files prevent conflicts
✅ **Graceful degradation** - Stalled jobs can be recovered or canceled
✅ **Better UX** - Clear messaging about job status

## Files Modified

| File | Status | Changes |
|------|--------|---------|
| `episodes/runtime.py` | ✅ Created | Job tracking module |
| `jobs/tasks/detect_embed.py` | 📋 Planned | Runtime tracking, 10s heartbeat, lock files |
| `app/pages/3_🗂️_Workspace.py` | 📋 Planned | Active job detection, Resume/Cancel UI |

## Backward Compatibility

✅ **Fully backward compatible**:
- Existing jobs without runtime.json work normally
- No breaking changes to envelope format
- Lock files are optional (no existing code depends on them)
- Resume/Cancel UI only shows when relevant

## Next Steps

1. **Review this plan** - Confirm approach is correct
2. **Implement Phase 2** - Update detect_embed.py
3. **Implement Phase 3** - Update Workspace UI
4. **Test all scenarios** - Run through testing checklist
5. **Document** - Update user-facing docs

## Estimated Time

- Phase 2 implementation: ~30 minutes
- Phase 3 implementation: ~45 minutes
- Testing: ~30 minutes
- **Total**: ~1.5-2 hours

## Questions for User

1. Should we apply this pattern to Track/Cluster stages too?
2. Should Cancel also kill the worker process, or just mark as canceled?
3. Should we show a "time since last heartbeat" indicator in the UI?
4. Should we add a manual "Check Status" button for debugging?

---

**Ready to proceed?** Let me know if this plan looks good and I'll start implementing Phase 2!
