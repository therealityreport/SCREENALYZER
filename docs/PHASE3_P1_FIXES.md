# Phase 3 P1 Fixes: Video Path Resolution & Upload Automation

## Summary

Fixed the `video_path` UnboundLocalError and enforced Phase 3 P1 automation by:
1. Making video_path sourcing deterministic from episode registry
2. Adding self-healing fallback chains
3. Removing legacy Prepare button for normal workflows
4. Adding auto-refresh polling for extraction status

---

## Issues Fixed

### Issue 1: `video_path` UnboundLocalError

**Symptom:**
```
Prepare failed: cannot access local variable 'video_path' where it is not associated with a value
```

**Root Cause:**
- Orchestrator only assigned `video_path` inside the harvest block (`if not artifacts["manifest"]`)
- If frames already existed, `video_path` was never set
- Later code at line 209 tried to use undefined `video_path`

**Fix:** [jobs/tasks/orchestrate.py](jobs/tasks/orchestrate.py:154-190)

Added deterministic video_path sourcing with 3-tier fallback:
```python
# Source 1: Episode registry (Phase 2+)
episode_key = job_manager.normalize_episode_key(episode_id)
registry_data = job_manager.load_episode_registry(episode_key)
if registry_data:
    video_path = registry_data.get("video_path")

# Source 2: Old registry (fallback for legacy episodes)
if not video_path:
    # Search configs/shows_seasons.json

# Source 3: diagnostics/episodes.json (last resort)
if not video_path:
    # Search diagnostics/episodes.json

if not video_path:
    raise ValueError(f"ERR_EPISODE_NOT_REGISTERED: {episode_id}")
```

---

### Issue 2: Legacy Prepare Button Still Visible

**Symptom:**
- After Phase 3 P1 (auto-extraction), Workspace still showed "Prepare Tracks & Stills" button
- Users could click it even though frames were already extracted
- Contradicted P1 automation goal

**Fix:** [app/pages/3_🗂️_Workspace.py](app/pages/3_🗂️_Workspace.py:278-342)

Made Prepare button conditional on extraction status:
```python
# Check extraction status from episode registry
episode_state = get_episode_state(current_ep)
extraction_ready = episode_state.get("states", {}).get("extracted_frames", False)

if extraction_ready:
    # Show secondary button for re-running detection
    # (Prepare becomes "rebuild" functionality)
else:
    # Show passive extraction status with manual retry option
```

---

### Issue 3: No Auto-Refresh Polling

**Symptom:**
- After upload, user had to manually refresh browser to see extraction complete
- No feedback that extraction was happening
- Stale UI state

**Fix:** [app/pages/3_🗂️_Workspace.py](app/pages/3_🗂️_Workspace.py:282-304)

Added auto-refresh polling:
```python
if not extraction_ready:
    validated = episode_state.get("states", {}).get("validated", False)

    if validated:
        # Poll every 2 seconds for completion
        st.info("⏳ Frame extraction in progress... Page will refresh automatically.")

        poll_interval = 2
        if elapsed >= poll_interval:
            st.rerun()  # Auto-refresh
```

---

### Issue 4: detect_embed.py Indentation Bug

**Symptom:**
- Code tried to use `episodes_data` variable outside its scope
- NameError or UnboundLocalError in self-healing fallback chain

**Fix:** [jobs/tasks/detect_embed.py](jobs/tasks/detect_embed.py:123-160)

Fixed indentation and added final safety check:
```python
# Properly indented fallback chain
else:
    logger.warning("No episode registry, attempting self-heal from episodes.json")
    from app.lib.registry import load_episodes_json
    episodes_data = load_episodes_json()

    # This block is now properly inside the else
    episode_match = None
    for ep in episodes_data.get("episodes", []):
        if ep.get("episode_id") in job_id:
            episode_match = ep
            video_path = episode_match.get("video_path")
            break

# Final safety check
if not video_path:
    raise ValueError(f"ERR_EPISODE_NOT_REGISTERED: {job_id}")
```

---

## Testing Checklist

### 1. Inspect Episode State
```bash
python tools/inspect_state.py rhobh_s05_e01

# Expected output:
# ✅ validated: true
# ✅ extracted_frames: true
# Video Path: videos/rhobh/s05/RHOBH_S05_E01_11062025.mp4
```

### 2. Upload → Auto-Extract → Workspace Flow
1. Go to Upload page
2. Upload video for new episode
3. Click "Validate Video"
4. Click "Start Upload"
5. **Auto-extraction runs** → See "Extracting frames..." message
6. **On success** → Navigate to Workspace
7. **Expected:** Workspace shows "✅ Assets Ready" (no Prepare button)
8. **Expected:** Frames preview grid visible
9. **Expected:** "Run Detect/Embed" button enabled

### 3. Legacy Episode (Frames Already Exist)
1. Select episode that was processed before Phase 3
2. Workspace loads
3. **Expected:** No UnboundLocalError
4. **Expected:** Video path resolved from registry
5. **Expected:** Prepare button shows "Re-run detection" (secondary style)

### 4. Manual Retry
1. Upload episode but force extraction to fail (e.g., corrupt video)
2. Workspace shows "⏳ Extracting frames..."
3. Click "⚠️ Retry Extraction" button
4. **Expected:** Manual extraction triggered
5. **Expected:** Success/error message shown

---

## Files Modified

### Backend
- `jobs/tasks/orchestrate.py` - Always load video_path from registry
- `jobs/tasks/detect_embed.py` - Fixed self-healing fallback chain
- `api/episodes.py` - Created (episode state API)
- `jobs/tasks/auto_extract.py` - Created (auto-extraction logic)

### Frontend
- `app/pages/1_📤_Upload.py` - Added auto-extraction call after upload
- `app/pages/3_🗂️_Workspace.py` - Conditional Prepare button + polling

### Registry/Cache
- `app/lib/registry.py` - Fixed save_registry() to work without Streamlit context
- `app/lib/episode_manager.py` - Fixed purge_all_episodes() to clear registry

---

## Architecture Changes

### Before Phase 3 P1
```
Upload → Validate → Upload Complete
                           ↓
                    User manually clicks
                    "Prepare Tracks & Stills"
                           ↓
                    Frames extracted
```

### After Phase 3 P1
```
Upload → Validate → Upload Complete
                           ↓
                    [AUTO: Extract Frames]
                           ↓
                    Workspace shows frames
                    (no manual Prepare needed)
```

### video_path Resolution Order
1. **Episode Registry** (`episodes/{episode_key}/state.json`)
2. **Old Registry** (`configs/shows_seasons.json`)
3. **Episodes JSON** (`diagnostics/episodes.json`)
4. **Error** (ERR_EPISODE_NOT_REGISTERED)

---

## Edge Cases Handled

### Case 1: Redis TTL Expired (24h)
- Job envelope persists to disk
- Workers self-heal from envelope or registry
- **Result:** No UnboundLocalError, pipeline continues

### Case 2: Registry Cleared (Purge All)
- New uploads create fresh registry entries
- Old registry used as fallback for legacy episodes
- **Result:** Both old and new episodes work

### Case 3: Extraction In Progress
- Workspace shows "⏳ Extracting frames..."
- Auto-refreshes every 2 seconds
- **Result:** User sees progress without manual refresh

### Case 4: Extraction Failed
- "⚠️ Retry Extraction" button available
- Manual trigger re-runs auto_extract.trigger_auto_extraction()
- **Result:** User can recover without support intervention

---

## Known Limitations

1. **Auto-refresh polling is active** - Creates network traffic every 2s while extraction runs
2. **No progress percentage** - Shows binary state (extracting/complete)
3. **Detect/Embed not automated** - Still requires manual click (planned for future phase)
4. **No cancellation** - Once extraction starts, cannot be stopped

---

## Future Improvements (Phase 3 P2+)

1. **WebSocket-based updates** instead of polling
2. **Progress bar** showing frame extraction percentage
3. **Auto-detect after extraction** - Fully hands-off pipeline
4. **Queue visibility** - Show position in extraction queue
5. **Bandwidth optimization** - Only poll when tab is visible

---

## Rollback Instructions

If Phase 3 P1 causes issues:

```bash
# 1. Restore orchestrator
git checkout HEAD~1 jobs/tasks/orchestrate.py

# 2. Restore Workspace UI
git checkout HEAD~1 app/pages/3_🗂️_Workspace.py

# 3. Remove auto-extract
rm jobs/tasks/auto_extract.py

# 4. Remove episode API
rm api/episodes.py

# 5. Revert Upload page
git checkout HEAD~1 app/pages/1_📤_Upload.py
```

Users will need to use the manual "Prepare Tracks & Stills" button again.

---

## Contact

For questions about Phase 3 P1 implementation, see:
- **Architecture**: [docs/PHASE3_ARCHITECTURE.md](PHASE3_ARCHITECTURE.md)
- **Testing**: [tools/test_registry.py](../tools/test_registry.py)
- **State Inspection**: `python tools/inspect_state.py <episode_key>`
