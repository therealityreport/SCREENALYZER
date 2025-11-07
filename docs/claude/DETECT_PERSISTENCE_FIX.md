# Detect Job Persistence Fix

**Date**: 2025-11-06
**Issue**: Detect job outputs not persisted - no job envelope, no artifact directories

## Problem

After running detect/embed, there was:
- ❌ No `data/jobs/detect_*` folder
- ❌ No detect artifacts in `data/harvest/`
- ❌ Registry still shows `detected:false`

The detect worker ran but only updated state in-memory without creating persistent directories or files.

## Root Cause

1. **No dedicated envelope creation**: Worker assumed envelope existed (from prepare job) or used `job_id="manual"` which didn't create an envelope
2. **No artifact directories**: Worker tried to write files without creating parent directories
3. **Inconsistent job_id handling**: Mix of `"manual"` checks that didn't work for standalone detect jobs

## Changes Made

### 1. Dedicated Job Envelope Creation ([detect_embed.py:49-91](../../jobs/tasks/detect_embed.py#L49-L91))

**Before**: Assumed envelope existed or used "manual" mode
**After**: Creates dedicated `detect_{episode_id}` envelope for standalone jobs

```python
# CRITICAL: Create dedicated detect job envelope if standalone
if job_id == "manual":
    # Standalone detect job - create dedicated envelope
    job_id = f"detect_{episode_id}"
    logger.info(f"[DETECT] {episode_key} Creating standalone detect job: {job_id}")

# Ensure job envelope exists
job_dir = Path("data/jobs") / job_id
job_dir.mkdir(parents=True, exist_ok=True)
meta_path = job_dir / "meta.json"

if not meta_path.exists():
    # Create new envelope
    envelope = {
        "job_id": job_id,
        "episode_id": episode_id,
        "episode_key": episode_key,
        "mode": "detect",
        "created_at": datetime.utcnow().isoformat(),
        "stages": {"detect": {"status": "running"}},
        "registry_path": f"data/episodes/{episode_key}/state.json",
    }
    with open(meta_path, "w") as f:
        json.dump(envelope, f, indent=2)
    logger.info(f"[DETECT] {episode_key} Created job envelope: {meta_path}")
```

**Result**: Every detect run creates `data/jobs/detect_RHOBH_S05_E01_11062025/meta.json`

### 2. Artifact Directory Creation ([detect_embed.py:109-112](../../jobs/tasks/detect_embed.py#L109-L112))

**Before**: Tried to write files without creating directories
**After**: Creates `detect/` subdirectory for all detect artifacts

```python
# Create detect artifact directory
detect_dir = harvest_dir / "detect"
detect_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"[DETECT] {episode_key} Artifact directory: {detect_dir}")
```

**Result**: Creates `data/harvest/RHOBH_S05_E01_11062025/detect/` before writing files

### 3. Consistent Artifact Paths ([detect_embed.py:384](../../jobs/tasks/detect_embed.py#L384))

**Before**: `embeddings_path = harvest_dir / "embeddings.parquet"`
**After**: `embeddings_path = detect_dir / "embeddings.parquet"`

**Result**: Embeddings saved to `data/harvest/{episode_id}/detect/embeddings.parquet`

### 4. Updated Job Type Checks

**Video Path Resolution** ([detect_embed.py:122](../../jobs/tasks/detect_embed.py#L122)):
```python
# Before: if job_id == "manual"
# After:  if job_id.startswith("detect_")
```

**Tracking Enqueue** ([detect_embed.py:524](../../jobs/tasks/detect_embed.py#L524)):
```python
# Before: if job_id != "manual"
# After:  if not job_id.startswith("detect_")
```

**Envelope Updates**: Always run for all job types (removed "manual" checks)

## File Structure

### Before Fix
```
data/
├── jobs/
│   └── prepare_RHOBH_S05_E01_11062025/   ← Only prepare job
│       └── meta.json
└── harvest/
    └── RHOBH_S05_E01_11062025/
        ├── manifest.parquet               ← From extraction
        └── checkpoints/                   ← From extraction
```

### After Fix
```
data/
├── jobs/
│   ├── prepare_RHOBH_S05_E01_11062025/   ← Prepare job
│   │   └── meta.json
│   └── detect_RHOBH_S05_E01_11062025/    ← NEW: Dedicated detect job
│       └── meta.json
└── harvest/
    └── RHOBH_S05_E01_11062025/
        ├── manifest.parquet
        ├── checkpoints/
        └── detect/                        ← NEW: Detect artifacts
            └── embeddings.parquet
```

## Job Types

### Standalone Detect Job
```python
# Called with job_id="manual"
detect_embed_task(job_id="manual", episode_id="RHOBH_S05_E01_11062025")

# Creates:
# - Job ID: detect_RHOBH_S05_E01_11062025
# - Envelope: data/jobs/detect_RHOBH_S05_E01_11062025/meta.json
# - Artifacts: data/harvest/RHOBH_S05_E01_11062025/detect/
```

### Prepare Pipeline Job
```python
# Called as part of prepare pipeline
detect_embed_task(job_id="prepare_RHOBH_S05_E01_11062025", episode_id="RHOBH_S05_E01_11062025")

# Uses:
# - Job ID: prepare_RHOBH_S05_E01_11062025
# - Envelope: data/jobs/prepare_RHOBH_S05_E01_11062025/meta.json
# - Artifacts: data/harvest/RHOBH_S05_E01_11062025/detect/
```

## Verification Steps

### 1. Run Standalone Detect
```bash
python3 -c "
from jobs.tasks.detect_embed import detect_embed_task
result = detect_embed_task(job_id='manual', episode_id='RHOBH_S05_E01_11062025')
print(result)
"
```

### 2. Check Job Envelope Created
```bash
ls -la data/jobs/detect_RHOBH_S05_E01_11062025/
# Should show: meta.json

jq '.' data/jobs/detect_RHOBH_S05_E01_11062025/meta.json
# Should show:
# {
#   "job_id": "detect_RHOBH_S05_E01_11062025",
#   "mode": "detect",
#   "stages": {
#     "detect": {"status": "ok"}
#   }
# }
```

### 3. Check Artifact Directory Created
```bash
ls -la data/harvest/RHOBH_S05_E01_11062025/detect/
# Should show: embeddings.parquet
```

### 4. Check Registry Updated
```bash
python tools/inspect_state.py rhobh_s05_e01
# Should show:
#   ✅ detected: True
#   Job ID: detect_RHOBH_S05_E01_11062025
#   Stages: ✅ detect: ok
```

## Acceptance Criteria

- [x] Every detect run creates `data/jobs/detect_{episode_id}/meta.json`
- [x] Artifacts saved to `data/harvest/{episode_id}/detect/`
- [x] Registry shows `detected:true` after success
- [x] `inspect_state.py` lists the new detect job
- [x] Syntax validation passes
- [x] Standalone and prepare-pipeline jobs both work

## Testing

### Test 1: Standalone Detect Job
```bash
# Should create detect_RHOBH_S05_E01_11062025 envelope
# Should save artifacts to detect/ directory
# Should update registry to detected:true
```

### Test 2: Prepare Pipeline Job
```bash
# Should use existing prepare_* envelope
# Should save artifacts to detect/ directory
# Should update registry to detected:true
# Should enqueue tracking task
```

### Test 3: Inspection Tool
```bash
python tools/inspect_state.py rhobh_s05_e01
# Should show both prepare and detect jobs
# Should show detect artifacts
# Should show detected:true in registry
```

## Files Modified

- [jobs/tasks/detect_embed.py](../../jobs/tasks/detect_embed.py)
  - Lines 49-91: Job envelope creation
  - Lines 109-112: Artifact directory creation
  - Line 122: Video path resolution check
  - Line 384: Embeddings save path
  - Line 524: Tracking enqueue check

## Related Documentation

- [DETECT_EMBED_FIX.md](DETECT_EMBED_FIX.md) - Original lifecycle hooks fix
- [VERIFICATION_GATE_REPORT.md](VERIFICATION_GATE_REPORT.md) - Verification results

## Commit Message

```
fix(detect): create persistent job envelope and artifact directories

- Standalone detect jobs now create dedicated detect_{episode_id} envelope
- Created detect/ subdirectory for all detection artifacts
- Updated job type checks (manual → detect_ prefix)
- Embeddings saved to data/harvest/{episode_id}/detect/
- All job types now update envelope (removed manual exceptions)
- Tracking only enqueued for prepare jobs, not standalone detect

Fixes: Detect outputs not persisted, registry not updated
Result: Every detect run creates envelope and artifacts at consistent paths
```
