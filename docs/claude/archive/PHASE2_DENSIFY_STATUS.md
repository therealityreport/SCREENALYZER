# Phase 2 Status: Densify 2-Pass (Partial Complete - Runtime Blocker)

**Status**: ⚠️ SETUP COMPLETE, EXECUTION PENDING (runtime exceeds phase budget)
**Time Spent**: 50 minutes (config + script development)
**Blocker**: Densify execution requires 30-60+ minutes (video decode + detection + embedding)

---

## What's Complete ✅

### 1. Configuration (pipeline.yaml)
**File**: [configs/pipeline.yaml](../configs/pipeline.yaml) (lines 178-227)

**Pass 1 (Conservative)**:
```yaml
local_densify:
  detection:
    min_confidence: 0.58   # ↓ from 0.70
    min_face_px: 44        # ↓ from 72
    scales: [1.0, 1.25, 1.5, 2.0]
  verify:
    min_sim: 0.86
    min_margin: 0.12
    min_consecutive: 4
```

**Pass 2 (Aggressive)**:
```yaml
local_densify_pass2:
  trigger_threshold: 4.5   # Only run if still >4.5s error after pass 1
  detection:
    min_confidence: 0.50   # ↓ more aggressive
    min_face_px: 36        # ↓ capture very small faces
    scales: [1.0, 1.35, 1.7, 2.2]
  verify:
    min_sim: 0.88          # ↑ tighter verification
    min_margin: 0.14       # ↑ wider margin
    min_consecutive: 5     # ↑ more evidence required
```

### 2. Execution Script
**File**: [jobs/tasks/densify_two_pass.py](densify_two_pass.py) (269 lines)

**Features**:
- Loads config from pipeline.yaml
- Skips frozen identities (KIM, KYLE, LVP)
- Pass 1: Runs conservative densify on all non-frozen
- Computes delta after Pass 1
- Pass 2: Runs aggressive densify ONLY on identities still >4.5s error
- Generates audit reports: `densify_pass1_audit.json`, `densify_pass2_audit.json`
- Reports seconds_recovered per identity

**Syntax**: ✅ Validated (imports correct, no errors)

---

## Current Baseline (Before Densify)

| Person   | Auto (ms) | GT (ms) | Δ (s)  | Error % | Status |
|----------|-----------|---------|--------|---------|--------|
| YOLANDA  | 15999     | 16002   | -0.00  | -0.0%   | ✅ PASS |
| KIM      | 49501     | 48004   | +1.50  | +3.1%   | ✅ PASS |
| KYLE     | 23751     | 21017   | +2.73  | +13.0%  | ❌ FAIL |
| RINNA    | 30084     | 25015   | +5.07  | +20.3%  | ❌ FAIL |
| EILEEN   | 14416     | 10001   | +4.42  | +44.1%  | ❌ FAIL |
| BRANDI   | 6585      | 10014   | -3.43  | -34.2%  | ❌ FAIL |
| LVP      | 3167      | 2018    | +1.15  | +56.9%  | ❌ FAIL |

**Pass Rate**: 2/7 (29%)

**Frozen** (will skip in densify): KIM, KYLE, LVP

**Pass 1 Targets**: YOLANDA, RINNA, EILEEN, BRANDI (4 identities)

---

## Expected Execution Plan

### Pass 1 (Conservative Densify)
**Targets**: YOLANDA, RINNA, EILEEN, BRANDI

**Process**:
1. Identify gap windows >2s for each identity from timeline.csv
2. Decode frames at 30fps in gap windows (padded ±800ms)
3. Run RetinaFace detection with min_conf=0.58, min_face_px=44
4. Embed faces with ArcFaceEmbedder
5. Verify against facebank prototype (sim≥0.86, margin≥0.12)
6. Apply negative gating (reject if sim_to_others within 0.06)
7. Create tracklets from ≥4 consecutive verified frames
8. Merge tracklets into tracks.json via ReID (sim≥0.82)

**Expected Recovery**: 1-2s per identity (conservative thresholds)

**Estimated Runtime**: 15-20 minutes (4 identities × multiple gap windows)

---

### Pass 2 (Aggressive Densify)
**Trigger**: Only run if identity still >4.5s error after Pass 1

**Expected Targets** (after Pass 1):
- RINNA: Likely still >4.5s (currently +5.07s)
- EILEEN: Likely still >4.5s (currently +4.42s)
- BRANDI: Possibly (currently -3.43s, may improve to ≤4.5s after pass 1)

**Process**: Same as Pass 1 but with:
- min_conf=0.50, min_face_px=36 (more faces detected)
- sim≥0.88, margin≥0.14 (stricter verification to prevent FP)
- min_consecutive=5 (require more evidence)

**Expected Recovery**: 0.5-1.5s per identity (aggressive thresholds, tighter verification)

**Estimated Runtime**: 10-15 minutes (2-3 identities × largest residual gaps only)

---

## Blocker: Runtime vs Budget

**Phase 2 Budget**: 60 minutes
**Time Spent on Setup**: 50 minutes
**Execution Time Needed**: 25-35 minutes (Pass 1 + Pass 2)

**Total**: ~75-85 minutes (exceeds budget by 15-25 min)

### Why Densify is Slow:
1. **Video Decoding**: Must decode frames at 30fps (3× baseline 10fps) for all gap windows
2. **Detection**: RetinaFace on 640×640 frames with multiple scales [1.0, 1.25, 1.5, 2.0]
3. **Embedding**: ArcFace 512-d embedding generation for all detected faces
4. **Verification**: Cosine similarity computation against all facebank prototypes
5. **Negative Gating**: Cross-identity similarity checks to prevent pollution

**Approximate Frame Count**:
- RINNA: ~8 gap windows × 3s avg × 30fps = ~720 frames
- EILEEN: ~6 gap windows × 2.5s avg × 30fps = ~450 frames
- BRANDI: ~10 gap windows × 2s avg × 30fps = ~600 frames
- YOLANDA: ~3 gap windows × 1.5s avg × 30fps = ~135 frames
- **Total**: ~1900 frames to decode + detect + embed

**Estimated**:
- Frame decode: 0.05s/frame × 1900 = 95s
- Detection: 0.3s/frame × 1900 = 570s
- Embedding: 0.1s/face × ~3000 faces = 300s
- **Total**: ~965s = **16 minutes** (Pass 1 only)

---

## Options

### Option A: Run Now (Blow Budget)
```bash
source .venv/bin/activate && python3 jobs/tasks/densify_two_pass.py
```

**Pros**: Get actual results for Pass 1 + Pass 2
**Cons**: Exceeds 60-min phase budget by 15-25 min

---

### Option B: Defer to Next Session (Recommended)
Document setup as complete, run densify in dedicated session.

**Pros**: Respects phase budget, allows monitoring
**Cons**: No actual recovery data for this phase

**Recommendation**: Option B - densify is a long-running background task better suited for dedicated execution slot

---

### Option C: Run Pass 1 Only Now (Compromise)
Run conservative pass only, defer aggressive pass.

**Pros**: Get some recovery data (~1-2s/identity)
**Cons**: Still ~16 min runtime, may not bring all identities to ≤4.5s

---

## Deliverables (Current Status)

| Item | Status | Location |
|------|--------|----------|
| Pass 1 config | ✅ DONE | configs/pipeline.yaml lines 178-203 |
| Pass 2 config | ✅ DONE | configs/pipeline.yaml lines 205-227 |
| 2-pass script | ✅ DONE | jobs/tasks/densify_two_pass.py |
| Syntax validation | ✅ PASS | No import errors |
| Execution | ⚠️ PENDING | Requires 25-35 min runtime |
| Pass 1 audit | ⏳ PENDING | data/harvest/.../densify_pass1_audit.json |
| Pass 2 audit | ⏳ PENDING | data/harvest/.../densify_pass2_audit.json |
| Updated delta_table.csv | ⏳ PENDING | After densify + analytics re-run |

---

## Recommendation

**Pause Phase 2 at setup complete, proceed to Phase 3**

**Rationale**:
1. Config + script ready (can be run anytime)
2. Densify runtime (25-35 min) better as dedicated background task
3. Remaining phases (3-6) are higher priority for session goals
4. Can circle back to run densify offline or in next session

**Alternative**: If user approves budget extension, run Pass 1 now in background (16 min) while proceeding with Phase 3 planning

---

## Next Steps

**If Running Densify**:
```bash
# Start in background
nohup python3 jobs/tasks/densify_two_pass.py > /tmp/densify_two_pass.log 2>&1 &

# Monitor progress
tail -f /tmp/densify_two_pass.log | grep -E "(Starting PASS|RESULTS|verified|tracklets)"
```

**If Deferring**:
- Mark Phase 2 as "Setup Complete"
- Proceed to Phase 3 (Identity-Guided Recall)
- Circle back to densify after Phase 6 or in next session

---

**Phase 2 Status**: Setup complete (config + script), execution deferred due to runtime
**Files Modified**: 1 (configs/pipeline.yaml)
**Files Created**: 2 (jobs/tasks/densify_two_pass.py, PHASE2_DENSIFY_STATUS.md)
**Blocker**: Execution time (25-35 min) exceeds remaining phase budget
