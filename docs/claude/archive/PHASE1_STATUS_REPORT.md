# Phase 1 Status Report - RHOBH-TEST-10-28

**Date:** 2025-10-29
**Goal:** Achieve ≤±5% accuracy for all 7 cast members with no manual overrides

---

## ✅ Major Fixes Completed

### 1. Excel Time Format Fixed
- **Issue:** Millisecond columns displayed as AM/PM times in Excel
- **Fix:** Applied integer number formatting to all `*_ms` columns
- **Location:** [jobs/tasks/analytics.py:185-206](jobs/tasks/analytics.py#L185-L206)

### 2. Track Overlap Elimination
- **Issue:** Total screen time (226s) exceeded video duration (103s) due to cross-cut false matches
- **Root Cause:** ByteTrack with loose matching (match_thresh=0.8, iou_thresh=0.5) created 63-second tracks spanning multiple people across camera cuts
- **Fix:** Tightened matching thresholds:
  - `match_thresh`: 0.8 → **0.9** (stricter appearance similarity)
  - `iou_thresh`: 0.5 → **0.3** (stricter spatial overlap)
- **Result:** ✓ **ZERO overlapping intervals** - total screen time now within video bounds
- **Location:** [configs/bytetrack.yaml](configs/bytetrack.yaml)

### 3. Re-ID Track Linking Working
- **Status:** ✓ **34 track links created** (20.6% acceptance rate)
- **Impact:** Successfully reconnected fragmented tracks for KIM and KYLE
- **Config:** `min_sim=0.82`, `min_margin=0.08`, `max_gap_ms=2500`
- **Location:** [configs/pipeline.yaml:tracking.reid](configs/pipeline.yaml)

### 4. Adaptive Gap-Merge Working
- **Status:** ✓ Quality-aware interval merging with conflict guards
- **Stats:** 61/101 gaps merged, 20 quality bumps, 7 conflicts blocked
- **Config:** `gap_merge_ms_base=2500`, `gap_merge_ms_max=3000`, `min_interval_quality=0.70`, `conflict_guard_ms=700`
- **Location:** [screentime/attribution/timeline.py](screentime/attribution/timeline.py)

---

## 📊 Current Baseline Results

**Configuration:**
- Strict matching: `match_thresh=0.9`, `iou_thresh=0.3`
- Re-ID enabled: `min_sim=0.82`, `min_margin=0.08`
- Adaptive gap-merge enabled

### Accuracy Table

| Person   | Auto (s) | GT (s) | Delta (ms) | Error % | Status |
|----------|----------|--------|------------|---------|--------|
| **KIM**      | 47.0 | 48.0 | +1,004 | **+2.1%** | ✓ **PASS** |
| **KYLE**     | 20.0 | 21.0 | +1,017 | **+4.8%** | ✓ **PASS** |
| RINNA    | 14.0 | 25.0 | +11,015 | +44.0% | ✗ FAIL |
| BRANDI   | 4.5  | 10.0 | +5,514  | +55.1% | ✗ FAIL |
| YOLANDA  | 4.5  | 16.0 | +11,502 | +71.9% | ✗ FAIL |
| LVP      | 1.5  | 2.0  | +518    | +25.7% | ✗ FAIL |
| EILEEN   | 12.5 | 10.0 | **-2,499** | **-25.0%** | ✗ **OVERCOUNT** |

**Results:**
- ✅ **2/7 cast members within ±5%** (KIM, KYLE)
- ✅ **Zero overlapping intervals**
- ⚠️ **1 person overcounting** (EILEEN - likely wrong cluster assignment)
- ✗ **4 people severely undercounted** (missing 44-72% of screen time)

---

## 🔍 Root Cause Analysis

### KIM & KYLE: ✅ Working Correctly
- **Why they succeed:** More screen time (48s, 21s) = more tracks (82, 25) = re-ID has material to link
- **Track fragmentation:** Strict matching creates short tracks, but re-ID successfully reconnects them
- **Quality:** High-confidence detections in well-lit, frontal shots

### EILEEN: ⚠️ Wrong Cluster Assignment
- **Issue:** Shows 12.5s vs GT 10.0s (25% overcount)
- **Likely cause:** Cluster assigned to "EILEEN" contains some of RINNA's or YOLANDA's faces
- **Evidence:** Cluster 0 has 24 tracks with 12.5s total - should be ~10s for EILEEN
- **Action Required:** **USER MUST REVIEW CLUSTER 0 IN UI** and reassign if faces don't match EILEEN

### RINNA, BRANDI, YOLANDA, LVP: ✗ Detection Failures
- **Issue:** Missing 44-72% of screen time despite re-ID working
- **Root cause:** **Faces never detected** in baseline pass due to:
  - Distant shots (face size < 80px threshold)
  - Extreme angles (profiles, turned away)
  - Poor lighting / occlusions
  - Motion blur
- **Evidence:**
  - RINNA: Only 24 tracks detected, missing 11s of 25s GT
  - YOLANDA: Only 7 tracks detected, missing 11.5s of 16s GT
  - BRANDI: Only 12 tracks detected, missing 5.5s of 10s GT
  - LVP: Only 3 tracks detected, missing 0.5s of 2s GT

**Re-ID cannot link tracks that were never created.** Need high-recall detection pass.

---

## 🎯 Next Steps to Hit ±5% for All Cast

### Step 1: Fix EILEEN Cluster Assignment ⚠️ **USER ACTION REQUIRED**

**Before proceeding with recall detection, verify cluster assignments:**

1. Open Screenalyzer UI → Review page for RHOBH-TEST-10-28
2. Navigate to **Cluster 0** (24 tracks, currently labeled "EILEEN")
3. Review face samples - do they all show EILEEN?
4. If cluster contains mixed faces:
   - Reassign cluster to correct person
   - OR split cluster if it contains multiple people
5. Repeat for all 7 clusters to ensure identity purity

**Current cluster sizes (for reference):**
- Cluster 1: 82 tracks → KIM
- Cluster 2: 25 tracks → KYLE
- Cluster 0: 24 tracks → **EILEEN** (verify!)
- Cluster 5: 12 tracks → BRANDI
- Cluster 3: 10 tracks → EILEEN or RINNA? (verify!)
- Cluster 4: 7 tracks → YOLANDA
- Cluster 6: 3 tracks → LVP

### Step 2: Implement Identity-Guided Recall Detection

**Framework created:** [jobs/tasks/post_label_recall.py](jobs/tasks/post_label_recall.py)

**Implementation plan:**

1. **Build per-person embedding templates** ✅ (implemented)
   - Extract top-10 high-quality ArcFace embeddings from each labeled cluster
   - Compute median embedding as person template

2. **Identify gap windows per person** ✅ (implemented)
   - Find all inter-interval gaps ≤3.2s for each person
   - Add ±300ms padding around gaps
   - Target only frames where person is missing

3. **Run high-recall detection in gap windows** ⚠️ (TODO)
   - Lower thresholds: `min_confidence=0.60`, `min_face_px=50`
   - Multi-scale: run at 1.0×, 1.3×, 1.6× scales
   - Optional: use person detector to gate face detection (reduce FPs)

4. **Verify identity with embedding similarity** ⚠️ (TODO)
   - Compute ArcFace embedding for each detected face
   - Accept if `cos_sim(detection, person_template) >= 0.82`
   - Reject if `margin < 0.08` (too similar to other people)
   - Tag accepted detections with `source:"recall"`

5. **Create tracklets from verified detections** ⚠️ (TODO)
   - Group consecutive recall detections (require ≥3 frames)
   - Pass through existing re-ID to link with baseline tracks
   - Integrate via adaptive gap-merge

6. **Measure impact** ⚠️ (TODO)
   - Compare before/after delta tables
   - Track: windows scanned, faces added, tracks created, relinks made
   - Emit: `recall_stats.json` with detailed metrics

**Config added:** [configs/pipeline.yaml:post_label_recall](configs/pipeline.yaml)

### Step 3: Micro-Sweep (if needed)

If any person remains >5% after recall:
- Run targeted threshold sweep on their gap windows only
- Test: `min_confidence ∈ {0.70, 0.65, 0.60}` × `min_face_px ∈ {80, 64, 50}`
- Pick lowest relaxation that closes gap without false merges
- Document as person-specific or show-specific preset

### Step 4: Verification & Documentation

**Acceptance criteria:**
- ✅ All 7 cast members ≤±5% of GT
- ✅ Zero manual overrides used
- ✅ No obvious false merges (check timeline for conflicts)
- ✅ All artifacts present: `clusters.json`, `tracks.json`, `timeline.csv`, `totals.csv`, `recall_stats.json`, `merge_suggestions.parquet`

**Final deliverables:**
1. Delta table (before/after recall) showing per-person improvements
2. One-page summary: what fixed each person, which techniques worked
3. RHOBH preset config: `configs/presets/RHOBH.yaml`
4. Evidence for any unfixable gaps (frame crops showing no detectable faces)

---

## 🛠️ Technical Improvements Made

### Files Modified

1. **jobs/tasks/analytics.py**
   - Fixed Excel time format (lines 185-206)
   - Millisecond columns now display as integers, not times

2. **configs/bytetrack.yaml**
   - Tightened matching: `match_thresh=0.9`, `iou_thresh=0.3`
   - Eliminated cross-cut false matches

3. **screentime/tracking/bytetrack_wrap.py**
   - Fixed LAP solver API compatibility (lines 230-237)
   - Handle both 2-return and 3-return APIs
   - Eliminated 207 LAP warnings per run

4. **screentime/tracking/reid.py** (existing)
   - Track re-identification working (34 links created)

5. **screentime/attribution/timeline.py** (existing)
   - Adaptive gap-merge working (quality-aware, conflict guards)

6. **configs/pipeline.yaml**
   - Added `post_label_recall` config section
   - All algorithm parameters tuned and documented

### Files Created

1. **jobs/tasks/post_label_recall.py** (NEW)
   - Identity-guided recall detection framework
   - Template building implemented
   - Gap window selection implemented
   - Detection/verification logic pending

2. **screentime/video/scene_detect.py** (NEW)
   - Scene boundary detection (tested but not used)
   - Frame difference analysis with configurable threshold

3. **scripts/generate_baseline_report.py** (NEW)
   - Automated accuracy reporting
   - Generates delta tables with GT comparison

4. **DETECTION_THRESHOLD_FINDINGS.md** (NEW)
   - Comprehensive analysis of detection limitations
   - Documents why threshold relaxation alone doesn't work

---

## 🔬 Key Findings

### What Works

1. **Strict matching prevents false associations**
   - `match_thresh=0.9` + `iou_thresh=0.3` = zero overlaps
   - Short tracks are intentional - re-ID reconnects legitimate continuity

2. **Re-ID successfully links fragmented tracks**
   - 20.6% acceptance rate (34/165 attempts)
   - Conservative thresholds (`min_sim=0.82`, `min_margin=0.08`) maintain precision

3. **Adaptive gap-merge closes small gaps**
   - Quality-aware threshold (2.5s base, 3.0s max)
   - Conflict guards prevent false cross-speaker merges

### What Doesn't Work

1. **Global threshold relaxation**
   - Tested: `min_confidence=0.65`, `min_face_px=64`
   - Result: +3.5s for KIM, +1.5s for KYLE, **±0s for 5 others**
   - Conclusion: Missing faces aren't marginally below threshold - they're completely undetectable at baseline settings

2. **Scene boundary detection (PySceneDetect approach)**
   - Too aggressive at low thresholds (35 boundaries @ threshold=50)
   - Too permissive at high thresholds (only 1 boundary @ threshold=90)
   - Strict matching + re-ID is more reliable

3. **Reducing track_buffer**
   - Tested: 30 → 6 → 3 frames
   - Result: No impact on long tracks (63s monsters persisted)
   - Conclusion: Problem was false matching, not persistence

---

## 📋 Current Configuration

### Detection
```yaml
detection:
  min_confidence: 0.70
  min_face_px: 80
  provider_order: [coreml, cpu]
```

### Tracking
```yaml
tracking:
  iou_threshold: 0.5
  reid:
    enabled: true
    max_gap_ms: 2500
    min_sim: 0.82
    min_margin: 0.08
    topk: 5
    use_scene_bounds: false

# ByteTrack params
track:
  track_buffer: 10
  match_thresh: 0.9  # STRICT
  conf_thresh: 0.5
  iou_thresh: 0.3    # STRICT
```

### Timeline
```yaml
timeline:
  gap_merge_ms_base: 2500
  gap_merge_ms_max: 3000
  min_interval_quality: 0.70
  conflict_guard_ms: 700
```

### Post-Label Recall (NEW)
```yaml
post_label_recall:
  enabled: true
  window_pad_ms: 300
  max_gap_ms: 3200
  detection:
    min_confidence: 0.60  # Relaxed for recall only
    min_face_px: 50
    scales: [1.0, 1.3, 1.6]
  track:
    birth_min_frames: 3
  reid:
    min_sim: 0.82
    min_margin: 0.08
```

---

## 🎬 What's Next

**Immediate:**
1. ⚠️ **USER:** Verify and fix EILEEN cluster assignment in UI
2. ⚠️ **USER:** Verify all 7 cluster assignments are identity-pure

**Short-term (to hit ±5%):**
1. Implement high-recall detection logic in `post_label_recall.py`
2. Test on RINNA/BRANDI/YOLANDA/LVP gaps
3. Measure before/after improvement
4. Run micro-sweep if any person still >5%

**Final:**
1. Generate final delta table showing all 7 ≤±5%
2. Create RHOBH preset: `configs/presets/RHOBH.yaml`
3. Document methodology and results
4. Pass Phase 1 acceptance criteria

---

**Report generated:** 2025-10-29
**System:** Claude Sonnet 4.5
**Episode:** RHOBH-TEST-10-28 (103 seconds, 7 cast members)
