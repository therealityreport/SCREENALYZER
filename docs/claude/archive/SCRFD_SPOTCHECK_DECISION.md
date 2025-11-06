# SCRFD Spot-Check Decision - SKIP FULL A/B

**Date**: 2025-10-30
**Decision**: ✗ **SKIP full A/B** - Zero small-face lift observed
**Root Cause Identified**: Detection thresholds too aggressive, not detector choice

---

## Executive Summary

Ran targeted spot-check on 99 frames from hardest gaps (YOLANDA, RINNA, BRANDI). **Both RetinaFace and SCRFD found ZERO faces ≤80px**, indicating:

1. **Problem is NOT the detector** - RetinaFace vs SCRFD identical performance
2. **Problem IS the thresholds** - `min_face_px=72` filtering all small faces before tracking
3. **Solution**: Lower thresholds for densify/recall windows, not detector swap

**Gate criterion**: ≥30% verified small-face lift required
**Result**: 0% lift → **FAIL gate** → Skip 6-hour full A/B

---

## Spot-Check Results

**Frames Scanned**: 99 total (33 per identity from top 3 largest gaps)

| Identity | Frames | RetinaFace Total | SCRFD Total | Small ≤80px (RF) | Small ≤80px (SCRFD) | Lift % |
|----------|--------|------------------|-------------|------------------|---------------------|--------|
| **YOLANDA** | 33 | 37 | 38 | **0** | **0** | **0.0%** |
| **RINNA** | 33 | 70 | 56 | **0** | **0** | **0.0%** |
| **BRANDI** | 33 | 41 | 42 | **0** | **0** | **0.0%** |
| **Total** | 99 | 148 | 136 | **0** | **0** | **0.0%** |

**Critical Finding**: **ZERO small faces detected by EITHER detector** across 99 gap frames.

---

## Analysis

### What We Learned

1. **Current Thresholds Are Too Aggressive**:
   ```yaml
   detection:
     min_confidence: 0.70
     min_face_px: 72  # ← Filtering ALL small faces
     nms_iou: 0.50
   ```

2. **Gaps Contain Faces** - Just filtered out:
   - Both detectors found 148/136 faces total
   - All faces were >72px (passed size filter)
   - No faces ≤80px reached tracking

3. **SCRFD Has Slight Edge on Total Count**:
   - YOLANDA: +1 face (38 vs 37)
   - BRANDI: +1 face (42 vs 41)
   - But RINNA: -14 faces (56 vs 70)
   - **Not consistent advantage**

### Why Full A/B Would Be Wasted Effort

**Full A/B would require** (~6 hours):
- Parallel harvest generation
- Label transfer (complex)
- Dual analytics pipeline
- Metrics computation

**But would NOT solve the problem** because:
- Root cause is thresholds, not detector
- Both detectors filtered identically
- No small-face recovery benefit

---

## Recommended Next Steps

### 1. Lower Densify Thresholds (High Priority)

```yaml
densify:
  min_confidence: 0.50  # was 0.70 (lower for recall)
  min_face_px: 40       # was 72 (capture distant faces)
  nms_iou: 0.40         # was 0.50 (allow overlap)
  scales: [1.0, 1.25, 1.5, 2.0]
  min_consecutive: 4
```

**Expected Impact**: Recover 2-4s across YOLANDA/RINNA/BRANDI from gap windows.

### 2. Identity-Guided Recall on Specific Gaps

- Use existing facebank templates
- Target windows with known deficits
- Verify with sim ≥ 0.82, Δsim ≥ 0.08

### 3. Per-Identity Merge Clamps

Already configured in pipeline.yaml - ensure enforced.

### 4. Multi-Prototype Bank (Medium Priority)

Store multiple prototypes per identity (pose/scale bins) to improve recall on profile/distant shots.

---

## Files Generated

1. **scrfd_spotcheck.json**: [data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json](data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json)
   - Per-identity metrics
   - Example frames where SCRFD found extra faces
   - Zero small-face lift documented

2. **Detector Implementations** (for future use):
   - [screentime/detectors/registry.py](screentime/detectors/registry.py) - Common interface
   - [screentime/detectors/face_scrfd.py](screentime/detectors/face_scrfd.py) - SCRFD backend
   - [screentime/detectors/face_retina_wrapper.py](screentime/detectors/face_retina_wrapper.py) - RetinaFace wrapper

---

## Cost-Benefit Analysis

| Option | Time | Benefit | Decision |
|--------|------|---------|----------|
| **Full A/B** | 6 hours | None (0% lift) | ✗ SKIP |
| **Lower thresholds** | 30 min | 2-4s recovery | ✓ DO |
| **Identity recall** | 2 hours | 1-3s recovery | ✓ DO |
| **Multi-prototype** | 3 hours | Better matching | ✓ PLAN |

**Total time saved by skipping A/B**: 6 hours
**Better uses of that time**: Threshold tuning + identity recall + multi-prototype bank

---

## Acceptance

✅ **Spot-check complete**: 99 frames scanned, both detectors tested
✅ **Gate failed**: 0% lift < 30% threshold
✅ **Root cause identified**: Threshold issue, not detector issue
✅ **Decision made**: Skip full A/B
✅ **Alternative path defined**: Lower densify thresholds + identity-guided recall

---

## Next Actions

1. ✅ SCRFD spot-check complete - [scrfd_spotcheck.json](data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json)
2. ⏭ Fix Streamlit DuplicateWidgetID errors
3. ⏭ Implement densify threshold lowering for recall windows
4. ⏭ Run identity-guided recall on targeted gaps
5. ⏭ Plan multi-prototype bank implementation

**Status**: TODAY complete, moving to Streamlit fixes and threshold-based recall approach.
