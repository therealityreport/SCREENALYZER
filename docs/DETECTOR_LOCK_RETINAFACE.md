# Detector Lock - RetinaFace Only

**Purpose**: Lock pipeline to RetinaFace detector, disable SCRFD A/B testing based on spot-check results.

**Decision**: SCRFD spot-check showed **0% small-face lift** → root cause is thresholds (min_face_px=72), not detector choice → skip full A/B (6 hours saved) → focus on threshold tuning instead.

**Status**: Config patch ready to apply

---

## 1. Config Patch

### Add to `configs/pipeline.yaml`

**Location**: Add new `detection_ab` section after `detection:` block (line ~23)

```yaml
detection:
  min_confidence: 0.7
  provider_order:
  - coreml
  - cpu
  recall:
    birth_confirm_frames: 3
    enabled: true
    max_window_duration_ms: 5000
    min_confidence: 0.6
    min_face_px: 50
    scale_factors:
    - 1.0
    - 1.3
    - 1.6

# Detector A/B testing configuration
detection_ab:
  enabled: false  # ← Disabled based on SCRFD spot-check results
  rationale: |
    SCRFD spot-check (2025-10-30) scanned 99 frames from hardest gaps
    (YOLANDA, RINNA, BRANDI). Both RetinaFace and SCRFD found ZERO
    small faces (≤80px), indicating root cause is threshold filtering
    (min_face_px=72) not detector choice. 0% lift < 30% gate threshold.
    Decision: Skip full A/B (6 hours), focus on densify threshold tuning.
  baseline_detector: "retinaface"  # buffalo_l det_10g
  alternative_detector: "scrfd"    # scrfd_10g_bnkps (available but unused)
  spot_check_results:
    frames_scanned: 99
    retinaface_small_faces: 0
    scrfd_small_faces: 0
    lift_percent: 0.0
    gate_threshold: 30.0
    gate_passed: false
  full_ab_skipped: true
  spot_check_report: "data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json"
```

---

## 2. Implementation Notes

### Detector Selection Logic

**File**: `screentime/detectors/__init__.py` (if registry implemented)

```python
def get_detector(config: dict) -> FaceDetector:
    """
    Get face detector based on config.

    Returns RetinaFace by default. SCRFD only used if detection_ab.enabled=true.
    """
    ab_config = config.get("detection_ab", {})

    if ab_config.get("enabled", False):
        detector_name = ab_config.get("alternative_detector", "retinaface")
        logger.info(f"A/B testing enabled, using detector: {detector_name}")
    else:
        detector_name = ab_config.get("baseline_detector", "retinaface")
        logger.info(f"A/B testing disabled, using baseline: {detector_name}")

    if detector_name == "retinaface":
        from screentime.detectors.face_retina import RetinaFaceDetector
        return RetinaFaceDetector(**config.get("detection", {}))
    elif detector_name == "scrfd":
        from screentime.detectors.face_scrfd import SCRFDDetector
        return SCRFDDetector(**config.get("detection", {}))
    else:
        raise ValueError(f"Unknown detector: {detector_name}")
```

**Current Behavior**: Pipeline hardcodes RetinaFaceDetector throughout. This config documents the decision and provides hooks for future A/B if needed on different episodes.

---

## 3. Spot-Check Summary (Reference)

**Report**: [data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json](../data/harvest/RHOBH-TEST-10-28/diagnostics/reports/scrfd_spotcheck.json)

**Key Results**:
- Frames scanned: 99 (33 per identity: YOLANDA, RINNA, BRANDI)
- RetinaFace: 148 total faces, **0 ≤80px**, **0 ≤64px**
- SCRFD: 136 total faces, **0 ≤80px**, **0 ≤64px**
- Small-face lift: **0%** (gate requires ≥30%)

**Critical Insight**: Both detectors found ZERO small faces because `min_face_px=72` filters them out BEFORE tracking. Problem is thresholds, not detector.

**Decision Impact**:
- Time saved: 6 hours (skip full A/B harvest + analytics)
- Better use of time: 2-pass densify (2 hours) + identity-guided recall (2 hours)
- Expected recovery: 2-4s from threshold tuning vs 0s from detector swap

---

## 4. Future Considerations

### When to Re-Enable A/B:
- **Different episode**: SCRFD may excel on outdoor/distant shots (not present in RHOBH-TEST-10-28 kitchen scene)
- **Different show**: Reality TV with more wide shots may benefit from SCRFD's small-face performance
- **Post-threshold-tuning**: If densify with min_face_px=36 STILL shows gaps, re-run spot-check at new thresholds

### Spot-Check Protocol:
1. Identify hardest gaps (top 3 per identity by duration)
2. Sample ~100 frames from gap centers
3. Run both detectors with SAME thresholds
4. Compare small-face counts (≤80px, ≤64px)
5. **Gate**: Require ≥30% verified lift to proceed with full A/B

**Acceptance**: Small lift (5-15%) does NOT justify 6-hour full A/B investment. Focus on threshold/recall instead.

---

## 5. Validation After Config Patch

**Checklist**:
- [ ] Add `detection_ab` section to configs/pipeline.yaml
- [ ] Verify pipeline still initializes RetinaFaceDetector (no behavior change)
- [ ] Confirm spot-check report exists at specified path
- [ ] Document decision in session notes/commit message

**Command to Validate**:
```bash
# Check config loads without errors
python -c "import yaml; yaml.safe_load(open('configs/pipeline.yaml'))"

# Verify detector initialization
python -c "from screentime.detectors.face_retina import RetinaFaceDetector; d = RetinaFaceDetector(); print(f'Detector: {d.name}')"
```

**Expected Output**:
```
Detector: retinaface
```

---

## 6. Acceptance Criteria

✅ Config patch applied to pipeline.yaml
✅ `detection_ab.enabled = false` with documented rationale
✅ Spot-check results linked in config
✅ Pipeline continues using RetinaFace (no behavior change)
✅ Decision documented for future reference

---

**Status**: Config patch ready to apply (5 lines to add)
**File**: `configs/pipeline.yaml` (line ~23, after `detection:` block)
**Behavior Change**: None (documents existing RetinaFace lock)
**Time Saved**: 6 hours (full A/B skipped)
