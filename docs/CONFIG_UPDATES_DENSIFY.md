# Densify Threshold Configuration - 2-Pass Strategy

**Purpose**: Recover 2-4s from gap windows using conservative then aggressive passes
**Baseline**: 10 fps global, 30 fps local densify only in flagged windows
**Policy**: No overrides, same pipeline for all identities

---

## Configuration Updates

### Add to `configs/pipeline.yaml`:

```yaml
# Global detection (baseline 10 fps)
detection:
  backend: "retinaface"
  min_confidence: 0.70
  min_face_px: 72
  nms_iou: 0.50
  provider_order: ["coreml", "cpu"]

# Local densify - PASS 1 (conservative)
local_densify:
  enabled: true
  min_confidence: 0.58      # Lower from 0.70
  min_face_px: 44           # Lower from 72
  nms_iou: 0.40             # Lower from 0.50 (allow overlap)
  scales: [1.0, 1.25, 1.5, 2.0]
  min_consecutive: 4        # Birth tracklet after 4 consecutive frames

  # Verification (prevent false positives)
  verify:
    min_similarity: 0.86    # Must match facebank template
    second_best_margin: 0.12  # Must beat next-best identity by this margin

  # Negative gating
  negative_to_others:
    enabled: true
    margin: 0.06            # sim_to_seed - best_other >= 0.06

# Local densify - PASS 2 (aggressive, only for residual gaps)
local_densify_pass2:
  enabled: true
  trigger_threshold: 4.5    # Only run if identity still >4.5s error after pass 1

  min_confidence: 0.50      # More aggressive
  min_face_px: 36           # Capture very small faces
  nms_iou: 0.40
  scales: [1.0, 1.35, 1.7, 2.2]  # More scale variations
  min_consecutive: 5        # Require more evidence (tighter to prevent FP)

  # Verification (TIGHTER when loosening detection)
  verify:
    min_similarity: 0.88    # Higher than pass 1
    second_best_margin: 0.14  # Wider margin

  # Negative gating (same)
  negative_to_others:
    enabled: true
    margin: 0.06

# A/B detector comparison (disabled after spot-check)
detection_ab:
  enabled: false            # Re-enable only if future spot-check shows SCRFD lift
  detectors: ["retinaface", "scrfd"]

  # Thresholds for A/B (if re-enabled)
  common:
    min_confidence: 0.70
    min_face_px: 72
    nms_iou: 0.50
```

---

## Implementation Notes

### Pass 1 Strategy
- Run on ALL gap windows for ALL identities
- Conservative thresholds to avoid false positives
- Verify each detection against facebank
- Expected recovery: 1-2s per identity with large gaps

### Pass 2 Strategy
- **Only run if identity still >4.5s error after Pass 1**
- **Only on largest residual gap** for that identity
- Aggressive thresholds (min_face_px=36) but TIGHTER verification (sim≥0.88)
- Expected recovery: 0.5-1.5s for hardest cases

### Negative Gating
Always enabled to prevent cross-identity pollution:
```python
accept = (
    sim_to_target >= min_similarity and
    sim_to_target - max(sim_to_others) >= margin
)
```

### Person ROI (Optional Enhancement)
If person bounding box available, restrict detection to that ROI to reduce background FP.

### Temporal Consistency
Require `min_consecutive` frames before birthing a tracklet:
- Pass 1: 4 frames (400ms at 10fps, 133ms at 30fps)
- Pass 2: 5 frames (500ms at 10fps, 167ms at 30fps)

---

## Acceptance Criteria

✅ Seconds added only where verified faces exist
✅ No ID drift (negative gating + verification enforced)
✅ Totals ≤ runtime (co-appearance credit working)
✅ No manual overrides
✅ Same pipeline for all identities

---

## Files Modified

1. `configs/pipeline.yaml` - Add local_densify and local_densify_pass2 sections
2. `jobs/tasks/local_densify.py` - Implement 2-pass logic (next session)
3. `screentime/tracking/reid.py` - Ensure negative gating enforced (verify)

---

## Testing Plan

1. Run Pass 1 on episode
2. Check delta_table.csv - identify identities still >4.5s
3. Run Pass 2 only on those identities
4. Verify no regressions (KIM/KYLE/LVP stay within bounds)
5. Check recall_stats.json for per-identity contributions

---

## Telemetry

Generate `diagnostics/reports/recall_stats.json`:

```json
{
  "episode_id": "RHOBH-TEST-10-28",
  "pass1": {
    "identities_processed": 7,
    "windows_scanned": 18,
    "faces_detected": 234,
    "faces_verified": 156,
    "tracklets_born": 42,
    "seconds_recovered": {
      "YOLANDA": 1.2,
      "RINNA": 0.8,
      "BRANDI": 0.5,
      "total": 2.5
    }
  },
  "pass2": {
    "identities_processed": 3,
    "windows_scanned": 3,
    "faces_detected": 45,
    "faces_verified": 23,
    "tracklets_born": 7,
    "seconds_recovered": {
      "RINNA": 0.7,
      "BRANDI": 0.4,
      "total": 1.1
    }
  },
  "small_face_histogram": {
    "0-40px": 12,
    "40-60px": 34,
    "60-80px": 67,
    "80-100px": 89,
    "100+px": 254
  }
}
```

---

**Status**: Config ready for next session implementation
**ETA**: 2-3 hours to implement both passes + telemetry
