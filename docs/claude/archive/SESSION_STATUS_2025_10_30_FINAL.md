# Session Status - October 30, 2025 (Final Summary)

## Overview

This session focused on implementing identity-agnostic cluster hygiene and auto-caps to eliminate hardcoded per-person tuning and improve clustering purity.

---

## ✅ Completed Work

### 1. Cluster Hygiene Pipeline (Phase A) - **COMPLETE**

**Implementation**: Fully integrated into [jobs/tasks/cluster.py](jobs/tasks/cluster.py)

**Components**:
- ✅ Face-only filtering (baseline harvest): Removes non-face chips
  - Thresholds: conf ≥0.65, size ≥72px, IoU ≤0.10
  - Result: 77.6% retention (272 non-face chips removed from 1,217)

- ✅ Top-K selection per track: Uses 10 best embeddings for centroids
  - Quality score: weighted by confidence, size, sharpness
  - Prevents centroid drift from poor frames

- ✅ Gallery integration: Uses `picked_samples.parquet` (faces-only)
  - [app/lib/data.py](app/lib/data.py:92-116) loads picked_samples first
  - Falls back to embeddings.parquet for backward compatibility

**Files Created**:
- [screentime/clustering/face_quality.py](screentime/clustering/face_quality.py) - 300 lines
- [screentime/clustering/contamination_audit.py](screentime/clustering/contamination_audit.py) - 400 lines

**Config**: [configs/pipeline.yaml](configs/pipeline.yaml:9-26)

**Test Results**:
- Picked samples: 945 (from 1,217 embeddings)
- Retention: 77.6%
- Tracks with samples: 316
- Avg samples per track: 3.0

---

### 2. Auto-Caps (Phase B) - **COMPLETE**

**Implementation**: Fully integrated into [jobs/tasks/analytics.py](jobs/tasks/analytics.py:64-113)

**Features**:
- ✅ Computes P80 safe gaps per identity
- ✅ Safe gap criteria: both sides visible_frac ≥0.60, no conflict ≥500ms
- ✅ Formula: `clamp(P80 × 1.2, 1200ms, 2500ms)`
- ✅ Identity-agnostic, data-driven

**Files Created**:
- [screentime/attribution/auto_caps.py](screentime/attribution/auto_caps.py) - 150 lines

**Config**: [configs/pipeline.yaml](configs/pipeline.yaml:90-98)

**Test Results** (RHOBH-TEST-10-28):
| Identity | auto_cap_ms | safe_gap_count | safe_gap_p80 |
|----------|-------------|----------------|--------------|
| RINNA | 1,200 | 15 | 250ms |
| KIM | 1,200 | 67 | 84ms |
| EILEEN | 1,200 | 3 | 983ms |
| Others | 1,200 | 0-3 | 0-83ms |

**Note**: All capped at 1200ms minimum due to short episode (102.5s) with few safe gaps.

---

### 3. Auto-Threshold Tuning (Phase C - Partial) - **COMPLETE**

**Implementation**: Integrated into [jobs/tasks/cluster.py](jobs/tasks/cluster.py:191-219)

**Features**:
- ✅ Self-tuning DBSCAN eps from k-NN distance curve
- ✅ Uses P75 of 5-NN distances as knee point
- ✅ Clamped to safe range [0.30, 0.45]
- ✅ Identity-agnostic, episode-specific

**Files Created**:
- [screentime/clustering/auto_threshold.py](screentime/clustering/auto_threshold.py) - 180 lines

**Test Results** (RHOBH-TEST-10-28):
- eps_raw: 0.271 (P75 of k-NN distances)
- eps_tuned: 0.300 (clamped to minimum)
- Clusters created: 11 (vs 5 with fixed eps=0.45)
- Impact: **-33% tighter clustering** → less mixing expected

**Diagnostics**: [data/harvest/RHOBH-TEST-10-28/diagnostics/cluster_threshold.json](data/harvest/RHOBH-TEST-10-28/diagnostics/cluster_threshold.json)

---

## 📊 Current Accuracy (2/7 Passing)

| person | delta_s | status | issue |
|--------|---------|--------|-------|
| YOLANDA | -0.00s | ✅ PASS | Perfect |
| KIM | +1.50s | ✅ PASS | Within threshold |
| LVP | +1.15s | ❌ FAIL | Slight over-merge |
| KYLE | +2.73s | ❌ FAIL | Over-merge |
| BRANDI | -3.43s | ❌ FAIL | Under-detection |
| EILEEN | +4.42s | ❌ FAIL | Over-merge |
| RINNA | +5.07s | ❌ FAIL | Over-merge |

**Pass Rate**: 2/7 (29%)

**Validation**:
- ✅ Totals: 128.1s = 102.5s runtime + 25.6s overlaps
- ✅ No hardcoded overrides
- ✅ Uniform pipeline for all identities

---

## 🎯 Current Status: Step 1 (Gallery QA)

**Top 5 Clusters** (by track count):
1. **Cluster 1**: 204 tracks, 559 samples
2. **Cluster 0**: 35 tracks, 154 samples
3. **Cluster 2**: 13 tracks, 76 samples
4. **Cluster 9**: 12 tracks, 39 samples
5. **Cluster 6**: 9 tracks, 10 samples

**Visual Inspection Needed**:
- User should inspect galleries at http://localhost:8501
- Check for mixed identities in top 5 clusters
- Check for non-face thumbnails (glasses, arms, background)

---

## 🔧 Recommended Next Steps

### If Visual Inspection Shows Mixing Persists → Step 2

**Strengthen Contamination Audit** (~90 min):

1. **Frame-level outliers** (within cluster):
   - Mark chips with `sim_to_medoid < 0.75` as outliers
   - Exclude from centroid computation

2. **Track-level cross-probing** (time-aware):
   - For each track segment: compute `sim_current` vs `sim_other_clusters`
   - If `(sim_other - sim_current) ≥ 0.10` for ≥5 frames → flag contamination span

3. **Auto-split logic**:
   - Split contaminated spans from tracks
   - Move to Unknown or better-matching cluster (if margin ≥0.12)
   - Log all actions to `contamination_audit.json`

**Expected Result**: Galleries show single identity per cluster, purity >90%

### If Mixing Still Persists → Step 3

**Extend Face-Quality to All Sources** (~60 min):
- Apply face_quality filter to entrance-injected tracks
- Apply face_quality filter to densify/recall outputs
- Currently only baseline harvest is filtered

### If Galleries Are Clean → Skip to Analytics

Proceed directly to Phase 5 (Analytics UI) and targeted densify for BRANDI if needed.

---

## 📂 Files Generated This Session

| File | Size | Purpose |
|------|------|---------|
| **cluster_threshold.json** | 0.5 KB | Auto-tuned eps diagnostics |
| **per_identity_caps.json** | 0.6 KB | Auto-computed merge caps |
| **contamination_audit.json** | 56 B | Contamination detection (0 spans - may need stricter thresholds) |
| **picked_samples.parquet** | 3.2 MB | Face-only samples for gallery |
| **clusters.json** | 5.0 KB | 11 clusters (vs 5 before) |

---

## 📝 Code Changes Summary

**Files Modified**:
1. [configs/pipeline.yaml](configs/pipeline.yaml) - Added face_quality + auto_caps config (+40 lines)
2. [jobs/tasks/cluster.py](jobs/tasks/cluster.py) - Integrated hygiene pipeline (+150 lines)
3. [jobs/tasks/analytics.py](jobs/tasks/analytics.py) - Integrated auto-caps (+50 lines)
4. [app/lib/data.py](app/lib/data.py) - Gallery uses picked_samples (+10 lines)

**Files Created**:
1. [screentime/clustering/face_quality.py](screentime/clustering/face_quality.py) - 300 lines
2. [screentime/clustering/contamination_audit.py](screentime/clustering/contamination_audit.py) - 400 lines
3. [screentime/attribution/auto_caps.py](screentime/attribution/auto_caps.py) - 150 lines
4. [screentime/clustering/auto_threshold.py](screentime/clustering/auto_threshold.py) - 180 lines
5. [.github/workflows/no-hardcoded-identities.sh](.github/workflows/no-hardcoded-identities.sh) - 80 lines (CI guard)

**Total New Code**: ~1,160 lines
**Total Modified**: ~250 lines

---

## 🔒 Guardrails Maintained

✅ **No Per-Person Hardcoding**:
- All thresholds in configs are global
- per_identity sections computed from data (auto-caps)
- CI guard prevents regression

✅ **Identity-Agnostic**:
- Same pipeline for all identities
- No manual tuning or overrides
- Data-driven thresholds

✅ **Validation**:
- Totals ≤ runtime + overlaps
- Co-appearances credited
- No timeline overlaps

✅ **Baseline Maintained**:
- RetinaFace detector
- 10fps sampling
- 30fps only in recall windows

---

## 🎬 Next Actions

**Immediate** (Step 1 - User Action):
1. Open Streamlit gallery at http://localhost:8501
2. Navigate to "Clusters" page
3. Inspect top 5 clusters for:
   - Mixed identities (different people in same cluster)
   - Non-face thumbnails (glasses, arms, background)
4. Report findings

**If Mixing Found** (Step 2 - ~90 min):
- Implement stronger contamination audit
- Add frame-level + track-level probing
- Implement auto-split logic

**If Clean** (Skip to Phase 5):
- Proceed to Analytics UI implementation
- Queue targeted densify for BRANDI if needed

---

## 📊 Summary Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| **Clustering** ||||
| DBSCAN eps | 0.45 (fixed) | 0.30 (auto) | -33% |
| Clusters | 5 | 11 | +6 |
| Cluster purity | Unknown | Pending QA | TBD |
| **Face Quality** ||||
| Embeddings | 1,217 | 945 | -22.4% |
| Non-face chips removed | 0 | 272 | +272 |
| Faces-only retention | 100% | 77.6% | -22.4% |
| **Auto-Caps** ||||
| Per-identity tuning | Manual | Auto | ✅ |
| Safe gaps identified | N/A | 0-67 per identity | ✅ |
| **Accuracy** ||||
| Pass rate (≤4.5s) | 2/7 | 2/7 | No change |
| Auto-caps impact | N/A | Minimal (short episode) | Expected |

---

## 💡 Key Insights

1. **Auto-Threshold Working**: eps tightened from 0.45 → 0.30, created 11 tighter clusters
2. **Face-Only Filtering Effective**: Removed 22.4% non-face chips from baseline harvest
3. **Auto-Caps Functional But Limited**: Short episode (102.5s) doesn't provide enough safe gap data
4. **Accuracy Issues Are Structural**: Under-detection (BRANDI) and over-merges (EILEEN/RINNA) not solved by caps/clustering alone
5. **Next Priority**: Visual QA → contamination audit → (if needed) extend face-quality to entrance/densify

---

## 🔗 Related Documentation

- [CLUSTER_HYGIENE_INTEGRATION_COMPLETE.md](CLUSTER_HYGIENE_INTEGRATION_COMPLETE.md) - Initial hygiene implementation
- [IDENTITY_AGNOSTIC_REFACTOR_STATUS.md](IDENTITY_AGNOSTIC_REFACTOR_STATUS.md) - Auto-caps refactor
- [configs/pipeline.yaml](configs/pipeline.yaml) - Current configuration

---

**Session Duration**: ~4 hours
**Status**: Phase A ✅ Complete, Phase B ✅ Complete, Phase C ⚠️ Partial (awaiting visual QA)
**Next**: User visual inspection → Step 2 or Phase 5
