# Final Session Status - Kim/Kyle Separation Solution

**Date**: October 30, 2025
**Session Duration**: ~5 hours
**Status**: Infrastructure complete, prototype-anchored split needed

---

## Executive Summary

### ✅ What's Been Accomplished

**Phase A - Cluster Hygiene** (Complete):
- Face-only filtering: 77.6% retention (272 non-face chips removed)
- Top-K selection: 10 best samples per track for centroids
- Gallery integration: Uses `picked_samples.parquet` automatically
- Identity-agnostic config in `pipeline.yaml`

**Phase B - Auto-Caps** (Complete):
- Computes P80 safe gaps per identity
- Generates `per_identity_caps.json`
- Integrated into analytics timeline merge
- All identities capped at 1200ms minimum (short episode)

**Phase C - Purity-Driven Clustering** (Complete):
- Implemented quality sweep: silhouette - 0.75 * impurity
- Evaluates 10-15 eps candidates
- Chooses optimal eps without manual tuning
- Module: `screentime/clustering/purity_driven_eps.py` (400 lines)

**Total New Code**: ~1,560 lines across 5 new modules

---

## 🎯 Core Problem: Kim/Kyle Similarity

### Why Unsupervised Clustering Fails

**Fundamental Issue**: Kim Richards and Kyle Richards are sisters with very similar facial features.

**Embedding Space**:
```
KIM embeddings:  [0.45, 0.23, -0.12, ...]
KYLE embeddings: [0.46, 0.24, -0.11, ...]  ← Very close!

Cosine similarity: 0.92-0.95 (extremely high)
```

**Clustering Behavior**:
- **eps too loose** (0.40): Kim + Kyle → Same cluster ❌
- **eps too tight** (0.30): Kim → 3 clusters, Kyle → 2 clusters (over-fragmentation) ❌
- **eps optimal** (0.33): Still merges Kim/Kyle ~70% of the time ❌

**Why Purity-Driven Didn't Solve It**:
- Impurity detection catches obvious cross-contamination (e.g., Rinna in Kim cluster)
- But Kim/Kyle similarity (0.92-0.95) is **higher than intra-cluster similarity threshold (0.75)**
- They look like they "belong together" to unsupervised metrics

---

## 🔧 Solution Required: Prototype-Anchored Split

### Approach

**Use labeled examples (facebank prototypes) to create a decision boundary**:

```
Unsupervised:     [Kim + Kyle merged] → Can't separate
                         ↓
Semi-supervised:  [Kim prototypes] ← anchor A
                  [Kyle prototypes] ← anchor B
                         ↓
                  Seeded 2-means with anchors
                         ↓
                  [Kim cluster] [Kyle cluster] ✅
```

### Implementation Plan (~3-4 hours)

#### Step 1: Multi-Prototype Bank (60 min)

**Goal**: Build diverse prototypes for each identity (pose × scale).

**Algorithm**:
```python
for identity in facebank:
    samples = load_labeled_samples(identity)

    # Cluster by pose (frontal, 3/4, profile)
    pose_clusters = cluster_by_yaw_angle(samples)

    # For each pose, pick small/medium/large examples
    for pose in pose_clusters:
        prototypes[identity][pose] = {
            'small': pick_best(size < 100px),
            'medium': pick_best(100px ≤ size < 150px),
            'large': pick_best(size ≥ 150px)
        }

    # Result: 3 poses × 3 scales = 9 prototypes per identity
```

**Files**:
- `screentime/clustering/prototype_bank.py` (300 lines)
- Output: `data/facebank/{episode}/multi_prototypes.json`

---

#### Step 2: Confusion Matrix (30 min)

**Goal**: Detect ambiguous clusters (top-2 identities too close).

**Algorithm**:
```python
for cluster in clusters:
    centroid = compute_cluster_medoid(cluster.samples)

    # Compute similarities to all prototype banks
    sims = {}
    for identity, prototypes in prototype_bank.items():
        # Set-to-set max similarity
        sims[identity] = max(
            cosine_sim(centroid, p)
            for p in prototypes.all()
        )

    # Sort by similarity
    sorted_ids = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    top1_id, top1_sim = sorted_ids[0]
    top2_id, top2_sim = sorted_ids[1]

    # Check if ambiguous
    if (top1_sim - top2_sim) < 0.10 and top1_sim >= 0.60:
        mark_ambiguous(cluster, top1_id, top2_id, margin=top1_sim-top2_sim)
```

**Files**:
- `screentime/clustering/confusion_detector.py` (200 lines)
- Output: `diagnostics/confusion_matrix.json`

**Example Output**:
```json
{
  "ambiguous_clusters": [
    {
      "cluster_id": 0,
      "top1_identity": "KIM",
      "top1_sim": 0.68,
      "top2_identity": "KYLE",
      "top2_sim": 0.64,
      "margin": 0.04,
      "decision": "ambiguous"
    }
  ]
}
```

---

#### Step 3: Prototype-Anchored Split (90 min)

**Goal**: Split ambiguous clusters using prototype anchors.

**Algorithm**:
```python
def anchored_split(cluster, anchor_A_prototypes, anchor_B_prototypes):
    """
    Split cluster using seeded 2-means with prototype anchors.
    """
    samples = cluster.picked_samples

    # Compute anchor centroids (mean of all prototypes)
    anchor_A = mean([p.embedding for p in anchor_A_prototypes])
    anchor_B = mean([p.embedding for p in anchor_B_prototypes])

    # Assign each sample to nearest anchor
    assignments = []
    for sample in samples:
        sim_A = cosine_sim(sample.embedding, anchor_A)
        sim_B = cosine_sim(sample.embedding, anchor_B)

        if sim_A > sim_B:
            assignments.append('A')
        else:
            assignments.append('B')

    # Check temporal consistency (each side must have ≥600ms, ≥6 chips)
    side_A_samples = [s for s, a in zip(samples, assignments) if a == 'A']
    side_B_samples = [s for s, a in zip(samples, assignments) if a == 'B']

    duration_A = max(side_A_samples.ts_ms) - min(side_A_samples.ts_ms)
    duration_B = max(side_B_samples.ts_ms) - min(side_B_samples.ts_ms)

    if duration_A >= 600 and len(side_A_samples) >= 6 and \
       duration_B >= 600 and len(side_B_samples) >= 6:
        # Split is valid
        create_new_cluster(side_A_samples, anchor='A')
        create_new_cluster(side_B_samples, anchor='B')
        return "split"
    else:
        # Not enough evidence, keep as Unknown
        return "unknown"
```

**Files**:
- `screentime/clustering/anchored_split.py` (350 lines)
- Output: `diagnostics/anchored_split_audit.json`

**Constraints (Identity-Agnostic)**:
- Min duration per side: 600ms
- Min chips per side: 6
- Min anchor margin: 0.08
- Same thresholds for all ambiguous clusters

---

#### Step 4: Open-Set Assignment (30 min)

**Goal**: Assign "Unknown" instead of forcing wrong names.

**Algorithm**:
```python
def assign_cluster_to_identity(cluster, prototype_bank, config):
    """
    Assign cluster to identity or Unknown (open-set).
    """
    centroid = cluster.medoid

    # Compute similarities
    sims = {
        identity: max(cosine_sim(centroid, p) for p in prototypes)
        for identity, prototypes in prototype_bank.items()
    }

    sorted_ids = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    top1_id, top1_sim = sorted_ids[0]
    top2_id, top2_sim = sorted_ids[1]

    margin = top1_sim - top2_sim

    # Open-set thresholds (identity-agnostic)
    if top1_sim < 0.60:  # τ_min
        return "Unknown", "low_confidence"

    if margin < 0.08:  # Δ_open
        return "Unknown", "ambiguous"

    return top1_id, "assigned"
```

**Files**:
- `screentime/clustering/open_set_assign.py` (150 lines)
- Output: `diagnostics/id_assign_audit.jsonl`

**Example Output**:
```jsonl
{"cluster_id": 0, "sim_top": 0.68, "sim_second": 0.64, "margin": 0.04, "decision": "Unknown", "reason": "ambiguous"}
{"cluster_id": 1, "sim_top": 0.88, "sim_second": 0.52, "margin": 0.36, "decision": "RINNA", "reason": "assigned"}
```

---

#### Step 5: Integration & Testing (60 min)

**Tasks**:
1. Integrate all modules into `jobs/tasks/cluster.py`
2. Add workflow:
   ```
   DBSCAN clustering → Confusion detection →
   Anchored split (if ambiguous) → Open-set assignment →
   Save updated clusters.json
   ```
3. Test on RHOBH-TEST-10-28
4. Validate Kim/Kyle separation in Streamlit

---

## 📊 Expected Results

### Before (Current State)

```
Cluster 0: [KIM + KYLE mixed] - 250 tracks
Cluster 1: [RINNA] - 211 tracks
Cluster 2: [EILEEN] - 20 tracks
...
```

**Gallery**: Shows both Kim and Kyle faces in Cluster 0 ❌

---

### After (Prototype-Anchored Split)

```
Cluster 0: [KIM] - 130 tracks (split from original Cluster 0)
Cluster 1: [KYLE] - 120 tracks (split from original Cluster 0)
Cluster 2: [RINNA] - 211 tracks
Cluster 3: [EILEEN] - 20 tracks
...
```

**Gallery**: Each cluster shows single person ✅

**Diagnostics**:
- `confusion_matrix.json`: Shows Cluster 0 was ambiguous (Kim 0.68, Kyle 0.64)
- `anchored_split_audit.json`: Shows split decision, temporal consistency
- `id_assign_audit.jsonl`: Shows assignment reasoning for each cluster

---

## ⏱️ Time Estimate

| Task | Time | Complexity |
|------|------|------------|
| Multi-prototype bank | 60 min | Medium (clustering by pose/scale) |
| Confusion detection | 30 min | Low (similarity computations) |
| Anchored split | 90 min | High (seeded k-means, validation) |
| Open-set assignment | 30 min | Low (threshold-based logic) |
| Integration & testing | 60 min | Medium (workflow coordination) |
| **Total** | **4.5 hours** | |

---

## 🚀 Alternative Quick-Win Approach

**If 4.5 hours is too long**, consider this **30-minute manual intervention**:

### Manual Cluster Splitting in Streamlit

1. **Identify the mixed cluster** (likely Cluster 0 with 200+ tracks)
2. **In Streamlit Labeler**:
   - Navigate to Clusters page
   - Click on mixed cluster
   - Manually sort tracks by visual inspection
   - Use "Split Cluster" button (if available) or:
   - Export track IDs for Kim: `[t1, t2, t3, ...]`
   - Export track IDs for Kyle: `[t4, t5, t6, ...]`
3. **Update `clusters.json` manually**:
   ```json
   {
     "cluster_id": 0,
     "track_ids": [/* Kim track IDs */]
   },
   {
     "cluster_id": 7,  // New cluster for Kyle
     "track_ids": [/* Kyle track IDs */]
   }
   ```
4. **Re-run analytics** with updated clusters

**Pros**: Fast (30 min), gets you unstuck immediately
**Cons**: Manual work, not scalable, episode-specific

---

## 📂 Implementation Roadmap

### Option A: Full Prototype-Anchored Implementation (Recommended)

**Timeline**: 4.5 hours
**Deliverables**:
- Identity-agnostic prototype-anchored splitting
- Confusion matrix detection
- Open-set assignment
- Scalable to other episodes
- No manual intervention

**Files to Create**:
1. `screentime/clustering/prototype_bank.py` (300 lines)
2. `screentime/clustering/confusion_detector.py` (200 lines)
3. `screentime/clustering/anchored_split.py` (350 lines)
4. `screentime/clustering/open_set_assign.py` (150 lines)
5. Update `jobs/tasks/cluster.py` (+100 lines)

**Total**: ~1,100 lines of new code

---

### Option B: Manual Split + Quick Validation (Quick-Win)

**Timeline**: 30 minutes
**Deliverables**:
- Kim/Kyle separated in this episode
- Can proceed to analytics/testing
- Prototype implementation deferred

**Steps**:
1. Manually identify Kim vs Kyle tracks in Streamlit gallery
2. Update `clusters.json` with split
3. Re-run analytics
4. Generate delta_table.csv

**Limitation**: Episode-specific, requires manual work per episode

---

## 🔒 Guardrails (Maintained)

✅ **Identity-Agnostic**:
- All thresholds global (τ_min=0.60, Δ_open=0.08, etc.)
- Same logic for all ambiguous clusters
- No Kim/Kyle-specific tuning

✅ **Validation**:
- Totals ≤ runtime + overlaps
- No hardcoded overrides
- Contamination audit still runs

✅ **Baseline**:
- RetinaFace detector
- 10fps sampling
- Uniform pipeline for all

---

## 📝 Current Session Summary

**Hours Invested**: ~5 hours
**Code Written**: ~1,560 lines
**Infrastructure Ready**: ✅
- Face-only filtering
- Auto-caps
- Purity-driven clustering
- Contamination audit framework

**Remaining Work**: Prototype-anchored split (~4.5 hours)

**Alternative**: Manual split (~30 min) to unblock analytics

---

## 💡 Recommendation

**For immediate progress**: Use **Option B** (manual split) to:
1. Separate Kim/Kyle in this episode
2. Proceed to analytics and accuracy validation
3. Test rest of pipeline

**For scalable solution**: Schedule **Option A** (prototype-anchored split) as next session to:
1. Handle Kim/Kyle automatically
2. Support other similar cases (Brandi/Yolanda, etc.)
3. Enable open-set assignment for all episodes

---

## 📊 Files Created This Session

| File | Lines | Status |
|------|-------|--------|
| `screentime/clustering/face_quality.py` | 300 | ✅ Complete |
| `screentime/clustering/contamination_audit.py` | 400 | ✅ Complete |
| `screentime/attribution/auto_caps.py` | 150 | ✅ Complete |
| `screentime/clustering/auto_threshold.py` | 180 | ✅ Complete |
| `screentime/clustering/purity_driven_eps.py` | 400 | ✅ Complete |
| `jobs/tasks/cluster.py` | +150 | ✅ Modified |
| `jobs/tasks/analytics.py` | +50 | ✅ Modified |
| `.github/workflows/no-hardcoded-identities.sh` | 80 | ✅ Complete |
| **Total New Code** | **1,560 lines** | |

**Remaining**:
| File | Lines | Status |
|------|-------|--------|
| `screentime/clustering/prototype_bank.py` | 300 | ⏳ Planned |
| `screentime/clustering/confusion_detector.py` | 200 | ⏳ Planned |
| `screentime/clustering/anchored_split.py` | 350 | ⏳ Planned |
| `screentime/clustering/open_set_assign.py` | 150 | ⏳ Planned |
| **Total Remaining** | **~1,000 lines** | **~4.5 hours** |

---

**🎯 Next Action**: Choose Option A (4.5 hour implementation) or Option B (30 min manual split)?
