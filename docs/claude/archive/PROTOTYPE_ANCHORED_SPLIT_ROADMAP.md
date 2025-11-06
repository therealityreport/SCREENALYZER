# Prototype-Anchored Split - Complete Implementation Roadmap

**Date**: October 30, 2025
**Estimated Time**: 4-5 hours
**Status**: Ready to implement (Option A chosen)

---

## 🎯 Goal

Separate Kim/Kyle (and other ambiguous clusters) using **labeled prototypes as anchors** for semi-supervised splitting.

---

## ⚠️ Critical Prerequisite: Seed Facebank

**Blocker**: Empty facebank (`data/facebank/` has no labeled examples)

**Solution - Bootstrap Approach**:

###Step 0: Manual Seed Facebank Creation (30 min - USER ACTION REQUIRED)

**In Streamlit Gallery (http://localhost:8501)**:

1. Navigate to "Clusters" page
2. For each cluster that looks "clean" (single person):
   - Identify the person visually
   - Select 5-10 best face examples (frontal, clear, diverse poses)
   - Export/save these as labeled examples

**Expected Output**:
```
data/facebank/RHOBH-TEST-10-28/
├── KIM/
│   ├── sample_001.jpg (face chip + embedding)
│   ├── sample_002.jpg
│   └── ... (5-10 examples)
├── KYLE/
│   ├── sample_001.jpg
│   └── ...
├── RINNA/
├── EILEEN/
├── BRANDI/
├── YOLANDA/
└── LVP/
```

**Alternative - Use Existing Cluster Labels**:

If you've already manually labeled clusters in a previous run, we can extract prototypes from those labeled clusters:

```python
# Extract prototypes from previously labeled clusters
for identity in ['KIM', 'KYLE', 'RINNA', ...]:
    cluster_id = get_cluster_for_identity(identity)  # From old cluster_assignments
    samples = get_cluster_samples(cluster_id)
    # Pick 10 best (highest confidence, frontal, large)
    prototypes = select_best_samples(samples, k=10)
    save_to_facebank(identity, prototypes)
```

---

## 📋 Implementation Steps

Once facebank seed exists:

### Step 1: Multi-Prototype Bank Builder (60 min)

**File**: `screentime/clustering/prototype_bank.py` (~300 lines)

**Purpose**: Build diverse prototypes per identity (pose × scale)

**Algorithm**:

```python
def build_multi_prototype_bank(episode_id, facebank_path):
    """
    Build multi-prototype bank from seed facebank.

    For each identity:
    1. Load seed samples from facebank
    2. Cluster by pose (frontal/3-4/profile) using yaw angle
    3. For each pose, bin by scale (small/medium/large)
    4. Pick best representative per (pose, scale) cell

    Returns:
        {
            'KIM': {
                'frontal': {'small': emb, 'medium': emb, 'large': emb},
                '3-4': {...},
                'profile': {...}
            },
            'KYLE': {...},
            ...
        }
    """

    prototypes = {}

    for identity in facebank_identities:
        seed_samples = load_facebank_samples(facebank_path, identity)

        # 1. Cluster by pose (using face landmarks yaw/pitch)
        pose_clusters = cluster_by_pose(seed_samples)
        # Expected: ~3 clusters (frontal, 3/4, profile)

        # 2. For each pose, bin by face size
        identity_prototypes = {}
        for pose_label, pose_samples in pose_clusters.items():
            size_bins = bin_by_size(pose_samples, bins=[
                ('small', 0, 100),
                ('medium', 100, 150),
                ('large', 150, 999)
            ])

            # 3. Pick best representative per bin
            pose_prototypes = {}
            for size_label, size_samples in size_bins.items():
                if size_samples:
                    best = select_best(size_samples, criteria='confidence')
                    pose_prototypes[size_label] = best.embedding

            identity_prototypes[pose_label] = pose_prototypes

        prototypes[identity] = identity_prototypes

    # Save to diagnostics
    save_prototype_bank(episode_id, prototypes)

    return prototypes
```

**Key Functions**:
- `cluster_by_pose()`: Use yaw angle to group frontal/3-4/profile
- `bin_by_size()`: Bin by face_px into small/medium/large
- `select_best()`: Pick highest confidence + sharpness

**Output**: `data/harvest/{episode}/diagnostics/multi_prototypes.json`

---

### Step 2: Confusion Matrix Detector (30 min)

**File**: `screentime/clustering/confusion_detector.py` (~200 lines)

**Purpose**: Detect ambiguous clusters (top-2 identities too close)

**Algorithm**:

```python
def detect_confusion(clusters_data, picked_samples, prototype_bank, config):
    """
    Detect ambiguous clusters where top-2 identities are too close.

    For each cluster:
    1. Compute cluster medoid
    2. Compute similarity to all prototype banks (set-to-set max)
    3. If (sim_top - sim_second) < margin AND sim_top >= threshold:
       → Mark as ambiguous

    Returns:
        {
            'ambiguous_clusters': [
                {
                    'cluster_id': 0,
                    'top1_identity': 'KIM',
                    'top1_sim': 0.68,
                    'top2_identity': 'KYLE',
                    'top2_sim': 0.64,
                    'margin': 0.04,
                    'decision': 'ambiguous'
                }
            ]
        }
    """

    ambiguous = []

    for cluster in clusters_data['clusters']:
        # Get cluster samples
        cluster_samples = picked_samples[picked_samples['cluster_id'] == cluster['cluster_id']]
        cluster_embeddings = cluster_samples['embedding'].tolist()

        # Compute medoid
        medoid = compute_cluster_medoid(cluster_embeddings)

        # Compute similarity to each prototype bank
        sims = {}
        for identity, prototypes in prototype_bank.items():
            # Set-to-set max similarity
            all_proto_embs = [
                proto for pose in prototypes.values()
                for proto in pose.values()
            ]
            sims[identity] = max(cosine_sim(medoid, p) for p in all_proto_embs)

        # Sort by similarity
        sorted_ids = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top1_id, top1_sim = sorted_ids[0]
        top2_id, top2_sim = sorted_ids[1]

        margin = top1_sim - top2_sim

        # Check ambiguity (identity-agnostic thresholds)
        if margin < config.confusion_margin and top1_sim >= config.min_sim:
            ambiguous.append({
                'cluster_id': cluster['cluster_id'],
                'top1_identity': top1_id,
                'top1_sim': float(top1_sim),
                'top2_identity': top2_id,
                'top2_sim': float(top2_sim),
                'margin': float(margin),
                'decision': 'ambiguous'
            })

    return {'ambiguous_clusters': ambiguous}
```

**Config (Identity-Agnostic)**:
```python
@dataclass
class ConfusionConfig:
    confusion_margin: float = 0.10  # Top-2 margin threshold
    min_sim: float = 0.60           # Minimum similarity to consider
```

**Output**: `diagnostics/confusion_matrix.json`

---

### Step 3: Prototype-Anchored Split (90 min)

**File**: `screentime/clustering/anchored_split.py` (~350 lines)

**Purpose**: Split ambiguous clusters using prototype anchors

**Algorithm**:

```python
def anchored_split(cluster, anchor_A_prototypes, anchor_B_prototypes, config):
    """
    Split cluster using seeded 2-means with prototype anchors.

    Steps:
    1. Compute anchor centroids (mean of all prototypes)
    2. Assign each sample to nearest anchor
    3. Check temporal consistency (each side ≥600ms, ≥6 chips)
    4. Validate margin (each side prefers its anchor by ≥0.08)
    5. If valid: split into 2 clusters, else: keep as Unknown

    Returns:
        {
            'decision': 'split' | 'unknown',
            'side_A': {
                'samples': [...],
                'duration_ms': int,
                'avg_margin': float
            },
            'side_B': {...}
        }
    """

    # 1. Compute anchor centroids
    anchor_A_embs = [p for pose in anchor_A_prototypes.values() for p in pose.values()]
    anchor_B_embs = [p for pose in anchor_B_prototypes.values() for p in pose.values()]

    anchor_A = np.mean(anchor_A_embs, axis=0)
    anchor_B = np.mean(anchor_B_embs, axis=0)

    # Normalize
    anchor_A = anchor_A / np.linalg.norm(anchor_A)
    anchor_B = anchor_B / np.linalg.norm(anchor_B)

    # 2. Assign each sample to nearest anchor
    assignments = []
    margins = []

    for sample in cluster.samples:
        sim_A = np.dot(sample.embedding, anchor_A)
        sim_B = np.dot(sample.embedding, anchor_B)

        margin = abs(sim_A - sim_B)
        margins.append(margin)

        if sim_A > sim_B:
            assignments.append('A')
        else:
            assignments.append('B')

    # 3. Group by assignment
    side_A_samples = [s for s, a in zip(cluster.samples, assignments) if a == 'A']
    side_B_samples = [s for s, a in zip(cluster.samples, assignments) if a == 'B']

    # 4. Check temporal consistency
    duration_A = max(s.ts_ms for s in side_A_samples) - min(s.ts_ms for s in side_A_samples)
    duration_B = max(s.ts_ms for s in side_B_samples) - min(s.ts_ms for s in side_B_samples)

    avg_margin_A = np.mean([m for s, a, m in zip(cluster.samples, assignments, margins) if a == 'A'])
    avg_margin_B = np.mean([m for s, a, m in zip(cluster.samples, assignments, margins) if a == 'B'])

    # 5. Validate split
    valid_A = (
        duration_A >= config.min_duration_ms and
        len(side_A_samples) >= config.min_chips and
        avg_margin_A >= config.min_margin
    )

    valid_B = (
        duration_B >= config.min_duration_ms and
        len(side_B_samples) >= config.min_chips and
        avg_margin_B >= config.min_margin
    )

    if valid_A and valid_B:
        # Split is valid
        return {
            'decision': 'split',
            'side_A': {
                'samples': side_A_samples,
                'duration_ms': duration_A,
                'avg_margin': avg_margin_A
            },
            'side_B': {
                'samples': side_B_samples,
                'duration_ms': duration_B,
                'avg_margin': avg_margin_B
            }
        }
    else:
        # Not enough evidence
        return {'decision': 'unknown', 'reason': 'insufficient_temporal_consistency'}
```

**Config (Identity-Agnostic)**:
```python
@dataclass
class AnchoredSplitConfig:
    min_duration_ms: int = 600  # Each side must have ≥600ms
    min_chips: int = 6           # Each side must have ≥6 samples
    min_margin: float = 0.08     # Each side must prefer its anchor by ≥0.08
```

**Output**: `diagnostics/anchored_split_audit.json`

---

### Step 4: Open-Set Assignment (30 min)

**File**: `screentime/clustering/open_set_assign.py` (~150 lines)

**Purpose**: Assign clusters to identities or "Unknown"

**Algorithm**:

```python
def assign_cluster(cluster, prototype_bank, config):
    """
    Assign cluster to identity or Unknown (open-set).

    Thresholds (identity-agnostic):
    - τ_min = 0.60: Minimum similarity to assign
    - Δ_open = 0.08: Minimum margin to avoid ambiguity

    Returns:
        {
            'cluster_id': int,
            'assigned_identity': str | 'Unknown',
            'sim_top': float,
            'sim_second': float,
            'margin': float,
            'reason': str
        }
    """

    medoid = cluster.medoid

    # Compute similarities to all prototype banks
    sims = {}
    for identity, prototypes in prototype_bank.items():
        all_proto_embs = [
            proto for pose in prototypes.values()
            for proto in pose.values()
        ]
        sims[identity] = max(cosine_sim(medoid, p) for p in all_proto_embs)

    # Sort by similarity
    sorted_ids = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    top1_id, top1_sim = sorted_ids[0]
    top2_id, top2_sim = sorted_ids[1]

    margin = top1_sim - top2_sim

    # Open-set decision (identity-agnostic)
    if top1_sim < config.tau_min:
        return {
            'cluster_id': cluster.id,
            'assigned_identity': 'Unknown',
            'sim_top': top1_sim,
            'reason': 'low_confidence'
        }

    if margin < config.delta_open:
        return {
            'cluster_id': cluster.id,
            'assigned_identity': 'Unknown',
            'sim_top': top1_sim,
            'sim_second': top2_sim,
            'margin': margin,
            'reason': 'ambiguous'
        }

    # Confident assignment
    return {
        'cluster_id': cluster.id,
        'assigned_identity': top1_id,
        'sim_top': top1_sim,
        'sim_second': top2_sim,
        'margin': margin,
        'reason': 'assigned'
    }
```

**Config (Identity-Agnostic)**:
```python
@dataclass
class OpenSetConfig:
    tau_min: float = 0.60    # Minimum similarity for assignment
    delta_open: float = 0.08  # Minimum margin to avoid ambiguity
```

**Output**: `diagnostics/id_assign_audit.jsonl` (one line per cluster)

---

### Step 5: Integration & Workflow (60 min)

**Update**: `jobs/tasks/cluster.py`

**New Workflow**:

```python
def cluster_task_with_anchored_split(job_id, episode_id):
    # ... existing clustering ...

    # STEP 4: Build Multi-Prototype Bank
    prototype_bank = build_multi_prototype_bank(episode_id, facebank_path)

    # STEP 5: Detect Confusion
    confusion_results = detect_confusion(
        clusters_data, picked_samples, prototype_bank, confusion_config
    )

    # STEP 6: Anchored Split (for ambiguous clusters only)
    for ambiguous in confusion_results['ambiguous_clusters']:
        cluster_id = ambiguous['cluster_id']
        top1_id = ambiguous['top1_identity']
        top2_id = ambiguous['top2_identity']

        # Get cluster data
        cluster = get_cluster(cluster_id, clusters_data, picked_samples)

        # Get prototype anchors
        anchor_A = prototype_bank[top1_id]
        anchor_B = prototype_bank[top2_id]

        # Attempt split
        split_result = anchored_split(cluster, anchor_A, anchor_B, split_config)

        if split_result['decision'] == 'split':
            # Create two new clusters from split
            create_cluster_from_samples(split_result['side_A']['samples'], f"{cluster_id}_A")
            create_cluster_from_samples(split_result['side_B']['samples'], f"{cluster_id}_B")
            # Remove original cluster
            remove_cluster(cluster_id)

    # STEP 7: Open-Set Assignment
    assignments = []
    for cluster in updated_clusters:
        assignment = assign_cluster(cluster, prototype_bank, openset_config)
        assignments.append(assignment)

    # Save results
    save_confusion_matrix(episode_id, confusion_results)
    save_anchored_split_audit(episode_id, split_results)
    save_id_assign_audit(episode_id, assignments)

    return updated_clusters
```

---

## 📊 Expected Results

### Before

```
Cluster 0: [KIM + KYLE mixed] - 250 tracks
  Gallery: Shows both Kim and Kyle faces ❌
```

### After

```
Cluster 0_A: [KIM] - 130 tracks
  Gallery: Only Kim faces ✅

Cluster 0_B: [KYLE] - 120 tracks
  Gallery: Only Kyle faces ✅
```

### Diagnostics

**confusion_matrix.json**:
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

**anchored_split_audit.json**:
```json
{
  "splits": [
    {
      "original_cluster_id": 0,
      "anchor_A": "KIM",
      "anchor_B": "KYLE",
      "decision": "split",
      "side_A_samples": 130,
      "side_B_samples": 120,
      "side_A_duration_ms": 45000,
      "side_B_duration_ms": 38000
    }
  ]
}
```

**id_assign_audit.jsonl**:
```jsonl
{"cluster_id": 0, "assigned_identity": "Unknown", "reason": "ambiguous"}
{"cluster_id": 1, "assigned_identity": "RINNA", "margin": 0.45, "reason": "assigned"}
{"cluster_id": 2, "assigned_identity": "EILEEN", "margin": 0.38, "reason": "assigned"}
```

---

## ⏱️ Time Estimate

| Step | Time | Complexity |
|------|------|------------|
| **Step 0: Seed Facebank (USER)** | 30 min | Manual labeling |
| **Step 1: Multi-Prototype Bank** | 60 min | Medium (pose clustering) |
| **Step 2: Confusion Detection** | 30 min | Low (similarity computations) |
| **Step 3: Anchored Split** | 90 min | High (seeded k-means + validation) |
| **Step 4: Open-Set Assignment** | 30 min | Low (threshold logic) |
| **Step 5: Integration** | 60 min | Medium (workflow coordination) |
| **Total** | **5 hours** | |

---

## 🚀 Next Action: Choose Path

### Path A: Full Implementation (5 hours)

**Steps**:
1. **YOU**: Create seed facebank (30 min manual labeling in Streamlit)
2. **ME**: Implement Steps 1-5 (4.5 hours coding)
3. **RESULT**: Scalable, identity-agnostic prototype-anchored splitting

### Path B: Quick Bootstrap (2 hours)

**Steps**:
1. **YOU**: Manually label current clusters in Streamlit (identify which is KIM, KYLE, etc.)
2. **ME**: Extract prototypes from labeled clusters (30 min)
3. **ME**: Implement simplified anchored split (90 min - just Steps 2-3-4)
4. **RESULT**: Kim/Kyle separated, works for this episode

---

## 📝 Decision Point

**Ready to proceed?**

- **Path A**: I'll wait for you to create seed facebank, then implement full solution
- **Path B**: Label clusters now in Streamlit, I'll build quick bootstrap version

**Which path do you prefer?**
