# Multi-Prototype Identity Bank - Design Document

**Purpose**: Enable robust identity matching across pose/scale variations by storing multiple representative prototypes per identity, rather than single centroid embeddings.

**Problem Solved**: Entrance recovery bridging fails (sim=0.144 YOLANDA Track 307 → Track 42) when pose/lighting differs between appearance windows.

**Status**: Architecture documented, ready for next session implementation (3 hours)

---

## 1. Core Architecture

### MultiProtoIdentityBank Class

**Location**: `screentime/recognition/multi_proto_bank.py`

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd

@dataclass
class PrototypeBin:
    """Single prototype in the bank (one pose × scale bin)."""
    embedding: np.ndarray       # 512-d ArcFace embedding
    pose_bin: str              # "frontal", "three_quarter", "profile"
    scale_bin: str             # "small" (<100px), "medium" (≥100px)
    source_track_id: int       # Track this prototype came from
    n_samples: int             # Number of embeddings averaged
    confidence: float          # Average detection confidence

class MultiProtoIdentityBank:
    """
    Multi-prototype identity bank with pose/scale stratification.

    Each identity has up to 6 prototypes:
    - 3 pose bins: frontal (±30°), three-quarter (30-60°), profile (>60°)
    - 2 scale bins: small (<100px), medium (≥100px)
    """

    def __init__(self):
        # identity_name -> list[PrototypeBin]
        self._bank: dict[str, list[PrototypeBin]] = {}

    def add_prototype(self, identity: str, embedding: np.ndarray,
                     pose_bin: str, scale_bin: str, **metadata):
        """Add or update a prototype bin."""
        if identity not in self._bank:
            self._bank[identity] = []

        # Check if bin exists, update if so
        for proto in self._bank[identity]:
            if proto.pose_bin == pose_bin and proto.scale_bin == scale_bin:
                # Average with existing
                proto.embedding = (proto.embedding * proto.n_samples + embedding) / (proto.n_samples + 1)
                proto.n_samples += 1
                return

        # Create new bin
        self._bank[identity].append(PrototypeBin(
            embedding=embedding,
            pose_bin=pose_bin,
            scale_bin=scale_bin,
            **metadata
        ))

    def match(self, query_embedding: np.ndarray,
             pose_bin: Optional[str] = None,
             scale_bin: Optional[str] = None,
             method: str = "max") -> tuple[str, float, float]:
        """
        Match query embedding to bank.

        Args:
            query_embedding: 512-d face embedding
            pose_bin: Optional pose hint for routing
            scale_bin: Optional scale hint for routing
            method: "max" (best prototype) or "mean" (average across bins)

        Returns:
            (best_identity, similarity, margin_to_second_best)
        """
        scores = {}

        for identity, prototypes in self._bank.items():
            # Filter prototypes by hint (if provided)
            candidates = prototypes
            if pose_bin:
                candidates = [p for p in candidates if p.pose_bin == pose_bin]
            if scale_bin:
                candidates = [p for p in candidates if p.scale_bin == scale_bin]

            if not candidates:
                candidates = prototypes  # Fallback to all

            # Compute similarity
            sims = [np.dot(query_embedding, p.embedding) for p in candidates]

            if method == "max":
                scores[identity] = max(sims)
            elif method == "mean":
                scores[identity] = np.mean(sims)

        # Find best and second-best
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_id, best_sim = sorted_scores[0]
        second_sim = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        margin = best_sim - second_sim

        return best_id, best_sim, margin

    def get_prototype_count(self, identity: str) -> int:
        """Return number of prototype bins for identity."""
        return len(self._bank.get(identity, []))

    def export_to_parquet(self, path: Path):
        """Save bank to parquet for persistence."""
        rows = []
        for identity, prototypes in self._bank.items():
            for proto in prototypes:
                rows.append({
                    'identity': identity,
                    'pose_bin': proto.pose_bin,
                    'scale_bin': proto.scale_bin,
                    'embedding': proto.embedding.tolist(),
                    'source_track_id': proto.source_track_id,
                    'n_samples': proto.n_samples,
                    'confidence': proto.confidence
                })
        df = pd.DataFrame(rows)
        df.to_parquet(path)

    @classmethod
    def load_from_parquet(cls, path: Path) -> 'MultiProtoIdentityBank':
        """Load bank from parquet."""
        df = pd.read_parquet(path)
        bank = cls()
        for _, row in df.iterrows():
            bank.add_prototype(
                identity=row['identity'],
                embedding=np.array(row['embedding'], dtype=np.float32),
                pose_bin=row['pose_bin'],
                scale_bin=row['scale_bin'],
                source_track_id=int(row['source_track_id']),
                n_samples=int(row['n_samples']),
                confidence=float(row['confidence'])
            )
        return bank
```

---

## 2. Pose Estimation (Simplified)

**Method**: Landmark-based yaw approximation (no external pose model)

```python
def estimate_pose_bin(kps: np.ndarray) -> str:
    """
    Estimate pose bin from 5-point facial landmarks.

    Landmarks (InsightFace order):
        0: left eye, 1: right eye, 2: nose, 3: left mouth, 4: right mouth

    Returns:
        "frontal" (±30°), "three_quarter" (30-60°), "profile" (>60°)
    """
    if kps is None or kps.shape[0] < 5:
        return "frontal"  # Default if no landmarks

    left_eye = kps[0]
    right_eye = kps[1]
    nose = kps[2]

    # Horizontal distances
    left_to_nose = np.linalg.norm(nose - left_eye)
    right_to_nose = np.linalg.norm(nose - right_eye)

    # Asymmetry ratio (1.0 = frontal, >2.0 = profile)
    if right_to_nose == 0:
        ratio = 999
    else:
        ratio = left_to_nose / right_to_nose

    if 0.7 < ratio < 1.4:
        return "frontal"
    elif 1.4 <= ratio < 2.0 or 0.5 < ratio <= 0.7:
        return "three_quarter"
    else:
        return "profile"

def estimate_scale_bin(face_size: int) -> str:
    """
    Classify face by size bin.

    Returns:
        "small" (<100px), "medium" (≥100px)
    """
    return "small" if face_size < 100 else "medium"
```

---

## 3. Set-to-Set Bridge Matching

**Problem**: Single medoid embeddings fail on pose variation (entrance vs later appearance)

**Solution**: Compare top-K embeddings from entrance cluster to top-K from downstream track

```python
def set_to_set_similarity(set_a: list[np.ndarray],
                          set_b: list[np.ndarray],
                          method: str = "max_topk",
                          k: int = 5) -> float:
    """
    Compute similarity between two sets of embeddings.

    Args:
        set_a: Entrance cluster embeddings
        set_b: Downstream track embeddings
        method: "max_topk" (average of top-K pairs) or "mean_topk"
        k: Number of top embeddings to use

    Returns:
        Similarity score (0.0-1.0)
    """
    if method == "max_topk":
        # For each embedding in set_a, find best match in set_b
        # Then take average of top-K such pairs
        all_pairs = []
        for emb_a in set_a[:k]:
            best_match = max(np.dot(emb_a, emb_b) for emb_b in set_b[:k])
            all_pairs.append(best_match)
        return np.mean(sorted(all_pairs, reverse=True)[:k])

    elif method == "mean_topk":
        # Average top-K from each set, then compare
        mean_a = np.mean(set_a[:k], axis=0)
        mean_b = np.mean(set_b[:k], axis=0)
        return float(np.dot(mean_a, mean_b))
```

**Usage in Entrance Recovery**:
```python
# In bridge logic (entrance_recovery.py)
entrance_topk = sorted(entrance_embeddings,
                       key=lambda e: e['similarity'],
                       reverse=True)[:5]

for downstream_track in candidate_tracks:
    track_topk = sorted(downstream_track['embeddings'],
                       key=lambda e: e['confidence'],
                       reverse=True)[:5]

    bridge_sim = set_to_set_similarity(
        [e['embedding'] for e in entrance_topk],
        [e['embedding'] for e in track_topk],
        method="max_topk",
        k=5
    )

    if bridge_sim >= 0.70:  # Accept bridge
        # Merge entrance track into downstream
```

---

## 4. Bank Population Strategy

### Phase 1: Bootstrap from Labeled Tracks (Post-Clustering)

```python
def populate_bank_from_clusters(episode_id: str, data_root: Path) -> MultiProtoIdentityBank:
    """
    Build multi-prototype bank from labeled tracks.

    For each identity:
    1. Load all embeddings from tracks in their cluster
    2. Bin by pose × scale
    3. Average within each bin to create prototype
    """
    bank = MultiProtoIdentityBank()

    # Load clusters and embeddings
    clusters = load_clusters(episode_id, data_root)
    embeddings_df = pd.read_parquet(data_root / "harvest" / episode_id / "embeddings.parquet")

    for cluster in clusters['clusters']:
        identity = cluster.get('name')
        if not identity:
            continue

        # Get all embeddings for this identity's tracks
        track_ids = cluster['track_ids']
        identity_embs = embeddings_df[embeddings_df['track_id'].isin(track_ids)]

        # Bin by pose × scale
        bins = {}
        for _, row in identity_embs.iterrows():
            pose_bin = estimate_pose_bin(np.array(row['kps']))
            scale_bin = estimate_scale_bin(int(row['face_size']))

            key = (pose_bin, scale_bin)
            if key not in bins:
                bins[key] = []
            bins[key].append(np.array(row['embedding']))

        # Create prototypes (average within bins)
        for (pose_bin, scale_bin), embs in bins.items():
            if len(embs) < 3:  # Skip sparse bins
                continue

            # Take top 10 by confidence, average
            top_embs = sorted(embs, key=lambda e: e['confidence'], reverse=True)[:10]
            prototype = np.mean([e['embedding'] for e in top_embs], axis=0)

            bank.add_prototype(
                identity=identity,
                embedding=prototype,
                pose_bin=pose_bin,
                scale_bin=scale_bin,
                source_track_id=track_ids[0],  # Representative
                n_samples=len(embs),
                confidence=np.mean([e['confidence'] for e in top_embs])
            )

    return bank
```

### Phase 2: Assimilate Entrance Tracks

When entrance recovery creates a new track (e.g., Track 307 YOLANDA):
1. Extract embeddings from entrance track
2. Bin by pose × scale
3. Add to bank as new prototypes (if bins don't exist) or update existing

```python
def assimilate_entrance_track(bank: MultiProtoIdentityBank,
                               identity: str,
                               entrance_embeddings: list[dict]):
    """
    Add entrance track embeddings to the identity's bank.

    This enriches the bank with early-appearance variations.
    """
    # Bin entrance embeddings
    bins = {}
    for emb_data in entrance_embeddings:
        pose_bin = estimate_pose_bin(emb_data.get('kps'))
        scale_bin = estimate_scale_bin(emb_data.get('face_size', 100))

        key = (pose_bin, scale_bin)
        if key not in bins:
            bins[key] = []
        bins[key].append(emb_data['embedding'])

    # Update bank
    for (pose_bin, scale_bin), embs in bins.items():
        if len(embs) < 2:
            continue

        avg_emb = np.mean(embs, axis=0)
        bank.add_prototype(
            identity=identity,
            embedding=avg_emb,
            pose_bin=pose_bin,
            scale_bin=scale_bin,
            source_track_id=entrance_embeddings[0]['track_id'],
            n_samples=len(embs),
            confidence=np.mean([e['confidence'] for e in embs])
        )
```

---

## 5. Integration Points

### A. Entrance Recovery Bridge Logic

**File**: `jobs/tasks/entrance_recovery.py` (line ~700, bridge attempt)

**Before** (single medoid):
```python
entrance_medoid = compute_medoid(entrance_cluster['embeddings'])
track_medoid = compute_medoid(downstream_track['embeddings'])
bridge_sim = np.dot(entrance_medoid, track_medoid)
```

**After** (set-to-set with bank):
```python
# Option 1: Set-to-set direct
bridge_sim = set_to_set_similarity(
    entrance_cluster['embeddings'][:5],
    downstream_track['embeddings'][:5],
    method="max_topk",
    k=5
)

# Option 2: Bank-assisted routing
entrance_pose = estimate_pose_bin(entrance_cluster['avg_kps'])
entrance_scale = estimate_scale_bin(entrance_cluster['avg_face_size'])

best_id, best_sim, margin = bank.match(
    entrance_medoid,
    pose_bin=entrance_pose,
    scale_bin=entrance_scale,
    method="max"
)

if best_id == target_identity and best_sim >= 0.70:
    bridge_success = True
```

---

### B. Clustering Stage (Identity Assignment)

**File**: `screentime/pipeline/clustering.py` (line ~400)

**Before** (single prototype per identity):
```python
# Load facebank prototypes
prototypes = load_identity_prototypes(episode_id, facebank_path)
for track in unlabeled_tracks:
    track_medoid = compute_medoid(track['embeddings'])
    best_id, best_sim = find_best_match(track_medoid, prototypes)
```

**After** (multi-prototype bank):
```python
bank = MultiProtoIdentityBank.load_from_parquet(facebank_path / "multi_proto_bank.parquet")

for track in unlabeled_tracks:
    track_medoid = compute_medoid(track['embeddings'])
    track_pose = estimate_pose_bin(track['avg_kps'])
    track_scale = estimate_scale_bin(track['avg_face_size'])

    best_id, best_sim, margin = bank.match(
        track_medoid,
        pose_bin=track_pose,
        scale_bin=track_scale,
        method="max"
    )

    if best_sim >= 0.82 and margin >= 0.08:
        assign_track_to_identity(track, best_id)
```

---

## 6. Data Schema

### Storage: `data/facebank/EPISODEID/multi_proto_bank.parquet`

| Column | Type | Description |
|--------|------|-------------|
| identity | str | Identity name (YOLANDA, KIM, etc.) |
| pose_bin | str | Pose category (frontal, three_quarter, profile) |
| scale_bin | str | Size category (small, medium) |
| embedding | list[float] | 512-d ArcFace embedding (stored as JSON list) |
| source_track_id | int | Representative track this prototype came from |
| n_samples | int | Number of embeddings averaged into this prototype |
| confidence | float | Average detection confidence |

**Example Row**:
```
identity: "YOLANDA"
pose_bin: "three_quarter"
scale_bin: "small"
embedding: [0.123, -0.456, ..., 0.789]  # 512 floats
source_track_id: 307
n_samples: 12
confidence: 0.87
```

---

## 7. Expected Impact

### Current Baseline (Single Prototype):
- YOLANDA Track 307 → Track 42 bridge: **sim=0.144** (FAIL)
- Entrance tracks created but **all bridges rejected**
- +0.42s to +1.50s recovered per identity (no bridging benefit)

### With Multi-Prototype Bank:
- Bridge using **set-to-set top-5** matching
- Expected bridge sim: **0.70-0.78** (PASS)
- Entrance tracks **merge into downstream**, extending coverage
- Expected additional recovery: **1-2s per identity** from successful bridges

### Bridge Success Rate Prediction:
- **Before**: 0/6 bridges succeeded (0%)
- **After**: 4-5/6 bridges expected (67-83%)
- **Failure mode**: LVP (only 0.75s recovered, may not have downstream track nearby)

---

## 8. Implementation Checklist

**Estimated Time**: 3 hours

### Files to Create:
- [ ] `screentime/recognition/multi_proto_bank.py` (200 lines) - Core class
- [ ] `jobs/tasks/build_multi_proto_bank.py` (150 lines) - Population from clusters
- [ ] `tests/test_multi_proto_bank.py` (100 lines) - Unit tests

### Files to Modify:
- [ ] `jobs/tasks/entrance_recovery.py` (line ~700) - Set-to-set bridging
- [ ] `screentime/pipeline/clustering.py` (line ~400) - Bank-assisted assignment
- [ ] `configs/pipeline.yaml` - Add multi_proto config section

### Configuration (Add to pipeline.yaml):
```yaml
multi_proto_bank:
  enabled: true
  pose_bins: ["frontal", "three_quarter", "profile"]
  scale_bins: ["small", "medium"]
  scale_threshold_px: 100
  min_samples_per_bin: 3
  topk_for_prototype: 10
  bridge:
    method: "max_topk"  # or "mean_topk"
    k: 5
    min_sim: 0.70
```

### Validation:
- [ ] Run `build_multi_proto_bank.py` on RHOBH-TEST-10-28
- [ ] Verify 4-6 bins per identity (pose × scale)
- [ ] Re-run entrance recovery with set-to-set bridging
- [ ] Check bridge success rate (expect 4-5/6)
- [ ] Verify delta table improvement (expect 4-6/7 PASS)

---

## 9. Acceptance Criteria

✅ Bank contains 4-6 prototypes per identity (pose × scale bins)
✅ Set-to-set bridge sim ≥ 0.70 for YOLANDA Track 307 → Track 42
✅ Bridge success rate ≥ 60% (4+/6 entrance tracks merge)
✅ Delta table shows ≥5/7 identities PASS (≤4.5s error)
✅ Bank persisted to parquet, reusable across episodes

---

**Status**: Design complete, ready for 3-hour implementation sprint next session
**Files**: 3 new (multi_proto_bank.py, build_multi_proto_bank.py, tests), 2 modified (entrance_recovery.py, clustering.py)
**Expected Impact**: +1-2s recovery per identity from successful entrance bridging
