# Season-Aware Flows + Multi-Format Image Ingest + Unknown Discovery

**Date**: November 4, 2025
**Status**: Ready for Implementation
**Estimated Time**: 12-15 hours

---

## Overview

Transform the system from episode-centric to season-aware with:
1. **Show/Season hierarchy** for organizing episodes
2. **Multi-format image ingest** (jpg/png/avif/webp/heic) with auto-normalization
3. **Season memory** - reusable face banks across episodes
4. **Unknown discovery** - surface and promote new faces during review
5. **Clean re-harvest** - identity-agnostic pipeline from scratch

**Goal**: Re-run RHOBH S05 cleanly with cast-seeded season bank; future episodes auto-leverage learned identities.

---

## Current State Assessment

**✅ Already Built (from October 30th work)**:
- Face-only filtering + Top-K selection
- Purity-driven clustering
- Auto-caps (identity-agnostic merge limits)
- Contamination audit framework
- Streamlit UI with Review/Analytics pages

**🔨 Needs Implementation**:
- Show/Season data model
- Multi-format image upload + auto-convert
- Cast Images UI with face validation
- Season bank (multi_prototypes.json)
- Unknown discovery panel
- Re-harvest button + pipeline integration

---

## Architecture Overview

### Directory Structure (NEW)

```
data/
  videos/
    <SHOW>/
      <SEASON>/
        episode.mp4
  facebank/
    <SHOW>/
      <SEASON>/
        multi_prototypes.json      # Season bank (pose × scale bins)
        <CAST_NAME>/
          seed_001.png              # Normalized seed images
          seed_002.png
          metadata.json             # Original format, quality metrics
  harvest/
    <SHOW>/
      <SEASON>/
        <EPISODE>/
          tracks.json
          clusters.json
          embeddings.parquet
          picked_samples.parquet
  outputs/
    <SHOW>/
      <SEASON>/
        <EPISODE>/
          timeline.csv
          totals.csv
          delta_table.csv

configs/
  shows_seasons.json                # Registry
  presets/
    <SHOW>/<SEASON>.yaml            # Optional season-specific overrides
  pipeline.yaml                      # Global config
```

### Registry: shows_seasons.json

```json
{
  "shows": [
    {
      "show_id": "rhobh",
      "show_name": "Real Housewives of Beverly Hills",
      "seasons": [
        {
          "season_id": "s05",
          "season_label": "Season 5",
          "season_number": 5,
          "cast": [
            {"name": "KIM", "seed_count": 12, "valid_seeds": 10},
            {"name": "KYLE", "seed_count": 10, "valid_seeds": 9},
            {"name": "LISA", "seed_count": 8, "valid_seeds": 8}
          ],
          "episodes": [
            {
              "episode_id": "e01-pilot",
              "video_path": "data/videos/rhobh/s05/e01-pilot.mp4",
              "status": "completed",
              "duration_sec": 2580
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Phase 1: Multi-Format Image Processing Infrastructure (1.5 hours)

### File: `screentime/image_utils.py` (NEW)

**Requirements**:
- Accept: `.jpg`, `.jpeg`, `.png`, `.avif`, `.webp`, `.heic`
- Auto-convert to normalized format:
  - RGB colorspace (8-bit)
  - Strip EXIF
  - Respect orientation
  - Save as PNG or lossless WebP
- Keep audit trail of original format

**Implementation**:

```python
"""
Image normalization utilities for multi-format ingestion.
"""

from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import pillow_heif  # For HEIC support

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.avif', '.webp', '.heic'}

def normalize_image(
    input_path: Path,
    output_path: Path,
    save_format: str = "png"
) -> dict:
    """
    Normalize image to analysis-ready format.

    Args:
        input_path: Original image path
        output_path: Where to save normalized image
        save_format: "png" or "webp" (lossless)

    Returns:
        Metadata dict with original format, size, etc.
    """
    # Register HEIF opener
    if input_path.suffix.lower() == '.heic':
        pillow_heif.register_heif_opener()

    # Open image
    img = Image.open(input_path)

    # Capture original metadata
    original_format = img.format or input_path.suffix.lstrip('.')
    original_size = img.size

    # Apply EXIF orientation
    if hasattr(img, '_getexif') and img._getexif() is not None:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)

    # Convert to RGB (strip alpha, handle CMYK, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Save normalized version
    if save_format == "png":
        img.save(output_path, format='PNG', optimize=False)
    elif save_format == "webp":
        img.save(output_path, format='WEBP', lossless=True, quality=100)
    else:
        raise ValueError(f"Unsupported save format: {save_format}")

    return {
        'original_path': str(input_path),
        'original_format': original_format,
        'original_size': original_size,
        'normalized_path': str(output_path),
        'normalized_format': save_format,
        'normalized_size': img.size
    }
```

**Dependencies to add**:
```bash
pip install pillow pillow-heif pillow-avif-plugin
```

**Acceptance**:
- Can load jpg/png/avif/webp/heic and save as normalized PNG
- EXIF orientation respected
- Original format tracked in metadata

---

## Phase 2: Show/Season Data Model + Registry (1.5 hours)

### File: `screentime/models/show_season.py` (NEW)

**No database** - use JSON registry for simplicity.

```python
"""
Show/Season registry management (file-based, no DB).
"""

import json
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime

REGISTRY_PATH = Path("configs/shows_seasons.json")

@dataclass
class CastMember:
    name: str
    seed_count: int = 0
    valid_seeds: int = 0

@dataclass
class Episode:
    episode_id: str
    video_path: str
    status: str = "uploaded"  # uploaded, processing, completed, failed
    duration_sec: Optional[float] = None
    created_at: str = None

@dataclass
class Season:
    season_id: str
    season_label: str
    season_number: int
    cast: List[CastMember] = None
    episodes: List[Episode] = None

    def __post_init__(self):
        if self.cast is None:
            self.cast = []
        if self.episodes is None:
            self.episodes = []

@dataclass
class Show:
    show_id: str
    show_name: str
    seasons: List[Season] = None

    def __post_init__(self):
        if self.seasons is None:
            self.seasons = []

class ShowSeasonRegistry:
    """Manage show/season registry."""

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self.registry_path = registry_path
        self.shows: List[Show] = []
        self.load()

    def load(self):
        """Load registry from JSON."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                self.shows = [self._dict_to_show(s) for s in data.get('shows', [])]
        else:
            self.shows = []

    def save(self):
        """Save registry to JSON."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump({'shows': [asdict(s) for s in self.shows]}, f, indent=2)

    def create_show(self, show_id: str, show_name: str) -> Show:
        """Create new show."""
        show = Show(show_id=show_id, show_name=show_name)
        self.shows.append(show)
        self.save()
        return show

    def create_season(self, show_id: str, season_number: int, season_label: str) -> Season:
        """Create new season."""
        show = self.get_show(show_id)
        if not show:
            raise ValueError(f"Show {show_id} not found")

        season_id = f"s{season_number:02d}"
        season = Season(
            season_id=season_id,
            season_label=season_label,
            season_number=season_number
        )
        show.seasons.append(season)
        self.save()
        return season

    def get_show(self, show_id: str) -> Optional[Show]:
        """Get show by ID."""
        return next((s for s in self.shows if s.show_id == show_id), None)

    def get_season(self, show_id: str, season_id: str) -> Optional[Season]:
        """Get season by ID."""
        show = self.get_show(show_id)
        if not show:
            return None
        return next((s for s in show.seasons if s.season_id == season_id), None)

    # ... helper methods for cast, episodes ...
```

**Acceptance**:
- Can create RHOBH → Season 5
- Registry persists to JSON
- Path helpers return correct directories

---

## Phase 3: Upload Page - Show/Season Selection (2 hours)

### File: `app/labeler.py` - Modify `render_upload_page()`

**Changes**:

1. **Add Show selector** (before video upload)
   - Dropdown with existing shows + "+ Add new"
   - Inline form for new show creation

2. **Add Season selector** (after show)
   - Scoped to selected show
   - "+ Add new" option with inline form

3. **Bind video to Show/Season**
   - Update registry with episode entry
   - Save to `data/videos/<SHOW>/<SEASON>/`

**UI Mock**:

```
📤 Upload Episode
─────────────────

1. Select Show
   [Dropdown: RHOBH ▼]  [+ Add new]

2. Select Season
   [Dropdown: Season 5 ▼]  [+ Add new]

3. Select Video File
   [Choose file: episode.mp4]

   Episode Name: [E01-PilotParty________]

   [Validate Video] [Start Upload]
```

**Acceptance**:
- Can create RHOBH → S05 at upload time
- Video saved to `data/videos/rhobh/s05/e01-pilot.mp4`
- Registry updated

---

## Phase 4: Cast Images Page - Multi-Format Upload + Face Validation (3 hours)

### File: `app/labeler.py` - Add `render_cast_images_page()` updates

**Requirements**:
1. **Show/Season selectors** at top
2. **Current cast display** with seed counts
3. **Add cast form**:
   - Cast Name input
   - Multi-file uploader (drag/drop)
   - Accept all formats
4. **Face validation on each upload**:
   - Detect faces (reject if 0 or >1, unless user picks)
   - Min confidence: 0.65
   - Min face height: 64-72px
   - Quality bar: small/med/large, frontal/¾/profile
5. **Auto-convert** to normalized PNG
6. **Gallery view** of seeds per cast

**Face Validation Logic**:

```python
def validate_cast_image(image_path: Path, min_face_px: int = 64) -> dict:
    """
    Validate cast image for face bank.

    Returns:
        {
            'valid': bool,
            'errors': List[str],
            'face_count': int,
            'face_quality': dict,  # confidence, size, pose estimate
            'normalized_path': Optional[Path]
        }
    """
    from screentime.detectors.face_retina import RetinaFaceDetector

    detector = RetinaFaceDetector()

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        return {'valid': False, 'errors': ['Cannot read image']}

    # Detect faces
    faces = detector.detect(img)

    if len(faces) == 0:
        return {'valid': False, 'errors': ['No face detected'], 'face_count': 0}

    if len(faces) > 1:
        return {
            'valid': False,
            'errors': [f'{len(faces)} faces detected - please crop to single face'],
            'face_count': len(faces)
        }

    face = faces[0]

    # Quality checks
    face_height = face['bbox'][3] - face['bbox'][1]
    if face_height < min_face_px:
        return {
            'valid': False,
            'errors': [f'Face too small ({face_height}px < {min_face_px}px)'],
            'face_count': 1
        }

    if face['confidence'] < 0.65:
        return {
            'valid': False,
            'errors': [f'Low confidence ({face["confidence"]:.2f})'],
            'face_count': 1
        }

    return {
        'valid': True,
        'errors': [],
        'face_count': 1,
        'face_quality': {
            'confidence': face['confidence'],
            'face_height': face_height,
            'pose': 'frontal' if face.get('yaw', 0) < 15 else 'profile'
        }
    }
```

**UI Flow**:

```
🎭 Cast Reference Images
────────────────────────

1. Select Show & Season
   [RHOBH ▼] [Season 5 ▼]

2. Current Cast
   ┌─────────────────────────────────────┐
   │ KIM          KYLE         LISA      │
   │ 10 seeds ✅   8 seeds ⚠️   12 seeds ✅│
   └─────────────────────────────────────┘

3. Add Cast Member
   Cast Name: [KIM____________]

   Drag and drop images here
   ┌───────────────────────────────────┐
   │  📁 img1.jpg  ✅ Valid            │
   │  📁 img2.heic ✅ Valid            │
   │  📁 img3.webp ❌ No face detected  │
   └───────────────────────────────────┘

   [Upload & Add to Season Bank]

4. Facebank Gallery
   ▼ KIM (10 seeds)
     [img] [img] [img] [img] [img]
     Small: 2  Med: 5  Large: 3
     Frontal: 7  Three-quarter: 2  Profile: 1
```

**Acceptance**:
- Can upload jpg/png/avif/webp/heic
- Non-face images rejected with toast
- Multi-face images prompt to pick face
- Seeds auto-converted to PNG
- Quality bar shows distribution

---

## Phase 5: Season Bank System (2 hours)

### File: `screentime/clustering/season_bank.py` (NEW)

**Requirements**:
1. **Build multi_prototypes.json** from seeds
2. **Pose × scale bins**:
   - Pose: frontal (yaw <15°), three_quarter (15-45°), profile (>45°)
   - Scale: small (<100px height), large (≥100px)
3. **Top-K per bin** (e.g., 5 best per pose/scale combo)
4. **Rolling updates** after confirmed episodes

**Structure**:

```json
{
  "season_id": "s05",
  "show_id": "rhobh",
  "last_updated": "2025-11-04T12:00:00Z",
  "identities": {
    "KIM": {
      "frontal_small": [
        {"embedding": [...], "source": "seed_003.png", "confidence": 0.92},
        ...
      ],
      "frontal_large": [...],
      "three_quarter_small": [...],
      ...
    },
    "KYLE": {...}
  }
}
```

**Implementation**:

```python
def build_season_bank_from_seeds(show_id: str, season_id: str) -> dict:
    """
    Build season bank from cast seed images.
    """
    from screentime.detectors.face_retina import RetinaFaceDetector
    from screentime.recognition.embed_arcface import ArcFaceEmbedder

    detector = RetinaFaceDetector()
    embedder = ArcFaceEmbedder()

    facebank_dir = Path(f"data/facebank/{show_id}/{season_id}")

    identities = {}

    for cast_dir in facebank_dir.iterdir():
        if not cast_dir.is_dir():
            continue

        cast_name = cast_dir.name
        seed_files = sorted(cast_dir.glob("seed_*.png"))

        # Bin by pose × scale
        bins = {
            'frontal_small': [],
            'frontal_large': [],
            'three_quarter_small': [],
            'three_quarter_large': [],
            'profile_small': [],
            'profile_large': []
        }

        for seed_file in seed_files:
            img = cv2.imread(str(seed_file))
            faces = detector.detect(img)

            if len(faces) != 1:
                continue

            face = faces[0]

            # Embed
            aligned = align_face(img, face)
            embedding = embedder.embed(aligned)

            # Determine bin
            yaw = face.get('yaw', 0)
            face_height = face['bbox'][3] - face['bbox'][1]

            if abs(yaw) < 15:
                pose = 'frontal'
            elif abs(yaw) < 45:
                pose = 'three_quarter'
            else:
                pose = 'profile'

            scale = 'small' if face_height < 100 else 'large'

            bin_key = f"{pose}_{scale}"

            bins[bin_key].append({
                'embedding': embedding.tolist(),
                'source': seed_file.name,
                'confidence': face['confidence'],
                'face_height': face_height,
                'yaw': yaw
            })

        # Keep top-K per bin (by confidence)
        for bin_key in bins:
            bins[bin_key] = sorted(
                bins[bin_key],
                key=lambda x: x['confidence'],
                reverse=True
            )[:5]

        identities[cast_name] = bins

    return {
        'season_id': season_id,
        'show_id': show_id,
        'last_updated': datetime.utcnow().isoformat(),
        'identities': identities
    }
```

**Acceptance**:
- Seeds binned by pose × scale
- Top-5 per bin saved to multi_prototypes.json
- Can load and use for assignment

---

## Phase 6: Unknown/Not-in-Cast Discovery Panel (2 hours)

### File: `app/labeler.py` - Add to `render_review_page()`

**Requirements**:
- Show clusters/tracks that are:
  - Unknown (open-set, not assigned)
  - OR accumulated ≥6s this season
- For each unknown:
  - Rep tiles (from picked_samples)
  - Time this episode / this season
  - Top-2 nearest season identities with sims (context only)
- Actions:
  - "Add to Season Cast" → prompt for name, add to facebank
  - "Mark Guest/Ignore"
  - "Merge with Existing"

**UI**:

```
🔍 Review & Label Faces
─────────────────────────

Episode: RHOBH S05E01

[All Faces] [Pairwise Review] [Low-Confidence] [Unclustered] [Unknown / Not in Cast] ← NEW TAB

▼ Unknown / Not in Cast (Season)

  ┌─────────────────────────────────────────────────┐
  │ Cluster 42 (Unknown)                            │
  │ [img] [img] [img]                               │
  │                                                  │
  │ This Episode: 12.4s  |  This Season: 45.2s     │
  │                                                  │
  │ Nearest Identities:                             │
  │   1. BRANDI (sim: 0.54)                         │
  │   2. RINNA (sim: 0.48)                          │
  │                                                  │
  │ [Add to Season Cast] [Mark Guest] [Merge with]  │
  └─────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────┐
  │ Cluster 58 (Unknown)                            │
  │ ...                                              │
  └─────────────────────────────────────────────────┘
```

**Logic**:

```python
def get_unknown_clusters(episode_id: str, show_id: str, season_id: str, min_season_sec: float = 6.0):
    """
    Find Unknown clusters that exceed season threshold.
    """
    from app.lib.data import load_clusters
    from screentime.utils import get_episode_path

    # Load current episode clusters
    clusters_data = load_clusters(episode_id)

    # Find all episodes in this season
    season_episodes = get_season_episodes(show_id, season_id)

    unknowns = []

    for cluster in clusters_data.get('clusters', []):
        if cluster.get('name') not in (None, 'Unknown'):
            continue

        # Calculate season-wide time for this cluster
        # (This requires cross-episode cluster tracking - simplified for now)
        total_season_sec = cluster.get('total_sec', 0)

        if total_season_sec >= min_season_sec:
            unknowns.append({
                'cluster_id': cluster['cluster_id'],
                'episode_sec': cluster.get('total_sec', 0),
                'season_sec': total_season_sec,
                'rep_samples': cluster.get('rep_samples', []),
                'nearest_identities': find_nearest_identities(cluster, show_id, season_id)
            })

    return unknowns
```

**Acceptance**:
- Unknown panel shows clusters ≥6s this season
- Can promote unknown to cast (adds to facebank, updates bank)
- Can mark as guest (excluded from future)

---

## Phase 7: Re-Harvest & Analyze Pipeline Integration (2 hours)

### Add "Re-Harvest" Button

**Location**: Upload page or Cast Images page

**UI**:

```
5. Re-Harvest & Analyze (Show/Season)

   Scope: [RHOBH S05 - E01-Pilot ▼]

   This will:
   • Re-harvest at 10fps baseline + entrance densify
   • Cluster with purity-driven eps
   • Assign using season bank (open-set if low margin)
   • Generate analytics (timeline, totals, delta_table)

   [Re-Harvest & Analyze]

   Status: [Progress bar]
```

**Pipeline Changes**:

```python
# jobs/tasks/harvest.py - Update to use Show/Season paths

def harvest_task(job_id: str, episode_id: str, show_id: str, season_id: str):
    """Harvest with Show/Season context."""

    # Load season bank if available
    season_bank_path = Path(f"data/facebank/{show_id}/{season_id}/multi_prototypes.json")
    season_bank = None
    if season_bank_path.exists():
        with open(season_bank_path) as f:
            season_bank = json.load(f)
        logger.info(f"[{job_id}] Loaded season bank for {show_id}/{season_id}")

    # ... existing harvest logic ...

    # Pass season_bank to clustering
    return {'season_bank': season_bank, ...}

# jobs/tasks/cluster.py - Update for season-aware clustering

def cluster_task(job_id: str, episode_id: str, season_bank: Optional[dict] = None):
    """Cluster with optional season bank."""

    # ... face-only filtering + top-K ...

    # Purity-driven DBSCAN
    clusters = purity_driven_cluster(picked_samples, config)

    if season_bank:
        # Open-set assignment
        assignments = assign_clusters_open_set(
            clusters,
            season_bank,
            min_sim=0.60,
            min_margin=0.08
        )

        # TODO: Prototype-anchored split for ambiguous clusters (future)
    else:
        # Legacy: all Unknown
        assignments = {c['cluster_id']: 'Unknown' for c in clusters}

    # Save with assignments
    save_clusters_with_names(clusters, assignments, episode_id)
```

**Acceptance**:
- Re-harvest button queues full pipeline
- Season bank loaded and used for assignment
- Unknowns remain Unknown (not forced)

---

## Phase 8: Config Updates + Global Settings (1 hour)

### File: `configs/pipeline.yaml` - Add new sections

```yaml
# ... existing config ...

upload:
  allow_formats: [".jpg", ".jpeg", ".png", ".avif", ".webp", ".heic"]
  normalize:
    colorspace: "RGB"
    bitdepth: 8
    strip_exif: true
    save_as: "png"      # or "webp"
  min_face_px_accept: 64
  multiface_policy: "prompt"   # "largest" as fallback

season_bank:
  enable: true
  pose_bins: ["frontal", "three_quarter", "profile"]
  scale_bins: ["small", "large"]         # small < 100px
  max_prototypes_per_bin: 5

open_set:
  min_sim: 0.60          # Minimum similarity to assign
  min_margin: 0.08       # Minimum margin over next-best

unknowns_panel:
  min_season_seconds: 6.0  # Threshold for promoting unknowns
```

**Acceptance**:
- Config values control upload, season bank, open-set
- Can adjust thresholds without code changes

---

## Phase 9: End-to-End Testing (1 hour)

### Test Workflow:

1. **Upload**:
   - Create RHOBH → Season 5
   - Upload RHOBH-TEST-10-28.mp4

2. **Cast Images**:
   - Add KIM with 10 seeds (mix of jpg/png/heic)
   - Add KYLE with 10 seeds
   - Add LISA, BRANDI, RINNA, EILEEN, YOLANDA (7 total)
   - Verify face validation rejects bad images

3. **Re-Harvest**:
   - Click "Re-Harvest & Analyze"
   - Wait for pipeline to complete

4. **Review**:
   - Check clusters - Kim ≠ Kyle
   - Check Unknown panel - new faces surfaced
   - Promote an unknown to cast

5. **Analytics**:
   - Verify totals ≤ runtime
   - Verify overlaps = 0
   - Check delta_table.csv

**Acceptance Criteria**:
- ✅ Kim and Kyle in separate clusters
- ✅ No Kim faces in Kyle cluster (and vice versa)
- ✅ Unknown faces shown with ≥6s threshold
- ✅ Can promote unknown → updates season bank
- ✅ Future episodes use updated bank
- ✅ Totals ≤ runtime, overlaps = 0

---

## Summary

**Total Time: 12-15 hours**

| Phase | Description | Time |
|-------|-------------|------|
| 1 | Multi-format image processing | 1.5h |
| 2 | Show/Season data model | 1.5h |
| 3 | Upload page UI | 2h |
| 4 | Cast Images page + validation | 3h |
| 5 | Season bank system | 2h |
| 6 | Unknown discovery panel | 2h |
| 7 | Re-harvest pipeline integration | 2h |
| 8 | Config updates | 1h |
| 9 | End-to-end testing | 1h |

**Key Benefits**:
- **Identity-agnostic**: No per-person tuning, all config global
- **Season-aware**: Future episodes leverage learned identities
- **Multi-format**: No manual conversion, accepts any format
- **Unknown discovery**: Surface new faces automatically
- **Clean pipeline**: Re-harvest from scratch with confidence

**Next Step**: Would you like me to proceed with implementation, starting with Phase 1? Or would you like to review/modify the plan first?
