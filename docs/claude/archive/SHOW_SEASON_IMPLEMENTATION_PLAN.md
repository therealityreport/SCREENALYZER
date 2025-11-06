# Show/Season + Cast Images Implementation Plan

**Date**: October 30, 2025
**Estimated Time**: 8-10 hours
**Status**: Complete roadmap, ready to implement

---

## 🎯 Overview

Build UI workflow to:
1. Organize episodes by Show/Season
2. Upload cast face seeds per season
3. Use seeds for prototype-anchored clustering
4. Re-harvest with clean, identity-agnostic pipeline

**Key Benefit**: Solves Kim/Kyle separation by providing labeled seed faces through UI instead of manual file creation.

---

## 📊 Current Session Summary (5 hours invested)

**✅ Infrastructure Complete**:
- Face-only filtering (77.6% retention)
- Auto-caps (identity-agnostic merge limits)
- Purity-driven clustering (quality-based eps)
- Contamination audit framework
- CI guard (prevents per-person hardcoding)

**⏳ Missing Component**: Facebank seed data → This implementation provides it!

---

## 🏗️ Implementation Phases

### **Phase 1: Database Schema** (1 hour)

**New Tables**:

```python
# api/models.py

class Show(Base):
    """TV show (e.g., Real Housewives of Beverly Hills)"""
    __tablename__ = "shows"

    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)  # "RHOBH"
    display_name = Column(String, nullable=False)  # "Real Housewives of Beverly Hills"
    created_at = Column(DateTime, default=datetime.utcnow)

    seasons = relationship("Season", back_populates="show")


class Season(Base):
    """Season within a show"""
    __tablename__ = "seasons"

    id = Column(Integer, primary_key=True)
    show_id = Column(Integer, ForeignKey("shows.id"), nullable=False)
    season_number = Column(Integer, nullable=False)  # 5
    label = Column(String, nullable=False)  # "Season 5" or "S05"
    created_at = Column(DateTime, default=datetime.utcnow)

    show = relationship("Show", back_populates="seasons")
    episodes = relationship("Episode", back_populates="season")
    cast_members = relationship("CastMember", back_populates="season")

    __table_args__ = (
        UniqueConstraint('show_id', 'season_number', name='unique_show_season'),
    )


class CastMember(Base):
    """Cast member for a season (seed facebank)"""
    __tablename__ = "cast_members"

    id = Column(Integer, primary_key=True)
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=False)
    name = Column(String, nullable=False)  # "KIM"
    seed_count = Column(Integer, default=0)  # Number of seed images
    created_at = Column(DateTime, default=datetime.utcnow)

    season = relationship("Season", back_populates="cast_members")

    __table_args__ = (
        UniqueConstraint('season_id', 'name', name='unique_season_cast'),
    )


# Update Episode model
class Episode(Base):
    # ... existing fields ...
    season_id = Column(Integer, ForeignKey("seasons.id"), nullable=True)  # Nullable for backward compat
    season = relationship("Season", back_populates="episodes")
```

**Migration**:
```python
# migrations/add_show_season.py

def upgrade():
    # Create shows table
    op.create_table('shows', ...)

    # Create seasons table
    op.create_table('seasons', ...)

    # Create cast_members table
    op.create_table('cast_members', ...)

    # Add season_id to episodes (nullable for backward compatibility)
    op.add_column('episodes', Column('season_id', Integer, ForeignKey('seasons.id'), nullable=True))

    # Migrate existing episodes to "Legacy" show/season if desired
    # Or leave season_id=null for backward compatibility
```

**Time**: 1 hour (schema + migration)

---

### **Phase 2: Upload Page - Show/Season Flow** (2 hours)

**File**: `app/pages/1_Upload.py` (modify existing)

**UI Changes**:

```python
import streamlit as st
from api.database import SessionLocal
from api.models import Show, Season, Episode

def upload_page():
    st.title("📤 Upload Video")

    db = SessionLocal()

    # ========================================
    # STEP 1: Select or Create Show
    # ========================================
    st.subheader("1. Select Show")

    shows = db.query(Show).all()
    show_options = {show.display_name: show.id for show in shows}
    show_options["+ Create New Show"] = None

    selected_show_name = st.selectbox(
        "Show",
        options=list(show_options.keys()),
        key="show_select"
    )

    # Create new show inline
    if selected_show_name == "+ Create New Show":
        with st.form("new_show_form"):
            show_name = st.text_input("Show Name (short)", placeholder="RHOBH")
            show_display = st.text_input("Display Name", placeholder="Real Housewives of Beverly Hills")

            if st.form_submit_button("Create Show"):
                new_show = Show(name=show_name, display_name=show_display)
                db.add(new_show)
                db.commit()
                st.success(f"✅ Created show: {show_display}")
                st.toast("Created new show! You can now add seasons.")
                st.rerun()
    else:
        selected_show_id = show_options[selected_show_name]

    # ========================================
    # STEP 2: Select or Create Season
    # ========================================
    if selected_show_name != "+ Create New Show":
        st.subheader("2. Select Season")

        seasons = db.query(Season).filter(Season.show_id == selected_show_id).all()
        season_options = {f"Season {s.season_number}": s.id for s in seasons}
        season_options["+ Create New Season"] = None

        selected_season_label = st.selectbox(
            "Season",
            options=list(season_options.keys()),
            key="season_select"
        )

        # Create new season inline
        if selected_season_label == "+ Create New Season":
            with st.form("new_season_form"):
                season_number = st.number_input("Season Number", min_value=1, value=1)
                season_label = st.text_input("Label", value=f"Season {season_number}")

                if st.form_submit_button("Create Season"):
                    new_season = Season(
                        show_id=selected_show_id,
                        season_number=season_number,
                        label=season_label
                    )
                    db.add(new_season)
                    db.commit()
                    st.success(f"✅ Created {season_label}")
                    st.toast(f"Created {season_label}! You can now seed cast faces on the Cast Images page.")
                    st.rerun()
        else:
            selected_season_id = season_options[selected_season_label]

    # ========================================
    # STEP 3: Upload Video (existing logic)
    # ========================================
    if selected_show_name != "+ Create New Show" and selected_season_label != "+ Create New Season":
        st.subheader("3. Upload Video")

        uploaded_file = st.file_uploader("Choose video file", type=['mp4', 'mov', 'avi'])

        if uploaded_file:
            episode_name = st.text_input("Episode Name", placeholder="E01-PilotParty")

            if st.button("Upload & Harvest"):
                # Save video to show/season structure
                show = db.query(Show).filter(Show.id == selected_show_id).first()
                season = db.query(Season).filter(Season.id == selected_season_id).first()

                video_dir = Path(f"data/videos/{show.name}/S{season.season_number:02d}")
                video_dir.mkdir(parents=True, exist_ok=True)

                video_path = video_dir / f"{episode_name}.mp4"
                with open(video_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Create episode in DB
                episode = Episode(
                    name=episode_name,
                    season_id=selected_season_id,
                    video_path=str(video_path),
                    status="uploaded"
                )
                db.add(episode)
                db.commit()

                st.success(f"✅ Uploaded {episode_name} to {show.display_name} {season.label}")

                # Queue harvest job
                from api.jobs import job_manager
                job_id = job_manager.create_job("harvest", episode_id=episode_name)

                st.info(f"🔄 Harvest job queued: {job_id}")

    db.close()
```

**File Path Structure**:
```
data/
  videos/RHOBH/S05/E01-PilotParty.mp4
  facebank/RHOBH/S05/KIM/seed_001.jpg
  harvest/RHOBH/S05/E01-PilotParty/tracks.json
  outputs/RHOBH/S05/E01-PilotParty/timeline.csv
```

**Time**: 2 hours (UI + file path refactoring)

---

### **Phase 3: Cast Images Page** (3 hours)

**File**: `app/pages/5_Cast_Images.py` (new page)

**UI Flow**:

```python
import streamlit as st
from pathlib import Path
import numpy as np
from api.database import SessionLocal
from api.models import Show, Season, CastMember
from screentime.face_detection import detect_faces, align_face, embed_face

def cast_images_page():
    st.title("🎭 Cast Images - Seed Facebank")

    db = SessionLocal()

    # ========================================
    # STEP 1: Select Show/Season
    # ========================================
    st.subheader("1. Select Show & Season")

    shows = db.query(Show).all()
    show_options = {show.display_name: show.id for show in shows}

    if not shows:
        st.warning("No shows found. Create a show on the Upload page first.")
        return

    selected_show_name = st.selectbox("Show", options=list(show_options.keys()))
    selected_show_id = show_options[selected_show_name]

    seasons = db.query(Season).filter(Season.show_id == selected_show_id).all()
    season_options = {f"Season {s.season_number}": s.id for s in seasons}

    if not seasons:
        st.warning("No seasons found. Create a season on the Upload page first.")
        return

    selected_season_label = st.selectbox("Season", options=list(season_options.keys()))
    selected_season_id = season_options[selected_season_label]

    # ========================================
    # STEP 2: Show Existing Cast
    # ========================================
    st.subheader("2. Current Cast")

    cast_members = db.query(CastMember).filter(CastMember.season_id == selected_season_id).all()

    if cast_members:
        cols = st.columns(len(cast_members))
        for i, cast in enumerate(cast_members):
            with cols[i]:
                st.metric(
                    cast.name,
                    f"{cast.seed_count} seeds",
                    delta="✅" if cast.seed_count >= 5 else "⚠️ needs more"
                )
    else:
        st.info("No cast members added yet.")

    # ========================================
    # STEP 3: Add Cast Member + Upload Seeds
    # ========================================
    st.subheader("3. Add Cast Member")

    with st.form("add_cast_form"):
        cast_name = st.text_input(
            "Cast Name",
            placeholder="KIM",
            help="Use all caps for consistency (e.g., KIM, KYLE, RINNA)"
        )

        uploaded_files = st.file_uploader(
            "Upload Face Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            help="Upload 8-12 clear face photos. Mix of frontal/3-4 view. Include 2-3 small faces."
        )

        if st.form_submit_button("Add Cast & Upload Seeds"):
            if not cast_name:
                st.error("Please enter a cast name")
            elif not uploaded_files:
                st.error("Please upload at least one image")
            else:
                # Process uploads
                show = db.query(Show).filter(Show.id == selected_show_id).first()
                season = db.query(Season).filter(Season.id == selected_season_id).first()

                facebank_dir = Path(f"data/facebank/{show.name}/S{season.season_number:02d}/{cast_name}")
                facebank_dir.mkdir(parents=True, exist_ok=True)

                valid_seeds = []
                rejected = []

                progress = st.progress(0)
                status = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status.text(f"Processing {uploaded_file.name}...")

                    # Read image
                    image_bytes = uploaded_file.read()
                    image = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

                    # Detect face
                    faces = detect_faces(image)

                    if len(faces) == 0:
                        rejected.append((uploaded_file.name, "No face detected"))
                        continue

                    if len(faces) > 1:
                        rejected.append((uploaded_file.name, "Multiple faces detected"))
                        continue

                    face = faces[0]

                    # Check quality
                    if face['confidence'] < 0.65:
                        rejected.append((uploaded_file.name, f"Low confidence: {face['confidence']:.2f}"))
                        continue

                    if face['face_size'] < 48:
                        rejected.append((uploaded_file.name, f"Face too small: {face['face_size']}px"))
                        continue

                    # Align and embed
                    aligned = align_face(image, face)
                    embedding = embed_face(aligned)

                    # Save
                    seed_path = facebank_dir / f"seed_{i+1:03d}.jpg"
                    cv2.imwrite(str(seed_path), aligned)

                    valid_seeds.append({
                        'path': str(seed_path),
                        'embedding': embedding.tolist(),
                        'confidence': face['confidence'],
                        'face_size': face['face_size']
                    })

                    progress.progress((i + 1) / len(uploaded_files))

                # Save metadata
                metadata_path = facebank_dir / "seeds.json"
                with open(metadata_path, 'w') as f:
                    json.dump({'seeds': valid_seeds}, f, indent=2)

                # Create or update cast member
                cast_member = db.query(CastMember).filter(
                    CastMember.season_id == selected_season_id,
                    CastMember.name == cast_name
                ).first()

                if cast_member:
                    cast_member.seed_count = len(valid_seeds)
                else:
                    cast_member = CastMember(
                        season_id=selected_season_id,
                        name=cast_name,
                        seed_count=len(valid_seeds)
                    )
                    db.add(cast_member)

                db.commit()

                # Report results
                st.success(f"✅ Added {len(valid_seeds)} seeds for {cast_name}")

                if rejected:
                    with st.expander(f"⚠️ Rejected {len(rejected)} images"):
                        for filename, reason in rejected:
                            st.text(f"• {filename}: {reason}")

                st.balloons()
                st.rerun()

    # ========================================
    # STEP 4: Facebank Gallery View
    # ========================================
    if cast_members:
        st.subheader("4. Facebank Gallery")

        for cast in cast_members:
            with st.expander(f"{cast.name} ({cast.seed_count} seeds)"):
                facebank_dir = Path(f"data/facebank/{show.name}/S{season.season_number:02d}/{cast.name}")

                if facebank_dir.exists():
                    seed_files = sorted(facebank_dir.glob("seed_*.jpg"))
                    cols = st.columns(min(len(seed_files), 6))

                    for i, seed_file in enumerate(seed_files[:12]):  # Show first 12
                        with cols[i % 6]:
                            st.image(str(seed_file), width=100)

    db.close()
```

**Quality Checks** (Identity-Agnostic):
- Confidence ≥ 0.65
- Face size ≥ 48px
- Single face only
- No blur (optional sharpness check)

**Time**: 3 hours (UI + face detection + quality checks)

---

### **Phase 4: Multi-Prototype Bank Builder** (1.5 hours)

**File**: `screentime/clustering/prototype_bank.py` (create)

**Integration with Cast Images**:

```python
def build_multi_prototype_bank_from_seeds(episode_id, show_name, season_number):
    """
    Build multi-prototype bank from uploaded seed images.

    Reads seeds from data/facebank/{show}/{season}/{cast}/seeds.json
    and creates pose × scale prototypes.
    """

    facebank_dir = Path(f"data/facebank/{show_name}/S{season_number:02d}")

    if not facebank_dir.exists():
        raise ValueError(f"No facebank found for {show_name} S{season_number:02d}")

    prototypes = {}

    for cast_dir in facebank_dir.iterdir():
        if not cast_dir.is_dir():
            continue

        cast_name = cast_dir.name
        seeds_file = cast_dir / "seeds.json"

        if not seeds_file.exists():
            logger.warning(f"No seeds.json for {cast_name}, skipping")
            continue

        with open(seeds_file) as f:
            seeds_data = json.load(f)

        seeds = seeds_data['seeds']

        # Convert to embeddings
        embeddings = [np.array(s['embedding']) for s in seeds]

        # Simple prototype: use all seeds (no pose/scale clustering for now)
        # Future: cluster by yaw angle and face size
        prototypes[cast_name] = {
            'all': embeddings  # Simplified: just use all seeds
        }

    return prototypes
```

**Simplified Approach** (for v1):
- Don't cluster by pose/scale initially
- Just use all uploaded seeds as prototypes
- Future enhancement: Add pose/scale clustering

**Time**: 1.5 hours (simplified version)

---

### **Phase 5: Re-Harvest Button + Integration** (1.5 hours)

**File**: `app/pages/1_Upload.py` or `app/pages/5_Cast_Images.py`

**UI Addition**:

```python
# On Cast Images page, after facebank gallery

st.subheader("5. Re-Harvest & Analyze")

episodes = db.query(Episode).filter(Episode.season_id == selected_season_id).all()

if episodes:
    episode_options = {ep.name: ep.id for ep in episodes}
    selected_episode = st.selectbox("Select Episode", options=list(episode_options.keys()))

    if st.button("🔄 Re-Harvest & Analyze (with Facebank)"):
        # Queue full pipeline job
        job_id = queue_full_pipeline(
            episode_name=selected_episode,
            show_name=show.name,
            season_number=season.season_number
        )

        st.success(f"✅ Queued full pipeline: {job_id}")
        st.info("Pipeline: Harvest → Cluster (purity-driven) → Prototype-anchored split → Analytics")
```

**Pipeline Integration** (update `jobs/tasks/cluster.py`):

```python
def cluster_task(job_id, episode_id):
    # ... existing clustering ...

    # NEW: Check if facebank exists for this episode's season
    episode = get_episode(episode_id)
    if episode.season:
        show_name = episode.season.show.name
        season_number = episode.season.season_number

        facebank_dir = Path(f"data/facebank/{show_name}/S{season_number:02d}")

        if facebank_dir.exists():
            logger.info(f"[{job_id}] Facebank found, using prototype-anchored split")

            # Build prototype bank
            prototype_bank = build_multi_prototype_bank_from_seeds(
                episode_id, show_name, season_number
            )

            # Detect confusion
            confusion_results = detect_confusion(
                clusters_data, picked_samples, prototype_bank, confusion_config
            )

            # Anchored split for ambiguous clusters
            for ambiguous in confusion_results['ambiguous_clusters']:
                # ... anchored split logic ...
                pass

            # Open-set assignment
            assignments = assign_clusters_to_identities(
                updated_clusters, prototype_bank, openset_config
            )
        else:
            logger.info(f"[{job_id}] No facebank found, using Unknown labels")
            # Fall back to Unknown for all clusters
    else:
        logger.info(f"[{job_id}] No season context, legacy mode")
```

**Time**: 1.5 hours (button + integration)

---

## ⏱️ Total Time Estimate

| Phase | Description | Time |
|-------|-------------|------|
| 1 | Database schema + migration | 1 hour |
| 2 | Upload page - Show/Season flow | 2 hours |
| 3 | Cast Images page | 3 hours |
| 4 | Multi-prototype bank builder | 1.5 hours |
| 5 | Re-Harvest button + integration | 1.5 hours |
| **Total** | | **9 hours** |

---

## 🎯 Expected Workflow

### User Flow:

1. **Upload** → Create "RHOBH" → "Season 5" → Upload "E01-PilotParty.mp4"
2. **Cast Images** → Select "RHOBH" → "Season 5" → Add "KIM" + upload 10 face photos → Add "KYLE" + upload 10 photos → ... (all 7 cast)
3. **Cast Images** → Click "Re-Harvest & Analyze" → Wait ~5-10 min
4. **Clusters** → See clean galleries: Kim ≠ Kyle, LVP not scattered
5. **Analytics** → See delta_table.csv with accurate results

### Pipeline Flow:

```
Upload Video
    ↓
Seed Facebank (Cast Images page)
    ↓
Re-Harvest & Analyze button
    ↓
Harvest (10fps baseline, 30fps recall)
    ↓
Face-only filtering + Top-K
    ↓
Purity-driven clustering
    ↓
Build prototype bank from seeds
    ↓
Detect confusion (Kim/Kyle ambiguous)
    ↓
Prototype-anchored split
    ↓
Open-set assignment
    ↓
Analytics (timeline, totals, delta_table)
```

---

## 📊 Validation Criteria

**UI**:
- ✅ Can create RHOBH → Season 5
- ✅ Can upload cast faces with quality checks
- ✅ Rejected images show clear reasons
- ✅ Facebank gallery shows thumbnails
- ✅ Badge shows "ready" when ≥5 seeds per cast

**Pipeline**:
- ✅ Kim and Kyle separated into different clusters
- ✅ LVP not scattered
- ✅ Galleries show single person per cluster
- ✅ Analytics header shows "Show: RHOBH, Season: 5"
- ✅ Totals ≤ runtime, overlaps = 0

**Diagnostics**:
- ✅ `confusion_matrix.json` shows Kim/Kyle ambiguity
- ✅ `anchored_split_audit.json` shows split decision
- ✅ `id_assign_audit.jsonl` shows assignments
- ✅ `cluster_threshold.json` shows purity-driven eps

---

## 🚀 Next Steps

**Option A: Full Implementation** (9 hours):
I'll implement all 5 phases sequentially, testing each before moving to next.

**Option B: Phased Rollout** (3 phases × 3 hours):
- Phase 1: Database + Upload page (3 hours) → Test Show/Season creation
- Phase 2: Cast Images page (3 hours) → Test seed upload
- Phase 3: Integration + Re-Harvest (3 hours) → Test full pipeline

**What's your preference?**

---

**This approach perfectly complements the 5 hours of infrastructure work we've already done. The facebank UI provides the missing seed data, and all the clustering/splitting logic is ready to use it.**
