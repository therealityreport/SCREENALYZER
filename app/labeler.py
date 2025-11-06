"""
Screenalyzer Admin UI - Streamlit Application

Reviewer-first interface for episode processing, review, and analytics.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from app.pairwise_review_redesign import render_pairwise_review_v2
from app.all_faces_redesign import render_all_faces_grid_v2
from app.cluster_split import render_cluster_split
from api.jobs import job_manager
from api.uploads import upload_manager
from app.lib.cluster_mutations import ClusterMutator
from app.lib.mutator_api import configure_workspace_mutator
from app.workspace.faces import render_faces_tab
from app.workspace.clusters import render_clusters_tab
from app.workspace.tracks import render_tracks_tab
from app.workspace.review import render_review_tab
from app.lib.data import (
    get_episode_summary,
    load_assets_thumbnail,
    load_clusters,
    load_lowconf_queue,
    load_merge_suggestions,
    load_tracks,
)
from app.lib.review_state import ReviewStateManager
from screentime.io_utils import validate_cast_image, validate_video
from screentime.types import UploadStatus
from screentime.utils import get_video_path, normalize_episode_id
from screentime.viz.thumbnails import ThumbnailGenerator

load_dotenv()


def wkey(*parts) -> str:
    """
    Generate unique widget key from parts.

    Ensures widget keys are unique across all UI contexts by combining
    all identifying information (episode, cluster, track, frame, action, etc.)

    Args:
        *parts: Variable parts to combine (episode_id, cluster_id, track_id, etc.)

    Returns:
        Unique widget key string
    """
    return "w_" + "_".join(str(p) for p in parts)


# Configuration
st.set_page_config(
    page_title="Screenalyzer Admin",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))

# Load pipeline config for UI preferences
def load_ui_config():
    """Load UI preferences from pipeline config."""
    config_path = Path("configs/pipeline.yaml")
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
            return config.get("ui", {})
    return {}

UI_CONFIG = load_ui_config()
PREFER_ASSETS_THUMBNAILS = UI_CONFIG.get("prefer_assets_thumbnails", True)


def get_cluster_thumbnail(cluster, cluster_assignments, video_path, tracks_data, episode_id, thumb_gen):
    """
    Get thumbnail for cluster using fallback chain:
    1. Assets (if prefer_assets_thumbnails=true and cluster has assigned name)
    2. Generated frame crops from video
    3. Placeholder (initials/emoji)

    Returns: (thumbnail_paths, source) where source is "assets", "generated", or "placeholder"
    """
    cluster_id = cluster["cluster_id"]
    person_name = cluster_assignments.get(cluster_id)

    # Try assets first if enabled and name assigned
    if PREFER_ASSETS_THUMBNAILS and person_name:
        assets_path = load_assets_thumbnail(cluster_id=cluster_id, person_name=person_name)
        if assets_path:
            return ([assets_path], "assets")

    # Try generated thumbnails if video available
    if video_path.exists() and tracks_data:
        try:
            thumbnails = thumb_gen.generate_cluster_thumbnail(
                video_path, cluster, tracks_data, episode_id, max_samples=3
            )
            if thumbnails and any(t.exists() for t in thumbnails):
                return (thumbnails, "generated")
        except Exception:
            pass  # Fall through to placeholder

    # Fallback: placeholder
    return ([], "placeholder")


def init_session_state():
    """Initialize session state variables."""
    if "upload_session_id" not in st.session_state:
        st.session_state.upload_session_id = None
    if "cast_images" not in st.session_state:
        st.session_state.cast_images = {}
    if "selected_episode" not in st.session_state:
        st.session_state.selected_episode = None
    if "review_mode" not in st.session_state:
        st.session_state.review_mode = "all_faces"  # all_faces, pairwise, lowconf
    if "current_suggestion_idx" not in st.session_state:
        st.session_state.current_suggestion_idx = 0
    if "review_state_manager" not in st.session_state:
        st.session_state.review_state_manager = None
    if "cluster_mutator" not in st.session_state:
        st.session_state.cluster_mutator = None
    if "thumbnail_generator" not in st.session_state:
        thumbnail_cache_dir = DATA_ROOT / "cache" / "thumbnails"
        st.session_state.thumbnail_generator = ThumbnailGenerator(thumbnail_cache_dir)
    if "cluster_filter" not in st.session_state:
        st.session_state.cluster_filter = "all"  # all, lowconf, high_quality
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
    if "viewing_cluster_id" not in st.session_state:
        st.session_state.viewing_cluster_id = None
    if "viewing_lowconf_cluster_id" not in st.session_state:
        st.session_state.viewing_lowconf_cluster_id = None
    if "workspace_tab" not in st.session_state:
        st.session_state.workspace_tab = "Faces"
    if "workspace_selected_person" not in st.session_state:
        st.session_state.workspace_selected_person = None
    if "workspace_selected_cluster" not in st.session_state:
        st.session_state.workspace_selected_cluster = None
    if "workspace_episode" not in st.session_state:
        st.session_state.workspace_episode = None


def render_sidebar():
    """Render sidebar navigation."""
    st.sidebar.title("Screanalyzer")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "📤 Upload",
            "🎭 Cast Images",
            "🗂️ Workspace",
            "📊 Analytics",
            "⚙️ Settings",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data root: {DATA_ROOT}")

    return page


def resolve_show_season(registry, episode_id: str) -> tuple[str, str]:
    """Best-effort mapping from episode_id to show/season identifiers."""
    try:
        for show in getattr(registry, "shows", []):
            for season in getattr(show, "seasons", []):
                for episode in getattr(season, "episodes", []):
                    if episode.episode_id == episode_id:
                        return show.show_id, season.season_id
    except Exception:
        pass

    # Fall back to first show/season in registry if available
    shows = getattr(registry, "shows", [])
    if shows:
        show = shows[0]
        seasons = getattr(show, "seasons", [])
        if seasons:
            return show.show_id, seasons[0].season_id
        return show.show_id, "s01"

    # Final fallback to defaults used historically
    return "rhobh", "s05"


def render_upload_page():
    """Render video upload page with chunked upload support."""
    from app.lib.metadata import MetadataManager

    st.title("📤 Upload Episode")

    st.markdown(
        """
        Upload a video file (MP4 H.264/H.265) for processing.

        **Requirements:**
        - Format: MP4 (H.264 or H.265 codec)
        - Max duration: 90 minutes
        - Max file size: 5 GB
        """
    )

    # Initialize metadata manager
    metadata_manager = MetadataManager(DATA_ROOT)

    # Step 1: Show/Season Selection
    st.subheader("1. Select Show and Season")

    col1, col2 = st.columns(2)

    with col1:
        # Show selection
        shows = metadata_manager.list_shows()
        show_options = ["[Create New Show]"] + [show.name for show in shows]

        selected_show_option = st.selectbox(
            "Show",
            options=show_options,
            help="Select existing show or create new one"
        )

        # Create new show
        if selected_show_option == "[Create New Show]":
            st.markdown("#### Create New Show")
            new_show_name = st.text_input(
                "Show Name (Short)",
                placeholder="e.g., RHOBH",
                help="Short identifier (uppercase recommended)"
            )
            new_show_display = st.text_input(
                "Display Name",
                placeholder="e.g., Real Housewives of Beverly Hills",
                help="Full show title"
            )

            if st.button("Create Show"):
                if new_show_name and new_show_display:
                    try:
                        metadata_manager.create_show(new_show_name, new_show_display)
                        st.success(f"✅ Created show: {new_show_display}")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                else:
                    st.warning("⚠️ Please enter both show name and display name")

    with col2:
        # Season selection (only if show selected)
        if selected_show_option != "[Create New Show]" and selected_show_option:
            selected_show_name = selected_show_option
            seasons = metadata_manager.list_seasons(selected_show_name)
            season_options = ["[Create New Season]"] + [f"Season {s.season_number}" for s in seasons]

            selected_season_option = st.selectbox(
                "Season",
                options=season_options,
                help="Select existing season or create new one"
            )

            # Create new season
            if selected_season_option == "[Create New Season]":
                st.markdown("#### Create New Season")
                new_season_number = st.number_input(
                    "Season Number",
                    min_value=1,
                    max_value=50,
                    value=1,
                    help="Season number (e.g., 5 for Season 5)"
                )
                new_season_label = st.text_input(
                    "Label (optional)",
                    placeholder=f"S{new_season_number:02d}",
                    help="Optional custom label (defaults to S##)"
                )

                if st.button("Create Season"):
                    try:
                        label = new_season_label if new_season_label else None
                        metadata_manager.create_season(selected_show_name, new_season_number, label)
                        st.success(f"✅ Created Season {new_season_number}")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")

    # Only show video upload if Show and Season are selected
    if selected_show_option == "[Create New Show]":
        st.info("👆 Please create a show first before uploading videos")
        return

    if selected_show_option and "selected_season_option" in locals():
        if selected_season_option == "[Create New Season]":
            st.info("👆 Please create a season first before uploading videos")
            return

        # Extract season number
        selected_season_number = None
        for season in seasons:
            if f"Season {season.season_number}" == selected_season_option:
                selected_season_number = season.season_number
                break

        # Video Upload Section
        st.markdown("---")
        st.subheader("2. Select Video File")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4"],
            help="Select an MP4 video file to upload",
        )

        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

            # Validation
            if st.button("Validate Video", type="primary"):
                with st.spinner("Validating video..."):
                    # Save temp file for validation
                    temp_path = DATA_ROOT / "videos" / f".temp_{uploaded_file.name}"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Validate
                    result = validate_video(temp_path)

                    if result.is_valid:
                        st.success("✅ Video validation passed!")

                        if result.metadata:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{result.metadata.duration_sec / 60:.1f} min")
                            with col2:
                                st.metric(
                                    "Resolution", f"{result.metadata.width}x{result.metadata.height}"
                                )
                            with col3:
                                st.metric("FPS", f"{result.metadata.fps:.1f}")

                        if result.warnings:
                            for warning in result.warnings:
                                st.warning(f"⚠️ {warning}")

                        # Upload button
                        if st.button("Start Upload", type="primary"):
                            with st.spinner("Uploading..."):
                                try:
                                    # Create upload session
                                    session = upload_manager.create_upload_session(
                                        filename=uploaded_file.name,
                                        total_size_bytes=len(uploaded_file.getvalue()),
                                    )

                                    # Upload in chunks
                                    chunk_size = session.chunk_size_bytes
                                    file_data = uploaded_file.getvalue()
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()

                                    for chunk_id in range(session.total_chunks):
                                        start = chunk_id * chunk_size
                                        end = min(start + chunk_size, len(file_data))
                                        chunk_data = file_data[start:end]

                                        upload_manager.upload_chunk(
                                            session.session_id, chunk_id, chunk_data
                                        )

                                        progress = (chunk_id + 1) / session.total_chunks
                                        progress_bar.progress(progress)
                                        status_text.text(
                                            f"Uploading chunk {chunk_id + 1}/{session.total_chunks}..."
                                        )

                                    progress_bar.progress(1.0)
                                    status_text.text("Upload complete!")
                                    st.success(
                                        f"✅ Upload complete! Session ID: `{session.session_id}`"
                                    )

                                    # Clean up temp file
                                    temp_path.unlink(missing_ok=True)

                                except Exception as e:
                                    st.error(f"❌ Upload failed: {str(e)}")

                    else:
                        st.error("❌ Video validation failed!")
                        for error in result.errors:
                            st.error(f"• {error}")

                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)

    # Resume upload section
    st.markdown("---")
    st.subheader("2. Resume Upload")

    session_id_input = st.text_input(
        "Enter session ID to resume",
        help="Paste the session ID from a previous upload",
    )

    if session_id_input and st.button("Resume Upload"):
        try:
            resume_info = upload_manager.resume_upload(session_id_input)
            st.info(f"📊 Resume Information:")
            st.json(resume_info)

            if resume_info["status"] == UploadStatus.COMPLETED.value:
                st.success("✅ Upload already completed!")
            else:
                st.info(
                    f"Progress: {resume_info['progress_pct']:.1f}% "
                    f"({len(resume_info['uploaded_chunks'])}/{resume_info['total_chunks']} chunks)"
                )
                st.warning("⚠️ Resume functionality requires re-uploading the file")

        except ValueError as e:
            st.error(f"❌ {str(e)}")


def render_cast_images_page():
    """Render cast images upload page with Show/Season integration."""
    from screentime.models import get_registry
    from screentime.image_utils import ImageNormalizer
    from dataclasses import asdict
    import cv2
    import numpy as np
    import tempfile

    st.title("🎭 Cast Reference Images")

    st.markdown(
        """
        Upload reference images for cast members (3-5 images per cast recommended).

        **Supported Formats:**
        - JPEG, PNG, WebP, AVIF, HEIC (auto-converts to PNG)

        **Requirements:**
        - Clear face visible (min 64px, confidence ≥ 0.65)
        - Single face per image (or select if multiple)
        """
    )

    # Get registry
    registry = get_registry()

    # ========================================
    # 1. Select Show & Season
    # ========================================
    st.subheader("1. Select Show & Season")

    if not registry.shows:
        st.warning("No shows found. Please create a show first.")
        st.info("Run: `python init_rhobh_s05.py` to initialize RHOBH Season 5")
        return

    # Show selector
    show_options = {show.show_name: show.show_id for show in registry.shows}

    # Default to RHOBH if available
    default_show_name = next((name for name, sid in show_options.items() if sid == "rhobh"), list(show_options.keys())[0])

    selected_show_name = st.selectbox(
        "Show",
        options=list(show_options.keys()),
        index=list(show_options.keys()).index(default_show_name),
        key="cast_show_select"
    )
    selected_show_id = show_options[selected_show_name]
    selected_show = registry.get_show(selected_show_id)

    # Guard against None
    if selected_show is None:
        st.error(f"Show {selected_show_id} not found in registry.")
        return

    if not getattr(selected_show, 'seasons', None):
        st.warning(f"No seasons found for {selected_show_name}.")
        return

    # Season selector
    season_options = {s.season_label: s.season_id for s in selected_show.seasons}

    # Default to S05 if available
    default_season_label = next((label for label, sid in season_options.items() if sid == "s05"), list(season_options.keys())[0])

    selected_season_label = st.selectbox(
        "Season",
        options=list(season_options.keys()),
        index=list(season_options.keys()).index(default_season_label),
        key="cast_season_select"
    )
    selected_season_id = season_options[selected_season_label]
    selected_season = registry.get_season(selected_show_id, selected_season_id)

    # Guard against None
    if selected_season is None:
        st.error(f"Season {selected_season_id} not found in registry.")
        return

    st.info(f"📁 Facebank: `data/facebank/{selected_show_id}/{selected_season_id}/`")

    # ========================================
    # 2. Current Cast Members
    # ========================================
    st.markdown("---")
    st.subheader("2. Current Cast")

    if getattr(selected_season, 'cast', None):
        cols = st.columns(min(len(selected_season.cast), 5))
        for idx, cast in enumerate(selected_season.cast):
            with cols[idx % 5]:
                status = "✅" if cast.valid_seeds >= 3 else "⚠️" if cast.valid_seeds > 0 else "❌"
                st.metric(
                    cast.name,
                    f"{cast.valid_seeds} seeds",
                    delta=status
                )
    else:
        st.info("No cast members added yet.")

    # ========================================
    # 3. Add Cast Member + Upload Seeds
    # ========================================
    st.markdown("---")
    st.subheader("3. Add Cast Member")

    # Initialize upload counter in session state (used to reset uploader)
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0

    cast_name = st.text_input(
        "Cast Name",
        placeholder="KIM",
        help="Use UPPERCASE for consistency (e.g., KIM, KYLE, LISA)",
        key=f"cast_name_input_{st.session_state.upload_counter}"
    )

    uploaded_files = st.file_uploader(
        "Upload Face Images (3-5 recommended)",
        type=None,  # Accept all files, validate format after upload
        accept_multiple_files=True,
        help="Accepts: jpg, jpeg, png, webp, avif, heic. Auto-converts to PNG.",
        key=f"cast_images_uploader_{st.session_state.upload_counter}"
    )

    if st.button("Process & Add to Season", type="primary", disabled=not cast_name or not uploaded_files):
        if cast_name and uploaded_files:
            with st.spinner(f"Processing {len(uploaded_files)} images for {cast_name}..."):
                # Initialize normalizer and face detector
                normalizer = ImageNormalizer()
                from screentime.detectors.face_retina import RetinaFaceDetector
                detector = RetinaFaceDetector()

                # Prepare directories
                cast_dir = registry.get_cast_dir(selected_show_id, selected_season_id, cast_name)
                cast_dir.mkdir(parents=True, exist_ok=True)

                valid_seeds = []
                rejected = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")

                    try:
                        # Validate file extension
                        file_ext = Path(uploaded_file.name).suffix.lower()
                        supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.heic'}

                        if file_ext not in supported_formats:
                            rejected.append((uploaded_file.name, f"Unsupported format: {file_ext}. Use: jpg/png/webp/avif/heic"))
                            continue

                        # Save to temp file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = Path(tmp.name)

                        # Normalize image first
                        normalized_path = cast_dir / f"seed_tmp_{file_idx:03d}.png"
                        metadata = normalizer.normalize_image(tmp_path, normalized_path)

                        # Load normalized image for face detection
                        img = cv2.imread(str(normalized_path))

                        if img is None:
                            rejected.append((uploaded_file.name, "Cannot read image"))
                            normalized_path.unlink(missing_ok=True)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        # Detect faces
                        faces = detector.detect(img)

                        if len(faces) == 0:
                            rejected.append((uploaded_file.name, "No face detected"))
                            normalized_path.unlink(missing_ok=True)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        if len(faces) > 1:
                            rejected.append((uploaded_file.name, f"{len(faces)} faces detected - use single face images"))
                            normalized_path.unlink(missing_ok=True)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        face = faces[0]

                        # Quality checks
                        face_height = face['bbox'][3] - face['bbox'][1]

                        if face['confidence'] < 0.65:
                            rejected.append((uploaded_file.name, f"Low confidence ({face['confidence']:.2f})"))
                            normalized_path.unlink(missing_ok=True)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        if face_height < 64:
                            rejected.append((uploaded_file.name, f"Face too small ({face_height:.0f}px < 64px)"))
                            normalized_path.unlink(missing_ok=True)
                            tmp_path.unlink(missing_ok=True)
                            continue

                        # Warn if small but acceptable
                        if face_height < 72:
                            st.warning(f"⚠️ {uploaded_file.name}: Small face ({face_height:.0f}px), but acceptable")

                        # Valid seed - rename to final name
                        final_path = cast_dir / f"seed_{len(valid_seeds) + 1:03d}.png"
                        normalized_path.rename(final_path)

                        # Convert ImageMetadata dataclass to dict for JSON serialization
                        valid_seeds.append({
                            'path': str(final_path),
                            'original_name': uploaded_file.name,
                            'confidence': float(face['confidence']),  # Ensure native Python float
                            'face_height': float(face_height),
                            'metadata': asdict(metadata)  # Convert dataclass to dict
                        })

                        # Clean up temp
                        tmp_path.unlink(missing_ok=True)

                    except Exception as e:
                        rejected.append((uploaded_file.name, str(e)))
                        if 'normalized_path' in locals() and normalized_path.exists():
                            normalized_path.unlink(missing_ok=True)
                        if 'tmp_path' in locals() and tmp_path.exists():
                            tmp_path.unlink(missing_ok=True)

                    progress_bar.progress((file_idx + 1) / len(uploaded_files))

                progress_bar.empty()
                status_text.empty()

                # Save metadata atomically (tmp → move)
                if valid_seeds:
                    metadata_path = cast_dir / "seeds_metadata.json"
                    tmp_path = cast_dir / "seeds_metadata.json.tmp"

                    with open(tmp_path, 'w') as f:
                        json.dump({'seeds': valid_seeds}, f, indent=2)

                    # Atomic replace
                    tmp_path.replace(metadata_path)

                # Update registry
                registry.add_cast_member(
                    selected_show_id,
                    selected_season_id,
                    cast_name,
                    seed_count=len(uploaded_files),
                    valid_seeds=len(valid_seeds)
                )

                # Display results
                if valid_seeds:
                    st.success(f"✅ Added {len(valid_seeds)} valid seeds for {cast_name}")
                else:
                    st.error(f"❌ No valid seeds for {cast_name}")

                if rejected:
                    with st.expander(f"⚠️ Rejected {len(rejected)} images"):
                        for filename, reason in rejected:
                            st.text(f"• {filename}: {reason}")

                # Increment counter to reset uploader and clear inputs
                st.session_state.upload_counter += 1
                st.rerun()

    # ========================================
    # 4. Facebank Gallery View
    # ========================================
    if getattr(selected_season, 'cast', None):
        st.markdown("---")
        st.subheader("4. Season Facebank Gallery")

        for cast in selected_season.cast:
            with st.expander(f"{cast.name} ({cast.valid_seeds} seeds)", expanded=False):
                cast_dir = registry.get_cast_dir(selected_show_id, selected_season_id, cast.name)

                if cast_dir.exists():
                    seed_files = sorted(cast_dir.glob("seed_*.png"))

                    if seed_files:
                        # Show images with individual delete buttons
                        for i, seed_file in enumerate(seed_files[:12]):  # Show first 12
                            col1, col2 = st.columns([4, 1])
                            with col1:
                                st.image(str(seed_file), width=100)
                            with col2:
                                if st.button("🗑️", key=f"delete_img_{cast.name}_{seed_file.stem}_{i}"):
                                    seed_file.unlink()
                                    # Update count in registry
                                    remaining_seeds = len(list(cast_dir.glob("seed_*.png"))) - 1
                                    cast.valid_seeds = remaining_seeds
                                    cast.seed_count = remaining_seeds
                                    registry.save()
                                    st.success(f"✅ Deleted {seed_file.name}")
                                    st.rerun()
                    else:
                        st.info("No seed images found")
                else:
                    st.info("Cast directory not found")

                # Delete button
                if st.button(f"🗑️ Delete {cast.name}", key=f"delete_{cast.name}_{selected_season_id}"):
                    import shutil
                    # Delete cast directory
                    if cast_dir.exists():
                        shutil.rmtree(cast_dir)
                    # Remove from registry
                    selected_season.cast.remove(cast)
                    registry.save()
                    st.success(f"✅ Deleted {cast.name} and all seeds")
                    st.rerun()


def render_review_page():
    """Render review page with All Faces, Pairwise Review, and Low-Conf Queue."""
    st.title("🔍 Review & Label Faces")

    # Episode selector
    harvest_dir = DATA_ROOT / "harvest"
    if not harvest_dir.exists():
        st.warning("No episodes found. Please upload and process a video first.")
        return

    episodes = [d.name for d in harvest_dir.iterdir() if d.is_dir()]
    if not episodes:
        st.warning("No episodes found. Please upload and process a video first.")
        return

    # Episode selection
    selected_episode = st.selectbox(
        "Select Episode",
        episodes,
        index=(
            0
            if st.session_state.selected_episode is None
            else (
                episodes.index(st.session_state.selected_episode)
                if st.session_state.selected_episode in episodes
                else 0
            )
        ),
    )

    if selected_episode != st.session_state.selected_episode:
        st.session_state.selected_episode = selected_episode
        st.session_state.review_state_manager = ReviewStateManager(selected_episode, DATA_ROOT)
        st.session_state.cluster_mutator = ClusterMutator(selected_episode, DATA_ROOT)
        st.rerun()

    # Initialize managers if needed
    if st.session_state.review_state_manager is None:
        st.session_state.review_state_manager = ReviewStateManager(selected_episode, DATA_ROOT)
    if st.session_state.cluster_mutator is None:
        st.session_state.cluster_mutator = ClusterMutator(selected_episode, DATA_ROOT)

    state_mgr = st.session_state.review_state_manager
    cluster_mutator = st.session_state.cluster_mutator

    # Load data
    clusters_data = load_clusters(selected_episode, DATA_ROOT)
    tracks_data = load_tracks(selected_episode, DATA_ROOT) or {}
    suggestions_df = load_merge_suggestions(selected_episode, DATA_ROOT)
    lowconf_df = load_lowconf_queue(selected_episode, DATA_ROOT)

    if clusters_data is None:
        st.error(
            f"No cluster data found for episode {selected_episode}. Please process the episode first."
        )
        return

    # RE-CLUSTER button (toolbar)
    # Import job_manager at function scope so it's available throughout
    from api.jobs import job_manager
    from screentime.models import get_registry

    # Check if manual assignments exist
    has_manual_assignments = False
    if 'clusters' in clusters_data:
        for cluster in clusters_data['clusters']:
            # Manual assignments have assignment_confidence = 1.0 or no assignment_confidence at all after manual edit
            if cluster.get('name') and cluster.get('name') != 'Unknown':
                conf = cluster.get('assignment_confidence', 1.0)
                if conf == 1.0:  # Manual assignment
                    has_manual_assignments = True
                    break

    button_label = "🔄 Re-Cluster (constrained)" if has_manual_assignments else "🔄 Re-Cluster"

    col_button, col_cancel, col_options, col_status = st.columns([1, 1, 2, 2])

    with col_options:
        use_constraints = st.checkbox(
            "Use manual constraints",
            value=has_manual_assignments,  # Default on if manual assignments exist
            help="Respect manual track assignments during re-clustering (must-link / cannot-link)",
            key=f"use_constraints_{selected_episode}"
        )
        use_season_bank = st.checkbox(
            "Use season bank",
            value=True,  # Always on by default
            disabled=True,  # Always on
            help="Use season bank for open-set assignment (min_sim=0.60, min_margin=0.08)"
        )

    with col_button:
        if st.button(button_label, help="Re-run clustering with season bank (no re-detection)"):
            # Queue re-cluster job
            registry = get_registry()

            # Try to infer show/season from episode_id
            # For now, default to rhobh/s05
            show_id = "rhobh"
            season_id = "s05"

            job_id = job_manager.create_job(
                "recluster",
                episode_id=selected_episode,
                show_id=show_id,
                season_id=season_id,
                sources=["baseline", "entrance", "densify"],
                use_constraints=use_constraints
            )

            st.success(f"✅ Re-clustering queued: Job {job_id}")
            if use_constraints:
                st.info("🔗 Clustering will respect manual track assignments")
            st.info("🔄 Refresh page when complete")

    with col_cancel:
        if st.button("🛑 Cancel All", help="Cancel all re-cluster jobs for this episode", type="secondary"):
            # Cancel all recluster jobs for this episode
            result = job_manager.cancel_all(f"recluster_{selected_episode}")
            cancelled_count = result.get("cancelled_count", 0)

            if cancelled_count > 0:
                # Clear session state
                st.session_state.last_recluster_job_running = False
                st.session_state.pop('active_recluster_job_id', None)

                # Mark analytics dirty
                from app.lib.analytics_dirty import mark_analytics_dirty
                mark_analytics_dirty(selected_episode, DATA_ROOT, reason="re-cluster cancelled")

                st.success(f"Cancelled {cancelled_count} re-cluster job(s) for {selected_episode}")
                st.toast(f"Analytics marked stale - run Analyze to rebuild", icon="⚠️")
                st.rerun()
            else:
                st.info("No active re-cluster jobs to cancel")

    with col_status:
        # ROBUST GUARD: Only show banner if job ACTUALLY exists and is running
        # This prevents phantom banners from stale state

        from rq.job import Job
        from redis import Redis
        import logging
        logger = logging.getLogger(__name__)

        redis_conn = Redis(host='localhost', port=6379, db=0)

        # Check for truly running job (validate in RQ, not just session state)
        truly_running_job = None
        active_jobs = job_manager.get_active_jobs()

        for job_info in active_jobs:
            if job_info.get('task') == 'recluster' and job_info.get('episode_id') == selected_episode:
                job_id = job_info.get('job_id')
                if job_id:
                    try:
                        rq_job = Job.fetch(job_id, connection=redis_conn)
                        if rq_job.get_status() in ['queued', 'started']:
                            truly_running_job = job_info
                            break  # Found a running job
                        else:
                            logger.info(f"Job {job_id} terminal: {rq_job.get_status()}")
                    except Exception as e:
                        logger.warning(f"Job {job_id} not found in RQ: {e}")

        # Add Dismiss button if banner would show (even phantom)
        if truly_running_job or st.session_state.get('last_recluster_job_running', False):
            dismiss_col, status_col = st.columns([1, 5])
            with dismiss_col:
                if st.button("✕ Dismiss", key="dismiss_banner", help="Clear banner"):
                    st.session_state.last_recluster_job_running = False
                    st.session_state.pop('active_recluster_job_id', None)
                    st.success("Banner dismissed")
                    st.rerun()

        if truly_running_job:
            constraints_used = truly_running_job.get('use_constraints', False)
            job_id = truly_running_job.get('job_id')

            # Poll job.meta for progress information
            step = "Re-clustering"
            percent = 0
            try:
                rq_job_id = truly_running_job.get('rq_job_id')
                if rq_job_id:
                    rq_job = Job.fetch(rq_job_id, connection=redis_conn)
                    meta = rq_job.meta
                    step = meta.get('step', 'Re-clustering')
                    percent = meta.get('percent', 0)
            except Exception as e:
                logger.warning(f"Failed to fetch job progress: {e}")

            status_msg = f"🔄 {step}"
            if constraints_used:
                status_msg += " [constrained]"

            with status_col if 'status_col' in locals() else st:
                st.info(status_msg)

                # Show progress bar
                progress_value = percent / 100.0 if percent > 0 else 0.01  # Min 1% to show bar
                st.progress(progress_value)
                st.caption(f"{percent}% · Job {job_id} · Refreshing every 2s")

            # Auto-refresh every 2 seconds while job is running
            time.sleep(2)
            st.rerun()
        elif st.session_state.get('last_recluster_job_running', False):
            # Job just completed - show success message and reset flag
            st.success("✅ Re-clustering complete! Data refreshed.")
            st.session_state.last_recluster_job_running = False
            # Auto-refresh to update the data views
            time.sleep(1)
            st.rerun()

        # Track if job is currently running
        st.session_state.last_recluster_job_running = truly_running_job is not None

    st.markdown("---")

    # Status panel
    with st.expander("📊 Episode Status", expanded=False):
        from app.lib.episode_status import get_enhanced_episode_status

        status = get_enhanced_episode_status(selected_episode, DATA_ROOT)

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Faces", f"{status['faces_total']:,} / {status['faces_used']:,}",
                     help="Total detected / Used in clustering (Top-K)")
        with col2:
            st.metric("Tracks", f"{status['tracks']:,}")
        with col3:
            st.metric("Clusters", f"{status['clusters']:,}")
        with col4:
            st.metric("Suggestions", f"{status['suggestions']:,}")

        # Constraints and Suppression
        col5, col6 = st.columns(2)
        with col5:
            st.metric("Constraints", f"ML:{status['constraints_ml']}  CL:{status['constraints_cl']}",
                     help="Must-Link and Cannot-Link pairs")
        with col6:
            st.metric("Suppressed", f"T:{status['suppressed_tracks']}  C:{status['suppressed_clusters']}",
                     help="Deleted tracks/clusters")

        # Job status
        try:
            job_ids = [
                d.name
                for d in (DATA_ROOT / "harvest" / selected_episode).iterdir()
                if d.name.startswith("job_")
            ]
            if job_ids:
                latest_job_id = sorted(job_ids)[-1]
                status = job_manager.get_job_status(latest_job_id)
                st.info(
                    f"Last Job: {latest_job_id} - Status: {status['status']} ({status['progress_pct']:.1f}%)"
                )
        except Exception:
            pass

    # Controls
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    with col1:
        review_mode = st.radio(
            "Review Mode",
            ["All Faces", "Pairwise Review", "Low-Confidence Queue", "Cast View", "Unclustered"],
            index=["All Faces", "Pairwise Review", "Low-Confidence Queue", "Cast View", "Unclustered"].index(
                {
                    "all_faces": "All Faces",
                    "pairwise": "Pairwise Review",
                    "lowconf": "Low-Confidence Queue",
                    "cast_view": "Cast View",
                    "unclustered": "Unclustered",
                }.get(st.session_state.review_mode, "All Faces")
            ),
            horizontal=True,
        )
        st.session_state.review_mode = {
            "All Faces": "all_faces",
            "Pairwise Review": "pairwise",
            "Low-Confidence Queue": "lowconf",
            "Cast View": "cast_view",
            "Unclustered": "unclustered",
        }[review_mode]

    with col2:
        st.write("")  # Spacing

    with col3:
        if st.button("↩️ Undo", disabled=not state_mgr.can_undo()):
            action = state_mgr.undo()
            if action:
                st.success(f"Undone: {action.action_type}")
                st.rerun()

    with col4:
        if st.button("💾 Save"):
            state_mgr.autosave()
            st.success("Saved!")

    st.markdown("---")

    # Check for page navigation FIRST (priority over review modes)
    nav_page = st.session_state.get('navigation_page')

    if nav_page == 'cluster_gallery':
        # Render Cluster Gallery page
        from app.review_pages import render_cluster_gallery_page
        cluster_id_raw = st.session_state.get('nav_cluster_id')
        if cluster_id_raw is not None:
            try:
                cluster_id = int(cluster_id_raw)
                render_cluster_gallery_page(selected_episode, cluster_id, DATA_ROOT)
                return
            except (ValueError, TypeError):
                st.error(f"Invalid cluster ID: {cluster_id_raw}")
                return

    elif nav_page == 'track_gallery':
        # Render Track Gallery page
        from app.review_pages import render_track_gallery_page
        track_id_raw = st.session_state.get('nav_track_id')
        cluster_id_raw = st.session_state.get('nav_cluster_id')
        if track_id_raw is not None:
            try:
                track_id = int(track_id_raw)
                cluster_id = int(cluster_id_raw) if cluster_id_raw is not None else None
                render_track_gallery_page(selected_episode, track_id, cluster_id, DATA_ROOT)
                return
            except (ValueError, TypeError):
                st.error(f"Invalid track/cluster ID: {track_id_raw}/{cluster_id_raw}")
                return

    elif nav_page == 'cast_view':
        # Render Cast View page
        from app.review_pages import render_cast_view_page
        cast_name = st.session_state.get('nav_cast_name')
        if cast_name:
            render_cast_view_page(selected_episode, cast_name, DATA_ROOT)
            return

    # Render based on review mode (if no navigation page active)
    if st.session_state.review_mode == "all_faces":
        render_all_faces_grid_v2(clusters_data, selected_episode, state_mgr, cluster_mutator, DATA_ROOT)
    elif st.session_state.review_mode == "pairwise":
        render_pairwise_review_v2(
            clusters_data, suggestions_df, selected_episode, state_mgr, cluster_mutator, DATA_ROOT
        )
    elif st.session_state.review_mode == "lowconf":
        render_lowconf_queue(
            clusters_data, lowconf_df, selected_episode, state_mgr, cluster_mutator
        )
    elif st.session_state.review_mode == "cast_view":
        # Cast View selector
        st.subheader("Cast View - Select Identity")

        # Get unique identity names from clusters
        identity_names = set()
        for cluster in clusters_data.get('clusters', []):
            name = cluster.get('name')
            if name:
                identity_names.add(name)

        # Sort alphabetically (Unknown at end)
        sorted_names = sorted([n for n in identity_names if n != 'Unknown'])
        if 'Unknown' in identity_names:
            sorted_names.append('Unknown')

        if sorted_names:
            cast_name = st.selectbox(
                "Select Identity:",
                sorted_names,
                key="cast_view_selector"
            )

            if st.button("View", key="view_cast_btn"):
                # Navigate to cast view page
                st.session_state.navigation_page = 'cast_view'
                st.session_state.nav_cast_name = cast_name
                st.rerun()
        else:
            st.info("No named clusters found. Assign names first in All Faces view.")

    elif st.session_state.review_mode == "unclustered":
        render_unclustered_faces(
            clusters_data, tracks_data, selected_episode, state_mgr, cluster_mutator
        )

    # Manual Add stub
    with st.expander("➕ Manual Add (Coming Soon)", expanded=False):
        st.info("Manual face addition will be available in a future update.")
        st.text_input("Frame timestamp (ms)", disabled=True)
        st.button("Draw bounding box", disabled=True)


def render_all_faces_grid(
    clusters_data: dict,
    episode_id: str,
    state_mgr: ReviewStateManager,
    cluster_mutator: ClusterMutator,
):
    """Render All Faces grid view."""
    clusters = clusters_data.get("clusters", [])

    # Check if we're viewing a cluster's gallery
    if st.session_state.viewing_cluster_id is not None:
        cluster = next(
            (c for c in clusters if c["cluster_id"] == st.session_state.viewing_cluster_id),
            None
        )
        if cluster:
            render_cluster_gallery(cluster, episode_id, state_mgr, cluster_mutator, clusters)
            return
        else:
            # Cluster not found, reset
            st.session_state.viewing_cluster_id = None
            st.rerun()

    st.subheader("All Detected Faces")

    # Load tracks for thumbnail generation
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    # Build cluster assignments map (cluster_id -> person_name)
    cluster_assignments = {c["cluster_id"]: c.get("name") for c in clusters}

    # Filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input(
            "🔍 Search cluster by ID or track ID", value=st.session_state.search_query
        )
        st.session_state.search_query = search_query

    with col2:
        cluster_filter = st.selectbox(
            "Filter",
            ["All", "Low Confidence", "High Quality"],
            index=["All", "Low Confidence", "High Quality"].index(
                {"all": "All", "lowconf": "Low Confidence", "high_quality": "High Quality"}.get(
                    st.session_state.cluster_filter, "All"
                )
            ),
        )
        st.session_state.cluster_filter = {
            "All": "all",
            "Low Confidence": "lowconf",
            "High Quality": "high_quality",
        }[cluster_filter]

    # Apply filters
    filtered_clusters = clusters

    if st.session_state.cluster_filter == "lowconf":
        filtered_clusters = [c for c in clusters if c.get("is_lowconf", False)]
    elif st.session_state.cluster_filter == "high_quality":
        filtered_clusters = [c for c in clusters if not c.get("is_lowconf", False)]

    if search_query:
        try:
            query_int = int(search_query)
            filtered_clusters = [
                c
                for c in filtered_clusters
                if c["cluster_id"] == query_int or query_int in c.get("track_ids", [])
            ]
        except ValueError:
            pass

    st.caption(f"Showing {len(filtered_clusters)} of {len(clusters)} clusters")

    # Grid display
    if not filtered_clusters:
        st.info("No clusters match your filters.")
        return

    # Paginate
    page_size = 50
    total_pages = (len(filtered_clusters) + page_size - 1) // page_size

    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("◀ Previous", disabled=st.session_state.current_page == 0):
            st.session_state.current_page -= 1
            st.rerun()
    with col2:
        st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
    with col3:
        if st.button("Next ▶", disabled=st.session_state.current_page >= total_pages - 1):
            st.session_state.current_page += 1
            st.rerun()

    # Display clusters on current page
    start_idx = st.session_state.current_page * page_size
    end_idx = min(start_idx + page_size, len(filtered_clusters))
    page_clusters = filtered_clusters[start_idx:end_idx]

    cols = st.columns(5)
    for idx, cluster in enumerate(page_clusters):
        with cols[idx % 5]:
            # Show cluster ID and name (if assigned)
            person_name = cluster.get("name")
            if person_name:
                st.markdown(f"**{person_name}**")
                st.caption(f"Cluster {cluster['cluster_id']}")
            else:
                st.markdown(f"**Cluster {cluster['cluster_id']}**")
            st.caption(f"Size: {cluster['size']} | Quality: {cluster['quality_score']:.2f}")
            if cluster.get("is_lowconf"):
                st.warning("⚠️ Low confidence")

            # Display thumbnails using fallback chain: assets → generated → placeholder
            thumbnails, source = get_cluster_thumbnail(
                cluster, cluster_assignments, video_path, tracks_data, episode_id, thumb_gen
            )

            if thumbnails:
                for thumb_path in thumbnails[:3]:
                    if thumb_path.exists():
                        st.image(str(thumb_path), use_column_width=True)
                if source == "assets":
                    st.caption("📁 Asset")
            else:
                # Placeholder: show initials or emoji
                person_name = cluster_assignments.get(cluster["cluster_id"])
                if person_name:
                    initials = "".join([w[0].upper() for w in person_name.split()[:2]])
                    st.markdown(f"### {initials}")
                else:
                    st.markdown("🎭")

            if st.button(f"Assign Name", key=f"assign_{cluster['cluster_id']}"):
                st.session_state[f"assigning_{cluster['cluster_id']}"] = True

            if st.session_state.get(f"assigning_{cluster['cluster_id']}", False):
                name = st.text_input("Name", key=f"name_{cluster['cluster_id']}")
                if st.button("Confirm", key=f"confirm_{cluster['cluster_id']}") and name:
                    try:
                        # Perform actual assignment
                        cluster_mutator.assign_name(cluster["cluster_id"], name)

                        # Record action for undo
                        state_mgr.record_action(
                            "assign",
                            {"cluster_id": cluster["cluster_id"], "name": name},
                        )

                        st.success(f"Assigned cluster {cluster['cluster_id']} to {name}")
                        st.session_state[f"assigning_{cluster['cluster_id']}"] = False
                        st.rerun()
                    except Exception as e:
                        st.error(f"Assignment failed: {str(e)}")

            # View Tracks button - opens gallery view
            if st.button(f"📋 View Tracks ({len(cluster.get('track_ids', []))})", key=f"view_tracks_{cluster['cluster_id']}"):
                st.session_state.viewing_cluster_id = cluster['cluster_id']
                st.rerun()


def load_lowconf_ignore(episode_id: str) -> set:
    """Load ignored cluster IDs from lowconf_ignore.json."""
    ignore_path = DATA_ROOT / "harvest" / episode_id / "diagnostics" / "lowconf_ignore.json"
    if not ignore_path.exists():
        return set()

    try:
        with open(ignore_path, 'r') as f:
            data = json.load(f)
            return set(data.get('ignored', []))
    except Exception:
        return set()


def save_lowconf_ignore(episode_id: str, ignored_ids: set):
    """Save ignored cluster IDs to lowconf_ignore.json (atomic write)."""
    diagnostics_dir = DATA_ROOT / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    ignore_path = diagnostics_dir / "lowconf_ignore.json"
    temp_path = ignore_path.with_suffix('.json.tmp')

    data = {
        'episode_id': episode_id,
        'ignored': sorted(list(ignored_ids))
    }

    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2)

    temp_path.rename(ignore_path)


def render_lowconf_queue(
    clusters_data: dict,
    lowconf_df,
    episode_id: str,
    state_mgr: ReviewStateManager,
    cluster_mutator: ClusterMutator,
):
    """Render Low-Confidence Queue interface."""
    clusters = clusters_data.get("clusters", [])

    # Initialize session state for ignore tracking
    if 'lowconf_ignore' not in st.session_state:
        st.session_state.lowconf_ignore = {}

    # Load ignore set for this episode
    if episode_id not in st.session_state.lowconf_ignore:
        st.session_state.lowconf_ignore[episode_id] = load_lowconf_ignore(episode_id)

    # Check if we're in split mode
    if st.session_state.get('splitting_cluster_id') is not None:
        render_cluster_split(
            st.session_state.splitting_cluster_id,
            clusters_data,
            episode_id,
            cluster_mutator,
            DATA_ROOT
        )
        return

    # Check if we're viewing a low-conf cluster's gallery
    if st.session_state.viewing_lowconf_cluster_id is not None:
        cluster = next(
            (c for c in clusters if c["cluster_id"] == st.session_state.viewing_lowconf_cluster_id),
            None
        )
        if cluster:
            render_cluster_gallery(cluster, episode_id, state_mgr, cluster_mutator, clusters)
            # Add back button to return to lowconf view
            if st.button("◀ Back to Low-Confidence Queue"):
                st.session_state.viewing_lowconf_cluster_id = None
                st.rerun()
            return
        else:
            # Cluster not found, reset
            st.session_state.viewing_lowconf_cluster_id = None
            st.rerun()

    st.subheader("Low-Confidence Clusters")

    # Show ignored count and Undo option
    ignored_ids = st.session_state.lowconf_ignore[episode_id]
    if ignored_ids:
        info_col1, info_col2 = st.columns([4, 1])
        with info_col1:
            st.info(
                f"⚠️ These clusters have low quality scores and may need review. "
                f"({len(ignored_ids)} hidden via 'Mark as Good')"
            )
        with info_col2:
            if st.button("🔄 Show All", key="reset_ignore"):
                st.session_state.lowconf_ignore[episode_id] = set()
                save_lowconf_ignore(episode_id, set())
                st.success("Reset! All clusters visible again")
                st.rerun()
    else:
        st.info(
            "⚠️ These clusters have low quality scores and may need review. "
            "They might contain mixed faces or poor quality detections."
        )

    if lowconf_df is None or len(lowconf_df) == 0:
        st.info("No low-confidence clusters found. Great job!")
        return

    # Filter out ignored clusters
    lowconf_df = lowconf_df[~lowconf_df['cluster_id'].isin(ignored_ids)]

    if len(lowconf_df) == 0:
        st.info("✅ All low-confidence clusters marked as good! (Click 'Show All' above to review again)")
        return

    st.caption(f"{len(lowconf_df)} clusters flagged for review")

    # Load data for thumbnails
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator
    cluster_assignments = {c["cluster_id"]: c.get("name") for c in clusters}

    # Sort by quality score (lowest first)
    lowconf_df = lowconf_df.sort_values("quality_score")

    # Display clusters
    for idx, row in lowconf_df.iterrows():
        cluster_id = row["cluster_id"]
        size = row["size"]
        quality_score = row["quality_score"]

        # Find cluster data
        cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)
        if not cluster:
            continue

        # Get person name or use cluster ID
        person_name = cluster.get("name", f"Cluster {cluster_id}")
        expander_title = f"{person_name}" if cluster.get("name") else f"Cluster {cluster_id}"
        expander_title += f" (Quality: {quality_score:.2f}, Size: {size})"

        with st.expander(expander_title, expanded=idx == 0):
            st.caption(f"Tracks: {row['track_ids']}")

            # Show ALL images from ALL tracks in smaller gallery
            if video_path.exists() and tracks_data:
                all_thumbs = []
                try:
                    for track_id in cluster.get("track_ids", []):
                        track = next(
                            (t for t in tracks_data.get("tracks", []) if t["track_id"] == track_id),
                            None
                        )
                        if track:
                            # Get up to 3 frames per track
                            for frame_ref in track.get("frame_refs", [])[:3]:
                                thumb_path = thumb_gen.generate_frame_thumbnail(
                                    video_path, frame_ref["frame_id"], frame_ref["bbox"], episode_id, track_id
                                )
                                if thumb_path and thumb_path.exists():
                                    all_thumbs.append(thumb_path)

                    # Display all thumbnails in smaller grid (8 columns)
                    if all_thumbs:
                        st.caption(f"Showing {len(all_thumbs)} sample images:")
                        # Display in rows of 8
                        for row_start in range(0, len(all_thumbs), 8):
                            cols = st.columns(8)
                            for col_idx, thumb_path in enumerate(all_thumbs[row_start:row_start + 8]):
                                with cols[col_idx]:
                                    st.image(str(thumb_path), use_column_width=True)
                    else:
                        st.caption("🎭 No thumbnails available")
                except Exception as e:
                    st.error(f"Failed to generate thumbnails: {str(e)}")

            st.markdown("---")
            st.markdown("**Actions:**")
            st.caption(
                "• **Mark as Good**: Remove low-confidence flag (cluster quality is acceptable)\n"
                "• **View Tracks**: Open gallery view to manage individual tracks\n"
                "• **Split Cluster**: Flag for manual splitting (contains multiple different people)"
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"✅ Mark as Good", key=f"good_{cluster_id}"):
                    # Add to ignore set
                    st.session_state.lowconf_ignore[episode_id].add(cluster_id)
                    save_lowconf_ignore(episode_id, st.session_state.lowconf_ignore[episode_id])

                    # Show toast
                    st.toast(f"✅ Cluster {cluster_id} hidden from Low-Confidence until next re-cluster", icon="✅")
                    st.rerun()

            with col2:
                if st.button(f"📋 View Tracks", key=f"view_lowconf_{cluster_id}"):
                    st.session_state.viewing_lowconf_cluster_id = cluster_id
                    st.rerun()

            with col3:
                if st.button(f"✂️ Split Cluster", key=f"split_{cluster_id}"):
                    st.session_state.splitting_cluster_id = cluster_id
                    st.session_state.split_selected_tracks = set()  # Clear any previous selection
        st.rerun()

    # Batch actions
    st.markdown("---")
    st.markdown("**Batch Actions**")
    if st.button("Mark All as Reviewed"):
        for _, row in lowconf_df.iterrows():
            state_mgr.record_action("batch_review", {"cluster_id": row["cluster_id"]})
        st.success(f"Marked {len(lowconf_df)} clusters as reviewed")
        st.rerun()


def render_workspace_page():
    """Render the Workspace shell (Faces · Clusters · Tracks · Review)."""
    st.title("🗂️ Workspace")

    harvest_dir = DATA_ROOT / "harvest"
    if not harvest_dir.exists():
        st.warning("No episodes found. Please upload and process a video first.")
        return

    episodes = sorted([d.name for d in harvest_dir.iterdir() if d.is_dir()])
    if not episodes:
        st.warning("No episodes found. Please upload and process a video first.")
        return

    default_episode = st.session_state.get("workspace_episode")
    if default_episode in episodes:
        default_index = episodes.index(default_episode)
    else:
        default_index = 0

    selected_episode = st.selectbox(
        "Select Episode",
        options=episodes,
        index=default_index,
    )

    if selected_episode != st.session_state.get("workspace_episode"):
        st.session_state.workspace_episode = selected_episode
        st.session_state.workspace_selected_person = None
        st.session_state.workspace_selected_cluster = None

    mutator = configure_workspace_mutator(selected_episode, DATA_ROOT)

    render_workspace_recluster_toolbar(selected_episode, mutator.clusters)

    st.markdown("---")

    tab_labels = ["Faces", "Clusters", "Tracks", "Review"]
    default_tab = st.session_state.get("workspace_tab", "Faces")
    default_index = tab_labels.index(default_tab) if default_tab in tab_labels else 0

    active_tab = st.radio(
        "",
        options=tab_labels,
        index=default_index,
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.workspace_tab = active_tab

    if active_tab == "Faces":
        render_faces_tab(mutator)
    elif active_tab == "Clusters":
        render_clusters_tab(mutator)
    elif active_tab == "Tracks":
        render_tracks_tab(mutator)
    elif active_tab == "Review":
        render_review_tab(mutator)


def render_workspace_recluster_toolbar(selected_episode: str, clusters_data: dict) -> None:
    """Toolbar for re-cluster jobs, lifted from legacy review page."""
    from screentime.models import get_registry
    from rq.job import Job
    from redis import Redis
    import logging

    has_manual_assignments = False
    for cluster in clusters_data.get("clusters", []):
        if cluster.get("name") and cluster.get("name") != "Unknown":
            conf = cluster.get("assignment_confidence", 1.0)
            if conf == 1.0:
                has_manual_assignments = True
                break

    button_label = "🔄 Re-Cluster (constrained)" if has_manual_assignments else "🔄 Re-Cluster"

    col_button, col_cancel, col_options, col_status = st.columns([1, 1, 2, 2])

    with col_options:
        use_constraints = st.checkbox(
            "Use manual constraints",
            value=has_manual_assignments,
            help="Respect manual track assignments during re-clustering (must-link / cannot-link)",
            key=f"use_constraints_{selected_episode}"
        )

    with col_button:
        if st.button(button_label, help="Re-run clustering with season bank (no re-detection)"):
            registry = get_registry()
            show_id, season_id = resolve_show_season(registry, selected_episode)

            job_id = job_manager.create_job(
                "recluster",
                episode_id=selected_episode,
                show_id=show_id,
                season_id=season_id,
                sources=["baseline", "entrance", "densify"],
                use_constraints=use_constraints,
            )

            st.success(f"✅ Re-clustering queued: Job {job_id}")
            if use_constraints:
                st.info("🔗 Clustering will respect manual track assignments")
            st.info("🔄 Refresh page when complete")

    with col_cancel:
        if st.button("🛑 Cancel All", help="Cancel all re-cluster jobs for this episode", type="secondary"):
            result = job_manager.cancel_all(f"recluster_{selected_episode}")
            cancelled_count = result.get("cancelled_count", 0)

            if cancelled_count > 0:
                st.session_state.last_recluster_job_running = False
                st.session_state.pop('active_recluster_job_id', None)

                from app.lib.analytics_dirty import mark_analytics_dirty

                mark_analytics_dirty(selected_episode, DATA_ROOT, reason="re-cluster cancelled")

                st.success(f"Cancelled {cancelled_count} re-cluster job(s) for {selected_episode}")
                st.toast("Analytics marked stale - run Analyze to rebuild", icon="⚠️")
                st.rerun()
            else:
                st.info("No active re-cluster jobs to cancel")

    with col_status:
        logger = logging.getLogger(__name__)
        redis_conn = Redis(host='localhost', port=6379, db=0)

        truly_running_job = None
        active_jobs = job_manager.get_active_jobs()

        for job_info in active_jobs:
            if job_info.get('task') == 'recluster' and job_info.get('episode_id') == selected_episode:
                job_id = job_info.get('job_id')
                if job_id:
                    try:
                        rq_job = Job.fetch(job_id, connection=redis_conn)
                        if rq_job.get_status() in ['queued', 'started']:
                            truly_running_job = job_info
                            break
                        else:
                            logger.info(f"Job {job_id} terminal: {rq_job.get_status()}")
                    except Exception as e:
                        logger.warning(f"Job {job_id} not found in RQ: {e}")

        if truly_running_job or st.session_state.get('last_recluster_job_running', False):
            dismiss_col, status_col = st.columns([1, 5])
            with dismiss_col:
                if st.button("✕ Dismiss", key="dismiss_workspace_banner", help="Clear banner"):
                    st.session_state.last_recluster_job_running = False
                    st.session_state.pop('active_recluster_job_id', None)
                    st.success("Banner dismissed")
                    st.rerun()
        else:
            status_col = st

        if truly_running_job:
            constraints_used = truly_running_job.get('use_constraints', False)
            job_id = truly_running_job.get('job_id')

            step = "Re-clustering"
            percent = 0
            try:
                rq_job_id = truly_running_job.get('rq_job_id')
                if rq_job_id:
                    rq_job = Job.fetch(rq_job_id, connection=redis_conn)
                    meta = rq_job.meta
                    step = meta.get('step', 'Re-clustering')
                    percent = meta.get('percent', 0)
            except Exception as e:
                logger.warning(f"Failed to fetch job progress: {e}")

            status_msg = f"🔄 {step}"
            if constraints_used:
                status_msg += " [constrained]"

            with status_col:
                st.info(status_msg)
                progress_value = percent / 100.0 if percent > 0 else 0.01
                st.progress(progress_value)
                st.caption(f"{percent}% · Job {job_id} · Refreshing every 2s")

            time.sleep(2)
            st.rerun()
        elif st.session_state.get('last_recluster_job_running', False):
            st.success("✅ Re-clustering complete! Data refreshed.")
            st.session_state.last_recluster_job_running = False
            time.sleep(1)
            st.rerun()

        st.session_state.last_recluster_job_running = truly_running_job is not None


def render_analytics_page():
    """Render analytics page with exports."""
    st.title("📊 Analytics & Exports")

    # Episode selector
    harvest_dir = DATA_ROOT / "harvest"
    if not harvest_dir.exists():
        st.warning("No episodes found. Please upload and process a video first.")
        return

    episodes = [d.name for d in harvest_dir.iterdir() if d.is_dir()]
    if not episodes:
        st.warning("No episodes found.")
        return

    selected_episode = st.selectbox("Select Episode", episodes, index=0)

    # Check Analytics freshness
    from app.lib.analytics_dirty import is_analytics_dirty, get_analytics_freshness
    is_dirty = is_analytics_dirty(selected_episode, DATA_ROOT)
    freshness_status = get_analytics_freshness(selected_episode, DATA_ROOT)

    # Show freshness indicator
    if is_dirty or freshness_status == 'stale':
        st.warning("⚠️ Analytics are **stale** - clusters have been modified since last Analyze run")
        st.info("**Recommended workflow for best results:**\n1. Go to **Manage** tab and run **RE-CLUSTER (constrained)**\n2. Return here and click **Analyze** to rebuild analytics from current state")
    elif freshness_status == 'fresh':
        st.success("✅ Analytics are **fresh** - up to date with current clusters")
    elif freshness_status == 'unknown':
        st.info("ℹ️ Analytics status: **unknown** - no previous Analyze run detected")

    # Check for analytics outputs
    outputs_dir = DATA_ROOT / "outputs" / selected_episode
    totals_csv = outputs_dir / "totals.csv"
    timeline_csv = outputs_dir / "timeline.csv"
    excel_file = outputs_dir / "totals.xlsx"

    # Check if files exist and are not empty
    files_valid = False
    if totals_csv.exists() and timeline_csv.exists():
        # Check if files have content (more than just header/empty)
        try:
            import os
            totals_size = os.path.getsize(totals_csv)
            timeline_size = os.path.getsize(timeline_csv)
            # Files should be more than 1 byte (not just newline)
            if totals_size > 10 and timeline_size > 10:
                # Try to read them
                test_df = pd.read_csv(totals_csv)
                if len(test_df) > 0:
                    files_valid = True
        except:
            files_valid = False

    if not files_valid:
        st.info(f"Analytics not yet generated for {selected_episode} or files are empty.")
        st.info("**Click below to rebuild analytics from current clusters** (always rebuilds from scratch, no cache)")

        # Button to generate analytics
        if st.button("📊 Analyze", type="primary", help="Rebuild analytics from current clusters.json (suppression-aware, always fresh)"):
            with st.spinner("Generating analytics..."):
                try:
                    # Load cluster assignments from clusters.json
                    clusters_data = load_clusters(selected_episode, DATA_ROOT)
                    cluster_assignments = {}
                    if clusters_data:
                        for cluster in clusters_data.get("clusters", []):
                            if "name" in cluster:
                                cluster_assignments[cluster["cluster_id"]] = cluster["name"]

                    from jobs.tasks.analytics import analytics_task

                    result = analytics_task("manual", selected_episode, cluster_assignments)
                    st.success(
                        f"✅ Analytics generated! {result['stats']['intervals_created']} intervals created."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate analytics: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    # Load analytics data
    try:
        totals_df = pd.read_csv(totals_csv)
        timeline_df = pd.read_csv(timeline_csv)
    except Exception as e:
        st.error(f"Failed to load analytics: {str(e)}")
        st.info("Try clicking 'Generate Analytics' above to regenerate the files.")
        return

    # Display summary
    st.subheader("Screen Time Summary")

    # Regenerate button
    if st.button("🔄 Rebuild Analytics", help="Rebuild analytics from current clusters.json (always rebuilds from scratch, suppression-aware)"):
        with st.spinner("Regenerating analytics..."):
            try:
                # Load cluster assignments from clusters.json
                clusters_data = load_clusters(selected_episode, DATA_ROOT)
                cluster_assignments = {}
                if clusters_data:
                    for cluster in clusters_data.get("clusters", []):
                        if "name" in cluster:
                            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

                from jobs.tasks.analytics import analytics_task

                result = analytics_task("manual", selected_episode, cluster_assignments)
                st.success(
                    f"Analytics regenerated! {result['stats']['intervals_created']} intervals created."
                )
                st.rerun()
            except Exception as e:
                st.error(f"Failed to regenerate analytics: {str(e)}")

    if len(totals_df) == 0:
        st.info("No people detected or assigned in this episode.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("People Detected", len(totals_df))
    with col2:
        total_time_sec = totals_df["total_sec"].sum()
        minutes = int(total_time_sec // 60)
        seconds = int(total_time_sec % 60)
        st.metric("Total Screen Time", f"{minutes:02d}:{seconds:02d}")
    with col3:
        st.metric("Total Intervals", len(timeline_df))
    with col4:
        mean_time_sec = totals_df["total_sec"].mean()
        st.metric("Avg Time/Person", f"{mean_time_sec:.1f} sec")

    # Totals table
    st.subheader("Top Cast Members")

    # Format display columns
    display_df = totals_df[["person_name", "total_sec", "appearances", "percent"]].copy()
    # Format as MM:SS:MS
    display_df["total_time"] = display_df["total_sec"].apply(
        lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}:{int((x % 1) * 1000):03d}"
    )
    display_df = display_df.rename(
        columns={
            "person_name": "Person",
            "total_time": "Screen Time",
            "appearances": "Appearances",
            "percent": "% of Episode",
        }
    )
    display_df = display_df[["Person", "Screen Time", "Appearances", "% of Episode"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bar chart
    st.subheader("Screen Time Distribution")

    import plotly.express as px

    fig = px.bar(
        totals_df.head(10),
        x="person_name",
        y="total_sec",
        labels={"person_name": "Person", "total_sec": "Screen Time (seconds)"},
        title="Top 10 Cast Members by Screen Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Timeline preview
    with st.expander("Timeline Preview (First 20 intervals)", expanded=False):
        timeline_preview = timeline_df.head(20)[
            ["person_name", "start_ms", "end_ms", "duration_ms", "confidence"]
        ]
        st.dataframe(timeline_preview, use_container_width=True, hide_index=True)

    # Export section
    st.markdown("---")
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download Totals (CSV)",
            data=open(totals_csv, "rb").read(),
            file_name=f"{selected_episode}_totals.csv",
            mime="text/csv",
        )

    with col2:
        st.download_button(
            label="📥 Download Timeline (CSV)",
            data=open(timeline_csv, "rb").read(),
            file_name=f"{selected_episode}_timeline.csv",
            mime="text/csv",
        )

    with col3:
        if excel_file.exists():
            st.download_button(
                label="📥 Download Excel",
                data=open(excel_file, "rb").read(),
                file_name=f"{selected_episode}_totals.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.button("📥 Download Excel", disabled=True)


def render_cluster_gallery(
    cluster: dict,
    episode_id: str,
    state_mgr: ReviewStateManager,
    cluster_mutator: ClusterMutator,
    all_clusters: list,
):
    """Render full gallery view for a cluster's tracks."""
    cluster_id = cluster["cluster_id"]
    person_name = cluster.get("name", f"Cluster {cluster_id}")

    st.title(f"📸 {person_name}")
    st.caption(f"Cluster {cluster_id} | {len(cluster.get('track_ids', []))} tracks")

    if st.button("◀ Back to Grid"):
        st.session_state.viewing_cluster_id = None
        st.rerun()

    st.markdown("---")

    # Load data
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    if not tracks_data:
        st.error("No tracks data found")
        return

    # Display each track
    for track_idx, track_id in enumerate(cluster.get("track_ids", [])):
        track = next(
            (t for t in tracks_data.get("tracks", []) if t["track_id"] == track_id),
            None
        )
        if not track:
            continue

        with st.expander(f"Track {track_id} ({len(track.get('frame_refs', []))} frames)", expanded=True):
            # Generate thumbnails for this track
            if video_path.exists():
                try:
                    frame_refs = track.get("frame_refs", [])  # Show ALL frames

                    # Display in fixed grid: always 5 columns per row for uniform sizing
                    for row_start in range(0, len(frame_refs), 5):
                        row_refs = frame_refs[row_start:row_start + 5]
                        cols = st.columns(5)  # Always 5 columns for uniform spacing

                        for col_idx, frame_ref in enumerate(row_refs):
                            with cols[col_idx]:
                                # Generate thumbnail for this specific frame
                                thumb_path = thumb_gen.generate_frame_thumbnail(
                                    video_path, frame_ref["frame_id"], frame_ref["bbox"], episode_id, track_id
                                )
                                if thumb_path and thumb_path.exists():
                                    # Per-frame unique key from all context
                                    frame_id = frame_ref["frame_id"]
                                    frame_idx = row_start + col_idx
                                    # Use actual ts_ms if available (entrance/densify tracks), else calculate
                                    ts_ms = frame_ref.get("ts_ms", track.get("start_ms", 0) + (frame_idx * 100))

                                    base_key = (episode_id, cluster_id, track_idx, track_id, frame_idx, frame_id, ts_ms)

                                    # Fixed width 160px (no key - not supported in Streamlit 1.38)
                                    st.image(str(thumb_path), width=160)

                                    # Per-frame delete button
                                    del_key = wkey("del", *base_key)
                                    del_state = wkey("deleting", *base_key)

                                    if st.button("🗑️", key=del_key, help="Delete this frame"):
                                        st.session_state[del_state] = True

                                    # Delete confirmation
                                    if st.session_state.get(del_state, False):
                                        confirm_key = wkey("confirm", *base_key)
                                        if st.button("✓ Confirm", key=confirm_key, type="primary"):
                                            # Delete frame from track
                                            try:
                                                cluster_mutator.delete_frame_from_track(track_id, frame_id)
                                                st.success(f"Deleted frame {frame_id}")
                                                st.session_state[del_state] = False
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Delete failed: {str(e)}")

                                    # Per-image reassignment: move to another cluster
                                    move_frame_key = wkey("move_frame", *base_key)
                                    cluster_names = [c.get("name", f"Cluster {c['cluster_id']}") for c in all_clusters if c["cluster_id"] != cluster_id]
                                    cluster_names.insert(0, "Move to...")

                                    selected_cluster_name = st.selectbox(
                                        "Move to:",
                                        cluster_names,
                                        key=move_frame_key,
                                        label_visibility="collapsed"
                                    )

                                    if selected_cluster_name != "Move to...":
                                        # Find target cluster ID
                                        target_cluster = next((c for c in all_clusters if c.get("name") == selected_cluster_name or f"Cluster {c['cluster_id']}" == selected_cluster_name), None)
                                        if target_cluster:
                                            # Move frame to target cluster
                                            try:
                                                # Create a new single-frame track in the target cluster
                                                cluster_mutator.move_frame_to_cluster(track_id, frame_id, target_cluster["cluster_id"])
                                                st.success(f"Moved frame to {selected_cluster_name}")
                                                st.rerun()
                                            except Exception as e:
                                                st.error(f"Move failed: {str(e)}")

                except Exception as e:
                    st.error(f"Failed to generate thumbnails: {str(e)}")

            # Actions for this track
            track_base_key = (episode_id, cluster_id, track_idx, track_id, track.get("start_ms", 0))

            col1, col2, col3 = st.columns(3)
            with col1:
                move_key = wkey("move", *track_base_key)
                move_state = wkey("moving", *track_base_key)
                if st.button(f"Move Track {track_id}", key=move_key):
                    st.session_state[move_state] = True

            with col2:
                skip_key = wkey("skip", *track_base_key)
                if st.button(f"Skip Track {track_id}", key=skip_key, type="secondary"):
                    # Move to SKIP cluster (or create it if it doesn't exist)
                    skip_cluster = next((c for c in all_clusters if c.get('name') == 'SKIP'), None)
                    if skip_cluster:
                        # Use existing move_track method
                        result = cluster_mutator.move_track(track_id, cluster_id, skip_cluster['cluster_id'])
                        st.success(f"Track {track_id} moved to SKIP")
                    else:
                        # Create new SKIP cluster by splitting current cluster
                        current_cluster = next((c for c in all_clusters if c['cluster_id'] == cluster_id), None)
                        if current_cluster:
                            other_tracks = [tid for tid in current_cluster['track_ids'] if tid != track_id]
                            if other_tracks:
                                # Split: keep other tracks in original cluster, create new cluster with this track
                                result = cluster_mutator.split_cluster(cluster_id, other_tracks, [track_id])
                                # Get the new cluster ID and assign "SKIP" name
                                new_cluster_id = max(c['cluster_id'] for c in result['clusters'])
                                cluster_mutator.assign_name(new_cluster_id, "SKIP")
                                st.success(f"Track {track_id} moved to new SKIP cluster")
                            else:
                                # This is the only track in the cluster, just rename it
                                cluster_mutator.assign_name(cluster_id, "SKIP")
                                st.success(f"Cluster {cluster_id} renamed to SKIP")
                    st.rerun()

            with col3:
                deltrack_key = wkey("deltrack", *track_base_key)
                deltrack_state = wkey("deleting_track", *track_base_key)
                if st.button(f"Delete Track {track_id}", key=deltrack_key, type="secondary"):
                    st.session_state[deltrack_state] = True

            # Move dialog
            move_state = wkey("moving", *track_base_key)
            if st.session_state.get(move_state, False):
                cluster_options = {
                    (c.get('name') or f"Cluster {c['cluster_id']}"): c['cluster_id']
                    for c in all_clusters if c['cluster_id'] != cluster_id
                }
                cluster_options["➕ Create New Cluster"] = -1

                selected = st.selectbox(
                    "Move to:",
                    options=list(cluster_options.keys()),
                    key=wkey("selmove", *track_base_key)
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Confirm", key=wkey("cfmmove", *track_base_key)):
                        try:
                            to_cluster_id = cluster_options[selected]
                            cluster_mutator.move_track(track_id, cluster_id, to_cluster_id)
                            state_mgr.record_action("move_track", {"track_id": track_id, "from": cluster_id, "to": to_cluster_id})
                            st.success(f"Moved track {track_id}")
                            st.session_state[move_state] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Move failed: {str(e)}")
                with col_b:
                    if st.button("Cancel", key=wkey("canmove", *track_base_key)):
                        st.session_state[move_state] = False
                        st.rerun()

            # Delete dialog
            deltrack_state = wkey("deleting_track", *track_base_key)
            if st.session_state.get(deltrack_state, False):
                st.warning(f"⚠️ Delete track {track_id}? This cannot be undone.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Confirm Delete", key=wkey("cfmdel", *track_base_key)):
                        try:
                            cluster_mutator.delete_track(track_id, cluster_id)
                            state_mgr.record_action("delete_track", {"track_id": track_id, "cluster_id": cluster_id})
                            st.success(f"Deleted track {track_id}")
                            st.session_state[deltrack_state] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {str(e)}")
                with col_b:
                    if st.button("Cancel", key=wkey("candel", *track_base_key)):
                        st.session_state[deltrack_state] = False
                        st.rerun()


def render_unclustered_faces(
    clusters_data: dict,
    tracks_data: dict,
    episode_id: str,
    state_mgr: ReviewStateManager,
    cluster_mutator: ClusterMutator,
):
    """Render unclustered faces - tracks not assigned to any cluster."""
    st.subheader("Unclustered Faces")
    st.info("These tracks were not assigned to any cluster during the clustering process.")

    if not tracks_data:
        st.error("No tracks data found")
        return

    clusters = clusters_data.get("clusters", [])
    all_track_ids = {t["track_id"] for t in tracks_data.get("tracks", [])}
    clustered_track_ids = set()

    # Find all clustered track IDs
    for cluster in clusters:
        clustered_track_ids.update(cluster.get("track_ids", []))

    # Find unclustered tracks
    unclustered_track_ids = all_track_ids - clustered_track_ids

    if not unclustered_track_ids:
        st.success("All tracks have been assigned to clusters!")
        return

    st.caption(f"{len(unclustered_track_ids)} unclustered tracks found")

    # Load data for thumbnails
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    # Display unclustered tracks
    for track_id in sorted(unclustered_track_ids):
        track = next(
            (t for t in tracks_data.get("tracks", []) if t["track_id"] == track_id),
            None
        )
        if not track:
            continue

        with st.expander(f"Track {track_id} ({len(track.get('frame_refs', []))} frames)"):
            # Show thumbnails
            if video_path.exists():
                try:
                    frame_refs = track.get("frame_refs", [])[:5]
                    cols = st.columns(len(frame_refs))

                    for idx, frame_ref in enumerate(frame_refs):
                        with cols[idx]:
                            thumb_path = thumb_gen.generate_frame_thumbnail(
                                video_path, frame_ref["frame_id"], frame_ref["bbox"], episode_id, track_id
                            )
                            if thumb_path and thumb_path.exists():
                                st.image(str(thumb_path), use_column_width=True, width=100)
                except Exception as e:
                    st.caption(f"Unable to load thumbnails: {str(e)}")

            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Assign to Cluster", key=f"assign_unclustered_{track_id}"):
                    st.session_state[f"assigning_unclustered_{track_id}"] = True

            with col2:
                if st.button(f"Delete", key=f"delete_unclustered_{track_id}"):
                    st.session_state[f"deleting_unclustered_{track_id}"] = True

            # Assign dialog
            if st.session_state.get(f"assigning_unclustered_{track_id}", False):
                cluster_options = {
                    (c.get('name') or f"Cluster {c['cluster_id']}"): c['cluster_id']
                    for c in clusters
                }
                cluster_options["➕ Create New Cluster"] = -1

                selected = st.selectbox(
                    "Assign to:",
                    options=list(cluster_options.keys()),
                    key=f"select_assign_{track_id}"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Confirm", key=f"confirm_assign_{track_id}"):
                        try:
                            to_cluster_id = cluster_options[selected]
                            # Add track to cluster
                            if to_cluster_id == -1:
                                # Create new cluster
                                max_cluster_id = max((c["cluster_id"] for c in clusters), default=0)
                                new_cluster = {
                                    "cluster_id": max_cluster_id + 1,
                                    "size": 1,
                                    "track_ids": [track_id],
                                    "variance": 0.0,
                                    "silhouette_score": 0.0,
                                    "quality_score": 0.5,
                                    "is_lowconf": True,
                                }
                                clusters.append(new_cluster)
                                clusters_data["clusters"] = clusters
                                clusters_data["total_clusters"] = len(clusters)
                                cluster_mutator._save_clusters_atomic(clusters_data)
                            else:
                                # Add to existing cluster
                                target_cluster = next((c for c in clusters if c["cluster_id"] == to_cluster_id), None)
                                if target_cluster:
                                    target_cluster["track_ids"].append(track_id)
                                    target_cluster["size"] = len(target_cluster["track_ids"])
                                    cluster_mutator._save_clusters_atomic(clusters_data)

                            state_mgr.record_action("assign_unclustered", {"track_id": track_id, "cluster_id": to_cluster_id})
                            st.success(f"Assigned track {track_id}")
                            st.session_state[f"assigning_unclustered_{track_id}"] = False
                            st.rerun()
                        except Exception as e:
                            st.error(f"Assignment failed: {str(e)}")
                with col_b:
                    if st.button("Cancel", key=f"cancel_assign_{track_id}"):
                        st.session_state[f"assigning_unclustered_{track_id}"] = False
                        st.rerun()

            # Delete dialog (mark as deleted, don't actually remove from tracks)
            if st.session_state.get(f"deleting_unclustered_{track_id}", False):
                st.warning(f"⚠️ Delete track {track_id}?")
                st.caption("Note: This will be marked as deleted. The track data remains in tracks.json.")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("Confirm Delete", key=f"confirm_del_uncl_{track_id}"):
                        # Just close the dialog - we don't actually delete unclustered tracks from tracks.json
                        state_mgr.record_action("delete_unclustered", {"track_id": track_id})
                        st.success(f"Track {track_id} marked as deleted")
                        st.session_state[f"deleting_unclustered_{track_id}"] = False
                        st.rerun()
                with col_b:
                    if st.button("Cancel", key=f"cancel_del_uncl_{track_id}"):
                        st.session_state[f"deleting_unclustered_{track_id}"] = False
                        st.rerun()


def render_settings_page():
    """Render settings page."""
    st.title("⚙️ Settings")

    st.subheader("System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Data Root", str(DATA_ROOT))
        st.metric("Redis URL", os.getenv("REDIS_URL", "Not configured"))

    with col2:
        videos_dir = DATA_ROOT / "videos"
        if videos_dir.exists():
            video_count = len(list(videos_dir.glob("*.mp4")))
            st.metric("Uploaded Videos", video_count)


def main():
    """Main application entry point."""
    init_session_state()

    # Add CSS for uniform thumbnail tiles
    st.markdown("""
        <style>
        /* Uniform thumbnail tiles - 160x160 fixed size with center-crop */
        .stImage > img {
            height: 160px !important;
            object-fit: cover !important;
            border-radius: 4px;
        }

        /* Hover effect for thumbnails */
        .stImage:hover {
            opacity: 0.9;
            transition: opacity 0.2s;
        }
        </style>
    """, unsafe_allow_html=True)

    # Render sidebar and get selected page
    page = render_sidebar()

    # Render selected page
    if "Upload" in page:
        render_upload_page()
    elif "Cast Images" in page:
        render_cast_images_page()
    elif "Workspace" in page:
        render_workspace_page()
    elif "Analytics" in page:
        render_analytics_page()
    elif "Settings" in page:
        render_settings_page()


if __name__ == "__main__":
    main()
