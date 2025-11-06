"""
Upload - Episode upload and processing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import streamlit as st
from dotenv import load_dotenv

from app.lib.metadata import MetadataManager
from app.utils.ui_keys import wkey, safe_rerun
from screentime.io_utils import validate_video
from screentime.types import UploadStatus
from screentime.utils import canonical_show_slug
from api.uploads import upload_manager

load_dotenv()

st.set_page_config(
    page_title="Upload",
    page_icon="📤",
    layout="wide",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))


def render_upload_page():
    """Render video upload page with chunked upload support."""
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
        # Show nickname with full name in parentheses for clarity
        show_options = ["[Create New Show]"] + [
            f"{show.name} ({show.display_name})" for show in shows
        ]
        show_map = {f"{show.name} ({show.display_name})": show.name for show in shows}

        selected_show_option = st.selectbox(
            "Show",
            options=show_options,
            help="Select existing show or create new one",
            key=wkey("upload_show_select")
        )

        # Create new show
        if selected_show_option == "[Create New Show]":
            st.markdown("#### Create New Show")
            new_show_name = st.text_input(
                "Nickname (Short)",
                placeholder="e.g., RHOBH",
                help="Short nickname for dropdowns (uppercase recommended)",
                key=wkey("upload_new_show_name")
            )
            new_show_display = st.text_input(
                "Full Name",
                placeholder="e.g., Real Housewives of Beverly Hills",
                help="Full show title for display",
                key=wkey("upload_new_show_display")
            )

            if st.button("Create Show", key=wkey("upload_create_show")):
                if new_show_name and new_show_display:
                    try:
                        metadata_manager.create_show(new_show_name, new_show_display)
                        st.success(f"✅ Created show: {new_show_display}")
                        safe_rerun()
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                else:
                    st.warning("⚠️ Please enter both show name and display name")

    with col2:
        # Season selection (only if show selected)
        if selected_show_option != "[Create New Show]" and selected_show_option:
            # Extract short name from "RHOBH (Real Housewives of Beverly Hills)" format
            selected_show_name = canonical_show_slug(show_map.get(selected_show_option, selected_show_option))
            seasons = metadata_manager.list_seasons(selected_show_name)
            season_options = ["[Create New Season]"] + [f"Season {s.season_number}" for s in seasons]

            selected_season_option = st.selectbox(
                "Season",
                options=season_options,
                help="Select existing season or create new one",
                key=wkey("upload_season_select")
            )

            # Create new season
            if selected_season_option == "[Create New Season]":
                st.markdown("#### Create New Season")
                new_season_number = st.number_input(
                    "Season Number",
                    min_value=1,
                    max_value=50,
                    value=1,
                    help="Season number (e.g., 5 for Season 5)",
                    key=wkey("upload_new_season_number")
                )
                new_season_label = st.text_input(
                    "Label (optional)",
                    placeholder=f"S{new_season_number:02d}",
                    help="Optional custom label (defaults to S##)",
                    key=wkey("upload_new_season_label")
                )

                if st.button("Create Season", key=wkey("upload_create_season")):
                    try:
                        label = new_season_label if new_season_label else None
                        metadata_manager.create_season(selected_show_name, new_season_number, label)
                        st.success(f"✅ Created Season {new_season_number}")
                        safe_rerun()
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
        st.subheader("2. Episode Details")

        # Episode number input
        episode_number = st.number_input(
            "Episode Number",
            min_value=1,
            max_value=99,
            value=1,
            help="Episode number (e.g., 1 for Episode 1)",
            key=wkey("upload_episode_number")
        )

        st.subheader("3. Select Video File")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=["mp4"],
            help="Select an MP4 video file to upload",
            key=wkey("upload_file_uploader")
        )

        if uploaded_file is not None:
            file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.info(f"📁 **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

            # Auto-generate episode ID with format: SHOW_S##_E##_MMDDYYYY
            from datetime import datetime
            timestamp = datetime.now().strftime("%m%d%Y")
            episode_id = f"{selected_show_name.upper()}_S{selected_season_number:02d}_E{episode_number:02d}_{timestamp}"
            st.info(f"ℹ️ Episode ID: `{episode_id}`")

            # Initialize validation state
            validation_key = f"validated_{episode_id}"
            if validation_key not in st.session_state:
                st.session_state[validation_key] = False

            # Validation
            if st.button("Validate Video", type="primary", key=wkey("upload_validate")):
                with st.spinner("Validating video..."):
                    # Save temp file for validation
                    temp_path = DATA_ROOT / "videos" / f".temp_{uploaded_file.name}"
                    temp_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())

                    # Validate
                    result = validate_video(temp_path)

                    # Store validation result in session state
                    st.session_state[validation_key] = result.is_valid
                    st.session_state[f"validation_result_{episode_id}"] = result

                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)

            # Show validation results if validated
            if st.session_state.get(validation_key):
                result = st.session_state.get(f"validation_result_{episode_id}")

                if result and result.is_valid:
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

            # Upload button - OUTSIDE validation button block
            if st.session_state.get(validation_key):
                if st.button("Start Upload", type="primary", key=wkey("upload_start")):
                    with st.spinner("Uploading..."):
                        try:
                            # Create upload session
                            session = upload_manager.create_upload_session(
                                filename=uploaded_file.name,
                                total_size_bytes=len(uploaded_file.getvalue()),
                                show_id=canonical_show_slug(selected_show_name),
                                season_number=selected_season_number,
                                episode_id=episode_id,
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
                                f"✅ Upload complete!\n\n"
                                f"**Episode:** `{selected_show_name.upper()} S{selected_season_number:02d} E{episode_number:02d}`\n\n"
                                f"**File:** `{episode_id}.mp4`\n\n"
                                f"**Location:** `data/videos/{canonical_show_slug(selected_show_name)}/s{selected_season_number:02d}/`"
                            )

                            # Phase 3 P1: Auto-extract frames after upload
                            st.info("🔄 Automatically extracting frames...")

                            from jobs.tasks.auto_extract import trigger_auto_extraction
                            from pathlib import Path

                            video_path = Path("data/videos") / canonical_show_slug(selected_show_name) / f"s{selected_season_number:02d}" / f"{episode_id}.mp4"

                            with st.spinner("Extracting frames... This may take a few minutes."):
                                try:
                                    extract_result = trigger_auto_extraction(
                                        episode_id=episode_id,
                                        video_path=video_path,
                                    )

                                    if extract_result.get("success"):
                                        st.success(
                                            f"✅ Frame extraction complete!\n\n"
                                            f"**Episode Key:** `{extract_result['episode_key']}`\n\n"
                                            f"You can now proceed to Workspace to prepare tracks."
                                        )

                                        # Store episode_id in session for redirect
                                        st.session_state["last_uploaded_episode"] = episode_id
                                    else:
                                        st.warning(
                                            f"⚠️ Frame extraction failed: {extract_result.get('error')}\n\n"
                                            f"You can retry from the Workspace page."
                                        )

                                except Exception as e:
                                    st.error(f"❌ Frame extraction error: {str(e)}")

                            # Clear validation state after successful upload
                            st.session_state[validation_key] = False

                        except Exception as e:
                            st.error(f"❌ Upload failed: {str(e)}")

    # Resume upload section
    st.markdown("---")
    st.subheader("3. Resume Upload")

    session_id_input = st.text_input(
        "Enter session ID to resume",
        help="Paste the session ID from a previous upload",
        key=wkey("upload_resume_session_id")
    )

    if session_id_input and st.button("Resume Upload", key=wkey("upload_resume")):
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


if __name__ == "__main__":
    render_upload_page()
