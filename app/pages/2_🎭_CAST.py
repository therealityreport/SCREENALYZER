"""
CAST - Manage cast member reference images.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import cv2
import json
import os
import re
import tempfile
import streamlit as st
from dataclasses import asdict
from dotenv import load_dotenv
from PIL import Image

from app.utils.ui_keys import wkey, safe_rerun
from app.lib.facebank_meta import resolve_person_thumbnail, get_person_meta, save_person_meta
from screentime.models import get_registry
from screentime.image_utils import ImageNormalizer

load_dotenv()

st.set_page_config(
    page_title="CAST",
    page_icon="🎭",
    layout="wide",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))


def get_live_seed_count(show_id: str, season_id: str, person: str) -> int:
    """Get live seed count from filesystem."""
    person_dir = DATA_ROOT / "facebank" / show_id / season_id / person
    if not person_dir.exists():
        return 0
    return len(list(person_dir.glob("seed_*.png")))






def get_show_code(show_id: str, show_name: str) -> str:
    """Generate show code from show_id or show_name."""
    if show_id:
        return show_id.upper()
    # Fallback: acronym from name including short words like "of"
    words = re.findall(r"[A-Za-z0-9]+", show_name)
    return "".join(w[0].upper() for w in words) or "SHOW"


def get_season_code(season_label: str) -> str:
    """Extract season code from label (e.g., 'Season 5' -> 'S5', 's05' -> 'S5')."""
    match = re.search(r"(\d+)", season_label)
    if match:
        return f"S{match.group(1)}"
    return season_label.upper()


def render_person_gallery(show_id: str, season_id: str, person: str, registry):
    """Render full Person Gallery view with seeds, star controls, upload, and delete."""
    # Inject 4:5 thumbnail CSS - FILL frame with cover (no letterboxing)
    st.markdown("""
    <style>
    /* Force all thumbnails to FILL 4:5 frame */
    .tile-45 {
        width: 160px;
        height: 200px;
        overflow: hidden;
        border-radius: 8px;
        background: #f6f6f6;
        display: inline-block;
    }
    .tile-45 img {
        width: 100%;
        height: 100%;
        object-fit: cover !important;
        object-position: center !important;
        display: block;
    }
    /* Apply to all st.image containers */
    div[data-testid="stImage"] {
        width: 160px !important;
        height: 200px !important;
        overflow: hidden !important;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # Back button ABOVE name
    if st.button("← Back to CAST", key=wkey("cast", "back", person)):
        st.session_state.pop('viewing_person', None)
        safe_rerun()

    # Get cast directory
    cast_dir = registry.get_cast_dir(show_id, season_id, person)
    person_meta_path = cast_dir / "person_meta.json"

    # Load person metadata
    person_meta = {}
    if person_meta_path.exists():
        try:
            with open(person_meta_path, 'r') as f:
                person_meta = json.load(f)
        except:
            pass

    featured_seed = person_meta.get('featured_seed')

    # Get live seed count
    if not cast_dir.exists():
        st.warning("Cast directory not found")
        return

    seed_files = sorted(cast_dir.glob("seed_*.png"))
    seed_count = len(seed_files)

    # Resolve featured avatar path
    featured_avatar_path = None
    if featured_seed:
        featured_path = cast_dir / featured_seed
        if featured_path.exists():
            featured_avatar_path = str(featured_path)
    if not featured_avatar_path and seed_files:
        featured_avatar_path = str(seed_files[0])

    # ========================================
    # Header Layout: Name/Details (left 70%) + Avatar (right 30%)
    # ========================================
    header_col1, header_col2 = st.columns([7, 3])

    with header_col1:
        # Person name (no emoji)
        st.markdown(f"## {person}")
        st.caption(f"{seed_count} seed images")

        st.markdown("---")

        # ========================================
        # Person Details with Edit/Save UX
        # ========================================
        st.subheader("Person Details")

        # Initialize edit mode state
        edit_mode_key = wkey("cast", "details", "edit_mode", show_id, season_id, person)
        if edit_mode_key not in st.session_state:
            st.session_state[edit_mode_key] = False

        edit_mode = st.session_state[edit_mode_key]

        # Edit/Save buttons
        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if not edit_mode:
                if st.button("Edit", key=wkey("cast", "details", "edit", person)):
                    st.session_state[edit_mode_key] = True
                    safe_rerun()
            else:
                # Save button - only show in edit mode
                pass

        with btn_col2:
            if edit_mode:
                save_clicked = st.button("💾 Save", type="primary", key=wkey("cast", "details", "save", person))
                cancel_clicked = st.button("Cancel", key=wkey("cast", "details", "cancel", person))

                if cancel_clicked:
                    st.session_state[edit_mode_key] = False
                    safe_rerun()

        st.markdown("")  # Spacing

        # Full Name
        current_full_name = person_meta.get('full_name', '')
        if edit_mode:
            full_name = st.text_input(
                "Full Name",
                value=current_full_name,
                placeholder="e.g., Lisa Rinna",
                key=wkey("cast", "details", "full_name", person)
            )
        else:
            st.text_input("Full Name", value=current_full_name, disabled=True, key=wkey("cast", "details", "full_name_display", person))
            full_name = current_full_name

        # Screen Name / Nickname
        current_screen_name = person_meta.get('screen_name', '')
        if edit_mode:
            screen_name = st.text_input(
                "Screen Name / Nickname",
                value=current_screen_name,
                placeholder="e.g., RINNA",
                key=wkey("cast", "details", "screen_name", person)
            )
        else:
            st.text_input("Screen Name / Nickname", value=current_screen_name, disabled=True, key=wkey("cast", "details", "screen_name_display", person))
            screen_name = current_screen_name

        # Shows (multi-select)
        all_show_ids = [s.show_id for s in registry.shows]
        current_appears_in = person_meta.get('appears_in', [])
        current_selected_shows = [entry['show_id'] for entry in current_appears_in]

        if edit_mode:
            selected_show_ids = st.multiselect(
                "Appears in Shows",
                options=all_show_ids,
                default=current_selected_shows,
                key=wkey("cast", "details", "shows", person)
            )
        else:
            st.multiselect("Appears in Shows", options=all_show_ids, default=current_selected_shows, disabled=True, key=wkey("cast", "details", "shows_display", person))
            selected_show_ids = current_selected_shows

        # Seasons per selected show
        appears_in_data = []
        if edit_mode and selected_show_ids:
            st.caption("Select seasons for each show:")
            for s_id in selected_show_ids:
                show_obj = next((s for s in registry.shows if s.show_id == s_id), None)
                if show_obj:
                    # Get available seasons
                    available_seasons = [seas.season_id for seas in show_obj.seasons]

                    # Get current selections for this show
                    current_show_entry = next((e for e in current_appears_in if e['show_id'] == s_id), None)
                    current_seasons = current_show_entry['seasons'] if current_show_entry else []

                    selected_seasons = st.multiselect(
                        f"{s_id.upper()} seasons",
                        options=available_seasons,
                        default=current_seasons,
                        key=wkey("cast", "details", "seasons", person, s_id)
                    )

                    if selected_seasons:
                        appears_in_data.append({"show_id": s_id, "seasons": selected_seasons})
        elif not edit_mode:
            # Display only
            for entry in current_appears_in:
                st.caption(f"{entry['show_id'].upper()}: {', '.join(entry['seasons'])}")

        # Save logic
        if edit_mode and 'save_clicked' in locals() and save_clicked:
            # Validation
            if not full_name.strip():
                st.error("Full Name is required")
            else:
                # Update person_meta
                person_meta['full_name'] = full_name.strip()
                person_meta['screen_name'] = screen_name.strip()
                person_meta['appears_in'] = appears_in_data

                save_person_meta(show_id, season_id, person, person_meta)
                st.session_state[edit_mode_key] = False
                st.success("Person details saved")
                safe_rerun()

    with header_col2:
        # Featured avatar with star overlay
        st.caption("Featured Seed")
        if featured_avatar_path:
            from app.lib.facebank_meta import get_thumbnail
            avatar_thumb = get_thumbnail(Path(featured_avatar_path), size=(200, 250))
            st.image(avatar_thumb, use_column_width=True)
            if featured_seed:
                st.caption(f"⭐ {featured_seed}")
        else:
            st.markdown("No seeds yet")

    st.markdown("---")

    # Initialize per-person upload counter
    counter_key = f"upload_counter_{show_id}_{season_id}_{person}"
    if counter_key not in st.session_state:
        st.session_state[counter_key] = 0

    # Upload section
    with st.expander("📤 Upload New Seeds", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload Face Images",
            type=None,
            accept_multiple_files=True,
            help="Accepts: jpg, jpeg, png, webp, avif, heic. Auto-converts to PNG.",
            key=wkey("cast", "upload", person, st.session_state[counter_key])
        )

        if st.button("Process & Add", type="primary", disabled=not uploaded_files, key=wkey("cast", "upload_btn", person, st.session_state[counter_key])):
            if uploaded_files:
                with st.spinner(f"Processing {len(uploaded_files)} images..."):
                    # Initialize normalizer and face detector
                    normalizer = ImageNormalizer()
                    from screentime.detectors.face_retina import RetinaFaceDetector
                    detector = RetinaFaceDetector()

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
                                rejected.append((uploaded_file.name, f"Unsupported format: {file_ext}"))
                                continue

                            # Save to temp file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                                tmp.write(uploaded_file.getvalue())
                                tmp_path = Path(tmp.name)

                            # Normalize image first
                            existing_seeds = len(list(cast_dir.glob("seed_*.png")))
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
                                rejected.append((uploaded_file.name, f"{len(faces)} faces detected"))
                                normalized_path.unlink(missing_ok=True)
                                tmp_path.unlink(missing_ok=True)
                                continue

                            face = faces[0]
                            face_height = face['bbox'][3] - face['bbox'][1]

                            if face['confidence'] < 0.65:
                                rejected.append((uploaded_file.name, f"Low confidence ({face['confidence']:.2f})"))
                                normalized_path.unlink(missing_ok=True)
                                tmp_path.unlink(missing_ok=True)
                                continue

                            if face_height < 64:
                                rejected.append((uploaded_file.name, f"Face too small ({face_height:.0f}px)"))
                                normalized_path.unlink(missing_ok=True)
                                tmp_path.unlink(missing_ok=True)
                                continue

                            # Valid seed - rename to final name
                            final_path = cast_dir / f"seed_{existing_seeds + len(valid_seeds) + 1:03d}.png"
                            normalized_path.rename(final_path)

                            valid_seeds.append({
                                'path': str(final_path),
                                'original_name': uploaded_file.name,
                                'confidence': float(face['confidence']),
                                'face_height': float(face_height),
                                'metadata': asdict(metadata)
                            })

                            tmp_path.unlink(missing_ok=True)

                        except Exception as e:
                            rejected.append((uploaded_file.name, str(e)))

                        progress_bar.progress((file_idx + 1) / len(uploaded_files))

                    progress_bar.empty()
                    status_text.empty()

                    # Update seeds_metadata.json
                    if valid_seeds:
                        metadata_path = cast_dir / "seeds_metadata.json"
                        
                        # Load existing metadata
                        existing_meta = {'seeds': []}
                        if metadata_path.exists():
                            try:
                                with open(metadata_path, 'r') as f:
                                    existing_meta = json.load(f)
                            except:
                                pass
                        
                        # Append new seeds
                        existing_meta['seeds'].extend(valid_seeds)
                        
                        # Save atomically
                        tmp_path = cast_dir / "seeds_metadata.json.tmp"
                        with open(tmp_path, 'w') as f:
                            json.dump(existing_meta, f, indent=2)
                        tmp_path.replace(metadata_path)

                        st.success(f"✅ Added {len(valid_seeds)} valid seeds")
                    else:
                        st.error(f"❌ No valid seeds")

                    if rejected:
                        with st.expander(f"⚠️ Rejected {len(rejected)} images"):
                            for filename, reason in rejected:
                                st.text(f"• {filename}: {reason}")

                    # Increment counter to clear upload queue
                    st.session_state[counter_key] += 1
                    safe_rerun()

    st.markdown("---")

    # Check if editing a seed
    if 'editing_seed' in st.session_state and st.session_state.editing_seed:
        # Render fullscreen edit view with View/Crop mode flow
        from streamlit_cropper import st_cropper
        from datetime import datetime
        from app.lib.facebank_meta import calculate_seed_quality_metrics

        seed_name = st.session_state.editing_seed
        seed_path = cast_dir / seed_name

        # Initialize crop mode state
        crop_mode_state_key = f"crop_mode_{show_id}_{season_id}_{person}_{seed_name}"
        if crop_mode_state_key not in st.session_state:
            st.session_state[crop_mode_state_key] = False  # Default: View mode

        in_crop_mode = st.session_state[crop_mode_state_key]

        st.subheader(f"{'Crop' if in_crop_mode else 'View'} {seed_name}")

        # Load seeds_metadata if available
        metadata_path = cast_dir / "seeds_metadata.json"
        seeds_metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    seeds_metadata = json.load(f)
            except:
                pass

        # Calculate quality metrics
        metrics = calculate_seed_quality_metrics(seed_path, seeds_metadata)

        # Is this seed currently featured?
        is_featured = featured_seed is not None and seed_name == featured_seed

        # Toolbar
        toolbar_cols = st.columns([1, 1, 1.5, 1, 1])

        with toolbar_cols[0]:
            # Star/Unstar button
            star_label = "★ Featured" if is_featured else "⭐ Set Featured"
            if st.button(
                star_label,
                key=wkey("cast", "edit", "star", show_id, season_id, person, seed_name),
                disabled=is_featured,
                use_container_width=True
            ):
                person_meta['featured_seed'] = seed_name
                person_meta['featured_seed_path'] = str(seed_path)
                save_person_meta(show_id, season_id, person, person_meta)
                st.success(f"⭐ {seed_name} set as featured")
                safe_rerun()

        with toolbar_cols[1]:
            # Crop Image button (toggle)
            crop_btn_label = "📋 View Details" if in_crop_mode else "✂️ Crop Image"
            if st.button(
                crop_btn_label,
                key=wkey("cast", "edit", "toggle_crop", show_id, season_id, person, seed_name),
                use_container_width=True
            ):
                st.session_state[crop_mode_state_key] = not in_crop_mode
                safe_rerun()

        with toolbar_cols[2]:
            # Replace original checkbox (only enabled in crop mode)
            replace_original = st.checkbox(
                "Replace original",
                value=False,
                disabled=not in_crop_mode,
                help="If checked, overwrites the original seed. Otherwise, saves as new seed.",
                key=wkey("cast", "edit", "replace", show_id, season_id, person, seed_name)
            )

        with toolbar_cols[3]:
            # Save button (only enabled in crop mode)
            save_disabled = not in_crop_mode  # Always bool
            save_clicked = st.button(
                "💾 Save",
                type="primary",
                disabled=save_disabled,
                key=wkey("cast", "edit", "save", show_id, season_id, person, seed_name),
                use_container_width=True
            )

        with toolbar_cols[4]:
            # Cancel button
            if st.button(
                "✖️ Cancel",
                key=wkey("cast", "edit", "cancel", show_id, season_id, person, seed_name),
                use_container_width=True
            ):
                st.session_state.pop('editing_seed', None)
                st.session_state.pop(crop_mode_state_key, None)
                safe_rerun()

        st.markdown("---")

        if not in_crop_mode:
            # VIEW MODE: Show metadata only
            st.subheader("📊 Seed Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Filename", metrics['filename'])
                st.metric("Dimensions", f"{metrics['dimensions'][0]}×{metrics['dimensions'][1]}")
                st.metric("File Size", f"{metrics['file_size_kb']:.1f} KB")

            with col2:
                st.metric("Quality Score", f"{metrics['quality_score']:.1f}/100")
                st.metric("Sharpness", f"{metrics['sharpness_score']:.1f}")
                st.metric("Brightness", f"{metrics['brightness_mean']:.1f}")

            with col3:
                st.metric("Contrast", f"{metrics['contrast_std']:.1f}")
                if metrics['coverage_pct'] > 0:
                    st.metric("Face Coverage", f"{metrics['coverage_pct']:.1f}%")
                if metrics['face_bbox']:
                    st.metric("Face Box", f"{metrics['face_bbox'][2]}×{metrics['face_bbox'][3]}")

            # Show cache-busted preview image
            from app.lib.facebank_meta import get_thumbnail
            preview_thumb = get_thumbnail(seed_path, size=(400, 500))
            st.image(preview_thumb, caption="Preview", use_column_width=True)

        else:
            # CROP MODE: Show cropper UI
            st.subheader("🖼️ Crop Image")

            crop_mode = st.radio(
                "Aspect Ratio",
                options=["4:5 (default)", "1:1 (square)", "Free"],
                horizontal=True,
                key=wkey("cast", "edit", "aspect", show_id, season_id, person, seed_name)
            )

            # Map crop mode to aspect ratio
            if crop_mode == "4:5 (default)":
                aspect_ratio = (4, 5)
            elif crop_mode == "1:1 (square)":
                aspect_ratio = (1, 1)
            else:
                aspect_ratio = None

            # Load image for cropper
            img = Image.open(seed_path)

            # Cropper
            cropped_img = st_cropper(
                img,
                realtime_update=True,
                box_color='#0000FF',
                aspect_ratio=aspect_ratio,
                key=wkey("cast", "edit", "cropper", show_id, season_id, person, seed_name)
            )

            # Preview cropped image
            st.image(cropped_img, caption="Cropped Preview", width=200)

            # Handle Save button click
            if save_clicked:
                from app.lib.facebank_meta import compute_quality_from_image

                # Generate filename
                if replace_original:
                    output_path = seed_path
                else:
                    # Find next available seed ID
                    existing_seeds = sorted(cast_dir.glob("seed_*.png"))
                    next_id = len(existing_seeds) + 1
                    timestamp = datetime.now().strftime("%Y%m%d%H%M")
                    new_name = f"seed_{next_id:03d}_crop{timestamp}.png"
                    output_path = cast_dir / new_name

                # Save cropped image
                cropped_img.save(output_path)

                # Compute quality metrics from the SAVED cropped image
                saved_img = Image.open(output_path)
                quality_metrics = compute_quality_from_image(saved_img)

                # Update seeds_metadata.json
                if 'seeds' not in seeds_metadata:
                    seeds_metadata['seeds'] = []

                if replace_original:
                    # Update existing seed metadata
                    for seed in seeds_metadata['seeds']:
                        if seed_name in seed.get('path', ''):
                            seed['quality_score'] = quality_metrics['quality_score']
                            seed['sharpness'] = quality_metrics['sharpness']
                            seed['brightness'] = quality_metrics['brightness']
                            seed['contrast'] = quality_metrics['contrast']
                            break
                else:
                    # Add new seed metadata
                    seeds_metadata['seeds'].append({
                        'path': str(output_path),
                        'original_name': f"crop_of_{seed_name}",
                        'confidence': 0.0,
                        'face_height': 0,
                        'quality_score': quality_metrics['quality_score'],
                        'sharpness': quality_metrics['sharpness'],
                        'brightness': quality_metrics['brightness'],
                        'contrast': quality_metrics['contrast']
                    })

                # Save metadata atomically
                tmp_path = cast_dir / "seeds_metadata.json.tmp"
                with open(tmp_path, 'w') as f:
                    json.dump(seeds_metadata, f, indent=2)
                tmp_path.replace(metadata_path)

                # If replaced original and it was featured, keep it featured
                if replace_original and is_featured:
                    person_meta['featured_seed'] = seed_name
                    person_meta['featured_seed_path'] = str(output_path)
                    save_person_meta(show_id, season_id, person, person_meta)

                # Clear thumbnail cache to force reload
                st.cache_data.clear()

                st.success(f"✅ Saved as {output_path.name}")
                st.session_state.pop('editing_seed', None)
                st.session_state.pop(crop_mode_state_key, None)
                safe_rerun()

        return

    # Normal seed gallery view
    st.subheader("Seed Gallery")

    # Seed grid with star controls
    if cast_dir.exists():
        seed_files = list(cast_dir.glob("seed_*.png"))

        if seed_files:
            # Load seeds_metadata.json for quality scores
            metadata_path = cast_dir / "seeds_metadata.json"
            seeds_metadata = {}
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        seeds_metadata = json.load(f)
                except:
                    pass

            # Calculate quality metrics for each seed and build list with scores
            from app.lib.facebank_meta import calculate_seed_quality_metrics, get_thumbnail, compute_quality_from_image

            seed_data = []
            for seed_file in seed_files:
                # Check if we have metadata for this seed
                existing_meta = next((s for s in seeds_metadata.get('seeds', []) if seed_file.name in s.get('path', '')), None)

                if existing_meta and 'quality_score' in existing_meta:
                    # Use cached quality score from metadata
                    quality_score = existing_meta['quality_score']
                else:
                    # Compute quality from the actual saved file
                    img = Image.open(seed_file)
                    quality_metrics = compute_quality_from_image(img)
                    quality_score = quality_metrics['quality_score']

                    # Update metadata with computed quality
                    if existing_meta:
                        existing_meta['quality_score'] = quality_score
                    else:
                        seeds_metadata.setdefault('seeds', []).append({
                            'path': str(seed_file),
                            'quality_score': quality_score
                        })

                created_ts = seed_file.stat().st_ctime
                seed_data.append({
                    'path': seed_file,
                    'quality_score': quality_score,
                    'created_ts': created_ts
                })

            # Sort by quality_score (desc), then created_ts (desc) as tiebreaker
            seed_data.sort(key=lambda x: (x['quality_score'], x['created_ts']), reverse=True)

            # Create lookup for quality scores
            quality_lookup = {s['path'].name: s['quality_score'] for s in seed_data}
            seed_files = [s['path'] for s in seed_data]

            # Grid layout: 4 columns
            cols_per_row = 4
            for row_start in range(0, len(seed_files), cols_per_row):
                row_seeds = seed_files[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)

                for col_idx, seed_file in enumerate(row_seeds):
                    with cols[col_idx]:
                        # Star indicator at top
                        is_starred = featured_seed is not None and seed_file.name == featured_seed

                        # Show cache-busted thumbnail (160x200, 4:5 fill)
                        thumb = get_thumbnail(seed_file, size=(160, 200))
                        st.image(thumb, use_column_width=False)

                        # Quality badge and star indicator
                        quality_score = quality_lookup.get(seed_file.name, 0.0)
                        # Normalize to 0-1 scale (quality_score is 0-100)
                        quality_norm = quality_score / 100.0

                        badge_parts = []
                        if is_starred:
                            badge_parts.append("⭐ Featured")
                        badge_parts.append(f"Quality: {quality_norm:.2f}")

                        st.caption(" | ".join(badge_parts))

                        # Action buttons - 3 columns
                        btn_col1, btn_col2, btn_col3 = st.columns(3)

                        with btn_col1:
                            # Star button
                            star_label = "⭐" if not is_starred else "★"
                            if st.button(
                                star_label,
                                key=wkey("cast", "star", show_id, season_id, person, seed_file.stem),
                                help="Set as featured thumbnail",
                                disabled=is_starred
                            ):
                                # Update person_meta.json
                                person_meta['featured_seed'] = seed_file.name
                                person_meta['featured_seed_path'] = str(seed_file)
                                save_person_meta(show_id, season_id, person, person_meta)

                                st.success(f"⭐ {seed_file.name} set as featured")
                                safe_rerun()

                        with btn_col2:
                            # Edit button
                            if st.button("✏️", key=wkey("cast", "edit", show_id, season_id, person, seed_file.stem)):
                                st.session_state.editing_seed = seed_file.name
                                safe_rerun()

                        with btn_col3:
                            # Delete button
                            if st.button("🗑️", key=wkey("cast", "delete", show_id, season_id, person, seed_file.stem)):
                                seed_file.unlink()

                                # If deleted image was featured, clear featured status
                                if featured_seed and seed_file.name == featured_seed:
                                    person_meta.pop('featured_seed', None)
                                    person_meta.pop('featured_seed_path', None)
                                    save_person_meta(show_id, season_id, person, person_meta)

                                st.success(f"✅ Deleted {seed_file.name}")
                                safe_rerun()
        else:
            st.info("No seed images found")


def render_cast_page():
    """Render cast images upload page with Show/Season integration."""
    # Inject 4:5 thumbnail CSS - FILL frame with cover (no letterboxing)
    st.markdown("""
    <style>
    /* Force all thumbnails to FILL 4:5 frame */
    .tile-45 {
        width: 160px;
        height: 200px;
        overflow: hidden;
        border-radius: 8px;
        background: #f6f6f6;
        display: inline-block;
    }
    .tile-45 img {
        width: 100%;
        height: 100%;
        object-fit: cover !important;
        object-position: center !important;
        display: block;
    }
    /* Apply to all st.image containers */
    div[data-testid="stImage"] {
        width: 160px !important;
        height: 200px !important;
        overflow: hidden !important;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎭 CAST")

    # Get registry
    registry = get_registry()

    # Check if viewing a specific person
    if 'viewing_person' in st.session_state:
        person_info = st.session_state.viewing_person
        render_person_gallery(
            person_info['show_id'],
            person_info['season_id'],
            person_info['person'],
            registry
        )
        return

    st.markdown(
        """
        Manage cast member reference images. Click any cast member to view their gallery.
        """
    )

    # ========================================
    # 1. Select Show & Season
    # ========================================
    st.subheader("1. Select Show & Season")

    if not registry.shows:
        st.warning("No shows found. Please create a show first.")
        st.info("Go to Upload page to create a show and season")
        return

    # Show and Season selectors side by side
    col1, col2 = st.columns(2)

    with col1:
        # Show selector
        show_options = {show.show_name: show.show_id for show in registry.shows}

        # Default to RHOBH if available
        default_show_name = next((name for name, sid in show_options.items() if sid == "rhobh"), list(show_options.keys())[0])

        selected_show_name = st.selectbox(
            "Show",
            options=list(show_options.keys()),
            index=list(show_options.keys()).index(default_show_name),
            key=wkey("cast_show_select")
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

    with col2:
        # Season selector
        season_options = {s.season_label: s.season_id for s in selected_show.seasons}

        # Default to S05 if available
        default_season_label = next((label for label, sid in season_options.items() if sid == "s05"), list(season_options.keys())[0])

        selected_season_label = st.selectbox(
            "Season",
            options=list(season_options.keys()),
            index=list(season_options.keys()).index(default_season_label),
            key=wkey("cast_season_select")
        )
        selected_season_id = season_options[selected_season_label]
        selected_season = registry.get_season(selected_show_id, selected_season_id)

    # Guard against None
    if selected_season is None:
        st.error(f"Season {selected_season_id} not found in registry.")
        return

    # Show current episodes for this season (if any)
    if getattr(selected_season, 'episodes', None) and len(selected_season.episodes) > 0:
        episode_options = [ep.episode_id for ep in selected_season.episodes]

        # Episode selector
        st.selectbox(
            "📹 Episodes",
            options=episode_options,
            key=wkey("cast_episode_select"),
            help="Select an episode to view details"
        )

    # ========================================
    # 2. Add Cast Member
    # ========================================
    st.markdown("---")
    st.subheader("2. Add New Cast Member")

    # Initialize upload counter in session state (used to reset uploader)
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0

    cast_name = st.text_input(
        "Cast Name",
        placeholder="KIM",
        help="Use UPPERCASE for consistency (e.g., KIM, KYLE, LISA)",
        key=wkey("cast_name_input", st.session_state.upload_counter)
    )

    uploaded_files = st.file_uploader(
        "Upload Face Images (3-5 recommended)",
        type=None,
        accept_multiple_files=True,
        help="Accepts: jpg, jpeg, png, webp, avif, heic. Auto-converts to PNG.",
        key=wkey("cast_images_uploader", st.session_state.upload_counter)
    )

    if st.button("Process & Add to Season", type="primary", disabled=not cast_name or not uploaded_files, key=wkey("cast_process_add")):
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
                            rejected.append((uploaded_file.name, f"Unsupported format: {file_ext}"))
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
                            rejected.append((uploaded_file.name, f"{len(faces)} faces detected"))
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
                            rejected.append((uploaded_file.name, f"Face too small ({face_height:.0f}px)"))
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
                            'confidence': float(face['confidence']),
                            'face_height': float(face_height),
                            'metadata': asdict(metadata)
                        })

                        # Clean up temp
                        tmp_path.unlink(missing_ok=True)

                    except Exception as e:
                        rejected.append((uploaded_file.name, str(e)))
                        if 'normalized_path' in locals() and Path(normalized_path).exists():
                            Path(normalized_path).unlink(missing_ok=True)
                        if 'tmp_path' in locals() and Path(tmp_path).exists():
                            Path(tmp_path).unlink(missing_ok=True)

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
                safe_rerun()




    # ========================================
    # 3. Cast of Show Season
    # ========================================
    st.markdown("---")
    # Generate dynamic codes for cleaner display
    show_code = get_show_code(selected_show_id, selected_show_name)
    season_code = get_season_code(selected_season_label)
    st.subheader(f"3. Cast of {show_code} {season_code}")

    if getattr(selected_season, 'cast', None):
        # Grid layout: 5 cards per row
        cols_per_row = 5
        cast_list = list(selected_season.cast)

        for row_start in range(0, len(cast_list), cols_per_row):
            row_cast = cast_list[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)

            for col_idx, cast in enumerate(row_cast):
                with cols[col_idx]:
                    # Get live seed count from filesystem
                    live_seed_count = get_live_seed_count(selected_show_id, selected_season_id, cast.name)
                    
                    # Resolve thumbnail
                    thumbnail_path = resolve_person_thumbnail(
                        selected_show_id,
                        selected_season_id,
                        cast.name
                    )

                    # Clickable thumbnail container with 4:5 fill
                    if thumbnail_path:
                        # Display cache-busted 4:5 thumbnail
                        from app.lib.facebank_meta import get_thumbnail
                        thumb = get_thumbnail(Path(thumbnail_path), size=(160, 200))
                        st.image(thumb, use_column_width=False)
                    else:
                        # Placeholder with 4:5 dimensions
                        st.markdown(
                            '<div style="width: 160px; height: 200px; display: flex; align-items: center; justify-content: center; background: #f6f6f6; border-radius: 8px;">'
                            '<div style="font-size: 64px;">👤</div>'
                            '</div>',
                            unsafe_allow_html=True
                        )

                    # Name and status
                    status = "✅" if live_seed_count >= 3 else "🚩"
                    st.markdown(f"**{cast.name}** {status}")
                    st.caption(f"{live_seed_count} seeds")

                    # Click to open gallery
                    if st.button(
                        "View Gallery",
                        key=wkey("cast", "card", selected_show_id, selected_season_id, cast.name),
                        use_container_width=True
                    ):
                        st.session_state.viewing_person = {
                            'show_id': selected_show_id,
                            'season_id': selected_season_id,
                            'person': cast.name
                        }
                        safe_rerun()
    else:
        st.info("No cast members added yet.")

if __name__ == "__main__":
    render_cast_page()
