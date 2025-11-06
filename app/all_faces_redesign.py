"""
Redesigned All Faces view with horizontal row layout (one cluster per row).
"""

import streamlit as st
from pathlib import Path
import pandas as pd


def render_all_faces_grid_v2(
    clusters_data: dict,
    episode_id: str,
    state_mgr,
    cluster_mutator,
    DATA_ROOT: Path
):
    """Render All Faces view with one cluster per row (horizontal track scrollers)."""
    from app.lib.data import load_tracks
    from screentime.utils import get_video_path
    from app.season_cast_helpers import get_season_cast_dropdown_options

    # Header with toggles
    header_col1, header_col2, header_col3 = st.columns([3, 1, 1])
    with header_col1:
        st.subheader("All Faces / Clusters")
    with header_col2:
        if 'group_by_identity' not in st.session_state:
            st.session_state.group_by_identity = False
        group_by_identity = st.checkbox("Group by identity", value=st.session_state.group_by_identity, key="group_toggle")
        st.session_state.group_by_identity = group_by_identity
    with header_col3:
        if 'compact_mode' not in st.session_state:
            st.session_state.compact_mode = False
        compact_mode = st.checkbox("Compact", value=st.session_state.compact_mode, key="compact_toggle")
        st.session_state.compact_mode = compact_mode

    # Load tracks data first (needed for filtering)
    tracks_data = load_tracks(episode_id, DATA_ROOT)

    # Build visible clusters (filtered + optionally grouped)
    from app.lib.cluster_filtering import build_visible_clusters

    clusters = build_visible_clusters(
        clusters_data,
        tracks_data,
        episode_id,
        DATA_ROOT,
        group_by_identity=group_by_identity
    )

    if not clusters:
        st.info("No clusters remaining (all suppressed or empty).")
        return

    # Load video path
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    # Try to load picked_samples for confidence sorting
    picked_samples_path = DATA_ROOT / "harvest" / episode_id / "picked_samples.parquet"
    picked_samples_df = None
    if picked_samples_path.exists():
        picked_samples_df = pd.read_parquet(picked_samples_path)

    # Initialize track gallery modal state
    if 'track_gallery_open' not in st.session_state:
        st.session_state.track_gallery_open = False
        st.session_state.track_gallery_track_id = None
        st.session_state.track_gallery_cluster_id = None

    # Initialize pagination state for each cluster
    if 'cluster_page' not in st.session_state:
        st.session_state.cluster_page = {}

    # Tile sizes based on compact mode
    tile_width = 125 if compact_mode else 150
    tiles_per_page = 16 if compact_mode else 12

    # Render each cluster as a row
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        cluster_name = cluster.get('name', f'Cluster {cluster_id}')
        size = cluster['size']
        quality = cluster.get('quality_score', 0)
        is_grouped = cluster.get('is_grouped', False)
        num_clusters = cluster.get('num_clusters', 1)
        cluster_ids = cluster.get('cluster_ids', [cluster_id])
        all_locked = cluster.get('all_locked', False)

        st.markdown("---")

        # Header row
        header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([3, 1, 1, 1, 1])

        with header_col1:
            # Format cluster IDs for title
            from app.lib.cluster_filtering import format_cluster_ids
            cluster_id_str = format_cluster_ids(cluster_ids)

            # Show lock indicator if all clusters are locked
            conf = cluster.get('assignment_confidence', 0.0)
            if is_grouped and all_locked:
                st.markdown(f"### **{cluster_name}** 🔒 ({cluster_id_str})")
                st.caption("(pending merge on re-cluster)")
            elif is_grouped:
                st.markdown(f"### **{cluster_name}** ({cluster_id_str})")
            elif conf == 1.0:
                st.markdown(f"### **{cluster_name}** 🔒 ({cluster_id_str})")
            else:
                st.markdown(f"### **{cluster_name}** ({cluster_id_str})")

        with header_col2:
            # Disable Assign Name for grouped rows (already have same name)
            if is_grouped:
                st.caption("(grouped)")
            else:
                if st.button("Assign Name", key=f"assign_name_{cluster_id}"):
                    st.session_state[f'assigning_{cluster_id}'] = True

        with header_col3:
            if st.button(f"View Tracks ({size})", key=f"view_tracks_{cluster_id}"):
                if is_grouped:
                    # Navigate to Cast View for grouped rows
                    st.session_state.navigation_page = 'cast_view'
                    st.session_state.nav_cast_name = cluster_name
                else:
                    # Navigate to cluster gallery page
                    st.session_state.navigation_page = 'cluster_gallery'
                    st.session_state.nav_cluster_id = cluster_id
                st.rerun()

        with header_col4:
            # Disable DELETE for grouped rows (turn off grouping first)
            if is_grouped:
                st.caption("(disable grouping)")
            else:
                # Show confirmation for named clusters
                cluster_name_display = cluster.get('name', f'Cluster {cluster_id}')
                is_named = cluster.get('name') and cluster.get('name') != 'Unknown'

                if is_named:
                    button_label = f"🗑️ Delete {cluster_name_display}"
                    button_help = f"⚠️ WARNING: This will delete ALL {size} tracks for {cluster_name_display}"
                else:
                    button_label = "🗑️ Delete"
                    button_help = f"Remove entire cluster ({size} tracks)"

                if st.button(button_label, key=f"delete_cluster_{cluster_id}", help=button_help, type="secondary"):
                    from app.lib.episode_status import load_suppress_data, save_suppress_data
                    from app.lib.analytics_dirty import mark_analytics_dirty

                    suppress_data = load_suppress_data(episode_id, DATA_ROOT)

                    # Deduplicate before adding (defensive)
                    if cluster_id not in suppress_data.get('deleted_clusters', []):
                        suppress_data['deleted_clusters'].append(cluster_id)
                    suppress_data['deleted_tracks'].extend(cluster['track_ids'])
                    save_suppress_data(episode_id, DATA_ROOT, suppress_data)

                    # Mark analytics dirty
                    mark_analytics_dirty(episode_id, DATA_ROOT, reason="cluster deleted")

                    st.toast(f"✅ Cluster {cluster_id} suppressed (won't return after RE-CLUSTER)", icon="🗑️")
                    st.rerun()

        with header_col5:
            st.caption(f"Size: {size} | Q: {quality:.2f}")

        # Show assign name modal if requested
        if st.session_state.get(f'assigning_{cluster_id}', False):
            with st.expander("🏷️ Assign Identity", expanded=True):
                cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
                new_name = st.selectbox(
                    "Identity:",
                    cast_options,
                    key=f"assign_identity_{cluster_id}"
                )

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Confirm", key=f"confirm_assign_{cluster_id}"):
                        try:
                            from app.lib.analytics_dirty import mark_analytics_dirty
                            import json
                            from pathlib import Path

                            # Assign name with conf=1.0 (locked for consolidation)
                            cluster_mutator.assign_name(cluster_id, new_name)

                            # Also set assignment_confidence=1.0 directly
                            clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                            with open(clusters_path, 'r') as f:
                                clusters_file = json.load(f)

                            for c in clusters_file.get('clusters', []):
                                if c['cluster_id'] == cluster_id:
                                    c['assignment_confidence'] = 1.0
                                    break

                            with open(clusters_path, 'w') as f:
                                json.dump(clusters_file, f, indent=2)

                            # Mark analytics dirty
                            mark_analytics_dirty(episode_id, DATA_ROOT, reason="cluster assigned")

                            st.session_state[f'assigning_{cluster_id}'] = False
                            st.success(f"✅ Assigned to {new_name} (locked 🔒)")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Assignment failed: {e}")

                with col2:
                    if st.button("❌ Cancel", key=f"cancel_assign_{cluster_id}"):
                        st.session_state[f'assigning_{cluster_id}'] = False
                        st.rerun()

        # Horizontal track strip (150×200 tiles)
        track_ids = cluster['track_ids']

        if not tracks_data or not video_path.exists():
            st.caption(f"Tracks: {track_ids[:10]}...")
            continue

        # Get track confidence from cluster track_metrics (if available)
        track_metrics = {}
        for tm in cluster.get('track_metrics', []):
            track_id = tm.get('track_id')
            if track_id:
                track_metrics[track_id] = {
                    'p25': tm.get('track_conf_p25', 0.0),
                    'mean': tm.get('track_conf_mean', 0.0),
                    'min': tm.get('track_conf_min', 0.0),
                    'n_low': tm.get('n_frames_low', 0)
                }

        # Fallback: Sort by face confidence from picked_samples if track_metrics not available
        if not track_metrics and picked_samples_df is not None:
            for track_id in track_ids:
                track_samples = picked_samples_df[picked_samples_df['track_id'] == track_id]
                if len(track_samples) > 0:
                    # Use median face confidence as fallback
                    conf = track_samples['confidence'].median()
                    track_metrics[track_id] = {'p25': conf, 'mean': conf, 'min': conf, 'n_low': 0}

        # Sort by track_conf_p25 descending (highest confidence first)
        sorted_track_ids = sorted(track_ids, key=lambda tid: track_metrics.get(tid, {}).get('p25', 0.0), reverse=True)

        # Initialize page for this cluster
        if cluster_id not in st.session_state.cluster_page:
            st.session_state.cluster_page[cluster_id] = 0

        current_page = st.session_state.cluster_page[cluster_id]
        total_pages = (len(sorted_track_ids) + tiles_per_page - 1) // tiles_per_page

        # Get tracks for current page
        start_idx = current_page * tiles_per_page
        end_idx = min(start_idx + tiles_per_page, len(sorted_track_ids))
        display_track_ids = sorted_track_ids[start_idx:end_idx]

        # Navigation arrows
        if total_pages > 1:
            nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
            with nav_col1:
                if current_page > 0:
                    if st.button("◀", key=f"prev_{cluster_id}"):
                        st.session_state.cluster_page[cluster_id] -= 1
                        st.rerun()
            with nav_col2:
                st.caption(f"Page {current_page + 1} of {total_pages} · {len(sorted_track_ids)} tracks total")
            with nav_col3:
                if current_page < total_pages - 1:
                    if st.button("▶", key=f"next_{cluster_id}"):
                        st.session_state.cluster_page[cluster_id] += 1
                        st.rerun()

        # Create horizontal layout
        cols_per_row = 8 if compact_mode else 6
        cols = st.columns(cols_per_row)

        for idx, track_id in enumerate(display_track_ids):
            col_idx = idx % cols_per_row

            with cols[col_idx]:
                track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
                if not track:
                    continue

                # Get best frame (middle frame)
                frame_refs = track.get('frame_refs', [])
                if not frame_refs:
                    continue

                mid_idx = len(frame_refs) // 2
                frame_ref = frame_refs[mid_idx]

                # Generate thumbnail
                thumb_path = thumb_gen.generate_frame_thumbnail(
                    video_path,
                    frame_ref['frame_id'],
                    frame_ref['bbox'],
                    episode_id,
                    track_id
                )

                if thumb_path and thumb_path.exists():
                    # Display tile with appropriate width
                    st.image(str(thumb_path), width=tile_width)

                    # Track ID and confidence badge
                    metrics = track_metrics.get(track_id, {})
                    p25 = metrics.get('p25', 0.0)
                    mean = metrics.get('mean', 0.0)
                    min_conf = metrics.get('min', 0.0)
                    n_low = metrics.get('n_low', 0)

                    # Primary caption with p25
                    st.caption(f"**Track {track_id}** · p25={p25:.2f}")

                    # Tooltip with full metrics
                    if mean > 0 or min_conf > 0:
                        st.caption(f"mean={mean:.2f} · min={min_conf:.2f} · low={n_low}", help=f"Track confidence metrics:\n• p25 (25th percentile): {p25:.3f}\n• mean: {mean:.3f}\n• min: {min_conf:.3f}\n• frames with conf<0.55: {n_low}")

                    # Click to open track gallery page
                    if st.button("🔍 View", key=f"view_track_{cluster_id}_{track_id}", help=f"Track {track_id} · p25={p25:.2f}"):
                        # Navigate to track gallery page
                        st.session_state.navigation_page = 'track_gallery'
                        st.session_state.nav_track_id = track_id
                        st.session_state.nav_cluster_id = cluster_id
                        st.session_state.nav_track_list = sorted_track_ids
                        st.rerun()

    # All inline galleries removed - navigation now goes to dedicated pages


def render_track_gallery_modal(
    track_id: int,
    cluster_id: int,
    episode_id: str,
    tracks_data: dict,
    video_path: Path,
    thumb_gen,
    cluster_mutator,
    DATA_ROOT: Path,
    clusters_data: dict
):
    """Render Track Gallery modal for a single track."""
    from app.season_cast_helpers import get_season_cast_dropdown_options
    from app.lib.data import load_clusters

    st.markdown("---")

    # Header with navigation and close button
    header_col1, header_col2, header_col3, header_col4 = st.columns([1, 1, 3, 1])

    with header_col1:
        # Previous track button
        track_list = st.session_state.get('track_gallery_track_list', [])
        if track_list and track_id in track_list:
            current_idx = track_list.index(track_id)
            if current_idx > 0:
                if st.button("◀ Prev", key="prev_track"):
                    st.session_state.track_gallery_track_id = track_list[current_idx - 1]
                    st.rerun()

    with header_col2:
        # Next track button
        if track_list and track_id in track_list:
            current_idx = track_list.index(track_id)
            if current_idx < len(track_list) - 1:
                if st.button("Next ▶", key="next_track"):
                    st.session_state.track_gallery_track_id = track_list[current_idx + 1]
                    st.rerun()

    with header_col3:
        st.markdown(f"### 🖼️ Track {track_id} Gallery")

    with header_col4:
        if st.button("✕ Close", key="close_track_gallery"):
            st.session_state.track_gallery_open = False
            st.rerun()

    track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
    if not track:
        st.error(f"Track {track_id} not found")
        return

    # Quick "Move to..." action at top
    st.markdown("#### 🚀 Quick Move")
    quick_col1, quick_col2 = st.columns([3, 1])

    with quick_col1:
        cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
        quick_target = st.selectbox(
            "Move track to:",
            ["-- Select --"] + cast_options,
            key=f"quick_move_{track_id}"
        )

    with quick_col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("✅ Move", key=f"confirm_quick_move_{track_id}") and quick_target != "-- Select --":
            # Same logic as below, but at top for convenience
            try:
                from screentime.clustering.constraints import save_track_level_constraints

                constraint_info = {
                    'must_link_moved': [],
                    'cannot_link': [],
                    'target_identity': quick_target,
                    'track_ids': [track_id],
                    'show_id': 'rhobh',
                    'season_id': 's05',
                    'episode_id': episode_id
                }

                save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                if quick_target == "Unknown":
                    cluster_mutator.move_track(track_id, cluster_id, -1)
                else:
                    clusters = clusters_data.get('clusters', [])
                    target_cluster = next((c for c in clusters if c.get('name') == quick_target), None)

                    if target_cluster:
                        cluster_mutator.move_track(track_id, cluster_id, target_cluster['cluster_id'])
                    else:
                        new_clusters_data = cluster_mutator.move_track(track_id, cluster_id, -1)
                        new_cluster_id = max(c['cluster_id'] for c in new_clusters_data.get('clusters', []))
                        cluster_mutator.assign_name(new_cluster_id, quick_target)

                st.success(f"✅ Moved Track {track_id} → {quick_target}")
                st.session_state.track_gallery_open = False
                st.rerun()

            except Exception as e:
                st.error(f"Move failed: {e}")

    st.markdown("---")

    # Show 3-8 face chips
    frame_refs = track.get('frame_refs', [])
    sample_count = min(8, len(frame_refs))

    if sample_count > 0 and video_path.exists():
        # Sample evenly
        step = max(1, len(frame_refs) // sample_count)
        sample_refs = [frame_refs[i] for i in range(0, len(frame_refs), step)][:sample_count]

        # Display in grid
        cols = st.columns(min(sample_count, 4))

        for idx, frame_ref in enumerate(sample_refs):
            col_idx = idx % 4
            with cols[col_idx]:
                thumb_path = thumb_gen.generate_frame_thumbnail(
                    video_path,
                    frame_ref['frame_id'],
                    frame_ref['bbox'],
                    episode_id,
                    track_id
                )

                if thumb_path and thumb_path.exists():
                    st.image(str(thumb_path), width=150)
                    st.caption(f"Frame {frame_ref['frame_id']}")

    st.markdown("---")

    # Track-level assign
    st.markdown("#### Assign Track to Identity")

    cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
    target_identity = st.selectbox(
        "Move track to:",
        ["-- Select --"] + cast_options,
        key=f"track_assign_{track_id}"
    )

    # Handle "Add a Cast Member..." selection
    if target_identity == "Add a Cast Member...":
        st.markdown("##### Create New Cast Member")
        new_cast_name = st.text_input(
            "Cast Member Name:",
            key=f"new_cast_name_{track_id}",
            placeholder="e.g., LISA"
        )

        if st.button(f"Create & Assign", key=f"create_cast_{track_id}") and new_cast_name:
            try:
                # Create new cast member directory
                facebank_dir = DATA_ROOT / "facebank" / "rhobh" / "s05" / new_cast_name.upper()
                facebank_dir.mkdir(parents=True, exist_ok=True)

                # Create empty seeds metadata
                seeds_metadata_path = facebank_dir / "seeds_metadata.json"
                if not seeds_metadata_path.exists():
                    import json
                    with open(seeds_metadata_path, 'w') as f:
                        json.dump({'seeds': []}, f, indent=2)

                st.success(f"✅ Created new cast member: {new_cast_name.upper()}")
                st.info("You can add seed images for this cast member in the Cast Images page")

                # Now assign the track
                target_identity = new_cast_name.upper()

                # Save track-level constraint
                from screentime.clustering.constraints import save_track_level_constraints

                constraint_info = {
                    'must_link_moved': [],
                    'cannot_link': [],
                    'target_identity': target_identity,
                    'track_ids': [track_id]
                }

                save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                # Find or create cluster for this identity
                clusters = clusters_data.get('clusters', [])
                target_cluster = next((c for c in clusters if c.get('name') == target_identity), None)

                if target_cluster:
                    # Move track to existing cluster
                    cluster_mutator.move_track(track_id, cluster_id, target_cluster['cluster_id'])
                else:
                    # Create new cluster by moving to -1, then assign name
                    cluster_mutator.move_track(track_id, cluster_id, -1)
                    # Reload to get new cluster ID
                    clusters_data = cluster_mutator.assign_name(
                        max(c['cluster_id'] for c in clusters) + 1,
                        target_identity
                    )

                st.success(f"✅ Assigned Track {track_id} → {target_identity}")
                st.session_state.track_gallery_open = False
                st.rerun()

            except Exception as e:
                st.error(f"Failed to create cast member: {e}")

    elif target_identity != "-- Select --":
        if st.button(f"✅ Assign Track {track_id}", key=f"confirm_track_assign_{track_id}"):
            try:
                # Save track-level constraint
                from screentime.clustering.constraints import save_track_level_constraints

                constraint_info = {
                    'must_link_moved': [],  # Single track
                    'cannot_link': [],  # Will be computed if needed
                    'target_identity': target_identity,
                    'track_ids': [track_id]
                }

                save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                # Assign track
                if target_identity == "Unknown":
                    # Move to new unnamed cluster
                    cluster_mutator.move_track(track_id, cluster_id, -1)
                else:
                    # Find or create cluster for this identity
                    clusters = clusters_data.get('clusters', [])
                    target_cluster = next((c for c in clusters if c.get('name') == target_identity), None)

                    if target_cluster:
                        # Move track to existing cluster
                        cluster_mutator.move_track(track_id, cluster_id, target_cluster['cluster_id'])
                    else:
                        # Create new cluster by moving to -1, then assign name
                        new_clusters_data = cluster_mutator.move_track(track_id, cluster_id, -1)
                        new_cluster_id = max(c['cluster_id'] for c in new_clusters_data.get('clusters', []))
                        cluster_mutator.assign_name(new_cluster_id, target_identity)

                st.success(f"✅ Assigned Track {track_id} → {target_identity}")
                st.session_state.track_gallery_open = False
                st.rerun()

            except Exception as e:
                st.error(f"Assignment failed: {e}")
