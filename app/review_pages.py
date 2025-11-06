"""
Dedicated page renderers for Review navigation.

Routes:
- Cluster Gallery: episode + cluster_id
- Track Gallery: episode + track_id
- Cast View: episode + cast_name
"""

import streamlit as st
from pathlib import Path
import pandas as pd


def render_cluster_gallery_page(episode_id: str, cluster_id: int, DATA_ROOT: Path):
    """
    Render dedicated Cluster Gallery page.

    Shows all tracks in a cluster with grid view and delete options.
    """
    from app.lib.data import load_clusters, load_tracks
    from screentime.utils import get_video_path
    from app.lib.episode_status import load_suppress_data, save_suppress_data

    # Load data
    clusters_data = load_clusters(episode_id, DATA_ROOT)
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    if not clusters_data:
        st.error("Cluster data not found")
        return

    # Find cluster
    cluster = next((c for c in clusters_data.get('clusters', []) if c['cluster_id'] == cluster_id), None)

    if not cluster:
        st.error(f"Cluster {cluster_id} not found")
        if st.button("← Back to Review"):
            st.session_state.navigation_page = None
            st.rerun()
        return

    # Breadcrumb
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Review** > All Faces > Cluster {cluster_id}")
    with col2:
        if st.button("← Back to Review", key="back_btn"):
            st.session_state.navigation_page = None
            st.rerun()

    st.markdown("---")

    # Header
    cluster_name = cluster.get('name', f'Cluster {cluster_id}')
    size = cluster['size']
    quality = cluster.get('quality_score', 0)
    conf = cluster.get('assignment_confidence', 0.0)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        # Show lock pill if approved
        if conf == 1.0:
            st.markdown(f"## {cluster_name} 🔒")
        else:
            st.markdown(f"## {cluster_name}")
    with col2:
        st.metric("Size", size)
        st.metric("Quality", f"{quality:.2f}")
    with col3:
        # Delete cluster button with warning for named clusters
        cluster_name_display = cluster.get('name', f'Cluster {cluster_id}')
        is_named = cluster.get('name') and cluster.get('name') != 'Unknown'

        if is_named:
            button_label = f"🗑️ Delete {cluster_name_display}"
            button_help = f"⚠️ WARNING: This will delete ALL {size} tracks for {cluster_name_display}"
        else:
            button_label = "🗑️ Delete Cluster"
            button_help = f"Remove entire cluster ({size} tracks)"

        if st.button(button_label, key=f"delete_cluster_{cluster_id}", help=button_help, type="secondary"):
            suppress_data = load_suppress_data(episode_id, DATA_ROOT)

            # Deduplicate before adding (defensive)
            if cluster_id not in suppress_data.get('deleted_clusters', []):
                suppress_data['deleted_clusters'].append(cluster_id)
            suppress_data['deleted_tracks'].extend(cluster.get('track_ids', []))
            save_suppress_data(episode_id, DATA_ROOT, suppress_data)

            st.success(f"✅ Cluster {cluster_id} deleted")
            st.info("Run RE-CLUSTER to finalize deletion")

            # Return to review
            st.session_state.navigation_page = None
            st.rerun()

    st.markdown("---")

    # Show all tracks in row-per-track layout (like Cast View)
    track_ids = cluster.get('track_ids', [])

    if not track_ids:
        st.info("No tracks in this cluster")
        return

    if not tracks_data or not video_path.exists():
        st.error("Track data or video not found")
        return

    # Compact mode toggle
    compact_mode = st.checkbox("Compact View", value=st.session_state.get('compact_mode', False))
    st.session_state.compact_mode = compact_mode

    tile_width = 125 if compact_mode else 150
    tiles_per_row = 8 if compact_mode else 6

    # Get cast options for splitting
    from app.season_cast_helpers import get_season_cast_dropdown_options
    from screentime.clustering.constraints import save_track_level_constraints
    cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)

    # Display tracks - one row per track with all frames
    st.markdown(f"### Tracks ({len(track_ids)})")

    # Render each track as a row
    for track_id in track_ids:
        track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
        if not track:
            continue

        frame_refs = track.get('frame_refs', [])
        if not frame_refs:
            continue

        st.markdown("---")

        # Track header
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        with header_col1:
            st.markdown(f"### Track {track_id}")
        with header_col2:
            st.caption(f"{len(frame_refs)} frames")
        with header_col3:
            # View detail button
            if st.button("🔍 View Detail", key=f"view_detail_{track_id}"):
                st.session_state.navigation_page = 'track_gallery'
                st.session_state.nav_track_id = track_id
                st.session_state.nav_cluster_id = cluster_id
                st.session_state.nav_track_list = track_ids
                st.rerun()

        # Frame selection controls
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 3])
        with sel_col1:
            if st.button("Select All", key=f"select_all_{track_id}"):
                st.session_state[f'selected_frames_cluster_{track_id}'] = set(range(len(frame_refs)))
                st.rerun()
        with sel_col2:
            if st.button("Select None", key=f"select_none_{track_id}"):
                st.session_state[f'selected_frames_cluster_{track_id}'] = set()
                st.rerun()
        with sel_col3:
            # Initialize selection state
            if f'selected_frames_cluster_{track_id}' not in st.session_state:
                st.session_state[f'selected_frames_cluster_{track_id}'] = set()

            selected_count = len(st.session_state[f'selected_frames_cluster_{track_id}'])
            if selected_count > 0:
                st.caption(f"✓ {selected_count} of {len(frame_refs)} frames selected")

        # Pagination for large tracks
        frames_per_page = 24
        total_pages = (len(frame_refs) + frames_per_page - 1) // frames_per_page

        if f'cluster_frame_page_{track_id}' not in st.session_state:
            st.session_state[f'cluster_frame_page_{track_id}'] = 0

        current_page = st.session_state[f'cluster_frame_page_{track_id}']

        if total_pages > 1:
            page_col1, page_col2, page_col3 = st.columns([1, 3, 1])
            with page_col1:
                if current_page > 0:
                    if st.button("◀", key=f"cluster_prev_{track_id}"):
                        st.session_state[f'cluster_frame_page_{track_id}'] -= 1
                        st.rerun()
            with page_col2:
                st.caption(f"Page {current_page + 1}/{total_pages}")
            with page_col3:
                if current_page < total_pages - 1:
                    if st.button("▶", key=f"cluster_next_{track_id}"):
                        st.session_state[f'cluster_frame_page_{track_id}'] += 1
                        st.rerun()

        start_idx = current_page * frames_per_page
        end_idx = min(start_idx + frames_per_page, len(frame_refs))
        page_frames = frame_refs[start_idx:end_idx]

        # Use form to batch selections
        with st.form(key=f"cluster_selection_form_{track_id}"):
            # Display frames in rows with checkboxes
            for row_start in range(0, len(page_frames), tiles_per_row):
                cols = st.columns(tiles_per_row)
                for col_idx in range(tiles_per_row):
                    frame_idx_in_page = row_start + col_idx
                    if frame_idx_in_page >= len(page_frames):
                        break

                    frame_idx = start_idx + frame_idx_in_page
                    frame_ref = page_frames[frame_idx_in_page]

                    with cols[col_idx]:
                        thumb_path = thumb_gen.generate_frame_thumbnail(
                            video_path,
                            frame_ref['frame_id'],
                            frame_ref['bbox'],
                            episode_id,
                            track_id
                        )

                        if thumb_path and thumb_path.exists():
                            st.image(str(thumb_path), width=tile_width)

                            # Checkbox for selection (inside form)
                            is_selected = frame_idx in st.session_state[f'selected_frames_cluster_{track_id}']
                            st.checkbox(
                                f"F{frame_ref['frame_id']}",
                                value=is_selected,
                                key=f"cluster_frame_{track_id}_{frame_idx}"
                            )

            # Form submit
            submitted = st.form_submit_button("✓ Update Selection")
            if submitted:
                new_selection = set()
                for frame_idx_in_page in range(len(page_frames)):
                    frame_idx = start_idx + frame_idx_in_page
                    if st.session_state.get(f"cluster_frame_{track_id}_{frame_idx}", False):
                        new_selection.add(frame_idx)
                st.session_state[f'selected_frames_cluster_{track_id}'] = new_selection
                st.rerun()

        # Split and Delete sections (if frames selected)
        if selected_count > 0:
            split_del_col1, split_del_col2 = st.columns(2)

            with split_del_col1:
                st.markdown(f"**🔀 Split {selected_count} frame(s)**")
                split_col1, split_col2 = st.columns([3, 1])
                with split_col1:
                    split_target = st.selectbox(
                        "Assign to:",
                        ["-- Select --"] + cast_options,
                        key=f"split_target_cluster_{track_id}"
                    )
                with split_col2:
                    st.write("")
                    if st.button("✅ Split", key=f"split_cluster_{track_id}") and split_target != "-- Select --":
                        try:
                            # Get selected frame indices
                            selected_indices = sorted(st.session_state[f'selected_frames_cluster_{track_id}'])
                            selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                            remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                            if len(remaining_indices) == 0:
                                # 100% of frames selected - offer Move Whole Track instead
                                st.warning(f"⚠️ All {len(frame_refs)} frames selected in Track {track_id}")
                                st.info(f"💡 Move the entire track to **{split_target}**?")

                                if st.button(f"✅ Move Track {track_id} to {split_target}", key=f"move_whole_cluster_{track_id}"):
                                    try:
                                        # Use assign_tracks_to_identity to move whole track
                                        result = cluster_mutator.assign_tracks_to_identity(
                                            [track_id],
                                            split_target,
                                            source_cluster_id=cluster_id
                                        )

                                        st.success(f"✅ Moved Track {track_id} to {split_target} (cluster {result['new_cluster_id']})")
                                        if result.get('orphans', 0) > 0:
                                            st.info(f"Note: {result['orphans']} orphan track(s) handled")

                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Move failed: {str(e)}")
                            else:
                                # Use same split logic as Cast View
                                import json
                                import shutil

                                # Create new track
                                new_track_id = max([t['track_id'] for t in tracks_data.get('tracks', [])]) + 1
                                new_track = {
                                    'track_id': new_track_id,
                                    'frame_refs': [frame_refs[i] for i in selected_indices]
                                }

                                # Update current track
                                updated_track = track.copy()
                                updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                                # Save tracks
                                tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]
                                if len(updated_track['frame_refs']) > 0:
                                    tracks_list.append(updated_track)
                                tracks_list.append(new_track)

                                tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                                if tracks_path.exists():
                                    shutil.copy(tracks_path, tracks_path.with_suffix('.json.bak'))

                                with open(tracks_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_tracks': len(tracks_list),
                                        'tracks': tracks_list
                                    }, f, indent=2)

                                # Update clusters
                                clusters = clusters_data.get('clusters', [])
                                target_cluster = next((c for c in clusters if c.get('name') == split_target), None)

                                if target_cluster:
                                    target_cluster['track_ids'].append(new_track_id)
                                    target_cluster['size'] = len(target_cluster['track_ids'])
                                else:
                                    new_cluster_id = max([c['cluster_id'] for c in clusters]) + 1 if clusters else 0
                                    clusters.append({
                                        'cluster_id': new_cluster_id,
                                        'size': 1,
                                        'track_ids': [new_track_id],
                                        'name': split_target,
                                        'assignment_confidence': 1.0,
                                        'quality_score': 0.0,
                                        'variance': 0.0,
                                        'silhouette_score': 0.0,
                                        'is_lowconf': False
                                    })

                                # Save clusters
                                clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                                if clusters_path.exists():
                                    shutil.copy(clusters_path, clusters_path.with_suffix('.json.bak'))

                                with open(clusters_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_clusters': len(clusters),
                                        'noise_tracks': clusters_data.get('noise_tracks', 0),
                                        'clusters': clusters
                                    }, f, indent=2)

                                # Save constraints
                                save_track_level_constraints(episode_id, DATA_ROOT, {
                                    'must_link_moved': [],
                                    'cannot_link': [],
                                    'target_identity': split_target,
                                    'track_ids': [new_track_id],
                                    'source_track_id': track_id,
                                    'selected_frame_ids': selected_frame_ids,
                                    'show_id': 'rhobh',
                                    'season_id': 's05',
                                    'episode_id': episode_id,
                                    'action': 'frame_split_cluster_view'
                                })

                                # Mark analytics dirty
                                from app.lib.analytics_dirty import mark_analytics_dirty
                                mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames split")

                                st.success(f"✅ Created Track {new_track_id} with {selected_count} frames → {split_target}")
                                st.rerun()

                        except Exception as e:
                            st.error(f"Split failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            with split_del_col2:
                st.markdown(f"**🗑️ Delete {selected_count} frame(s)**")
                st.caption("Permanently remove selected frames from track")
                if st.button("🗑️ Delete Frames", key=f"delete_frames_cluster_{track_id}"):
                    try:
                        # Get remaining frame indices
                        selected_indices = sorted(st.session_state[f'selected_frames_cluster_{track_id}'])
                        selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                        remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                        import json
                        import shutil

                        if len(remaining_indices) == 0:
                            # Delete entire track
                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]

                            # Also remove from cluster
                            current_cluster = next((c for c in clusters_data.get('clusters', []) if c['cluster_id'] == cluster_id), None)
                            if current_cluster:
                                current_cluster['track_ids'] = [tid for tid in current_cluster['track_ids'] if tid != track_id]
                                current_cluster['size'] = len(current_cluster['track_ids'])

                            st.info(f"All frames deleted - Track {track_id} removed entirely")
                        else:
                            # Update track to keep only remaining frames
                            updated_track = track.copy()
                            updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]
                            tracks_list.append(updated_track)

                        # Save tracks
                        tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                        if tracks_path.exists():
                            shutil.copy(tracks_path, tracks_path.with_suffix('.json.bak'))

                        with open(tracks_path, 'w') as f:
                            json.dump({
                                'episode_id': episode_id,
                                'total_tracks': len(tracks_list),
                                'tracks': tracks_list
                            }, f, indent=2)

                        # Save clusters if track was removed
                        if len(remaining_indices) == 0:
                            clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                            if clusters_path.exists():
                                shutil.copy(clusters_path, clusters_path.with_suffix('.json.bak'))

                            with open(clusters_path, 'w') as f:
                                json.dump({
                                    'episode_id': episode_id,
                                    'total_clusters': len(clusters_data.get('clusters', [])),
                                    'noise_tracks': clusters_data.get('noise_tracks', 0),
                                    'clusters': clusters_data.get('clusters', [])
                                }, f, indent=2)

                        # Log deletion
                        save_track_level_constraints(episode_id, DATA_ROOT, {
                            'must_link_moved': [],
                            'cannot_link': [],
                            'target_identity': 'DELETED',
                            'track_ids': [track_id],
                            'deleted_frame_ids': selected_frame_ids,
                            'show_id': 'rhobh',
                            'season_id': 's05',
                            'episode_id': episode_id,
                            'action': 'frame_delete_cluster_view'
                        })

                        # Mark analytics dirty
                        from app.lib.analytics_dirty import mark_analytics_dirty
                        mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames deleted")

                        st.success(f"✅ Deleted {selected_count} frames from Track {track_id}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


def render_track_gallery_page(episode_id: str, track_id: int, cluster_id: int = None, DATA_ROOT: Path = None):
    """
    Render dedicated Track Gallery page.

    Shows 3-8 face chips for a track with navigation and assignment options.
    """
    from app.lib.data import load_tracks, load_clusters
    from screentime.utils import get_video_path
    from app.season_cast_helpers import get_season_cast_dropdown_options
    from app.lib.cluster_mutations import ClusterMutator
    from screentime.clustering.constraints import save_track_level_constraints

    # Load data
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    clusters_data = load_clusters(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator
    cluster_mutator = st.session_state.cluster_mutator

    if not tracks_data:
        st.error("Track data not found")
        return

    track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
    if not track:
        st.error(f"Track {track_id} not found")
        if st.button("← Back"):
            st.session_state.navigation_page = 'cluster_gallery' if cluster_id else None
            st.rerun()
        return

    # Breadcrumb
    col1, col2 = st.columns([4, 1])
    with col1:
        if cluster_id:
            st.markdown(f"**Review** > All Faces > Cluster {cluster_id} > Track {track_id}")
        else:
            st.markdown(f"**Review** > Track {track_id}")
    with col2:
        if st.button("← Back", key="back_btn"):
            if cluster_id:
                st.session_state.navigation_page = 'cluster_gallery'
                st.session_state.nav_cluster_id = cluster_id
            else:
                st.session_state.navigation_page = None
            st.rerun()

    st.markdown("---")

    # Navigation between tracks (if track_list available)
    track_list = st.session_state.get('nav_track_list', [])
    if track_list and track_id in track_list:
        current_idx = track_list.index(track_id)

        nav_col1, nav_col2, nav_col3, nav_col4 = st.columns([1, 1, 3, 1])
        with nav_col1:
            if current_idx > 0:
                if st.button("◀ Prev"):
                    st.session_state.nav_track_id = track_list[current_idx - 1]
                    st.rerun()
        with nav_col2:
            if current_idx < len(track_list) - 1:
                if st.button("Next ▶"):
                    st.session_state.nav_track_id = track_list[current_idx + 1]
                    st.rerun()
        with nav_col3:
            st.markdown(f"### Track {track_id} ({current_idx + 1} of {len(track_list)})")
    else:
        st.markdown(f"### Track {track_id}")

    st.markdown("---")

    # Quick Move at top
    st.markdown("#### 🚀 Quick Move")
    move_col1, move_col2 = st.columns([3, 1])

    with move_col1:
        cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
        target = st.selectbox(
            "Move track to:",
            ["-- Select --"] + cast_options,
            key=f"quick_move_{track_id}"
        )

    with move_col2:
        st.write("")
        st.write("")
        if st.button("✅ Move", key=f"confirm_move_{track_id}") and target != "-- Select --":
            try:
                # Save constraints
                constraint_info = {
                    'must_link_moved': [],
                    'cannot_link': [],
                    'target_identity': target,
                    'track_ids': [track_id],
                    'show_id': 'rhobh',
                    'season_id': 's05',
                    'episode_id': episode_id
                }
                save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                # Move track
                if target == "Unknown":
                    cluster_mutator.move_track(track_id, cluster_id, -1)
                else:
                    clusters = clusters_data.get('clusters', [])
                    target_cluster = next((c for c in clusters if c.get('name') == target), None)

                    if target_cluster:
                        cluster_mutator.move_track(track_id, cluster_id, target_cluster['cluster_id'])
                    else:
                        new_clusters_data = cluster_mutator.move_track(track_id, cluster_id, -1)
                        new_cluster_id = max(c['cluster_id'] for c in new_clusters_data.get('clusters', []))
                        cluster_mutator.assign_name(new_cluster_id, target)

                # Mark analytics dirty
                from app.lib.analytics_dirty import mark_analytics_dirty
                mark_analytics_dirty(episode_id, DATA_ROOT, reason="track moved")

                st.success(f"✅ Moved Track {track_id} → {target}")
                # Return to cluster gallery
                if cluster_id:
                    st.session_state.navigation_page = 'cluster_gallery'
                else:
                    st.session_state.navigation_page = None
                st.rerun()

            except Exception as e:
                st.error(f"Move failed: {e}")

    st.markdown("---")

    # Show ALL face chips with selection capability
    frame_refs = track.get('frame_refs', [])

    if len(frame_refs) > 0 and video_path.exists():
        st.markdown("#### Frame Selection")

        # Selection controls
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 3])
        with sel_col1:
            if st.button("Select All", key=f"select_all_frames_{track_id}"):
                st.session_state[f'selected_frames_{track_id}'] = set(range(len(frame_refs)))
                st.rerun()
        with sel_col2:
            if st.button("Select None", key=f"select_none_frames_{track_id}"):
                st.session_state[f'selected_frames_{track_id}'] = set()
                st.rerun()
        with sel_col3:
            # Initialize selection state
            if f'selected_frames_{track_id}' not in st.session_state:
                st.session_state[f'selected_frames_{track_id}'] = set()

            selected_count = len(st.session_state[f'selected_frames_{track_id}'])
            st.caption(f"{selected_count} of {len(frame_refs)} frames selected")

        st.markdown("---")

        # Display frames with pagination to improve performance
        frames_per_page = 20
        total_pages = (len(frame_refs) + frames_per_page - 1) // frames_per_page

        # Page navigation
        if f'frame_page_{track_id}' not in st.session_state:
            st.session_state[f'frame_page_{track_id}'] = 0

        current_page = st.session_state[f'frame_page_{track_id}']

        if total_pages > 1:
            page_col1, page_col2, page_col3 = st.columns([1, 3, 1])
            with page_col1:
                if current_page > 0:
                    if st.button("◀ Prev", key=f"prev_page_{track_id}"):
                        st.session_state[f'frame_page_{track_id}'] -= 1
                        st.rerun()
            with page_col2:
                st.caption(f"Page {current_page + 1} of {total_pages} · Showing frames {current_page * frames_per_page + 1}-{min((current_page + 1) * frames_per_page, len(frame_refs))} of {len(frame_refs)}")
            with page_col3:
                if current_page < total_pages - 1:
                    if st.button("Next ▶", key=f"next_page_{track_id}"):
                        st.session_state[f'frame_page_{track_id}'] += 1
                        st.rerun()

        # Get frames for current page
        start_idx = current_page * frames_per_page
        end_idx = min(start_idx + frames_per_page, len(frame_refs))
        page_frames = frame_refs[start_idx:end_idx]

        # Use form to batch checkbox selections (prevents rerun on each click)
        with st.form(key=f"frame_selection_form_{track_id}"):
            # Display frames in grid with checkboxes (4 per row)
            for row_start in range(0, len(page_frames), 4):
                cols = st.columns(4)
                for col_idx in range(4):
                    frame_idx_in_page = row_start + col_idx
                    if frame_idx_in_page >= len(page_frames):
                        break

                    frame_idx = start_idx + frame_idx_in_page
                    frame_ref = page_frames[frame_idx_in_page]

                    with cols[col_idx]:
                        # Generate thumbnail
                        thumb_path = thumb_gen.generate_frame_thumbnail(
                            video_path,
                            frame_ref['frame_id'],
                            frame_ref['bbox'],
                            episode_id,
                            track_id
                        )

                        if thumb_path and thumb_path.exists():
                            # Display image
                            st.image(str(thumb_path), width=150)

                            # Checkbox for selection (inside form - no rerun until submit)
                            is_selected = frame_idx in st.session_state[f'selected_frames_{track_id}']
                            st.checkbox(
                                f"Frame {frame_ref['frame_id']}",
                                value=is_selected,
                                key=f"frame_check_{track_id}_{frame_idx}"
                            )

            # Form submit button
            submitted = st.form_submit_button("✓ Update Selection")

            if submitted:
                # Update session state based on form checkboxes
                new_selection = set()
                for frame_idx_in_page in range(len(page_frames)):
                    frame_idx = start_idx + frame_idx_in_page
                    if st.session_state.get(f"frame_check_{track_id}_{frame_idx}", False):
                        new_selection.add(frame_idx)

                st.session_state[f'selected_frames_{track_id}'] = new_selection
                st.rerun()

        st.markdown("---")

        # Split and Delete selected frames section
        if selected_count > 0:
            split_del_col1, split_del_col2 = st.columns(2)

            with split_del_col1:
                st.markdown("#### 🔀 Split Selected Frames")
                st.caption(f"Move {selected_count} frame(s) to a new track")

                split_col1, split_col2 = st.columns([3, 1])

                with split_col1:
                    split_target = st.selectbox(
                        "Assign selected frames to:",
                        ["-- Select --"] + cast_options,
                        key=f"split_target_{track_id}"
                    )

                with split_col2:
                    st.write("")
                    st.write("")
                    if st.button("✅ Split & Assign", key=f"confirm_split_{track_id}") and split_target != "-- Select --":
                        try:
                            # Get selected frame indices
                            selected_indices = sorted(st.session_state[f'selected_frames_{track_id}'])
                            selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                            remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                            if len(remaining_indices) == 0:
                                # 100% of frames selected - offer Move Whole Track instead
                                st.warning(f"⚠️ All {len(frame_refs)} frames selected in Track {track_id}")
                                st.info(f"💡 Move the entire track to **{split_target}**?")

                                if st.button(f"✅ Move Track {track_id} to {split_target}", key=f"move_whole_track_{track_id}"):
                                    try:
                                        # Use assign_tracks_to_identity to move whole track
                                        result = cluster_mutator.assign_tracks_to_identity(
                                            [track_id],
                                            split_target,
                                            source_cluster_id=cluster_id if 'cluster_id' in locals() else None
                                        )

                                        st.success(f"✅ Moved Track {track_id} to {split_target} (cluster {result['new_cluster_id']})")
                                        if result.get('orphans', 0) > 0:
                                            st.info(f"Note: {result['orphans']} orphan track(s) handled")

                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Move failed: {str(e)}")
                            else:
                                # Create new track with selected frames
                                new_track_id = max([t['track_id'] for t in tracks_data.get('tracks', [])]) + 1

                                new_track = {
                                    'track_id': new_track_id,
                                    'frame_refs': [frame_refs[i] for i in selected_indices]
                                }

                                # Update current track to remove selected frames
                                updated_track = track.copy()
                                updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                                # Save updated tracks.json
                                tracks_list = tracks_data.get('tracks', [])

                                # Remove old track
                                tracks_list = [t for t in tracks_list if t['track_id'] != track_id]

                                # Add updated track (if has frames)
                                if len(updated_track['frame_refs']) > 0:
                                    tracks_list.append(updated_track)

                                # Add new track
                                tracks_list.append(new_track)

                                # Save tracks.json
                                import json
                                tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                                tracks_backup_path = tracks_path.with_suffix('.json.bak')

                                # Backup
                                if tracks_path.exists():
                                    import shutil
                                    shutil.copy(tracks_path, tracks_backup_path)

                                # Write updated tracks
                                updated_tracks_data = {
                                    'episode_id': tracks_data.get('episode_id', episode_id),
                                    'total_tracks': len(tracks_list),
                                    'tracks': tracks_list
                                }

                                with open(tracks_path, 'w') as f:
                                    json.dump(updated_tracks_data, f, indent=2)

                                # Now handle cluster assignment for the new track
                                # Find or create cluster for target identity
                                clusters = clusters_data.get('clusters', [])
                                target_cluster = next((c for c in clusters if c.get('name') == split_target), None)

                                if target_cluster:
                                    # Add new track to existing cluster
                                    target_cluster['track_ids'].append(new_track_id)
                                    target_cluster['size'] = len(target_cluster['track_ids'])
                                    target_cluster_id = target_cluster['cluster_id']
                                else:
                                    # Create new cluster
                                    new_cluster_id = max([c['cluster_id'] for c in clusters]) + 1 if clusters else 0
                                    new_cluster = {
                                        'cluster_id': new_cluster_id,
                                        'size': 1,
                                        'track_ids': [new_track_id],
                                        'name': split_target,
                                        'assignment_confidence': 1.0,
                                        'quality_score': 0.0,
                                        'variance': 0.0,
                                        'silhouette_score': 0.0,
                                        'is_lowconf': False
                                    }
                                    clusters.append(new_cluster)
                                    target_cluster_id = new_cluster_id

                                # Update original cluster to reflect removed frames
                                if cluster_id is not None:
                                    orig_cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)
                                    if orig_cluster:
                                        if len(remaining_indices) == 0:
                                            # Remove track entirely from cluster
                                            orig_cluster['track_ids'] = [tid for tid in orig_cluster['track_ids'] if tid != track_id]
                                            orig_cluster['size'] = len(orig_cluster['track_ids'])
                                        # else: track still exists with remaining frames, keep it in cluster

                                # Save updated clusters.json
                                clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                                clusters_backup_path = clusters_path.with_suffix('.json.bak')

                                # Backup
                                if clusters_path.exists():
                                    import shutil
                                    shutil.copy(clusters_path, clusters_backup_path)

                                # Write updated clusters
                                updated_clusters_data = {
                                    'episode_id': clusters_data.get('episode_id', episode_id),
                                    'total_clusters': len(clusters),
                                    'noise_tracks': clusters_data.get('noise_tracks', 0),
                                    'clusters': clusters
                                }

                                with open(clusters_path, 'w') as f:
                                    json.dump(updated_clusters_data, f, indent=2)

                                # Save constraints
                                constraint_info = {
                                    'must_link_moved': [],
                                    'cannot_link': [],
                                    'target_identity': split_target,
                                    'track_ids': [new_track_id],
                                    'source_track_id': track_id,
                                    'selected_frame_ids': selected_frame_ids,
                                    'show_id': 'rhobh',
                                    'season_id': 's05',
                                    'episode_id': episode_id,
                                    'action': 'frame_split'
                                }
                                save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                                # Mark analytics dirty
                                from app.lib.analytics_dirty import mark_analytics_dirty
                                mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames split")

                                st.success(f"✅ Split {selected_count} frames into new Track {new_track_id} → {split_target}")
                                st.info(f"Original Track {track_id} kept {len(remaining_indices)} frames")

                                # Clear selection
                                st.session_state[f'selected_frames_{track_id}'] = set()

                                # Return to cluster gallery
                                if cluster_id:
                                    st.session_state.navigation_page = 'cluster_gallery'
                                    st.session_state.nav_cluster_id = cluster_id
                                else:
                                    st.session_state.navigation_page = None
                                st.rerun()

                        except Exception as e:
                            st.error(f"Split failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            with split_del_col2:
                st.markdown("#### 🗑️ Delete Selected Frames")
                st.caption(f"Permanently remove {selected_count} frame(s)")

                if st.button("🗑️ Delete Frames", key=f"delete_frames_{track_id}"):
                    try:
                        # Get remaining frame indices
                        selected_indices = sorted(st.session_state[f'selected_frames_{track_id}'])
                        selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                        remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                        import json
                        import shutil

                        if len(remaining_indices) == 0:
                            # Delete entire track
                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]

                            # Also remove from cluster
                            if cluster_id is not None:
                                clusters = clusters_data.get('clusters', [])
                                current_cluster = next((c for c in clusters if c['cluster_id'] == cluster_id), None)
                                if current_cluster:
                                    current_cluster['track_ids'] = [tid for tid in current_cluster['track_ids'] if tid != track_id]
                                    current_cluster['size'] = len(current_cluster['track_ids'])

                                # Save clusters
                                clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                                if clusters_path.exists():
                                    shutil.copy(clusters_path, clusters_path.with_suffix('.json.bak'))

                                with open(clusters_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_clusters': len(clusters),
                                        'noise_tracks': clusters_data.get('noise_tracks', 0),
                                        'clusters': clusters
                                    }, f, indent=2)

                            st.info(f"All frames deleted - Track {track_id} removed entirely")
                        else:
                            # Update track to keep only remaining frames
                            updated_track = track.copy()
                            updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]
                            tracks_list.append(updated_track)

                        # Save tracks
                        tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                        if tracks_path.exists():
                            shutil.copy(tracks_path, tracks_path.with_suffix('.json.bak'))

                        with open(tracks_path, 'w') as f:
                            json.dump({
                                'episode_id': episode_id,
                                'total_tracks': len(tracks_list),
                                'tracks': tracks_list
                            }, f, indent=2)

                        # Log deletion
                        save_track_level_constraints(episode_id, DATA_ROOT, {
                            'must_link_moved': [],
                            'cannot_link': [],
                            'target_identity': 'DELETED',
                            'track_ids': [track_id],
                            'deleted_frame_ids': selected_frame_ids,
                            'show_id': 'rhobh',
                            'season_id': 's05',
                            'episode_id': episode_id,
                            'action': 'frame_delete_track_gallery'
                        })

                        # Mark analytics dirty
                        from app.lib.analytics_dirty import mark_analytics_dirty
                        mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames deleted")

                        st.success(f"✅ Deleted {selected_count} frames from Track {track_id}")

                        # Clear selection and navigate back
                        st.session_state[f'selected_frames_{track_id}'] = set()
                        if cluster_id:
                            st.session_state.navigation_page = 'cluster_gallery'
                            st.session_state.nav_cluster_id = cluster_id
                        else:
                            st.session_state.navigation_page = None
                        st.rerun()

                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())


def render_cast_view_page(episode_id: str, cast_name: str, DATA_ROOT: Path):
    """
    Render CAST VIEW page showing all clusters and tracks for a specific identity.

    Useful for:
    - Seeing all instances of a person across multiple clusters
    - Confirming what will be consolidated on re-cluster
    - Bulk track operations
    """
    from app.lib.data import load_clusters, load_tracks
    from screentime.utils import get_video_path
    from app.lib.episode_status import load_suppress_data, save_suppress_data

    # Load data
    clusters_data = load_clusters(episode_id, DATA_ROOT)
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    if not clusters_data:
        st.error("Cluster data not found")
        return

    # Breadcrumb
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f"**Review** > Cast View > {cast_name}")
    with col2:
        if st.button("← Back to Review", key="back_btn"):
            st.session_state.navigation_page = None
            st.rerun()

    st.markdown("---")

    # Find all clusters for this cast member
    cast_clusters = [c for c in clusters_data.get('clusters', []) if c.get('name') == cast_name]

    if not cast_clusters:
        st.warning(f"No clusters found for {cast_name}")
        return

    # Collect all tracks
    all_track_ids = []
    for cluster in cast_clusters:
        all_track_ids.extend(cluster.get('track_ids', []))

    # Header with stats
    total_tracks = len(all_track_ids)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f"## {cast_name}")
    with col2:
        st.metric("Clusters", len(cast_clusters))
        st.metric("Tracks", total_tracks)

    st.markdown("---")

    # Show clusters list (pre-consolidation view)
    st.markdown(f"### Clusters for {cast_name}")
    st.caption(f"These {len(cast_clusters)} clusters will consolidate into 1 on RE-CLUSTER (if all approved)")

    for cluster in cast_clusters:
        cluster_id = cluster['cluster_id']
        size = cluster['size']
        quality = cluster.get('quality_score', 0)
        conf = cluster.get('assignment_confidence', 0.0)

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            # Show lock if approved
            if conf == 1.0:
                st.markdown(f"**Cluster {cluster_id}** 🔒")
            else:
                st.markdown(f"**Cluster {cluster_id}**")
        with col2:
            st.caption(f"Size: {size}")
        with col3:
            st.caption(f"Quality: {quality:.2f}")
        with col4:
            if st.button("View", key=f"view_cluster_{cluster_id}"):
                st.session_state.navigation_page = 'cluster_gallery'
                st.session_state.nav_cluster_id = cluster_id
                st.rerun()

    st.markdown("---")

    # All tracks - one row per track with all frames
    st.markdown(f"### All Tracks for {cast_name}")

    if not all_track_ids:
        st.info("No tracks found")
        return

    if not tracks_data or not video_path.exists():
        st.error("Track data or video not found")
        return

    # Compact mode toggle
    compact_mode = st.checkbox("Compact View", value=st.session_state.get('compact_mode', False))
    st.session_state.compact_mode = compact_mode

    tile_width = 125 if compact_mode else 150
    tiles_per_row = 8 if compact_mode else 6

    # Get cast options for splitting
    from app.season_cast_helpers import get_season_cast_dropdown_options
    from screentime.clustering.constraints import save_track_level_constraints
    cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)

    # Render each track as a row
    for track_id in all_track_ids:
        track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
        if not track:
            continue

        frame_refs = track.get('frame_refs', [])
        if not frame_refs:
            continue

        st.markdown("---")

        # Track header
        header_col1, header_col2, header_col3 = st.columns([2, 1, 1])
        with header_col1:
            st.markdown(f"### Track {track_id}")
        with header_col2:
            st.caption(f"{len(frame_refs)} frames")
        with header_col3:
            # View detail button
            if st.button("🔍 View Detail", key=f"view_detail_{track_id}"):
                st.session_state.navigation_page = 'track_gallery'
                st.session_state.nav_track_id = track_id
                st.session_state.nav_cluster_id = None
                st.session_state.nav_track_list = all_track_ids
                st.rerun()

        # Frame selection controls
        sel_col1, sel_col2, sel_col3 = st.columns([1, 1, 3])
        with sel_col1:
            if st.button("Select All", key=f"select_all_{track_id}"):
                st.session_state[f'selected_frames_cast_{track_id}'] = set(range(len(frame_refs)))
                st.rerun()
        with sel_col2:
            if st.button("Select None", key=f"select_none_{track_id}"):
                st.session_state[f'selected_frames_cast_{track_id}'] = set()
                st.rerun()
        with sel_col3:
            # Initialize selection state
            if f'selected_frames_cast_{track_id}' not in st.session_state:
                st.session_state[f'selected_frames_cast_{track_id}'] = set()

            selected_count = len(st.session_state[f'selected_frames_cast_{track_id}'])
            if selected_count > 0:
                st.caption(f"✓ {selected_count} of {len(frame_refs)} frames selected")

        # Pagination for large tracks
        frames_per_page = 24
        total_pages = (len(frame_refs) + frames_per_page - 1) // frames_per_page

        if f'cast_frame_page_{track_id}' not in st.session_state:
            st.session_state[f'cast_frame_page_{track_id}'] = 0

        current_page = st.session_state[f'cast_frame_page_{track_id}']

        if total_pages > 1:
            page_col1, page_col2, page_col3 = st.columns([1, 3, 1])
            with page_col1:
                if current_page > 0:
                    if st.button("◀", key=f"cast_prev_{track_id}"):
                        st.session_state[f'cast_frame_page_{track_id}'] -= 1
                        st.rerun()
            with page_col2:
                st.caption(f"Page {current_page + 1}/{total_pages}")
            with page_col3:
                if current_page < total_pages - 1:
                    if st.button("▶", key=f"cast_next_{track_id}"):
                        st.session_state[f'cast_frame_page_{track_id}'] += 1
                        st.rerun()

        start_idx = current_page * frames_per_page
        end_idx = min(start_idx + frames_per_page, len(frame_refs))
        page_frames = frame_refs[start_idx:end_idx]

        # Use form to batch selections
        with st.form(key=f"cast_selection_form_{track_id}"):
            # Display frames in rows with checkboxes
            for row_start in range(0, len(page_frames), tiles_per_row):
                cols = st.columns(tiles_per_row)
                for col_idx in range(tiles_per_row):
                    frame_idx_in_page = row_start + col_idx
                    if frame_idx_in_page >= len(page_frames):
                        break

                    frame_idx = start_idx + frame_idx_in_page
                    frame_ref = page_frames[frame_idx_in_page]

                    with cols[col_idx]:
                        thumb_path = thumb_gen.generate_frame_thumbnail(
                            video_path,
                            frame_ref['frame_id'],
                            frame_ref['bbox'],
                            episode_id,
                            track_id
                        )

                        if thumb_path and thumb_path.exists():
                            st.image(str(thumb_path), width=tile_width)

                            # Checkbox for selection (inside form)
                            is_selected = frame_idx in st.session_state[f'selected_frames_cast_{track_id}']
                            st.checkbox(
                                f"F{frame_ref['frame_id']}",
                                value=is_selected,
                                key=f"cast_frame_{track_id}_{frame_idx}"
                            )

            # Form submit
            submitted = st.form_submit_button("✓ Update Selection")
            if submitted:
                new_selection = set()
                for frame_idx_in_page in range(len(page_frames)):
                    frame_idx = start_idx + frame_idx_in_page
                    if st.session_state.get(f"cast_frame_{track_id}_{frame_idx}", False):
                        new_selection.add(frame_idx)
                st.session_state[f'selected_frames_cast_{track_id}'] = new_selection
                st.rerun()

        # Split and Delete sections (if frames selected)
        if selected_count > 0:
            split_del_col1, split_del_col2 = st.columns(2)

            with split_del_col1:
                st.markdown(f"**🔀 Split {selected_count} frame(s)**")

                split_col1, split_col2 = st.columns([3, 1])
                with split_col1:
                    split_target = st.selectbox(
                        "Assign to:",
                        ["-- Select --"] + cast_options,
                        key=f"split_target_cast_{track_id}"
                    )
                with split_col2:
                    st.write("")
                    if st.button("✅ Split", key=f"split_cast_{track_id}") and split_target != "-- Select --":
                        try:
                            # Get selected frame indices
                            selected_indices = sorted(st.session_state[f'selected_frames_cast_{track_id}'])
                            selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                            remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                            if len(remaining_indices) == 0:
                                # 100% of frames selected - offer Move Whole Track instead
                                st.warning(f"⚠️ All {len(frame_refs)} frames selected in Track {track_id}")
                                st.info(f"💡 Move the entire track to **{split_target}**?")

                                if st.button(f"✅ Move Track {track_id} to {split_target}", key=f"move_whole_cast_{track_id}"):
                                    try:
                                        # Use assign_tracks_to_identity to move whole track
                                        result = cluster_mutator.assign_tracks_to_identity(
                                            [track_id],
                                            split_target,
                                            source_cluster_id=None  # Cast view doesn't have cluster context
                                        )

                                        st.success(f"✅ Moved Track {track_id} to {split_target} (cluster {result['new_cluster_id']})")
                                        if result.get('orphans', 0) > 0:
                                            st.info(f"Note: {result['orphans']} orphan track(s) handled")

                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Move failed: {str(e)}")
                            else:
                                # Use same split logic as track gallery
                                import json
                                import shutil

                                # Create new track
                                new_track_id = max([t['track_id'] for t in tracks_data.get('tracks', [])]) + 1
                                new_track = {
                                    'track_id': new_track_id,
                                    'frame_refs': [frame_refs[i] for i in selected_indices]
                                }

                                # Update current track
                                updated_track = track.copy()
                                updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                                # Save tracks
                                tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]
                                if len(updated_track['frame_refs']) > 0:
                                    tracks_list.append(updated_track)
                                tracks_list.append(new_track)

                                tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                                if tracks_path.exists():
                                    shutil.copy(tracks_path, tracks_path.with_suffix('.json.bak'))

                                with open(tracks_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_tracks': len(tracks_list),
                                        'tracks': tracks_list
                                    }, f, indent=2)

                                # Update clusters
                                clusters = clusters_data.get('clusters', [])
                                target_cluster = next((c for c in clusters if c.get('name') == split_target), None)

                                if target_cluster:
                                    target_cluster['track_ids'].append(new_track_id)
                                    target_cluster['size'] = len(target_cluster['track_ids'])
                                else:
                                    new_cluster_id = max([c['cluster_id'] for c in clusters]) + 1 if clusters else 0
                                    clusters.append({
                                        'cluster_id': new_cluster_id,
                                        'size': 1,
                                        'track_ids': [new_track_id],
                                        'name': split_target,
                                        'assignment_confidence': 1.0,
                                        'quality_score': 0.0,
                                        'variance': 0.0,
                                        'silhouette_score': 0.0,
                                        'is_lowconf': False
                                    })

                                # Save clusters
                                clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                                if clusters_path.exists():
                                    shutil.copy(clusters_path, clusters_path.with_suffix('.json.bak'))

                                with open(clusters_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_clusters': len(clusters),
                                        'noise_tracks': clusters_data.get('noise_tracks', 0),
                                        'clusters': clusters
                                    }, f, indent=2)

                                # Save constraints
                                save_track_level_constraints(episode_id, DATA_ROOT, {
                                    'must_link_moved': [],
                                    'cannot_link': [],
                                    'target_identity': split_target,
                                    'track_ids': [new_track_id],
                                    'source_track_id': track_id,
                                    'selected_frame_ids': selected_frame_ids,
                                    'show_id': 'rhobh',
                                    'season_id': 's05',
                                    'episode_id': episode_id,
                                    'action': 'frame_split_cast_view'
                                })

                                # Mark analytics dirty
                                from app.lib.analytics_dirty import mark_analytics_dirty
                                mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames split")

                                st.success(f"✅ Created Track {new_track_id} with {selected_count} frames → {split_target}")
                                st.rerun()

                        except Exception as e:
                            st.error(f"Split failed: {e}")
                            import traceback
                            st.code(traceback.format_exc())

            with split_del_col2:
                st.markdown(f"**🗑️ Delete {selected_count} frame(s)**")
                st.caption("Permanently remove selected frames")

                if st.button("🗑️ Delete Frames", key=f"delete_frames_cast_{track_id}"):
                    try:
                        # Get remaining frame indices
                        selected_indices = sorted(st.session_state[f'selected_frames_cast_{track_id}'])
                        selected_frame_ids = [frame_refs[i]['frame_id'] for i in selected_indices]
                        remaining_indices = [i for i in range(len(frame_refs)) if i not in selected_indices]

                        import json
                        import shutil

                        # Find which cluster this track belongs to
                        track_cluster_id = None
                        for cluster in clusters_data.get('clusters', []):
                            if track_id in cluster.get('track_ids', []):
                                track_cluster_id = cluster['cluster_id']
                                break

                        if len(remaining_indices) == 0:
                            # Delete entire track
                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]

                            # Also remove from cluster
                            if track_cluster_id is not None:
                                clusters = clusters_data.get('clusters', [])
                                current_cluster = next((c for c in clusters if c['cluster_id'] == track_cluster_id), None)
                                if current_cluster:
                                    current_cluster['track_ids'] = [tid for tid in current_cluster['track_ids'] if tid != track_id]
                                    current_cluster['size'] = len(current_cluster['track_ids'])

                                # Save clusters
                                clusters_path = DATA_ROOT / "harvest" / episode_id / "clusters.json"
                                if clusters_path.exists():
                                    shutil.copy(clusters_path, clusters_path.with_suffix('.json.bak'))

                                with open(clusters_path, 'w') as f:
                                    json.dump({
                                        'episode_id': episode_id,
                                        'total_clusters': len(clusters),
                                        'noise_tracks': clusters_data.get('noise_tracks', 0),
                                        'clusters': clusters
                                    }, f, indent=2)

                            st.info(f"All frames deleted - Track {track_id} removed entirely")
                        else:
                            # Update track to keep only remaining frames
                            updated_track = track.copy()
                            updated_track['frame_refs'] = [frame_refs[i] for i in remaining_indices]

                            tracks_list = [t for t in tracks_data.get('tracks', []) if t['track_id'] != track_id]
                            tracks_list.append(updated_track)

                        # Save tracks
                        tracks_path = DATA_ROOT / "harvest" / episode_id / "tracks.json"
                        if tracks_path.exists():
                            shutil.copy(tracks_path, tracks_path.with_suffix('.json.bak'))

                        with open(tracks_path, 'w') as f:
                            json.dump({
                                'episode_id': episode_id,
                                'total_tracks': len(tracks_list),
                                'tracks': tracks_list
                            }, f, indent=2)

                        # Log deletion
                        save_track_level_constraints(episode_id, DATA_ROOT, {
                            'must_link_moved': [],
                            'cannot_link': [],
                            'target_identity': 'DELETED',
                            'track_ids': [track_id],
                            'deleted_frame_ids': selected_frame_ids,
                            'show_id': 'rhobh',
                            'season_id': 's05',
                            'episode_id': episode_id,
                            'action': 'frame_delete_cast_view'
                        })

                        # Mark analytics dirty
                        from app.lib.analytics_dirty import mark_analytics_dirty
                        mark_analytics_dirty(episode_id, DATA_ROOT, reason="frames deleted")

                        st.success(f"✅ Deleted {selected_count} frames from Track {track_id}")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Delete failed: {e}")
                        import traceback
                        st.code(traceback.format_exc())
