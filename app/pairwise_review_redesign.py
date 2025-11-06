"""
Redesigned Pairwise Review with row-based layout and track-level splitting.
"""

import streamlit as st
from pathlib import Path


def render_pairwise_review_v2(
    clusters_data: dict,
    suggestions_df,
    episode_id: str,
    state_mgr,
    cluster_mutator,
    DATA_ROOT: Path = None,
    *args,
    **kwargs
):
    """
    Render Pairwise Review with horizontal track scrollers and split controls.

    Permissive signature to tolerate older call sites.
    - clusters_data: dict
    - suggestions_df: DataFrame or None
    - episode_id: str
    - state_mgr: ReviewStateManager
    - cluster_mutator: ClusterMutator
    - DATA_ROOT: Path (optional, defaults to Path("data"))
    *args/**kwargs: ignored for forward-compatibility
    """
    import logging
    from app.lib.data import load_tracks
    from screentime.utils import get_video_path
    from app.season_cast_helpers import get_season_cast_dropdown_options

    logger = logging.getLogger(__name__)

    if args or kwargs:
        logger.info(f"Pairwise v2 received extra args (ignored): args={args}, kwargs={kwargs}")

    # Set default DATA_ROOT if not provided
    if DATA_ROOT is None:
        DATA_ROOT = Path("data")

    st.subheader("Pairwise Merge Review")

    if suggestions_df is None or len(suggestions_df) == 0:
        st.info("No merge suggestions available for this episode.")
        return

    # Queue progress
    total_suggestions = len(suggestions_df)
    current_idx = st.session_state.current_suggestion_idx

    if current_idx >= total_suggestions:
        st.success("🎉 All suggestions reviewed!")
        if st.button("Reset Queue"):
            st.session_state.current_suggestion_idx = 0
            st.rerun()
        return

    st.progress((current_idx + 1) / total_suggestions)
    st.caption(f"Suggestion {current_idx + 1} of {total_suggestions}")

    # Current suggestion
    suggestion = suggestions_df.iloc[current_idx]
    cluster_a_id = suggestion["cluster_a_id"]
    cluster_b_id = suggestion["cluster_b_id"]
    similarity = suggestion["similarity"]

    st.markdown(f"### Similarity: {similarity:.2%}")

    # Load data
    clusters = clusters_data.get("clusters", [])
    cluster_a = next((c for c in clusters if c["cluster_id"] == cluster_a_id), None)
    cluster_b = next((c for c in clusters if c["cluster_id"] == cluster_b_id), None)

    if not cluster_a or not cluster_b:
        st.error("Clusters not found")
        return

    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    # Initialize track selection state
    if 'selected_tracks' not in st.session_state:
        st.session_state.selected_tracks = set()

    # Render each cluster as a horizontal row
    def render_cluster_row(cluster):
        cluster_id = cluster['cluster_id']
        cluster_name = cluster.get('name', f'Cluster {cluster_id}')
        size = cluster['size']
        quality = cluster.get('quality_score', 0)

        # Header with Assign Name and View Tracks buttons
        header_col1, header_col2, header_col3, header_col4 = st.columns([3, 1, 1, 3])

        with header_col1:
            st.markdown(f"**{cluster_name}** (Cluster {cluster_id})")

        with header_col2:
            if st.button("Assign Name", key=f"assign_{cluster_id}"):
                st.session_state[f'assigning_{cluster_id}'] = True

        with header_col3:
            if st.button(f"View Tracks ({size})", key=f"view_{cluster_id}"):
                st.session_state.viewing_cluster_id = cluster_id
                st.rerun()

        with header_col4:
            st.caption(f"Size: {size} | Quality: {quality:.2f}")

        # Show assignment modal if needed
        if st.session_state.get(f'assigning_{cluster_id}', False):
            with st.expander("🏷️ Assign Identity", expanded=True):
                cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
                new_name = st.selectbox("Identity:", cast_options, key=f"new_name_{cluster_id}")

                confirm_col1, confirm_col2 = st.columns(2)
                with confirm_col1:
                    if st.button("✅ Confirm", key=f"confirm_assign_{cluster_id}"):
                        try:
                            cluster_mutator.assign_cluster_name(cluster_id, new_name)
                            st.session_state[f'assigning_{cluster_id}'] = False
                            st.success(f"Assigned {cluster_name} → {new_name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Assignment failed: {e}")

                with confirm_col2:
                    if st.button("❌ Cancel", key=f"cancel_assign_{cluster_id}"):
                        st.session_state[f'assigning_{cluster_id}'] = False
                        st.rerun()

        # Horizontal scroller with 150×150 track tiles
        st.markdown("---")

        if not tracks_data or not video_path.exists():
            st.caption(f"Tracks: {cluster['track_ids'][:10]}...")
            return

        track_ids = cluster['track_ids']

        # Create scrollable container with tracks
        st.markdown(
            f'<div style="display: flex; overflow-x: auto; gap: 10px; padding: 10px 0;">',
            unsafe_allow_html=True
        )

        # Use columns for horizontal layout
        cols = st.columns(min(len(track_ids), 20))  # Limit to 20 visible

        for idx, track_id in enumerate(track_ids[:20]):  # Show first 20 tracks
            if idx >= len(cols):
                break

            with cols[idx]:
                track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
                if not track:
                    continue

                # Get best frame from track
                frame_refs = track.get('frame_refs', [])
                if not frame_refs:
                    continue

                # Use middle frame as representative
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

                # Track ID label
                st.caption(f"Track {track_id}")

                if thumb_path and thumb_path.exists():
                    # Checkbox for selection
                    is_selected = track_id in st.session_state.selected_tracks

                    if st.checkbox(
                        "",  # No label, just checkbox
                        value=is_selected,
                        key=f"select_{cluster_id}_{track_id}"
                    ):
                        st.session_state.selected_tracks.add(track_id)
                    else:
                        st.session_state.selected_tracks.discard(track_id)

                    # Display 150×150 image
                    st.image(str(thumb_path), width=150)

                    # Click to open full track gallery
                    if st.button("🔍", key=f"open_{cluster_id}_{track_id}", help="View full track"):
                        st.session_state.viewing_track_id = track_id
                        st.session_state.viewing_track_cluster_id = cluster_id

        st.markdown('</div>', unsafe_allow_html=True)

    # Render both clusters
    st.markdown("### Cluster A")
    render_cluster_row(cluster_a)

    st.markdown("### Cluster B")
    render_cluster_row(cluster_b)

    st.markdown("---")

    # Track-level split controls
    if len(st.session_state.selected_tracks) > 0:
        st.markdown("### Track-level Actions")
        st.caption(f"{len(st.session_state.selected_tracks)} track(s) selected")

        action_col1, action_col2, action_col3, action_col4 = st.columns([2, 2, 1, 1])

        with action_col1:
            # Move to cast member dropdown
            cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
            move_to = st.selectbox("Move selected to:", ["-- Select --"] + cast_options, key="move_to_identity")

        with action_col2:
            if st.button("✅ Confirm Move") and move_to != "-- Select --":
                try:
                    selected = list(st.session_state.selected_tracks)

                    # Save track-level constraints
                    from screentime.clustering.constraints import save_track_level_constraints

                    # Group by original cluster
                    tracks_a = [tid for tid in selected if tid in cluster_a['track_ids']]
                    tracks_b = [tid for tid in selected if tid in cluster_b['track_ids']]
                    all_selected = selected

                    # Build ML/CL constraints
                    ml_within = [(all_selected[i], all_selected[j]) for i in range(len(all_selected)) for j in range(i+1, len(all_selected))]

                    # CL between moved tracks and remaining tracks
                    remaining_a = [tid for tid in cluster_a['track_ids'] if tid not in selected]
                    remaining_b = [tid for tid in cluster_b['track_ids'] if tid not in selected]

                    cl_pairs = []
                    for moved_tid in selected:
                        for remain_tid in remaining_a + remaining_b:
                            cl_pairs.append((min(moved_tid, remain_tid), max(moved_tid, remain_tid)))

                    constraint_info = {
                        'must_link_moved': ml_within,
                        'cannot_link': cl_pairs,
                        'target_identity': move_to
                    }

                    save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                    # Move tracks to target identity
                    if move_to == "Unknown":
                        cluster_mutator.move_tracks_to_unknown(selected, cluster_a_id)
                    else:
                        # Create or update cluster for this identity
                        cluster_mutator.assign_tracks_to_identity(selected, move_to)

                    st.success(f"✅ Moved {len(selected)} tracks to {move_to}")
                    st.session_state.selected_tracks = set()  # Clear selection
                    st.session_state.current_suggestion_idx += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Move failed: {e}")

        with action_col3:
            if st.button("Clear Selection"):
                st.session_state.selected_tracks = set()
                st.rerun()

        with action_col4:
            # Bulk select options
            with st.expander("Bulk"):
                if st.button("Select All A"):
                    st.session_state.selected_tracks.update(cluster_a['track_ids'])
                    st.rerun()
                if st.button("Select All B"):
                    st.session_state.selected_tracks.update(cluster_b['track_ids'])
                    st.rerun()

    st.markdown("---")

    # Quick Actions (for non-split cases)
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Merge", type="primary"):
            try:
                cluster_mutator.merge_clusters(cluster_a_id, cluster_b_id)
                state_mgr.record_action("merge", {"cluster_a_id": cluster_a_id, "cluster_b_id": cluster_b_id})
                st.session_state.current_suggestion_idx += 1
                st.success(f"Merged clusters {cluster_a_id} and {cluster_b_id}")
                st.rerun()
            except Exception as e:
                st.error(f"Merge failed: {e}")

    with col2:
        if st.button("❌ Not Same"):
            state_mgr.record_action("reject_merge", {"cluster_a_id": cluster_a_id, "cluster_b_id": cluster_b_id})
            st.session_state.current_suggestion_idx += 1
            st.rerun()

    with col3:
        if st.button("⏭️ Skip"):
            st.session_state.current_suggestion_idx += 1
            st.rerun()
