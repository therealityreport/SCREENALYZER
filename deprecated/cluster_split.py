"""
Cluster Split view for manually splitting mixed clusters.
"""

import streamlit as st
from pathlib import Path


def render_cluster_split(
    cluster_id: int,
    clusters_data: dict,
    episode_id: str,
    cluster_mutator,
    DATA_ROOT: Path
):
    """Render cluster split interface with multi-select tracks."""
    from app.lib.data import load_tracks
    from screentime.utils import get_video_path
    from app.season_cast_helpers import get_season_cast_dropdown_options

    st.subheader(f"✂️ Split Cluster {cluster_id}")

    # Find the cluster
    clusters = clusters_data.get("clusters", [])
    cluster = next((c for c in clusters if c["cluster_id"] == cluster_id), None)

    if not cluster:
        st.error(f"Cluster {cluster_id} not found")
        if st.button("← Back to Low-Confidence Queue"):
            st.session_state.splitting_cluster_id = None
            st.rerun()
        return

    cluster_name = cluster.get("name", f"Cluster {cluster_id}")
    size = cluster["size"]
    quality = cluster.get("quality_score", 0)

    st.info(
        f"**{cluster_name}** (Size: {size}, Quality: {quality:.2f})\n\n"
        f"Select tracks that belong to the same person and assign them to an identity. "
        f"You can make multiple assignments to split this cluster into separate groups."
    )

    # Load data
    tracks_data = load_tracks(episode_id, DATA_ROOT)
    video_path = get_video_path(episode_id, DATA_ROOT)
    thumb_gen = st.session_state.thumbnail_generator

    # Initialize selection state
    if 'split_selected_tracks' not in st.session_state:
        st.session_state.split_selected_tracks = set()

    track_ids = cluster.get("track_ids", [])

    if not track_ids:
        st.warning("No tracks in this cluster")
        if st.button("← Back to Low-Confidence Queue"):
            st.session_state.splitting_cluster_id = None
            st.rerun()
        return

    # Horizontal track display with checkboxes
    st.markdown("### Select Tracks")
    st.caption(f"{len(st.session_state.split_selected_tracks)} of {len(track_ids)} tracks selected")

    # Bulk select buttons
    bulk_col1, bulk_col2, bulk_col3 = st.columns(3)
    with bulk_col1:
        if st.button("Select All"):
            st.session_state.split_selected_tracks = set(track_ids)
            st.rerun()
    with bulk_col2:
        if st.button("Clear Selection"):
            st.session_state.split_selected_tracks = set()
            st.rerun()
    with bulk_col3:
        st.write("")  # Spacing

    st.markdown("---")

    # Display tracks in rows of 8
    if not tracks_data or not video_path.exists():
        st.warning("Track data or video not available")
    else:
        # Display tracks in rows
        tracks_per_row = 8
        for row_start in range(0, len(track_ids), tracks_per_row):
            row_track_ids = track_ids[row_start:row_start + tracks_per_row]
            cols = st.columns(tracks_per_row)

            for col_idx, track_id in enumerate(row_track_ids):
                with cols[col_idx]:
                    track = next((t for t in tracks_data.get('tracks', []) if t['track_id'] == track_id), None)
                    if not track:
                        st.caption(f"Track {track_id}")
                        st.caption("(no data)")
                        continue

                    # Get middle frame for display
                    frame_refs = track.get('frame_refs', [])
                    if frame_refs:
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
                            # Checkbox for selection
                            is_selected = track_id in st.session_state.split_selected_tracks
                            if st.checkbox(
                                f"Track {track_id}",
                                value=is_selected,
                                key=f"split_track_{track_id}"
                            ):
                                st.session_state.split_selected_tracks.add(track_id)
                            else:
                                st.session_state.split_selected_tracks.discard(track_id)

                            # Display thumbnail
                            st.image(str(thumb_path), width=120)
                        else:
                            st.caption(f"Track {track_id}")
                            st.caption("(no thumb)")
                    else:
                        st.caption(f"Track {track_id}")
                        st.caption("(no frames)")

    st.markdown("---")

    # Assignment controls
    if len(st.session_state.split_selected_tracks) > 0:
        st.markdown("### Assign Selected Tracks")
        st.caption(f"{len(st.session_state.split_selected_tracks)} track(s) selected")

        assign_col1, assign_col2 = st.columns([3, 1])

        with assign_col1:
            cast_options = get_season_cast_dropdown_options("rhobh", "s05", DATA_ROOT)
            target_identity = st.selectbox(
                "Move selected tracks to:",
                ["-- Select Identity --"] + cast_options,
                key="split_target_identity"
            )

        with assign_col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("✅ Confirm Assignment", type="primary") and target_identity != "-- Select Identity --":
                try:
                    selected = list(st.session_state.split_selected_tracks)

                    # Save track-level constraints
                    from screentime.clustering.constraints import save_track_level_constraints

                    # Build ML constraints (selected tracks must be linked)
                    ml_pairs = []
                    for i in range(len(selected)):
                        for j in range(i + 1, len(selected)):
                            ml_pairs.append((selected[i], selected[j]))

                    # Build CL constraints (selected vs remaining)
                    remaining = [tid for tid in track_ids if tid not in selected]
                    cl_pairs = []
                    for moved_tid in selected:
                        for remain_tid in remaining:
                            cl_pairs.append((min(moved_tid, remain_tid), max(moved_tid, remain_tid)))

                    constraint_info = {
                        'must_link_moved': ml_pairs,
                        'cannot_link': cl_pairs,
                        'target_identity': target_identity,
                        'track_ids': selected,
                        'show_id': 'rhobh',
                        'season_id': 's05',
                        'episode_id': episode_id
                    }

                    save_track_level_constraints(episode_id, DATA_ROOT, constraint_info)

                    # Show constraint counts
                    st.success(f"✅ Constraint counts: ML:+{len(ml_pairs)}, CL:+{len(cl_pairs)}")

                    # Load current clusters to find target
                    from app.lib.data import load_clusters
                    current_clusters_data = load_clusters(episode_id, DATA_ROOT)
                    clusters = current_clusters_data.get('clusters', [])

                    # Find or create target cluster
                    if target_identity == "Unknown":
                        # Move each track to its own new cluster
                        for track_id in selected:
                            cluster_mutator.move_track(track_id, cluster_id, -1)
                    else:
                        # Find existing cluster with this identity
                        target_cluster = next((c for c in clusters if c.get('name') == target_identity), None)

                        if target_cluster:
                            # Move all tracks to existing cluster
                            for track_id in selected:
                                cluster_mutator.move_track(track_id, cluster_id, target_cluster['cluster_id'])
                        else:
                            # Create new cluster for first track, then move others to it
                            first_track = selected[0]
                            new_clusters_data = cluster_mutator.move_track(first_track, cluster_id, -1)
                            new_cluster_id = max(c['cluster_id'] for c in new_clusters_data.get('clusters', []))
                            cluster_mutator.assign_name(new_cluster_id, target_identity)

                            # Move remaining tracks to this new cluster
                            for track_id in selected[1:]:
                                cluster_mutator.move_track(track_id, cluster_id, new_cluster_id)

                    st.success(f"✅ Assigned {len(selected)} track(s) to {target_identity}")

                    # Clear selection
                    st.session_state.split_selected_tracks = set()

                    # Check if cluster is empty now
                    remaining_count = len(remaining)
                    if remaining_count == 0:
                        st.info("All tracks have been assigned. Returning to Low-Confidence Queue.")
                        st.session_state.splitting_cluster_id = None

                    st.rerun()

                except Exception as e:
                    st.error(f"Assignment failed: {e}")

    st.markdown("---")

    # Navigation
    nav_col1, nav_col2 = st.columns(2)
    with nav_col1:
        if st.button("← Back to Low-Confidence Queue"):
            st.session_state.splitting_cluster_id = None
            st.session_state.split_selected_tracks = set()
            st.rerun()

    with nav_col2:
        # Offer RE-CLUSTER if constraints exist
        constraints_path = DATA_ROOT / "harvest" / episode_id / "diagnostics" / "track_constraints.jsonl"
        if constraints_path.exists():
            if st.button("🔄 Re-Cluster with Constraints"):
                from api.jobs import job_manager

                job_id = job_manager.create_job(
                    "recluster",
                    episode_id=episode_id,
                    show_id="rhobh",
                    season_id="s05",
                    sources=["baseline", "entrance", "densify"],
                    use_constraints=True
                )

                st.success(f"Started RE-CLUSTER job: {job_id}")
                st.info("Check the job status in the Review page header")
                st.rerun()
