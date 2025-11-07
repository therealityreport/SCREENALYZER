"""Clusters workspace tab."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
import streamlit as st

from app.components.strip import render_strip
from app.media.thumbnails import get_track_still_path, thumbnail_coverage
from app.workspace.common import render_stills_progress
from app.lib.mutator_api import WorkspaceMutator
from app.workspace.actions import (
    identity_options,
    render_cluster_actions,
    render_track_actions,
)
from app.workspace.common import (
    PLACEHOLDER_DATA_URI,
    ensure_workspace_styles,
    get_thumbnail_generator,
    track_frame_images,
)
from app.lib.registry import get_episode_hash
from app.utils.ui_keys import wkey
from screentime.utils import get_video_path


def render_clusters_tab(mutator: WorkspaceMutator) -> None:
    # Preflight check
    from app.workspace.common import check_artifacts_status
    from app.utils.ui_keys import safe_rerun

    artifacts = check_artifacts_status(mutator.episode_id, str(mutator.data_root))

    if artifacts["needs_detect"] or artifacts["needs_track"] or artifacts["needs_cluster"]:
        st.warning(f"⚠️ {artifacts['message']}")

        if artifacts["next_action"]:
            if st.button(f"▶️ {artifacts['next_action']}", key="clusters_preflight_btn", type="primary"):
                st.session_state["_trigger_cluster"] = True
                safe_rerun()

        with st.expander("Artifact status"):
            st.json(artifacts["artifacts"])

        return

    clusters_df = mutator.cluster_metrics_df
    if clusters_df.empty:
        st.info("No clusters available. Re-run clustering to populate this view.")
        return

    ensure_workspace_styles()

    generated_thumbs, total_thumbs, placeholders = thumbnail_coverage(mutator.episode_id)
    if total_thumbs > 0:
        coverage_ratio = generated_thumbs / float(total_thumbs)
    else:
        coverage_ratio = 0.0

    if total_thumbs == 0 or coverage_ratio < 0.1:
        st.warning(
            "Track thumbnails are missing for this episode. Click **Generate thumbnails** "
            "in the workspace header (install ffmpeg for faster, exact seeks)."
        )

    # Check if we're rendering a dedicated Person Clusters view
    clusters_person = st.session_state.get("clusters_person")
    if clusters_person:
        render_person_clusters(mutator, clusters_person)
        return

    identities = identity_options(mutator.person_metrics_df)
    thresholds = mutator.thresholds
    track_map = _build_track_map(mutator.tracks)
    cluster_map = {int(c["cluster_id"]): c for c in mutator.clusters.get("clusters", [])}
    thumb_gen = get_thumbnail_generator()
    video_path = get_video_path(mutator.episode_id, mutator.data_root)

    # Check if coming from Faces tab with person filter (legacy)
    person_filter = st.session_state.pop("cluster_person_filter", None)
    if person_filter and person_filter in identities:
        default_filter = [person_filter]
    else:
        default_filter = []

    filter_cols = st.columns([2, 1, 1])
    with filter_cols[0]:
        selected_people = st.multiselect(
            "Filter by identity",
            options=identities,
            default=default_filter,
            help="Leave empty to show all clusters.",
            key=wkey("clusters", "filter_identity"),
        )
    with filter_cols[1]:
        min_tracks = st.number_input(
            "Min tracks",
            min_value=0,
            value=0,
            step=1,
            key=wkey("clusters", "filter_min_tracks"),
        )
    with filter_cols[2]:
        min_p25 = st.slider(
            "Min median p25",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key=wkey("clusters", "filter_min_p25"),
        )

    filtered_clusters = clusters_df.copy()
    if selected_people:
        filtered_clusters = filtered_clusters[filtered_clusters["name"].isin(selected_people)]
    if min_tracks > 0:
        filtered_clusters = filtered_clusters[filtered_clusters["n_tracks"] >= min_tracks]
    if min_p25 > 0.0:
        filtered_clusters = filtered_clusters[
            filtered_clusters["tracks_conf_p25_median"] >= min_p25
        ]

    low_df = mutator.get_low_confidence_clusters()
    if selected_people:
        low_df = low_df[low_df["name"].isin(selected_people)]

    # Phase 3 P2: Unassigned clusters filtering
    unassigned_df = filtered_clusters[
        (filtered_clusters["name"].isna()) |
        (filtered_clusters["name"] == "Unknown") |
        (filtered_clusters["name"] == "Unassigned") |
        (filtered_clusters["name"] == "Other") |
        (filtered_clusters["assignment_conf"] < 0.3)
    ].copy()

    # Phase 3 P2: Sorting dropdown
    sort_by = st.selectbox(
        "Sort by",
        options=[
            "Assigned Name (A→Z)",
            "Assignment Confidence (High→Low)",
            "Cluster Confidence (High→Low)",
            "Cluster Size (Large→Small)",
        ],
        key=wkey("clusters", "sort_by"),
        help="Sort clusters in all tabs by selected criteria"
    )

    # Apply sorting to all dataframes
    def _apply_sorting(df: pd.DataFrame, sort_option: str) -> pd.DataFrame:
        if df.empty:
            return df
        if sort_option == "Assigned Name (A→Z)":
            return df.sort_values("name", na_position="last")
        elif sort_option == "Assignment Confidence (High→Low)":
            return df.sort_values("assignment_conf", ascending=False, na_position="last")
        elif sort_option == "Cluster Confidence (High→Low)":
            return df.sort_values("tracks_conf_p25_median", ascending=False, na_position="last")
        elif sort_option == "Cluster Size (Large→Small)":
            return df.sort_values("n_tracks", ascending=False)
        return df

    filtered_clusters = _apply_sorting(filtered_clusters, sort_by)
    low_df = _apply_sorting(low_df, sort_by)
    unassigned_df = _apply_sorting(unassigned_df, sort_by)

    # Phase 3 P2: Four sub-views
    all_tab, pairwise_tab, low_tab, unassigned_tab = st.tabs([
        "All Clusters",
        "Pairwise Review",
        "Low-Confidence",
        "Unassigned"
    ])
    selected_cluster = st.session_state.get("workspace_selected_cluster")

    with all_tab:
        st.caption(f"{len(filtered_clusters)} clusters")
        for _, cluster_row in filtered_clusters.iterrows():
            _render_cluster_card(
                mutator,
                cluster_row,
                identities,
                thresholds,
                cluster_map,
                track_map,
                selected_cluster,
                context="all",
            )

    with pairwise_tab:
        st.info("🔍 Pairwise Review")
        st.caption(
            "This view will show cluster pairs that are potential duplicates for manual merge review. "
            "Coming in Phase 3 (Refine Clusters)."
        )
        st.markdown("**Features in development:**")
        st.markdown("- Centroid distance < 0.35 detection")
        st.markdown("- Silhouette score improvement estimation")
        st.markdown("- Side-by-side cluster comparison")
        st.markdown("- One-click merge confirmation")

    with low_tab:
        if low_df.empty:
            st.success("No low-confidence clusters — great job!")
        else:
            low_threshold = float(thresholds.get("cluster_low_p25", 0.6))
            contam_threshold = float(thresholds.get("cluster_contam_high", 0.2))
            for _, cluster_row in low_df.iterrows():
                reasons = _cluster_reasons(cluster_row, low_threshold, contam_threshold)
                _render_cluster_card(
                    mutator,
                    cluster_row,
                    identities,
                    thresholds,
                    cluster_map,
                    track_map,
                    selected_cluster,
                    context="low",
                    reasons=reasons,
                )

    with unassigned_tab:
        st.caption(
            "Clusters with no cast assignment or marked as 'Unknown', 'Unassigned', or 'Other'. "
            "These are **excluded** from analytics."
        )
        if unassigned_df.empty:
            st.success("✅ All clusters are assigned to cast members!")
        else:
            st.caption(f"{len(unassigned_df)} unassigned clusters")
            for _, cluster_row in unassigned_df.iterrows():
                _render_cluster_card(
                    mutator,
                    cluster_row,
                    identities,
                    thresholds,
                    cluster_map,
                    track_map,
                    selected_cluster,
                    context="unassigned",
                )

    if selected_cluster is not None:
        _render_tracks_for_cluster(
            mutator,
            selected_cluster,
            identities,
            thresholds,
            track_map,
            thumb_gen,
            video_path,
        )


def render_person_clusters(mutator: WorkspaceMutator, person_name: str) -> None:
    """Render dedicated Person Clusters view."""
    clusters_df = mutator.cluster_metrics_df
    person_df = mutator.person_metrics_df
    thresholds = mutator.thresholds
    identities = identity_options(person_df)
    track_map = _build_track_map(mutator.tracks)
    cluster_map = {int(c["cluster_id"]): c for c in mutator.clusters.get("clusters", [])}
    thumb_gen = get_thumbnail_generator()
    video_path = get_video_path(mutator.episode_id, mutator.data_root)

    # Back button
    if st.button("← Back to Faces", key=wkey("person_clusters", "back")):
        st.session_state.pop("clusters_person", None)
        st.session_state["workspace_tab"] = "Faces"
        st.rerun()

    # Person header and metrics
    st.markdown(f"### Person · {person_name}")

    person_row = person_df[person_df["person"] == person_name]
    if not person_row.empty:
        row = person_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Clusters", int(row.get("n_clusters", 0)))
        with col2:
            st.metric("Tracks", int(row.get("n_tracks", 0)))
        with col3:
            bank_p25 = row.get("bank_conf_median_p25", 0.0)
            st.metric("Bank p25", f"{bank_p25:.2f}")
        with col4:
            contam = row.get("bank_contam_rate", 0.0)
            st.metric("Contamination", f"{contam:.2f}")

    st.markdown("---")

    # Filter clusters to this person
    person_clusters = clusters_df[clusters_df["name"] == person_name]

    if person_clusters.empty:
        st.info("No clusters currently assigned to this person.")
        return

    # Render tabs for All, Pairwise, and Low-Confidence clusters
    low_df = mutator.get_low_confidence_clusters()
    low_df = low_df[low_df["name"] == person_name]

    all_tab, pair_tab, low_tab = st.tabs(["All Clusters", "Pairwise Review", "Low-Confidence Clusters"])
    selected_cluster = st.session_state.get("workspace_selected_cluster")

    with all_tab:
        st.caption(f"{len(person_clusters)} clusters")
        for _, cluster_row in person_clusters.sort_values("cluster_id").iterrows():
            _render_cluster_card(
                mutator,
                cluster_row,
                identities,
                thresholds,
                cluster_map,
                track_map,
                selected_cluster,
                context="person_all",
            )

    with pair_tab:
        _render_pairwise_review(
            mutator,
            person_name,
            person_clusters,
            identities,
            cluster_map,
            track_map,
            thumb_gen,
            video_path,
        )

    with low_tab:
        if low_df.empty:
            st.success("No low-confidence clusters for this person")
        else:
            low_threshold = float(thresholds.get("cluster_low_p25", 0.6))
            contam_threshold = float(thresholds.get("cluster_contam_high", 0.2))
            for _, cluster_row in low_df.sort_values("tracks_conf_p25_median").iterrows():
                reasons = _cluster_reasons(cluster_row, low_threshold, contam_threshold)
                _render_cluster_card(
                    mutator,
                    cluster_row,
                    identities,
                    thresholds,
                    cluster_map,
                    track_map,
                    selected_cluster,
                    context="person_low",
                    reasons=reasons,
                )

    # Render tracks if a cluster is selected
    if selected_cluster is not None:
        _render_tracks_for_cluster(
            mutator,
            selected_cluster,
            identities,
            thresholds,
            track_map,
            thumb_gen,
            video_path,
        )


def _render_cluster_card(
    mutator: WorkspaceMutator,
    cluster_row: pd.Series,
    identities: Sequence[str],
    thresholds: dict,
    cluster_map: Dict[int, dict],
    track_map: Dict[int, dict],
    selected_cluster: int | None,
    *,
    context: str = "default",
    reasons: Sequence[str] | None = None,
) -> None:
    cluster_id = int(cluster_row["cluster_id"])
    cluster_dict = cluster_map.get(cluster_id, {})
    track_ids = [int(tid) for tid in cluster_dict.get("track_ids", [])]

    _render_cluster_header(cluster_row, thresholds, reasons)

    strip_id = f"c{cluster_id}_{context}"
    full_prefix = f"ws_clusters_{strip_id}"

    if track_ids:
        # Get episode_hash for cache busting
        episode_hash = get_episode_hash(mutator.episode_id, mutator.data_root)
        
        def _page_loader(batch_ids: Sequence[int | str]) -> List[Path]:
            return [get_track_still_path(mutator.episode_id, int(tid), episode_hash) for tid in batch_ids]

        render_strip(
            track_ids,
            ids=track_ids,
            key_prefix="ws_clusters",
            strip_id=strip_id,
            selectable=False,
            image_loader=_page_loader,
        )
    else:
        render_strip(
            [PLACEHOLDER_DATA_URI],
            ids=[f"placeholder-{cluster_id}"],
            key_prefix="ws_clusters",
            strip_id=strip_id,
            selectable=False,
        )

    page_state = st.session_state.get(wkey(full_prefix, "page"), 0)
    person_name = cluster_row.get("name", "Unknown")
    suffix = f"{mutator.episode_id}_person_{person_name}_cid{cluster_id}_ctx{context}_page{page_state}"

    view_clicked, key_map = render_cluster_actions(
        mutator,
        cluster_row,
        identities,
        key_suffix=suffix,
        selectable_tracks=track_ids,
        show_view_button=True,
        is_selected=selected_cluster == cluster_id,
    )
    key_map = dict(key_map)
    key_map["page_state"] = str(page_state)
    key_map["suffix"] = suffix

    with st.expander("Dev keys", expanded=False):
        formatted = "\n".join(f"{label}: {value}" for label, value in key_map.items())
        st.code(formatted or "(none)")

    if view_clicked:
        st.session_state["workspace_selected_cluster"] = cluster_id
        st.rerun()

    st.markdown("---")


def _render_tracks_for_cluster(
    mutator: WorkspaceMutator,
    cluster_id: int,
    identities: Sequence[str],
    thresholds: dict,
    track_map: Dict[int, dict],
    thumb_gen,
    video_path: Path,
) -> None:
    tracks_df = mutator.track_metrics_df
    cluster_tracks = tracks_df[tracks_df["cluster_id"] == cluster_id]
    if cluster_tracks.empty:
        return

    st.subheader(f"Tracks · Cluster {cluster_id}")
    for _, track_row in cluster_tracks.sort_values("conf_p25", ascending=False).iterrows():
        _render_track_card(
            track_row,
            identities,
            thresholds,
            track_map,
            thumb_gen,
            video_path,
            mutator.episode_id,
        )


def _render_track_card(
    track_row: pd.Series,
    identities: Sequence[str],
    thresholds: dict,
    track_map: Dict[int, dict],
    thumb_gen,
    video_path: Path,
    episode_id: str,
    *,
    reasons: Sequence[str] | None = None,
) -> None:
    track_id = int(track_row["track_id"])
    track_dict = track_map.get(track_id)

    _render_track_header(track_row, thresholds, reasons)

    frame_ids, frame_images = track_frame_images(
        track_dict,
        video_path,
        episode_id,
        thumb_gen,
    )

    if frame_ids:
        _, selected_frames = render_strip(
            frame_images,
            ids=frame_ids,
            key_prefix="ws_tracks",
            strip_id=f"t{track_id}",
            selectable=True,
        )
    else:
        st.info("No frame thumbnails available for this track.")
        selected_frames = []

    if selected_frames:
        st.caption(f"✓ {len(selected_frames)} frame(s) selected")

    render_track_actions(
        track_row,
        identities,
        selected_frames=selected_frames,
        episode_id=episode_id,
        context="clusters_track",
    )
    st.markdown("---")


def _render_pairwise_review(
    mutator: WorkspaceMutator,
    person_name: str,
    person_clusters: pd.DataFrame,
    identities: Sequence[str],
    cluster_map: Dict[int, dict],
    track_map: Dict[int, dict],
    thumb_gen,
    video_path: Path,
) -> None:
    """Render pairwise review section for ambiguous cluster merges."""
    st.caption("Review candidate pairs for potential merges or contamination")

    # For now, identify candidate pairs as clusters with contamination > threshold
    # In the future, this could load from contamination_audit.json or use centroid similarity
    contam_threshold = 0.1
    candidates = person_clusters[person_clusters["contam_rate"] > contam_threshold]

    if len(candidates) < 2:
        st.info("No ambiguous pairs detected. All clusters look distinct.")
        return

    # Generate pairs from high-contamination clusters
    cluster_ids = candidates["cluster_id"].tolist()
    pairs = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, min(i + 3, len(cluster_ids))):  # Limit to top few pairs
            pairs.append((int(cluster_ids[i]), int(cluster_ids[j])))

    if not pairs:
        st.info("No pairs to review.")
        return

    episode_id = mutator.episode_id
    episode_hash = get_episode_hash(episode_id, mutator.data_root)

    for idx, (cid_a, cid_b) in enumerate(pairs):
        st.markdown(f"#### Pair {idx + 1}: Cluster {cid_a} vs Cluster {cid_b}")

        # Get track previews for each cluster
        cluster_a = cluster_map.get(cid_a, {})
        cluster_b = cluster_map.get(cid_b, {})
        tracks_a = [int(tid) for tid in cluster_a.get("track_ids", [])][:6]  # Limit to 6
        tracks_b = [int(tid) for tid in cluster_b.get("track_ids", [])][:6]

        col_a, col_b = st.columns(2)

        with col_a:
            st.caption(f"Cluster {cid_a} ({len(tracks_a)} tracks)")
            if tracks_a:
                images_a = [get_track_still_path(episode_id, tid, episode_hash) for tid in tracks_a]
                render_strip(
                    images_a,
                    ids=tracks_a,
                    key_prefix="ws_pair",
                    strip_id=f"p{idx}_a{cid_a}",
                    selectable=False,
                )

        with col_b:
            st.caption(f"Cluster {cid_b} ({len(tracks_b)} tracks)")
            if tracks_b:
                images_b = [get_track_still_path(episode_id, tid, episode_hash) for tid in tracks_b]
                render_strip(
                    images_b,
                    ids=tracks_b,
                    key_prefix="ws_pair",
                    strip_id=f"p{idx}_b{cid_b}",
                    selectable=False,
                )

        # Action buttons
        action_cols = st.columns([1, 1, 1, 3])
        with action_cols[0]:
            if st.button(
                "Merge →",
                key=wkey("ws_pair", episode_id, person_name, cid_a, cid_b, "merge"),
                help="Merge these clusters together",
            ):
                st.info(f"Merge action for clusters {cid_a} and {cid_b} - TODO: implement merge logic")
                # TODO: Implement merge via mutator API

        with action_cols[1]:
            if st.button(
                "Not This Person",
                key=wkey("ws_pair", episode_id, person_name, cid_a, cid_b, "reject"),
                help="Move one cluster to different identity",
            ):
                st.info("Select which cluster to reassign - TODO: show identity picker")
                # TODO: Show identity picker and move cluster

        with action_cols[2]:
            if st.button(
                "Open Cluster A",
                key=wkey("ws_pair", episode_id, person_name, cid_a, cid_b, "open_a"),
            ):
                st.session_state["workspace_selected_cluster"] = cid_a
                st.rerun()

        st.markdown("---")


# Helper utilities ---------------------------------------------------------


def _build_track_map(tracks_data: dict) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for track in tracks_data.get("tracks", []) or []:
        mapping[int(track["track_id"])] = track
    return mapping


def _cluster_reasons(
    cluster_row: pd.Series,
    low_threshold: float,
    contam_threshold: float,
) -> List[str]:
    reasons: List[str] = []
    p25 = float(cluster_row.get("tracks_conf_p25_median", 0.0))
    contam = float(cluster_row.get("contam_rate", 0.0))
    if p25 < low_threshold:
        reasons.append(f"LOW p25 ({p25:.2f} < {low_threshold:.2f})")
    if contam >= contam_threshold:
        reasons.append(f"HIGH contam ({contam:.2f} ≥ {contam_threshold:.2f})")
    return reasons


def _format_cluster_metrics(p25, min_track, contam) -> str:
    parts: List[str] = []
    parts.append(_format_metric("Median p25", p25))
    parts.append(_format_metric("Min track p25", min_track))
    parts.append(_format_metric("Contam", contam))
    return " · ".join(part for part in parts if part)


def _format_track_metrics(conf_p25, conf_mean, conf_min, conflict_frac) -> str:
    parts: List[str] = []
    parts.append(_format_metric("p25", conf_p25))
    parts.append(_format_metric("mean", conf_mean))
    parts.append(_format_metric("min", conf_min))
    parts.append(_format_metric("conflict", conflict_frac))
    return " · ".join(part for part in parts if part)


def _format_metric(label: str, value) -> str:
    if value is None:
        return f"**{label}:** n/a"
    try:
        return f"**{label}:** {float(value):.2f}"
    except (TypeError, ValueError):
        return f"**{label}:** n/a"


def _render_cluster_header(
    cluster_row: pd.Series,
    thresholds: dict,
    reasons: Sequence[str] | None,
) -> None:
    cluster_id = int(cluster_row["cluster_id"])
    n_tracks = int(cluster_row.get("n_tracks", 0))
    name = cluster_row.get("name", "Unknown")
    p25 = cluster_row.get("tracks_conf_p25_median")
    min_track = cluster_row.get("min_track_conf_p25")
    contam = cluster_row.get("contam_rate")

    st.markdown(f"#### Cluster {cluster_id}")
    st.caption(f"{name} · {n_tracks} tracks")

    metrics = _format_cluster_metrics(p25, min_track, contam)
    if metrics:
        st.markdown(metrics)

    if reasons:
        st.code("\n".join(reasons))


def _render_track_header(
    track_row: pd.Series,
    thresholds: dict,
    reasons: Sequence[str] | None,
) -> None:
    track_id = int(track_row["track_id"])
    person = track_row.get("person", "Unknown")
    cluster_id = track_row.get("cluster_id", "–")
    n_frames = int(track_row.get("n_frames", 0))
    conf_p25 = track_row.get("conf_p25")
    conf_mean = track_row.get("conf_mean")
    conf_min = track_row.get("conf_min")
    conflict_frac = track_row.get("conflict_frac")

    st.markdown(f"#### Track {track_id}")
    st.caption(f"{person} · Cluster {cluster_id} · {n_frames} frames")

    metrics = _format_track_metrics(conf_p25, conf_mean, conf_min, conflict_frac)
    if metrics:
        st.markdown(metrics)

    if reasons:
        st.code("\n".join(reasons))
