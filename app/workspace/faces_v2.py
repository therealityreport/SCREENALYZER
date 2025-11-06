"""
Faces workspace tab - Phase 3 P2 redesign.

Two sections:
1. Cast Faces - Known identities from facebank
2. Other Faces - Unassigned clusters
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.tiles import render_person_tile
from app.lib.facebank_loader import load_facebank_identities, merge_with_cluster_metrics
from app.lib.mutator_api import WorkspaceMutator
from app.workspace.cluster_ops import refine_clusters, get_cluster_stats
from app.media.thumbnails import thumbnail_coverage
from app.workspace.common import render_stills_progress


def _infer_show_season_from_episode(episode_id: str) -> tuple[str, str]:
    """Infer show_id and season_id from episode_id."""
    # TODO: Parse episode_id or look up in registry
    return "rhobh", "s05"


def render_cast_faces_section(
    merged_identities: list[dict],
    mutator: WorkspaceMutator,
    show_id: str,
    season_id: str,
) -> None:
    """
    Render Cast Faces section (assigned identities).

    Args:
        merged_identities: List of facebank identities with metrics
        mutator: Workspace mutator
        show_id: Show ID
        season_id: Season ID
    """
    st.markdown("### 👥 Cast Faces")
    st.caption("Tap a person to view their clusters and tracks")

    if not merged_identities:
        st.info("No cast members found. Go to CAST page to add seeds.")
        return

    # Filter to only assigned identities (those with clusters)
    cast_with_clusters = [
        identity for identity in merged_identities
        if identity.get("n_clusters", 0) > 0
    ]

    if not cast_with_clusters:
        st.info("No clusters assigned yet. Run clustering first.")
        return

    # Render grid
    cols_per_row = 4
    for row_start in range(0, len(cast_with_clusters), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, record in enumerate(cast_with_clusters[row_start : row_start + cols_per_row]):
            with cols[idx]:
                tile_key = f"cast_tile_{record['person']}"
                if render_person_tile(record, mutator.thresholds, tile_key, show_id, season_id):
                    # Navigate to Clusters tab with person filter
                    st.session_state["workspace_tab"] = "Clusters"
                    st.session_state["clusters_person"] = record["person"]
                    st.session_state["workspace_selected_cluster"] = None
                    st.rerun()


def render_other_faces_section(
    mutator: WorkspaceMutator,
) -> None:
    """
    Render Other Faces section (unassigned clusters).

    Args:
        mutator: Workspace mutator
    """
    st.markdown("---")
    st.markdown("### 🕵️ Other Faces")
    st.caption("Unassigned clusters excluded from analytics")

    # Get cluster stats
    stats = get_cluster_stats(mutator.episode_id, mutator.data_root)

    unassigned_count = stats.get("unassigned_clusters", 0)

    if unassigned_count == 0:
        st.success("✅ All clusters assigned!")
        return

    st.info(f"Found {unassigned_count} unassigned cluster(s)")

    # Load unassigned clusters
    import json
    clusters_file = Path(mutator.data_root) / "harvest" / mutator.episode_id / "clusters.json"

    if not clusters_file.exists():
        st.warning("Clusters file not found")
        return

    with open(clusters_file, "r") as f:
        clusters_data = json.load(f)

    clusters = clusters_data.get("clusters", [])
    unassigned_clusters = [
        c for c in clusters
        if not c.get("person_id") or c.get("person_id") == "unassigned"
    ]

    # Show first few unassigned clusters
    display_limit = 12
    displayed = 0

    cols_per_row = 4
    row_cols = st.columns(cols_per_row)
    col_idx = 0

    for cluster in unassigned_clusters[:display_limit]:
        cluster_id = cluster.get("cluster_id")
        tracks = cluster.get("tracks", [])

        if not tracks:
            continue

        # Get representative track (first one)
        track = tracks[0]

        with row_cols[col_idx]:
            # Show thumbnail (if available)
            st.markdown(f"**Cluster {cluster_id}**")
            st.caption(f"{len(tracks)} track(s)")

            # Button to view cluster
            if st.button("View", key=f"other_face_{cluster_id}"):
                st.session_state["workspace_tab"] = "Clusters"
                st.session_state["workspace_selected_cluster"] = cluster_id
                st.rerun()

        displayed += 1
        col_idx = (col_idx + 1) % cols_per_row

        # Start new row
        if col_idx == 0 and displayed < len(unassigned_clusters):
            row_cols = st.columns(cols_per_row)

    if len(unassigned_clusters) > display_limit:
        st.info(f"Showing {display_limit} of {len(unassigned_clusters)} unassigned clusters. Go to Clusters tab to see all.")


def render_refine_clusters_button(mutator: WorkspaceMutator) -> None:
    """
    Render Refine Clusters button.

    Args:
        mutator: Workspace mutator
    """
    st.markdown("---")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("### ✨ Refine Clusters")
        st.caption("Recompute centroids, eject outliers, and merge duplicates")

    with col2:
        if st.button("Refine", type="primary", key="refine_clusters_btn", use_container_width=True):
            with st.spinner("Refining clusters..."):
                result = refine_clusters(mutator.episode_id, mutator.data_root)

                if result.get("success"):
                    st.success(
                        f"✅ Refinement complete!\n\n"
                        f"- Centroids recomputed: {result['centroids_recomputed']}\n"
                        f"- Outliers ejected: {result['outliers_ejected']}\n"
                        f"- Clusters merged: {result['clusters_merged']}\n"
                        f"- Clusters updated: {result['clusters_updated']}"
                    )

                    # Reload data
                    st.session_state["_needs_reload"] = True
                    st.rerun()
                else:
                    st.error(f"❌ Refinement failed: {result.get('error')}")


def render_faces_tab(mutator: WorkspaceMutator) -> None:
    """
    Render Faces tab with Cast and Other sections.

    Args:
        mutator: Workspace mutator
    """
    # Preflight check
    from app.workspace.common import check_artifacts_status
    from app.utils.ui_keys import safe_rerun

    artifacts = check_artifacts_status(mutator.episode_id, str(mutator.data_root))

    if artifacts["needs_detect"] or artifacts["needs_track"] or artifacts["needs_cluster"]:
        st.warning(f"⚠️ {artifacts['message']}")

        if artifacts["next_action"]:
            if st.button(f"▶️ {artifacts['next_action']}", key="faces_preflight_btn", type="primary"):
                st.session_state["_trigger_cluster"] = True
                safe_rerun()

        with st.expander("Artifact status"):
            st.json(artifacts["artifacts"])

        return

    # Show stills progress
    done, total, coverage = render_stills_progress(
        mutator.episode_id,
        mutator.data_root,
        key_suffix="faces"
    )

    # Check thumbnail coverage
    generated_thumbs, total_thumbs, _ = thumbnail_coverage(mutator.episode_id)
    coverage_ratio = (generated_thumbs / float(total_thumbs)) if total_thumbs else 0.0

    if total_thumbs == 0 or coverage_ratio < 0.1:
        st.warning(
            "Track thumbnails are missing. Click **Generate thumbnails** in the "
            "workspace header (install ffmpeg for faster seeks)."
        )

    # Infer show/season
    episode_id = st.session_state.get("workspace_episode") or st.session_state.get("episode_id") or ""
    show_id, season_id = _infer_show_season_from_episode(episode_id)

    # Load facebank identities
    facebank_identities = load_facebank_identities(show_id, season_id, mutator.data_root)

    if not facebank_identities:
        st.info(f"No cast members found for {show_id} {season_id}. Go to CAST page to add seeds.")
        return

    # Merge with cluster metrics
    person_metrics_list = mutator.person_metrics_df.to_dict(orient="records") if not mutator.person_metrics_df.empty else []
    merged_identities = merge_with_cluster_metrics(facebank_identities, person_metrics_list)

    # Render Cast Faces section
    render_cast_faces_section(merged_identities, mutator, show_id, season_id)

    # Render Other Faces section
    render_other_faces_section(mutator)

    # Render Refine Clusters button
    render_refine_clusters_button(mutator)
