"""Faces workspace tab."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app.components.tiles import render_person_tile
from app.lib.facebank_loader import load_facebank_identities, merge_with_cluster_metrics
from app.lib.mutator_api import WorkspaceMutator
from app.workspace.actions import (
    identity_options,
    render_cluster_actions,
    render_track_actions,
)
from app.media.thumbnails import thumbnail_coverage
from app.workspace.common import render_stills_progress


def _infer_show_season_from_episode(episode_id: str) -> tuple[str, str]:
    """
    Infer show_id and season_id from episode_id.

    For now, defaults to rhobh/s05. In the future, could parse episode_id
    or look up in registry.
    """
    # TODO: Parse episode_id or look up in registry
    # For now, default to RHOBH S05
    return "rhobh", "s05"


def render_faces_tab(mutator: WorkspaceMutator) -> None:
    """
    Render Faces tab with person overview and drill-down.

    This tab is driven by the facebank (seeded cast members), not clusters.
    All cast members with seeds will appear, even if they have 0 clusters.
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

    thresholds = mutator.thresholds

    # Infer show/season from episode
    episode_id = st.session_state.get("workspace_episode") or st.session_state.get("episode_id") or ""
    show_id, season_id = _infer_show_season_from_episode(episode_id)

    # Show stills progress if incomplete
    done, total, coverage = render_stills_progress(mutator.episode_id, mutator.data_root, key_suffix="faces")

    generated_thumbs, total_thumbs, _ = thumbnail_coverage(mutator.episode_id)
    coverage_ratio = (generated_thumbs / float(total_thumbs)) if total_thumbs else 0.0
    if total_thumbs == 0 or coverage_ratio < 0.1:
        st.warning(
            "Track thumbnails are missing for this episode. Click **Generate thumbnails** in the "
            "workspace header (install ffmpeg for faster, exact seeks)."
        )

    # Load all facebank identities (this is the source of truth)
    facebank_identities = load_facebank_identities(show_id, season_id, mutator.data_root)

    if not facebank_identities:
        st.info(f"No seeded cast members found for {show_id} {season_id}. Go to CAST page to add seeds.")
        return

    # Merge with cluster metrics if available
    person_metrics_list = mutator.person_metrics_df.to_dict(orient="records") if not mutator.person_metrics_df.empty else []
    merged_identities = merge_with_cluster_metrics(facebank_identities, person_metrics_list)

    st.caption("Tap a person to open their clusters and tracks.")

    cols_per_row = 4
    for row_start in range(0, len(merged_identities), cols_per_row):
        cols = st.columns(cols_per_row)
        for idx, record in enumerate(merged_identities[row_start : row_start + cols_per_row]):
            with cols[idx]:
                tile_key = f"person_tile_{record['person']}"
                if render_person_tile(record, thresholds, tile_key, show_id, season_id):
                    # Navigate to Clusters tab with person filter
                    st.session_state["workspace_tab"] = "Clusters"
                    st.session_state["clusters_person"] = record["person"]
                    st.session_state["workspace_selected_cluster"] = None
                    st.rerun()


def _render_person_summary(person_df: pd.DataFrame, person_name: str) -> None:
    subset = person_df[person_df["person"] == person_name]
    if subset.empty:
        return

    row = subset.iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Clusters", int(row.get("n_clusters", 0)))
        st.metric("Tracks", int(row.get("n_tracks", 0)))
    with col2:
        st.metric("Bank p25 (median)", f"{row.get('bank_conf_median_p25', 0.0):.2f}")
    with col3:
        st.metric("Contamination", f"{row.get('bank_contam_rate', 0.0):.2f}")


def _render_track_section(mutator: WorkspaceMutator, identities) -> None:
    cluster_id = st.session_state.get("workspace_selected_cluster")
    tracks_df = mutator.track_metrics_df
    cluster_tracks = tracks_df[tracks_df["cluster_id"] == cluster_id]

    if cluster_tracks.empty:
        st.info("No tracks remaining in this cluster.")
        return

    st.markdown("---")
    st.subheader(f"Tracks · Cluster {cluster_id}")

    episode_id = mutator.episode_id

    for _, track_row in cluster_tracks.iterrows():
        render_track_actions(
            track_row,
            identities,
            episode_id=episode_id,
            context="faces_track",
        )
