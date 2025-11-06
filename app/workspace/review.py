"""Workspace review summary tab."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.lib.mutator_api import WorkspaceMutator
from app.lib.analytics_dirty import get_analytics_freshness, is_analytics_dirty
from jobs.tasks.analytics import analytics_task


def render_review_tab(mutator: WorkspaceMutator) -> None:
    # Preflight check
    from app.workspace.common import check_artifacts_status
    from app.utils.ui_keys import safe_rerun

    artifacts = check_artifacts_status(mutator.episode_id, str(mutator.data_root))

    if artifacts["needs_detect"] or artifacts["needs_track"] or artifacts["needs_cluster"]:
        st.warning(f"⚠️ {artifacts['message']}")

        if artifacts["next_action"]:
            if st.button(f"▶️ {artifacts['next_action']}", key="review_preflight_btn", type="primary"):
                st.session_state["_trigger_cluster"] = True
                safe_rerun()

        with st.expander("Artifact status"):
            st.json(artifacts["artifacts"])

        return

    low_clusters = mutator.get_low_confidence_clusters()
    low_tracks = mutator.get_low_confidence_tracks()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Low-Confidence Clusters", len(low_clusters))
        if st.button("Open Clusters Queue"):
            st.session_state["workspace_tab"] = "Clusters"
            st.rerun()
    with col2:
        st.metric("Low-Confidence Tracks", len(low_tracks))
        if st.button("Open Tracks Queue"):
            st.session_state["workspace_tab"] = "Tracks"
            st.rerun()

    st.markdown("---")
    st.subheader("Analytics Freshness")

    episode_id = mutator.episode_id
    data_root = mutator.data_root

    dirty_flag, dirty_reason = is_analytics_dirty(episode_id, data_root)
    freshness = get_analytics_freshness(episode_id, data_root)

    if dirty_flag:
        st.warning(f"Analytics marked stale ({dirty_reason}). Rebuild recommended.")
    elif freshness.get("is_fresh"):
        st.success("Analytics are fresh.")
    else:
        st.info("Analytics status unknown. Consider rebuilding to verify outputs.")

    if st.button("Rebuild Analytics", help="Run analytics pipeline using current clusters"):
        _trigger_analytics_rebuild(mutator)


def _trigger_analytics_rebuild(mutator: WorkspaceMutator) -> None:
    episode_id = mutator.episode_id
    data_root: Path = mutator.data_root
    try:
        clusters_data = mutator.clusters
        cluster_assignments = {
            cluster["cluster_id"]: cluster.get("name", "Unknown")
            for cluster in clusters_data.get("clusters", [])
        }
        with st.spinner("Rebuilding analytics..."):
            result = analytics_task("workspace", episode_id, cluster_assignments)
        st.success(
            f"Analytics rebuilt — {result['stats']['intervals_created']} intervals generated."
        )
    except Exception as exc:
        st.error(f"Failed to rebuild analytics: {exc}")
