"""Faces workspace tab."""

from __future__ import annotations

from pathlib import Path
import json

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


def _render_frames_preview(episode_id: str, data_root: Path) -> None:
    """
    Render a small grid of sample frames to confirm assets are ready.

    Shows frames from manifest or frames directory if available.
    """
    st.markdown("### 📁 Assets Ready")

    # Try to load manifest for frame count
    manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"

    if manifest_path.exists():
        try:
            manifest_df = pd.read_parquet(manifest_path)
            frame_count = len(manifest_df)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"✅ {frame_count:,} frames extracted and indexed")
            with col2:
                if st.button("📊 View Manifest", key="view_manifest_btn", use_container_width=True):
                    with st.expander("Frame Manifest", expanded=True):
                        st.dataframe(manifest_df.head(20), use_container_width=True)

            # Show sample frames if frames directory exists
            frames_dir = data_root / "harvest" / episode_id / "frames"
            if frames_dir.exists():
                frame_files = sorted(frames_dir.glob("*.jpg"))[:6]  # Show first 6 frames

                if frame_files:
                    st.caption("Sample frames:")
                    cols = st.columns(6)
                    for idx, frame_path in enumerate(frame_files):
                        with cols[idx]:
                            st.image(str(frame_path), use_column_width=True, caption=f"Frame {idx}")
        except Exception as e:
            st.warning(f"Could not load manifest: {e}")
    else:
        st.info("No frame manifest found. Run detection first.")

    st.markdown("---")


def _split_cast_and_other(merged_identities: list[dict], thresholds: dict) -> tuple[list[dict], list[dict]]:
    """
    Split identities into Cast Faces and Other Faces.

    Cast Faces: High-confidence assignments to facebank members
    Other Faces: Low-confidence, unassigned, or explicitly marked as "other"

    Args:
        merged_identities: List of person records with metrics
        thresholds: Confidence thresholds

    Returns:
        (cast_faces, other_faces) tuple of lists
    """
    cast_faces = []
    other_faces = []

    # Threshold for what counts as "confirmed cast"
    min_assignment_conf = thresholds.get("min_assignment_conf", 0.45)
    min_bank_conf = thresholds.get("min_bank_conf_p25", 0.35)

    for record in merged_identities:
        person = record.get("person", "")
        bank_conf = record.get("bank_conf_median_p25", 0.0)
        n_clusters = record.get("n_clusters", 0)

        # Mark as "other" if:
        # - Person is "Unknown" or "Unassigned"
        # - Low bank confidence
        # - No clusters assigned
        # - Explicitly marked as "other_faces" (future enhancement)

        if person in ("Unknown", "Unassigned", "Other"):
            other_faces.append(record)
        elif n_clusters == 0:
            # Facebank member with no assigned clusters yet
            cast_faces.append(record)
        elif bank_conf < min_bank_conf:
            other_faces.append(record)
        else:
            cast_faces.append(record)

    return cast_faces, other_faces


def render_faces_tab(mutator: WorkspaceMutator) -> None:
    """
    Render Faces tab with Cast and Other sections.

    Phase 3 P2: Split into:
    - Cast Faces: Confirmed cast members with high-confidence assignments
    - Other Faces: Unassigned or low-confidence clusters (excluded from analytics)
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

    # Phase 3 P2: Frames Preview
    _render_frames_preview(mutator.episode_id, mutator.data_root)

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

    # Phase 3 P2: Split into Cast vs Other
    cast_faces, other_faces = _split_cast_and_other(merged_identities, thresholds)

    # Phase 3 P2 Section 1: Cast Faces
    st.markdown("## 👥 Cast Faces")
    st.caption(f"{len(cast_faces)} confirmed cast members • Click a person to view their clusters")

    if cast_faces:
        cols_per_row = 4
        for row_start in range(0, len(cast_faces), cols_per_row):
            cols = st.columns(cols_per_row)
            for idx, record in enumerate(cast_faces[row_start : row_start + cols_per_row]):
                with cols[idx]:
                    tile_key = f"cast_tile_{record['person']}"
                    if render_person_tile(record, thresholds, tile_key, show_id, season_id):
                        # Navigate to Clusters tab with person filter
                        st.session_state["workspace_tab"] = "Clusters"
                        st.session_state["clusters_person"] = record["person"]
                        st.session_state["workspace_selected_cluster"] = None
                        st.rerun()
    else:
        st.info("No cast members assigned yet. Add seeds on the CAST page and run clustering.")

    st.markdown("---")

    # Phase 3 P2 Section 2: Other Faces (excluded from analytics)
    with st.expander(f"🕵️ Other Faces (Excluded from Analytics) • {len(other_faces)} unassigned/uncertain", expanded=False):
        st.caption(
            "These are clusters with low confidence or no cast assignment. "
            "They are **excluded** from analytics computations. "
            "Reassign them to cast members to include in screen time totals."
        )

        if other_faces:
            cols_per_row = 4
            for row_start in range(0, len(other_faces), cols_per_row):
                cols = st.columns(cols_per_row)
                for idx, record in enumerate(other_faces[row_start : row_start + cols_per_row]):
                    with cols[idx]:
                        tile_key = f"other_tile_{record['person']}"
                        if render_person_tile(record, thresholds, tile_key, show_id, season_id):
                            # Navigate to Clusters tab with person filter
                            st.session_state["workspace_tab"] = "Clusters"
                            st.session_state["clusters_person"] = record["person"]
                            st.session_state["workspace_selected_cluster"] = None
                            st.rerun()
        else:
            st.success("✅ All faces are confidently assigned to cast members!")


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
