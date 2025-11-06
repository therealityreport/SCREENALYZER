"""Tracks workspace tab."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import streamlit as st

from app.components.strip import render_strip
from app.lib.mutator_api import WorkspaceMutator
from app.workspace.actions import identity_options, render_track_actions
from app.workspace.common import (
    confidence_badge,
    contamination_badge,
    ensure_workspace_styles,
    get_thumbnail_generator,
    track_frame_images,
    track_preview_image,
)
from app.utils.ui_keys import wkey
from screentime.utils import get_video_path


def render_tracks_tab(mutator: WorkspaceMutator) -> None:
    # Preflight check
    from app.workspace.common import check_artifacts_status
    from app.utils.ui_keys import safe_rerun

    artifacts = check_artifacts_status(mutator.episode_id, str(mutator.data_root))

    if artifacts["needs_detect"] or artifacts["needs_track"] or artifacts["needs_cluster"]:
        st.warning(f"⚠️ {artifacts['message']}")

        if artifacts["next_action"]:
            if st.button(f"▶️ {artifacts['next_action']}", key="tracks_preflight_btn", type="primary"):
                st.session_state["_trigger_cluster"] = True
                safe_rerun()

        with st.expander("Artifact status"):
            st.json(artifacts["artifacts"])

        return

    tracks_df = mutator.track_metrics_df
    if tracks_df.empty:
        st.info("No tracks available. Re-run clustering to populate this view.")
        return

    ensure_workspace_styles()

    identities = identity_options(mutator.person_metrics_df)
    thresholds = mutator.thresholds
    track_map = _build_track_map(mutator.tracks)
    thumb_gen = get_thumbnail_generator()
    video_path = get_video_path(mutator.episode_id, mutator.data_root)

    cluster_options = sorted({int(cid) for cid in tracks_df["cluster_id"].unique().tolist()})

    filter_cols = st.columns([2, 2, 1])
    with filter_cols[0]:
        selected_people = st.multiselect(
            "Filter by identity",
            options=identities,
            default=[],
            key=wkey("tracks", "filter_identity"),
        )
        selected_clusters = st.multiselect(
            "Filter by cluster",
            options=cluster_options,
            default=[],
            key=wkey("tracks", "filter_cluster"),
        )
    with filter_cols[1]:
        conf_range = st.slider(
            "Confidence p25 range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            key=wkey("tracks", "filter_conf_range"),
        )
        conflict_max = st.slider(
            "Max conflict fraction",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            key=wkey("tracks", "filter_conflict"),
        )
    with filter_cols[2]:
        min_frames = st.number_input(
            "Min frames",
            min_value=0,
            value=0,
            step=1,
            key=wkey("tracks", "filter_min_frames"),
        )

    filtered_tracks = _apply_track_filters(
        tracks_df,
        selected_people,
        selected_clusters,
        conf_range,
        conflict_max,
        min_frames,
    )

    low_df = mutator.get_low_confidence_tracks()
    low_df = _apply_track_filters(
        low_df,
        selected_people,
        selected_clusters,
        conf_range,
        conflict_max,
        min_frames,
    )

    all_tab, low_tab = st.tabs(["All Tracks", "Low-Confidence Tracks"])

    with all_tab:
        st.caption(f"{len(filtered_tracks)} tracks")
        if filtered_tracks.empty:
            st.info("No tracks match the current filters.")
        else:
            for _, track_row in filtered_tracks.sort_values("conf_p25", ascending=False).iterrows():
                _render_track_card(
                    track_row,
                    identities,
                    thresholds,
                    track_map,
                    thumb_gen,
                    video_path,
                    mutator.episode_id,
                    context="all",
                )

    with low_tab:
        if low_df.empty:
            st.success("No low-confidence tracks 🎉")
        else:
            low_threshold = float(thresholds.get("track_low_p25", 0.55))
            conflict_high = float(thresholds.get("track_conflict_frac_high", 0.2))
            for _, track_row in low_df.sort_values("conf_p25").iterrows():
                reasons = _track_reasons(track_row, low_threshold, conflict_high)
                _render_track_card(
                    track_row,
                    identities,
                    thresholds,
                    track_map,
                    thumb_gen,
                    video_path,
                    mutator.episode_id,
                    context="low",
                    reasons=reasons,
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
    context: str = "default",
    reasons: Sequence[str] | None = None,
) -> None:
    track_id = int(track_row["track_id"])
    track_dict = track_map.get(track_id)

    header_html = _track_header_html(track_row, thresholds, reasons)
    st.markdown(header_html, unsafe_allow_html=True)

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
            strip_id=f"t{track_id}_{context}",
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
        context=context,
    )
    st.markdown("---")


# Helper utilities ---------------------------------------------------------


def _build_track_map(tracks_data: dict) -> Dict[int, dict]:
    mapping: Dict[int, dict] = {}
    for track in tracks_data.get("tracks", []) or []:
        mapping[int(track["track_id"])] = track
    return mapping


def _apply_track_filters(
    df: pd.DataFrame,
    selected_people: List[str],
    selected_clusters: List[int],
    conf_range: tuple[float, float],
    conflict_max: float,
    min_frames: int,
) -> pd.DataFrame:
    filtered = df.copy()
    if filtered.empty:
        return filtered

    if selected_people:
        filtered = filtered[filtered["person"].isin(selected_people)]
    if selected_clusters:
        filtered = filtered[filtered["cluster_id"].isin(selected_clusters)]
    if conf_range != (0.0, 1.0):
        filtered = filtered[
            (filtered["conf_p25"] >= conf_range[0])
            & (filtered["conf_p25"] <= conf_range[1])
        ]
    if conflict_max < 1.0:
        filtered = filtered[filtered["conflict_frac"] <= conflict_max]
    if min_frames > 0:
        filtered = filtered[filtered["n_frames"] >= min_frames]
    return filtered


def _track_reasons(
    track_row: pd.Series,
    low_threshold: float,
    conflict_high: float,
) -> List[str]:
    reasons: List[str] = []
    p25 = float(track_row.get("conf_p25", 0.0))
    conflict = float(track_row.get("conflict_frac", 0.0))
    if p25 < low_threshold:
        reasons.append(f"LOW p25 ({p25:.2f} < {low_threshold:.2f})")
    if conflict >= conflict_high:
        reasons.append(f"HIGH conflict ({conflict:.2f} ≥ {conflict_high:.2f})")
    return reasons


def _track_header_html(
    track_row: pd.Series,
    thresholds: dict,
    reasons: Sequence[str] | None,
) -> str:
    track_id = int(track_row["track_id"])
    person = track_row.get("person", "Unknown")
    cluster_id = track_row.get("cluster_id", "–")
    n_frames = int(track_row.get("n_frames", 0))
    conf_p25 = track_row.get("conf_p25")
    conf_mean = track_row.get("conf_mean")
    conf_min = track_row.get("conf_min")
    conflict_frac = track_row.get("conflict_frac")

    high = float(thresholds.get("track_high_p25", 0.7))
    mid = float(thresholds.get("track_low_p25", 0.55))
    conflict_high = float(thresholds.get("track_conflict_frac_high", 0.2))

    badges = "".join(
        [
            confidence_badge("p25", conf_p25, mid, high),
            confidence_badge("mean", conf_mean, mid, high),
            confidence_badge("min", conf_min, mid, high),
            contamination_badge("conflict", conflict_frac, conflict_high),
        ]
    )

    pills = ""
    if reasons:
        pills = "".join(f'<span class="why-pill">{reason}</span>' for reason in reasons)

    return f"""
    <div class="workspace-card-header">
        <div>
            <h4>Track {track_id}</h4>
            <div>{person} · Cluster {cluster_id} · {n_frames} frames</div>
            {pills}
        </div>
        <div class="badge-row">{badges}</div>
    </div>
    """
