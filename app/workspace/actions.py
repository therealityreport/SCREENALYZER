"""Workspace action helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import pandas as pd
import streamlit as st

from app.lib.mutator_api import (
    assign_name,
    assign_tracks_to_identity,
    delete_cluster,
    delete_track,
    move_track,
    split_frames_and_assign,
)
from app.utils.ui_keys import wkey


def identity_options(person_df: pd.DataFrame) -> List[str]:
    options = sorted({str(name) for name in person_df.get("person", []).tolist()})
    if "Unknown" not in options:
        options.append("Unknown")
    return options


def render_cluster_actions(
    mutator,
    cluster_row: pd.Series,
    identities: Sequence[str],
    *,
    key_suffix: str,
    selectable_tracks: Sequence[int] | None = None,
    show_view_button: bool = False,
    is_selected: bool = False,
) -> tuple[bool, Dict[str, str]]:
    """
    Render action controls for a cluster card.

    Args:
        key_suffix: Unique suffix for widget keys (e.g., "cid9_tabClusters")

    Returns:
        Tuple of (view_clicked, key_map)
    """
    cluster_id = int(cluster_row["cluster_id"])
    identities = list(identities)
    if not identities:
        identities = ["Unknown"]
    view_clicked = False

    action_cols = st.columns([1.4, 1.8, 1])

    assign_key = wkey("ws", "clusters", key_suffix, "assign")
    lock_key = wkey("ws", "clusters", key_suffix, "lock")
    assign_btn_key = wkey("ws", "clusters", key_suffix, "assign_button")
    move_tracks_key = wkey("ws", "clusters", key_suffix, "move_tracks")
    move_identity_key = wkey("ws", "clusters", key_suffix, "move_identity")
    move_button_key = wkey("ws", "clusters", key_suffix, "move_button")
    delete_key = wkey("ws", "clusters", key_suffix, "delete")
    view_key = wkey("ws", "clusters", key_suffix, "view")

    key_map = {
        "assign_select": assign_key,
        "lock": lock_key,
        "assign_button": assign_btn_key,
        "move_tracks": move_tracks_key,
        "move_identity": move_identity_key,
        "move_button": move_button_key,
        "delete": delete_key,
        "view": view_key,
    }

    with action_cols[0]:
        current_name = cluster_row.get("name", "Unknown")
        default_index = identities.index(current_name) if current_name in identities else 0

        target_identity = st.selectbox(
            "Assign identity",
            options=identities,
            index=default_index,
            key=assign_key,
        )
        lock_cluster = st.checkbox(
            "Lock assignment",
            value=True,
            key=lock_key,
        )
        if st.button(
            "Assign",
            key=assign_btn_key,
        ):
            result = assign_name(cluster_id, target_identity, lock_cluster)
            _handle_result(result, f"Assigned cluster {cluster_id} to {target_identity}")

    with action_cols[1]:
        track_choices = list(selectable_tracks or [])
        selected_tracks = st.multiselect(
            "Tracks to move",
            options=track_choices,
            key=move_tracks_key,
            help="Select tracks to reassign to another identity.",
        )
        target_identity_move = st.selectbox(
            "Target identity",
            options=identities,
            key=move_identity_key,
        )
        if st.button(
            "Move selected",
            key=move_button_key,
            disabled=len(selected_tracks) == 0,
        ):
            result = assign_tracks_to_identity(
                selected_tracks,
                target_identity_move,
                source_cluster_id=cluster_id,
            )
            _handle_result(
                result,
                f"Moved {len(selected_tracks)} track(s) to {target_identity_move}",
            )

    with action_cols[2]:
        if st.button(
            "Delete cluster",
            key=delete_key,
            help="Remove entire cluster (tracks will be removed).",
        ):
            result = delete_cluster(cluster_id)
            _handle_result(result, f"Deleted cluster {cluster_id}")

        if show_view_button:
            label = "Tracks shown" if is_selected else "View tracks"
            if st.button(label, key=view_key):
                view_clicked = True

    return view_clicked, key_map


def render_track_actions(
    track_row: pd.Series,
    identities: Sequence[str],
    *,
    selected_frames: Iterable[int] | None = None,
    episode_id: str | None = None,
    context: str = "default",
) -> None:
    track_id = int(track_row["track_id"])
    identities = list(identities)
    if not identities:
        identities = ["Unknown"]
    selected_frames = list(selected_frames or [])

    # Build unique key prefix using episode_id and context to avoid duplicates
    if episode_id:
        key_prefix = (episode_id, "tracks", track_id, context)
    else:
        key_prefix = ("tracks", track_id, context)

    move_identity = st.selectbox(
        "Move whole track",
        options=identities,
        key=wkey(*key_prefix, "move_identity"),
    )
    if st.button("Move", key=wkey(*key_prefix, "move_button")):
        cluster_id = int(track_row.get("cluster_id"))
        result = assign_tracks_to_identity(
            [track_id],
            move_identity,
            source_cluster_id=cluster_id,
        )
        _handle_result(result, f"Moved track {track_id} to {move_identity}")

    split_cols = st.columns([2, 1])
    with split_cols[0]:
        st.caption(
            "Select frames in the strip above to enable split assignment."
        )
    with split_cols[1]:
        split_identity = st.selectbox(
            "Assign split to",
            options=identities,
            key=wkey(*key_prefix, "split_identity"),
        )

    if st.button(
        "Split & Assign",
        key=wkey(*key_prefix, "split_button"),
        disabled=len(selected_frames) == 0,
    ):
        cluster_id = int(track_row.get("cluster_id"))
        if len(selected_frames) >= int(track_row.get("n_frames", 0)):
            # All frames selected – treat as move whole track
            result = assign_tracks_to_identity(
                [track_id],
                split_identity,
                source_cluster_id=cluster_id,
            )
            _handle_result(
                result,
                f"Moved entire track {track_id} to {split_identity}",
            )
        else:
            result = split_frames_and_assign(track_id, selected_frames, split_identity)
            _handle_result(
                result,
                f"Split {len(selected_frames)} frame(s) from track {track_id}",
            )

    delete_cols = st.columns(2)
    with delete_cols[0]:
        if st.button("Delete track", key=wkey(*key_prefix, "delete")):
            cluster_id = int(track_row.get("cluster_id"))
            result = delete_track(track_id, cluster_id)
            _handle_result(result, f"Deleted track {track_id}")
    with delete_cols[1]:
        dest_cluster = st.number_input(
            "Move to cluster",
            min_value=0,
            step=1,
            key=wkey(*key_prefix, "move_cluster_id"),
        )
        if st.button("Direct move", key=wkey(*key_prefix, "direct_move")):
            from_cluster = int(track_row.get("cluster_id"))
            result = move_track(track_id, from_cluster, int(dest_cluster))
            _handle_result(
                result,
                f"Moved track {track_id} to cluster {int(dest_cluster)}",
            )


def _handle_result(result: dict, success_message: str) -> None:
    if result.get("ok"):
        st.success(success_message)
        st.rerun()
    else:
        st.error(result.get("error", "Unknown error"))
