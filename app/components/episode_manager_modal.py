"""
Episode Manager Modal - Phase 2

Streamlit modal with tabs for episode operations:
- Info: Episode metadata and health
- Move: Move to different show/season
- Remove: Soft archive with confirmation
- Restore: Unarchive episode
- Rehash: Validate files and update cache key

Features:
- Single actionable banner at top
- Progress tracking with ETA
- Safety confirmations
- Maintenance mode gating
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Callable, Dict, Any

from app.lib.episode_manager import (
    move_episode,
    remove_episode,
    restore_episode,
    rehash_episode,
    list_episodes as list_eps_manager,
)
from app.lib.pipeline import (
    check_pipeline_can_run,
    get_maintenance_reason,
    read_pipeline_state,
)
from app.lib.registry import load_registry
from app.utils.ui_keys import safe_rerun


def _get_episode_info(episode_id: str, data_root: Path) -> Dict[str, Any]:
    """Get episode metadata and health information."""
    # Find episode in registry
    reg = load_registry()
    ep_data = None
    show_id = None
    season_id = None
    season_num = None

    for show in reg.get("shows", []):
        for season in show.get("seasons", []):
            for ep in season.get("episodes", []):
                if ep.get("episode_id") == episode_id:
                    ep_data = ep
                    show_id = show.get("show_id")
                    season_id = season.get("season_id")
                    season_num = season.get("season_number")
                    break

    if not ep_data:
        return {"error": "Episode not found in registry"}

    # Check file health
    harvest_dir = data_root / "harvest" / episode_id
    files_health = {
        "manifest": (harvest_dir / "manifest.parquet").exists(),
        "embeddings": (harvest_dir / "embeddings.parquet").exists(),
        "tracks": (harvest_dir / "tracks.json").exists(),
        "clusters": (harvest_dir / "clusters.json").exists(),
        "stills": (harvest_dir / "stills").exists(),
    }

    # Get pipeline state
    pipeline_state = read_pipeline_state(episode_id, data_root)

    return {
        "episode_id": episode_id,
        "show_id": show_id,
        "season_id": season_id,
        "season_number": season_num,
        "archived": ep_data.get("archived", False),
        "episode_hash": ep_data.get("episode_hash"),
        "last_rehash": ep_data.get("last_rehash"),
        "files_health": files_health,
        "maintenance_mode": pipeline_state.get("maintenance_mode", False),
        "active_op": pipeline_state.get("active_op"),
        "pipeline_state": pipeline_state,
    }


def _render_progress_banner(episode_id: str, data_root: Path):
    """Render progress banner if operation is active."""
    state = read_pipeline_state(episode_id, data_root)

    if not state.get("maintenance_mode"):
        return

    active_op = state.get("active_op", {})
    op_type = active_op.get("type", "unknown")

    progress_data = state.get("active_op_progress", {})
    current = progress_data.get("current", 0)
    total = progress_data.get("total", 1)
    stage = progress_data.get("stage", "")

    # Get ETA if available
    eta_seconds = state.get("estimated_eta_seconds")
    eta_str = ""
    if eta_seconds and eta_seconds > 0:
        if eta_seconds < 60:
            eta_str = f" (ETA: {int(eta_seconds)}s)"
        else:
            eta_str = f" (ETA: {int(eta_seconds / 60)}m {int(eta_seconds % 60)}s)"

    op_names = {
        "move": "Moving",
        "remove": "Archiving",
        "restore": "Restoring",
        "rehash": "Rehashing",
    }
    op_name = op_names.get(op_type, op_type.title())

    # Progress bar
    progress_pct = current / max(total, 1) if total > 0 else 0.0

    st.info(f"🔄 {op_name} episode: {stage}{eta_str}")
    st.progress(progress_pct, text=f"Step {current} of {total}")


def _render_info_tab(episode_id: str, data_root: Path):
    """Render Info tab with episode metadata and health."""
    info = _get_episode_info(episode_id, data_root)

    if "error" in info:
        st.error(info["error"])
        return

    st.markdown("### Episode Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Episode ID:** `{info['episode_id']}`")
        st.markdown(f"**Show:** {info['show_id']}")
        st.markdown(f"**Season:** S{info['season_number']:02d} ({info['season_id']})")

        if info['archived']:
            st.warning("⚠️ **Status:** Archived")
        else:
            st.success("✅ **Status:** Active")

    with col2:
        st.markdown(f"**Cache Key:** `{info['episode_hash']}`")

        if info['last_rehash']:
            st.markdown(f"**Last Rehash:** {info['last_rehash'][:19]}")
        else:
            st.markdown("**Last Rehash:** Never")

        if info['maintenance_mode']:
            op_type = info['active_op'].get('type', 'unknown') if info['active_op'] else 'unknown'
            st.warning(f"⚠️ **Maintenance:** {op_type} in progress")

    st.markdown("---")
    st.markdown("### File Health")

    all_healthy = all(info['files_health'].values())

    for file_key, exists in info['files_health'].items():
        status = "✅" if exists else "❌"
        st.markdown(f"{status} **{file_key}**")

    if all_healthy:
        st.success("All files present")
    else:
        st.warning("Some files missing - consider running Rehash to validate")


def _render_move_tab(episode_id: str, data_root: Path, can_run: bool, block_reason: str):
    """Render Move tab."""
    st.markdown("### Move Episode")
    st.warning("⚠️ This will update the registry. All data will be preserved on disk.")

    # Get current location
    info = _get_episode_info(episode_id, data_root)
    if "error" not in info:
        st.info(f"Current location: **{info['show_id']}** / **S{info['season_number']:02d}**")

    # Get available shows and seasons
    reg = load_registry()
    shows = [(s.get("show_id", ""), s.get("seasons", [])) for s in reg.get("shows", [])]

    col1, col2 = st.columns(2)

    with col1:
        show_ids = [s[0] for s in shows]
        dst_show = st.selectbox(
            "Destination Show",
            options=show_ids,
            key="modal_move_show",
        )

    with col2:
        # Get seasons for selected show
        seasons = []
        for show_id, show_seasons in shows:
            if show_id == dst_show:
                seasons = [s.get("season_number") for s in show_seasons]
                break

        if seasons:
            dst_season = st.selectbox(
                "Destination Season",
                options=seasons,
                key="modal_move_season",
            )
        else:
            st.warning("No seasons available in selected show")
            dst_season = 1

    reason = st.text_input("Reason (optional)", key="modal_move_reason")

    if not can_run:
        st.error(f"🚫 Blocked: {block_reason}")

    if st.button(
        "✅ Confirm Move",
        key="modal_confirm_move_btn",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    ):
        try:
            with st.spinner(f"Moving {episode_id} to {dst_show}/S{dst_season:02d}..."):
                move_episode(episode_id, dst_show, dst_season, actor="user", reason=reason)
            st.success(f"✅ Moved {episode_id} successfully!")
            st.session_state["_modal_close"] = True
            safe_rerun()
        except Exception as e:
            st.error(f"❌ Move failed: {str(e)}")


def _render_remove_tab(episode_id: str, data_root: Path, can_run: bool, block_reason: str):
    """Render Remove tab."""
    st.markdown("### Archive Episode")
    st.warning("⚠️ Episode will be marked as archived but all data will be preserved.")

    st.markdown("""
    **What happens:**
    - Episode marked as `archived` in registry
    - Timestamp added: `archived_at`
    - Episode hidden from default listings
    - ALL data remains on disk (no files deleted)
    - Can be restored at any time
    """)

    understand = st.checkbox(
        "I understand this will archive the episode",
        key="modal_remove_understand"
    )

    confirm_id = st.text_input(
        f"Type `{episode_id}` to confirm:",
        key="modal_remove_confirm"
    )

    reason = st.text_input("Reason (optional)", key="modal_remove_reason")

    if not can_run:
        st.error(f"🚫 Blocked: {block_reason}")

    confirm_valid = understand and confirm_id == episode_id

    if st.button(
        "✅ Confirm Archive",
        key="modal_confirm_remove_btn",
        type="primary",
        disabled=not can_run or not confirm_valid,
        use_container_width=True,
    ):
        try:
            with st.spinner(f"Archiving {episode_id}..."):
                remove_episode(episode_id, soft=True, actor="user", reason=reason)
            st.success(f"✅ Archived {episode_id} successfully!")
            st.session_state["_modal_close"] = True
            safe_rerun()
        except Exception as e:
            st.error(f"❌ Archive failed: {str(e)}")


def _render_restore_tab(episode_id: str, data_root: Path, can_run: bool, block_reason: str):
    """Render Restore tab."""
    st.markdown("### Restore Episode")
    st.info("ℹ️ This will clear the archived flag and make the episode visible again.")

    st.markdown("""
    **What happens:**
    - `archived` flag cleared
    - `archived_at` timestamp removed
    - Episode becomes visible in default listings
    - No data changes (all files already on disk)
    """)

    reason = st.text_input("Reason (optional)", key="modal_restore_reason")

    if not can_run:
        st.error(f"🚫 Blocked: {block_reason}")

    if st.button(
        "✅ Restore Episode",
        key="modal_confirm_restore_btn",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    ):
        try:
            with st.spinner(f"Restoring {episode_id}..."):
                restore_episode(episode_id, actor="user", reason=reason)
            st.success(f"✅ Restored {episode_id} successfully!")
            st.session_state["_modal_close"] = True
            safe_rerun()
        except Exception as e:
            st.error(f"❌ Restore failed: {str(e)}")


def _render_rehash_tab(episode_id: str, data_root: Path, can_run: bool, block_reason: str):
    """Render Rehash tab."""
    st.markdown("### Rehash Episode")
    st.info("ℹ️ This will validate key files and update the episode hash for cache busting.")

    st.markdown("""
    **What happens:**
    - Validates all key files (manifest, embeddings, tracks, clusters, stills)
    - Computes new MD5 hash from file contents
    - Updates `episode_hash` in registry
    - Adds `last_rehash` timestamp
    - **Invalidates all caches** (thumbnails will regenerate on next load)
    - **Triggers thumbnail prewarm** in background
    """)

    reason = st.text_input("Reason (optional)", key="modal_rehash_reason")

    if not can_run:
        st.error(f"🚫 Blocked: {block_reason}")

    if st.button(
        "✅ Rehash Episode",
        key="modal_confirm_rehash_btn",
        type="primary",
        disabled=not can_run,
        use_container_width=True,
    ):
        try:
            with st.spinner(f"Rehashing {episode_id}..."):
                result = rehash_episode(episode_id, actor="user", reason=reason)

            st.success("✅ Rehash complete!")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Old hash:** `{result.old_hash}`")
            with col2:
                st.markdown(f"**New hash:** `{result.new_hash}`")

            st.markdown("---")
            st.markdown("**File Validation:**")

            for file, valid in result.validated_files.items():
                status = "✅" if valid else "❌"
                st.markdown(f"{status} {file}")

            if result.errors:
                st.warning(f"⚠️ Found {len(result.errors)} validation errors:")
                for err in result.errors:
                    st.markdown(f"- {err}")

            # Don't auto-close so user can see results
        except Exception as e:
            st.error(f"❌ Rehash failed: {str(e)}")


@st.dialog("Manage Episode", width="large")
def episode_manager_modal(episode_id: str, data_root: Path = Path("data")):
    """
    Episode Manager Modal - Phase 2

    Args:
        episode_id: Episode to manage
        data_root: Data root directory
    """
    # Check if should close
    if st.session_state.get("_modal_close"):
        st.session_state["_modal_close"] = False
        st.rerun()

    # Show progress banner if operation is active
    _render_progress_banner(episode_id, data_root)

    # Check if pipeline can run
    pipeline_check = check_pipeline_can_run(episode_id, data_root)
    can_run = pipeline_check.get("can_run", True)
    block_reason = pipeline_check.get("reason", "")

    # Tabs
    tab = st.radio(
        "Operation",
        ["Info", "Move", "Remove", "Restore", "Rehash"],
        horizontal=True,
        label_visibility="collapsed",
        key="modal_tab_selector",
    )

    st.markdown("---")

    # Render selected tab
    if tab == "Info":
        _render_info_tab(episode_id, data_root)
    elif tab == "Move":
        _render_move_tab(episode_id, data_root, can_run, block_reason)
    elif tab == "Remove":
        _render_remove_tab(episode_id, data_root, can_run, block_reason)
    elif tab == "Restore":
        _render_restore_tab(episode_id, data_root, can_run, block_reason)
    else:  # Rehash
        _render_rehash_tab(episode_id, data_root, can_run, block_reason)

    # Close button at bottom
    st.markdown("---")
    if st.button("Close", key="modal_close_btn", use_container_width=True):
        st.session_state["_modal_close"] = True
        st.rerun()
