"""
Workspace - Unified face/cluster/track review with internal tabs.

This page provides thumbnail-first review cards with real metrics from
cluster_metrics and track_metrics.
"""

import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
try:
    from streamlit_autorefresh import st_autorefresh
except ModuleNotFoundError:  # pragma: no cover - use vendored component
    from app.vendor.streamlit_autorefresh import st_autorefresh

from app.lib.mutator_api import configure_workspace_mutator
from app.lib.pipeline import (
    check_artifacts_status,
    is_pipeline_running,
    check_pipeline_can_run,
    get_maintenance_reason,
    calculate_eta,
    format_eta,
    cancel_pipeline,
)
from app.components.episode_manager_modal import episode_manager_modal
from app.workspace.faces import render_faces_tab
from app.workspace.clusters import render_clusters_tab
from app.workspace.tracks import render_tracks_tab

# Configure workspace debug logging
workspace_logger = logging.getLogger("workspace_ui")
workspace_logger.setLevel(logging.DEBUG)

# Add file handler for workspace debug log
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
workspace_log_file = log_dir / "workspace_debug.log"

if not any(isinstance(h, logging.FileHandler) for h in workspace_logger.handlers):
    file_handler = logging.FileHandler(workspace_log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    workspace_logger.addHandler(file_handler)

# Also add console handler
if not any(isinstance(h, logging.StreamHandler) for h in workspace_logger.handlers):
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    workspace_logger.addHandler(console_handler)
from app.workspace.review import render_review_tab
from app.utils.ui_keys import safe_rerun
from app.lib.registry import (
    load_registry,
    get_all_episodes,
    recover_episodes_from_fs,
    ensure_episode_in_registry,
    get_default_episode,
    load_episodes_json
)
from app.lib.episode_manager import purge_all_episodes, delete_video_file

# Page config
st.set_page_config(
    page_title="Workspace",
    page_icon="🗂️",
    layout="wide",
)

# Constants
DATA_ROOT = Path("data")


def read_recent_logs(log_path: Path, episode_id: str | None, run_iso_ts: str | None, limit: int = 100) -> list[str]:
    """Read recent workspace debug lines preferring those after a run-start marker or timestamp."""
    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []

    if not lines:
        return []

    marker_index = None
    marker_token = f"ts={run_iso_ts}" if run_iso_ts else None
    if episode_id:
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            if "[RUN-START]" in line and f"episode={episode_id}" in line:
                if marker_token and marker_token in line:
                    marker_index = idx
                    break
                if marker_token is None:
                    marker_index = idx
                    break

    candidate = lines[marker_index + 1 :] if marker_index is not None else lines[-limit * 3 :]

    if run_iso_ts:
        normalized = run_iso_ts.replace("Z", "+00:00") if "Z" in run_iso_ts else run_iso_ts
        try:
            run_dt = datetime.fromisoformat(normalized)
        except Exception:
            run_dt = None

        if run_dt:
            ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]")
            filtered: list[str] = []
            for line in candidate:
                match = ts_pattern.match(line)
                if match:
                    ts_str = match.group(1).replace(" ", "T")
                    try:
                        ts_dt = datetime.fromisoformat(ts_str)
                    except Exception:
                        ts_dt = None
                    if ts_dt and ts_dt >= run_dt:
                        filtered.append(line)
                else:
                    if filtered:
                        filtered.append(line)
            if filtered:
                candidate = filtered

    if not candidate:
        return []

    return candidate[-limit:]


def render_primary_progress(progress_state: dict | None) -> None:
    """Render a top-level progress indicator using pre-seeded session state."""
    if not progress_state:
        return

    stage = progress_state.get("stage") or "Full Pipeline"
    message = progress_state.get("message") or ""

    pct_val = progress_state.get("pct", 0.0) or 0.0
    try:
        pct_float = float(pct_val)
    except (TypeError, ValueError):
        pct_float = 0.0
    pct_clamped = max(0.0, min(pct_float, 1.0))

    text = f"⏳ {stage}"
    if message:
        text = f"{text} • {message}"

    st.progress(pct_clamped, text=text)


def schedule_auto_refresh(episode_id: str | None) -> None:
    """Trigger dual cadence auto-refresh while a job is active."""
    active_jobs = st.session_state.get("active_polling_jobs", {})
    active_job_id = st.session_state.get("active_job")

    if not episode_id or not (active_job_id or active_jobs):
        st.session_state.fast_refresh = False
        return

    now = time.time()
    last_run_start_ts = st.session_state.get("last_run_start_ts", now)
    first_heartbeat = st.session_state.get("first_heartbeat_seen", False)

    fast_window_active = (
        bool(active_job_id)
        and not first_heartbeat
        and (now - last_run_start_ts) < 5
    )

    if fast_window_active:
        st.session_state.fast_refresh = True
        st_autorefresh(interval=500, key=f"fast_{episode_id}")
        return

    if active_job_id and not first_heartbeat:
        st.session_state.first_heartbeat_seen = True

    st.session_state.fast_refresh = False
    st_autorefresh(interval=2000, key=f"steady_{episode_id}")


def init_workspace_state():
    """Initialize workspace session state."""
    if "workspace_tab" not in st.session_state:
        st.session_state.workspace_tab = "Clusters"
    if "workspace_episode" not in st.session_state:
        st.session_state.workspace_episode = None
    if "workspace_selected_person" not in st.session_state:
        st.session_state.workspace_selected_person = None
    if "workspace_selected_cluster" not in st.session_state:
        st.session_state.workspace_selected_cluster = None


@st.dialog("⚠️ Purge All Episodes", width="large")
def purge_all_episodes_dialog():
    """Confirmation dialog for purging all episodes."""
    st.warning("**WARNING:** This will permanently archive ALL episodes!")

    st.markdown("""
    **What will happen:**
    - All episodes will be removed from `diagnostics/episodes.json`
    - `data/harvest/<episode>/` will be moved to `data/archive/episodes/<episode>_<timestamp>/`
    - `data/outputs/<episode>/` will be moved to `data/archive/episodes/<episode>_<timestamp>/`
    - Optionally archive video files (you can choose below)
    - **Facebank and show registry will NOT be touched**

    **This cannot be easily undone.**
    """)

    archive_videos = st.checkbox(
        "Also archive video files (recommended to keep videos)",
        value=False,
        key="purge_archive_videos"
    )

    understand = st.checkbox(
        "I understand this will archive ALL episodes",
        key="purge_understand"
    )

    confirm_text = st.text_input(
        'Type "PURGE ALL" to confirm:',
        key="purge_confirm_text"
    )

    reason = st.text_area(
        "Reason (optional)",
        key="purge_reason",
        placeholder="e.g., Starting fresh with new episode uploads"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Cancel", key="purge_cancel_btn", use_container_width=True):
            st.session_state["_show_purge_dialog"] = False
            st.rerun()

    with col2:
        confirm_valid = understand and confirm_text == "PURGE ALL"

        if st.button(
            "✅ Confirm Purge",
            key="purge_confirm_btn",
            type="primary",
            disabled=not confirm_valid,
            use_container_width=True,
        ):
            try:
                with st.spinner("Purging all episodes..."):
                    stats = purge_all_episodes(
                        archive_videos=archive_videos,
                        actor="user",
                        reason=reason or "Purged all episodes to start fresh"
                    )

                st.success(f"✅ Purged {stats['episodes_purged']} episodes successfully!")

                if stats['errors']:
                    st.warning(f"⚠️ Encountered {len(stats['errors'])} errors:")
                    for err in stats['errors']:
                        st.markdown(f"- {err}")

                # Show summary
                with st.expander("Purge Summary"):
                    st.json(stats)

                st.session_state["_show_purge_dialog"] = False
                st.session_state["_purge_complete"] = True

                # Clear all caches to force registry reload
                st.cache_data.clear()
                st.cache_resource.clear()

                # Clear episode selection
                st.session_state.pop("workspace_episode", None)
                st.session_state.pop("episode_id", None)

                st.rerun()

            except Exception as e:
                st.error(f"❌ Purge failed: {str(e)}")


def main():
    """Main workspace page."""
    init_workspace_state()

    # Inject 4:5 thumbnail CSS - FILL frame with cover (no letterboxing)
    st.markdown("""
    <style>
    /* Force all thumbnails to FILL 4:5 frame */
    .tile-45 {
        width: 160px;
        height: 200px;
        overflow: hidden;
        border-radius: 8px;
        background: #f6f6f6;
        display: inline-block;
    }
    .tile-45 img {
        width: 100%;
        height: 100%;
        object-fit: cover !important;
        object-position: center !important;
        display: block;
    }
    /* Apply to all st.image containers */
    div[data-testid="stImage"] {
        width: 160px !important;
        height: 200px !important;
        overflow: hidden !important;
    }
    div[data-testid="stImage"] img {
        width: 100% !important;
        height: 100% !important;
        object-fit: cover !important;
        object-position: center !important;
        display: block !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🗂️ Workspace")

    # Load registry using canonical loader
    reg = load_registry()
    eps_in_registry = get_all_episodes(reg)
    eps_on_disk = recover_episodes_from_fs()

    # Recovery UI if registry is empty but episodes exist on disk
    if not eps_in_registry and eps_on_disk:
        st.warning("⚠️ No episodes in registry. Found harvests on disk.")
        st.info("📁 Select an episode below to add it to the registry and continue.")

        selected = st.selectbox(
            "Recover episode",
            options=eps_on_disk,
            key="recover_ep_select"
        )

        col1, col2 = st.columns(2)
        with col1:
            show_id = st.text_input("Show ID", "rhobh", key="recover_show_id")
        with col2:
            season_id = st.text_input("Season ID", "s05", key="recover_season_id")

        if st.button("Add to registry & continue", key="recover_ep_btn", type="primary") and selected:
            ensure_episode_in_registry(selected, show_id, season_id)
            st.stop()

        st.stop()

    elif not eps_in_registry and not eps_on_disk:
        st.error("❌ No episodes found in registry or data/harvest.")
        st.info("Please upload an episode first using the Upload page.")
        st.stop()

    # Episode selector (normal path)
    # Extract just episode_ids from tuples
    episode_ids = [ep_id for _, _, ep_id in eps_in_registry]

    if not episode_ids:
        st.warning("No episodes found in registry.")

        # Show recovery option if harvest episodes exist
        if eps_on_disk:
            st.info("Found episodes in data/harvest. Use recovery option above.")
        return

    # Surface thumbnail generation summary if present
    thumb_summary = st.session_state.pop("_thumb_summary", None)
    if thumb_summary:
        if thumb_summary.get("error_message"):
            st.error(thumb_summary["error_message"])
        else:
            st.success(
                f"Generated {thumb_summary['generated']} of {thumb_summary['total_tracks']} thumbnails"
                f" (placeholders: {thumb_summary['placeholders']})."
            )
            if thumb_summary.get("errors"):
                st.info(
                    f"{thumb_summary['errors']} tracks could not be thumbnailed."
                    " See thumbnails_stats.json for details."
                )

    # Header row: Pipeline buttons + Episode selector
    header_cols = st.columns([1.2, 1, 1, 2.5, 0.5])

    # Import pipeline helpers

    # Get default episode using smart selection
    default_ep = get_default_episode(episode_ids)
    current_ep = st.session_state.get("episode_id", default_ep)
    pipeline_running = is_pipeline_running(current_ep, DATA_ROOT) if current_ep else False
    artifact_status = check_artifacts_status(current_ep, DATA_ROOT) if current_ep else {
        "prepared": False,
        "has_clusters": False
    }


    # Check if pipeline can run (maintenance mode gate)
    pipeline_check = check_pipeline_can_run(current_ep, DATA_ROOT) if current_ep else {"can_run": True, "reason": ""}
    can_run = pipeline_check.get("can_run", True)
    block_reason = pipeline_check.get("reason", "")

    # CRITICAL: Check for active detect job (for resume/cancel functionality)
    active_detect_job = None
    detect_is_stalled = False
    episode_key = None
    if current_ep:
        from api.jobs import job_manager
        episode_key = job_manager.normalize_episode_key(current_ep)

        from episodes.runtime import get_active_job, check_job_stalled
        active_detect_job = get_active_job(episode_key, "detect", DATA_ROOT)

        if active_detect_job:
            detect_is_stalled = check_job_stalled(active_detect_job, DATA_ROOT)

    # Phase 3 P1: Check extraction status from episode registry
    extraction_ready = False
    episode_state = None
    if current_ep:
        from api.episodes import get_episode_state
        episode_state = get_episode_state(current_ep)
        extraction_ready = episode_state.get("states", {}).get("extracted_frames", False)

        # Auto-refresh while extraction is in progress
        if not extraction_ready:
            # Check if extraction is actually running or just pending
            validated = episode_state.get("states", {}).get("validated", False)

            if validated:
                # Validated but not extracted yet - poll for completion
                # Add auto-refresh banner
                st.info("⏳ Frame extraction in progress... Page will refresh automatically.")

                # Wait a bit then refresh (simulates polling)
                poll_interval = 2  # seconds
                if "last_poll_time" not in st.session_state:
                    st.session_state["last_poll_time"] = time.time()

                elapsed = time.time() - st.session_state["last_poll_time"]

                if elapsed >= poll_interval:
                    st.session_state["last_poll_time"] = time.time()
                    st.rerun()

                # Show progress placeholder
                with st.empty():
                    st.caption(f"Checking again in {max(0, poll_interval - int(elapsed))} seconds...")
                    time.sleep(0.5)
                    st.rerun()

    with header_cols[0]:
        from app.workspace.constants import STAGE_LABELS, STAGE_HELP

        # Phase 3 P1: Conditional Prepare based on extraction status
        if extraction_ready:
            # Frames ready - show full pipeline button
            prepare_help = STAGE_LABELS.get("full_pipeline", "Run full pipeline: detect → embed → track → cluster")
            prepare_disabled = not can_run

            # Disable if active job exists and is not stalled
            if active_detect_job and not detect_is_stalled:
                prepare_disabled = True
                prepare_help = f"Detect job already running: {active_detect_job}"
            elif active_detect_job and detect_is_stalled:
                prepare_help = "Detect job stalled. Use Resume or Cancel buttons below."
            elif not can_run:
                prepare_help = f"Blocked: {block_reason}"

            if st.button(
                STAGE_LABELS["full_pipeline"],
                type="primary",
                help=prepare_help,
                key="workspace_prepare_btn",
                use_container_width=True,
                disabled=prepare_disabled,
            ):
                st.session_state["_trigger_prepare"] = True
        else:
            # Frames not extracted yet - show passive message
            st.info("⏳ Extracting frames...")
            st.caption("This happens automatically after upload validation")

            # Add manual trigger in case auto-extraction failed
            if st.button(
                "⚠️ Retry Extraction",
                type="primary",
                use_container_width=True,
                help="Manual frame extraction if auto-extraction failed"
            ):
                from jobs.tasks.auto_extract import trigger_auto_extraction
                with st.spinner("Extracting frames..."):
                    try:
                        from pathlib import Path
                        video_path = episode_state.get("video_path", "")
                        if not video_path or not Path(video_path).exists():
                            st.error(f"❌ Video file not found: {video_path}")
                        else:
                            result = trigger_auto_extraction(current_ep, video_path)
                            if result.get("success"):
                                st.success("✅ Extraction complete!")
                                st.rerun()
                            else:
                                st.error(f"❌ Extraction failed: {result.get('error')}")
                    except Exception as e:
                        st.error(f"❌ Extraction error: {str(e)}")


    with header_cols[1]:
        from app.workspace.constants import STAGE_LABELS, STAGE_HELP

        # TASK 6: UX Guardrails for Cluster button
        # Disable if: detected=false OR any active job exists
        cluster_disabled = not can_run or not artifact_status.get("detected", False)
        cluster_help = STAGE_HELP.get("cluster", "Group face tracks by identity")

        # Check for active jobs
        active_jobs_check = st.session_state.get("active_polling_jobs", {})
        has_active_jobs = bool(active_jobs_check)

        if not can_run:
            cluster_help = f"Blocked: {block_reason}"
        elif not artifact_status.get("detected", False):
            cluster_help = f"Requires {STAGE_LABELS['detect']} first (detection not complete)"
            cluster_disabled = True
        elif has_active_jobs:
            cluster_help = f"Active job running: {', '.join(active_jobs_check.keys())} - wait for completion"
            cluster_disabled = True

        if st.button(
            STAGE_LABELS["cluster_button"],
            help=cluster_help,
            key="workspace_cluster_btn",
            use_container_width=True,
            disabled=cluster_disabled,
        ):
            st.session_state["_trigger_cluster"] = True

    with header_cols[2]:
        from app.workspace.constants import STAGE_LABELS, STAGE_HELP

        analyze_disabled = not can_run or not artifact_status["has_clusters"]
        analyze_help = STAGE_HELP.get("analytics", "Compute per-person screen time from labeled clusters")
        if not can_run:
            analyze_help = f"Blocked: {block_reason}"
        elif not artifact_status["has_clusters"]:
            analyze_help = "Requires clustering first"

        if st.button(
            STAGE_LABELS["analytics_button"],
            help=analyze_help,
            key="workspace_analyze_btn",
            use_container_width=True,
            disabled=analyze_disabled,
        ):
            st.session_state["_trigger_analyze"] = True


    with header_cols[3]:
        # Use get_default_episode for smart selection
        default_idx = 0
        if default_ep and default_ep in episode_ids:
            default_idx = episode_ids.index(default_ep)
        elif st.session_state.workspace_episode in episode_ids:
            default_idx = episode_ids.index(st.session_state.workspace_episode)

        selected_episode = st.selectbox(
            "Episode",
            options=episode_ids,
            index=default_idx,
            key="workspace_episode_selector",
            label_visibility="collapsed"
        )

    with header_cols[4]:
        if st.button("↻", key="refresh_episodes_btn", help="Refresh episode list", use_container_width=True):
            # Clear episode cache and reload
            st.session_state.pop("episode_id", None)
            st.cache_data.clear()
            safe_rerun()


    # Manage Episode button - opens modal
    col_manage, col_purge = st.columns([1, 1])

    with col_manage:
        if current_ep:
            if st.button("🔧 Manage Episode", key="manage_episode_btn", use_container_width=False):
                episode_manager_modal(current_ep, DATA_ROOT)

    with col_purge:
        if st.button("🗑️ Remove ALL Episodes", key="purge_all_btn", use_container_width=False, type="secondary"):
            st.session_state["_show_purge_dialog"] = True

    # Show purge dialog if triggered
    if st.session_state.get("_show_purge_dialog"):
        purge_all_episodes_dialog()

    # Show success message if purge just completed
    if st.session_state.pop("_purge_complete", False):
        st.success("✅ All episodes have been purged. Episode dropdowns will be empty until new episodes are uploaded.")
        safe_rerun()

    # Update session state if episode changed
    if selected_episode != st.session_state.workspace_episode:
        st.session_state.workspace_episode = selected_episode
        st.session_state.episode_id = selected_episode  # For wkey()
        st.session_state.workspace_selected_person = None
        st.session_state.workspace_selected_cluster = None
        safe_rerun()

    # Store episode_id for wkey
    st.session_state.episode_id = selected_episode

    # Get show_id and season_id for the selected episode
    show_id, season_id = None, None
    for s_id, ss_id, ep_id in eps_in_registry:
        if ep_id == selected_episode:
            show_id, season_id = s_id, ss_id
            break

    # TASK 2: Legacy Job ID Migration
    # After episode is selected, check for and migrate legacy prepare_* job IDs
    if current_ep and episode_key:
        from episodes.runtime import migrate_legacy_job_id
        migrated = migrate_legacy_job_id(episode_key, current_ep, DATA_ROOT)
        if migrated:
            st.toast("🔄 Legacy job ID migrated to detect_<EPISODE_ID>", icon="✅")
            workspace_logger.info(f"[MIGRATION] Legacy prepare_* job migrated for {episode_key}")

    # TASK 1: Auto-Polling on Page Load
    # Check for active jobs and start polling automatically (no Resume button needed)
    if current_ep and episode_key:
        from episodes.runtime import get_all_active_jobs
        active_jobs = get_all_active_jobs(episode_key, DATA_ROOT)

        if active_jobs:
            # Store active jobs in session state for polling
            st.session_state.active_polling_jobs = active_jobs
            workspace_logger.info(f"[AUTO-POLL] Found active jobs: {active_jobs} | Episode={episode_key}")
        else:
            # Clear active polling jobs if none found
            st.session_state.active_polling_jobs = {}

    # AUTO-REFRESH: Enable live progress updates while jobs are running
    schedule_auto_refresh(selected_episode)

    # CRITICAL: Show active job status and Resume/Cancel controls
    if active_detect_job:
        if detect_is_stalled:
            st.warning(f"⚠️ Detect job appears stalled (no heartbeat >30s): {active_detect_job}")
            st.caption("The job may have crashed or the worker may be stuck. Use Resume to reattach or Cancel to clear.")
        else:
            st.info(f"ℹ️ Detect job active: {active_detect_job}")
            st.caption("This job is currently running. Progress will update automatically.")

        # Resume/Cancel buttons
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Resume Detect", key="resume_detect", help="Reattach to running job", use_container_width=True):
                st.session_state["_resume_detect"] = active_detect_job
                st.rerun()

        with col2:
            if st.button("❌ Cancel Detect", key="cancel_detect", help="Cancel and clear job", use_container_width=True):
                st.session_state["_cancel_detect"] = active_detect_job
                st.rerun()

    # Handle Resume button click
    if st.session_state.pop("_resume_detect", None):
        workspace_logger.info(f"[UI] Clicked 'Resume Detect' button | Episode={episode_key} | job_id={active_detect_job}")

        st.info("🔄 **Resuming Detect job...**")
        st.write(f"Reattaching to job: {active_detect_job}")
        st.caption("The progress polling will now track this job automatically.")
        st.success("✅ Resumed! Polling for progress...")
        time.sleep(1)
        st.rerun()

    # Handle Cancel button click
    if st.session_state.pop("_cancel_detect", None):
        workspace_logger.info(f"[UI] Clicked 'Cancel Detect' button | Episode={episode_key} | job_id={active_detect_job}")

        st.warning("❌ **Canceling Detect job...**")

        from episodes.runtime import clear_active_job
        from api.jobs import job_manager

        try:
            # Mark job as canceled in envelope
            job_manager.update_stage_status(active_detect_job, "detect", "canceled")

            # Clear from runtime
            clear_active_job(episode_key, "detect", DATA_ROOT)

            # Remove lock file if exists
            lock_path = DATA_ROOT / "jobs" / active_detect_job / ".lock"
            if lock_path.exists():
                lock_path.unlink()

            workspace_logger.info(f"[UI] Successfully canceled Detect job | Episode={episode_key} | job_id={active_detect_job}")

            st.success("✅ Detect job canceled. You can start a new run.")
            st.rerun()
        except Exception as e:
            workspace_logger.error(f"[UI] Failed to cancel Detect job | Episode={episode_key} | job_id={active_detect_job} | Error={e}")
            st.error(f"Failed to cancel: {e}")

    # Progress polling UI
    from app.workspace.common import read_pipeline_state
    pipeline_state = read_pipeline_state(selected_episode, DATA_ROOT)

    # TASK 3: Polling Fallback
    # If no active jobs found BUT detected=false, enable polling anyway
    if current_ep and episode_key:
        active_jobs_check = st.session_state.get("active_polling_jobs", {})
        states = artifact_status if artifact_status else {}

        # Enable fallback polling if no active jobs but detection is incomplete
        if not active_jobs_check and states.get("detected") == False:
            workspace_logger.info(f"[POLL-FALLBACK] No active jobs but detected=false, polling pipeline_state | Episode={episode_key}")
            # Poll pipeline_state directly and attach to whatever job updates it
            # The polling will happen naturally through the existing auto-refresh logic below

    # Log pipeline state poll
    if pipeline_state:
        pct_raw = pipeline_state.get("pct")
        try:
            pct_float = float(pct_raw) if pct_raw is not None else 0.0
        except (TypeError, ValueError):
            pct_float = 0.0

        st.session_state.progress_state = {
            "stage": pipeline_state.get("current_step", "Full Pipeline") or "Full Pipeline",
            "pct": max(0.0, min(pct_float, 1.0)),
            "message": pipeline_state.get("message", ""),
        }

        heartbeat_present = any(
            pipeline_state.get(key)
            for key in ("heartbeat", "heartbeat_ts", "heartbeat_seq")
        )
        if pct_float > 0 or heartbeat_present:
            st.session_state.first_heartbeat_seen = True

        workspace_logger.debug(f"[POLL] Episode={selected_episode} | stage={pipeline_state.get('current_step', 'N/A')} | pct={pipeline_state.get('pct', 0)*100 if pipeline_state.get('pct') else 'N/A'}% | status={pipeline_state.get('status', 'N/A')} | msg={pipeline_state.get('message', '')[:80]}")

        # Check for JSON corruption in error message
        if pipeline_state.get("status") == "error":
            error_msg = pipeline_state.get("message", "")
            if "Extra data" in error_msg or "JSONDecodeError" in error_msg:
                workspace_logger.warning(f"[UI] JSON corruption detected | Episode={selected_episode} | Error={error_msg}")
                st.warning("⚠️ **JSON corruption detected** — auto-recovery in progress. Please wait...")
                st.caption("The system is automatically repairing corrupted pipeline state files. This may take a few seconds.")

                # Trigger automatic recovery by re-reading with safe_load_json
                try:
                    from screentime.diagnostics.utils import safe_load_json
                    state_path = DATA_ROOT / "harvest" / selected_episode / "diagnostics" / "pipeline_state.json"
                    
                    if state_path.exists():
                        # This will trigger automatic recovery if needed
                        recovered_state = safe_load_json(state_path)

                        if recovered_state:
                            st.toast("⚠️ Pipeline prerequisites repaired (auto-recovered JSON)", icon="✅")
                            workspace_logger.info(f"[UI] JSON recovery successful | Episode={selected_episode}")
                            
                            # Refresh to show recovered state
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Recovery failed. Please check logs for details.")
                            workspace_logger.error(f"[UI] JSON recovery failed | Episode={selected_episode}")
                except Exception as e:
                    st.error(f"❌ Recovery failed: {e}")
                    workspace_logger.error(f"[UI] JSON recovery exception | Episode={selected_episode} | Error={e}")

    # Handle cancelled state (show toast once)
    if pipeline_state and pipeline_state.get("status") == "cancelled":
        cancelled_key = f"_pipeline_cancelled_{selected_episode}"
        if cancelled_key not in st.session_state:
            st.info("✅ Pipeline was cancelled")
            st.session_state[cancelled_key] = True

            # Archive the cancelled state to prevent re-showing
            state_file = DATA_ROOT / "harvest" / selected_episode / "diagnostics" / "pipeline_state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        final_state = json.load(f)
                    final_state["status"] = "archived"
                    with open(state_file, "w") as f:
                        json.dump(final_state, f, indent=2)
                except Exception:
                    pass

    # Handle done state (show toast once)
    if pipeline_state and pipeline_state.get("status") == "done":
        done_key = f"_pipeline_done_{selected_episode}"
        if done_key not in st.session_state:
            # Show success toast with summary
            extra = pipeline_state.get("extra", {})
            result = extra.get("result", {}) if extra else {}

            summary_parts = []
            if "total_detections" in result:
                summary_parts.append(f"{result['total_detections']} faces detected")
            if "n_tracks" in result:
                summary_parts.append(f"{result['n_tracks']} tracks")
            if "n_clusters" in result:
                summary_parts.append(f"{result['n_clusters']} clusters")
            if "generated" in result:
                summary_parts.append(f"{result['generated']} stills")

            # Check for analytics totals
            totals_by_identity = result.get("totals_by_identity", {})
            if totals_by_identity:
                identity_count = len([k for k in totals_by_identity.keys() if k != "Unknown"])
                if identity_count > 0:
                    summary_parts.append(f"{identity_count} identities tracked")

            summary_msg = ", ".join(summary_parts) if summary_parts else "completed"
            st.toast(f"✅ Pipeline complete: {summary_msg}", icon="✅")
            st.session_state[done_key] = True

            # Archive the state file to prevent re-showing
            try:
                from screentime.diagnostics.utils import archive_pipeline_state
                archive_pipeline_state(selected_episode)
            except Exception:
                # Fallback: just mark as archived
                state_file = DATA_ROOT / "harvest" / selected_episode / "diagnostics" / "pipeline_state.json"
                if state_file.exists():
                    with open(state_file, "r") as f:
                        final_state = json.load(f)
                    final_state["status"] = "archived"
                    with open(state_file, "w") as f:
                        json.dump(final_state, f, indent=2)

    # Check if we should show progress bars (either pipeline_state says running OR session state has active jobs)
    should_show_progress = (
        st.session_state.get("active_job") or 
        st.session_state.get("active_polling_jobs") or
        (pipeline_state and pipeline_state.get("status") not in (None, "done", "archived", "cancelled"))
    )
    
    if should_show_progress:
        st.markdown("---")
        render_primary_progress(st.session_state.get("progress_state"))
        state_view = pipeline_state or {}

        current_step = state_view.get("current_step", "Unknown")
        step_index = state_view.get("step_index", 0)
        total_steps = state_view.get("total_steps", 4)
        status = state_view.get("status", "unknown")
        message = state_view.get("message", "")

        if status == "error":
            st.error(f"❌ Pipeline error: {message}")

            with st.expander("Error details"):
                st.json(state_view)

            # Check if error is in Stills stage
            is_stills_error = "stills" in current_step.lower() or "generate" in current_step.lower()

            if is_stills_error:
                # Stills-specific retry options
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔁 Resume Stills", key="resume_stills_btn", help="Continue from where it stalled"):
                        with st.spinner("Resuming stills generation..."):
                            try:
                                from jobs.tasks.orchestrate import run_stills_only
                                result = run_stills_only(selected_episode, data_root=DATA_ROOT, resume=True, force=False)
                                if result.get("status") == "ok":
                                    st.success("Stills generation resumed successfully!")
                                    safe_rerun()
                                else:
                                    st.error(f"Resume failed: {result.get('error', 'Unknown error')}")
                            except Exception as exc:
                                st.error(f"Resume failed: {exc}")
                with col2:
                    if st.button("🔄 Force Re-run Stills", key="force_stills_btn", help="Regenerate all stills from scratch"):
                        with st.spinner("Regenerating all stills..."):
                            try:
                                from jobs.tasks.orchestrate import run_stills_only
                                result = run_stills_only(selected_episode, data_root=DATA_ROOT, resume=False, force=True)
                                if result.get("status") == "ok":
                                    st.success("Stills regenerated successfully!")
                                    safe_rerun()
                                else:
                                    st.error(f"Regeneration failed: {result.get('error', 'Unknown error')}")
                            except Exception as exc:
                                st.error(f"Regeneration failed: {exc}")
                with col3:
                    if st.button("❌ Clear Error", key="clear_error_btn"):
                        state_file = DATA_ROOT / "harvest" / selected_episode / "diagnostics" / "pipeline_state.json"
                        if state_file.exists():
                            state_file.unlink()
                        safe_rerun()
            else:
                # General retry options
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Retry Pipeline", key="retry_pipeline_btn"):
                        st.session_state["_trigger_cluster"] = True
                        safe_rerun()
                with col2:
                    if st.button("❌ Clear Error", key="clear_error_btn_gen"):
                        state_file = DATA_ROOT / "harvest" / selected_episode / "diagnostics" / "pipeline_state.json"
                        if state_file.exists():
                            state_file.unlink()
                        safe_rerun()
        else:
            # Running state
            overall_pct = (step_index / total_steps) if total_steps > 0 else 0.0

            col_status, col_cancel = st.columns([4, 1])
            with col_status:
                st.info(f"🔄 {current_step} ({step_index}/{total_steps})")
            with col_cancel:
                if st.button("⏹️ Cancel Pipeline", key="cancel_pipeline_btn", type="secondary", use_container_width=True):
                    if cancel_pipeline(selected_episode, DATA_ROOT):
                        st.success("✅ Pipeline cancelled")
                        safe_rerun()
                    else:
                        st.error("❌ Failed to cancel pipeline")

            if message:
                st.caption(message)

            # Per-stage progress bars
            stages = ["1. RetinaFace + ArcFace (Detect & Embed)", "2. ByteTrack (Track Faces)", "3. Agglomerative Clustering (Group Tracks)", "4. Generate Stills", "5. Screenalyzer Analytics"]
            # Map display names to step keys used in stats
            step_keys = ["detect", "track", "cluster", "stills", "analytics"]

            for i, stage_name in enumerate(stages, start=1):
                if i < step_index:
                    # Completed
                    st.progress(1.0, text=f"✅ {stage_name}")
                elif i == step_index:
                    # Current - add ETA and frames_done/frames_total if available
                    stage_pct = state_view.get("pct", 0.5)
                    if stage_pct is None:
                        stage_pct = 0.5

                    # Extract frames_done/frames_total from envelope if available
                    frames_text = ""
                    try:
                        # Try to load job envelope to get frames_done/frames_total
                        from api.jobs import job_manager
                        from episodes.runtime import get_active_job

                        episode_key = job_manager.normalize_episode_key(selected_episode)
                        step_key = step_keys[i - 1]

                        # Get active job for this stage
                        active_job_id = get_active_job(episode_key, step_key, DATA_ROOT)

                        if active_job_id:
                            envelope = job_manager.load_job_envelope(active_job_id)
                            if envelope and "stages" in envelope:
                                stage_data = envelope["stages"].get(step_key, {})
                                stage_result = stage_data.get("result", {})

                                frames_done = stage_result.get("frames_done")
                                frames_total = stage_result.get("frames_total")
                                faces_detected = stage_result.get("faces_detected")
                                tracks_active = stage_result.get("tracks_active")

                                if frames_done is not None and frames_total is not None:
                                    frames_text = f" • {frames_done}/{frames_total} frames"
                                    if faces_detected is not None:
                                        frames_text += f" • {faces_detected} faces"
                                    if tracks_active is not None:
                                        frames_text += f" • {tracks_active} tracks"
                    except Exception:
                        pass  # Could not load envelope data, frames_text will remain empty

                    # Calculate ETA for current step
                    try:
                        step_key = step_keys[i - 1]
                        operation = "prepare"  # Default operation

                        # Determine operation type from pipeline state
                        if "cluster" in current_step.lower():
                            operation = "cluster"
                        elif "analyt" in current_step.lower():
                            operation = "analytics"

                        # Get remaining steps
                        remaining_steps = [step_keys[j] for j in range(i, len(step_keys))]

                        eta_info = calculate_eta(
                            selected_episode,
                            operation,
                            step_key,
                            remaining_steps,
                            current_step_elapsed_s=0.0,
                            data_root=DATA_ROOT
                        )

                        eta_seconds = eta_info.get("eta_seconds", 0)
                        confidence = eta_info.get("confidence", "none")

                        if confidence == "none" or eta_seconds <= 0:
                            eta_text = " • learning ETA..."
                        else:
                            eta_formatted = format_eta(eta_seconds)
                            eta_text = f" • ETA: ~{eta_formatted}"
                    except Exception:
                        eta_text = ""

                    # Show progress with percentage and frames info
                    pct_display = f"{int(stage_pct * 100)}%" if stage_pct > 0 else ""
                    progress_text = f"⏳ {stage_name} {pct_display}{frames_text}{eta_text}"
                    st.progress(float(stage_pct), text=progress_text)
                else:
                    # Pending
                    st.progress(0.0, text=f"⏸️ {stage_name}")

        # Debug Console
        with st.expander("🪵 Debug Console", expanded=False):
            # TASK 4 & 5: Per-run filtering and Clear Console button
            # Initialize last_run_ts in session state
            if "last_run_ts" not in st.session_state:
                st.session_state.last_run_ts = None

            # Header row with caption and Clear Console button
            col_caption, col_clear = st.columns([4, 1])
            with col_caption:
                if st.session_state.last_run_ts:
                    st.caption(f"Logs since last run: {st.session_state.last_run_ts}")
                else:
                    st.caption("Real-time workspace debug log (last 500 lines)")
            with col_clear:
                if st.button("Clear Console", key="clear_console_btn", help="Reset log view to current timestamp"):
                    st.session_state.last_run_ts = datetime.utcnow().isoformat()
                    st.session_state.pop("initial_debug_view", None)
                    workspace_logger.info(f"[CONSOLE] Cleared console, reset last_run_ts to {st.session_state.last_run_ts}")
                    st.rerun()

            initial_lines = st.session_state.pop("initial_debug_view", None)
            if initial_lines is not None:
                initial_content = "".join(initial_lines)
                if not initial_content.strip():
                    initial_content = "⏳ Initializing debug console for this run..."
                st.code(initial_content, language="log")
            elif workspace_log_file.exists():
                try:
                    with open(workspace_log_file, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        last_500 = lines[-500:] if len(lines) > 500 else lines

                        # TASK 4: Filter logs to only show entries since last_run_ts
                        if st.session_state.last_run_ts:
                            filtered_lines = []
                            last_run_dt = datetime.fromisoformat(st.session_state.last_run_ts.replace('Z', '+00:00') if 'Z' in st.session_state.last_run_ts else st.session_state.last_run_ts)

                            for line in last_500:
                                # Parse timestamp from log format: [YYYY-MM-DDTHH:MM:SS.ffffff] or [YYYY-MM-DD HH:MM:SS]
                                match = re.match(r'\[(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?)\]', line)
                                if match:
                                    try:
                                        timestamp_str = match.group(1).replace(' ', 'T')  # Normalize to ISO format
                                        line_dt = datetime.fromisoformat(timestamp_str)
                                        if line_dt >= last_run_dt:
                                            filtered_lines.append(line)
                                    except Exception:
                                        # If we can't parse the timestamp, include the line
                                        filtered_lines.append(line)
                                else:
                                    # No timestamp found, might be continuation line
                                    if filtered_lines:  # Only include if we have previous filtered lines
                                        filtered_lines.append(line)

                                # Also check for [RUN-START] markers and extract ts=
                                if "[RUN-START]" in line and "ts=" in line:
                                    ts_match = re.search(r'ts=([\d\-T:.]+)', line)
                                    if ts_match:
                                        try:
                                            run_start_ts = ts_match.group(1)
                                            # Update session state with the most recent run start
                                            st.session_state.last_run_ts = run_start_ts
                                            workspace_logger.debug(f"[CONSOLE] Found RUN-START marker, updated last_run_ts to {run_start_ts}")
                                        except Exception:
                                            pass

                            log_content = "".join(filtered_lines) if filtered_lines else "No logs since last run. Click 'Clear Console' to reset."
                        else:
                            log_content = "".join(last_500)

                    st.code(log_content or "No debug log output yet.", language="log")
                except Exception as e:
                    st.error(f"Could not read debug log: {e}")
            else:
                st.info("No debug log available yet. Debug events will appear here during pipeline runs.")

        st.markdown("---")

    # Handle Prepare button click
    if st.session_state.pop("_trigger_prepare", False):
        from app.workspace.constants import STAGE_LABELS

        workspace_logger.info(f"[UI] Clicked 'Full Pipeline' button | Episode={selected_episode}")

        # CRITICAL FIX: Set session state BEFORE calling backend to enable immediate UI updates
        st.session_state.active_job = f"full_{selected_episode}"
        st.session_state.last_run_ts = datetime.utcnow().isoformat()
        st.session_state.last_run_start_ts = time.time()
        st.session_state.active_polling_jobs = {"full": f"full_{selected_episode}"}
        st.session_state.fast_refresh = True
        st.session_state.first_heartbeat_seen = False
        st.session_state.progress_state = {
            "stage": "Full Pipeline",
            "pct": 0.0,
            "message": "Starting Detect/Embed → Track → Cluster → Stills…",
        }
        st.session_state.initial_debug_view = read_recent_logs(
            workspace_log_file,
            selected_episode,
            st.session_state.last_run_ts,
            limit=100,
        )
        schedule_auto_refresh(selected_episode)

        workspace_logger.info(f"[UI] Session state updated for full pipeline start | Episode={selected_episode} | last_run_ts={st.session_state.last_run_ts}")

        st.info("🔄 **Starting Full Pipeline...**")
        st.write("Running: Detect/Embed → Track → Generate Face Stills")

        # Show immediate feedback
        st.toast("🚀 Full pipeline started — progress will update live", icon="✅")

        # Mark pipeline as starting
        from app.workspace.common import read_pipeline_state
        diagnostics_dir = DATA_ROOT / "harvest" / selected_episode / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        state_file = diagnostics_dir / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump({
                "episode": selected_episode,
                "current_step": "Starting",
                "step_index": 0,
                "total_steps": 4,
                "status": "running",
                "message": "Initializing full pipeline (detect → track → stills)...",
            }, f, indent=2)

        try:
            from jobs.tasks.orchestrate import orchestrate_prepare

            with st.spinner("Running full pipeline... This may take several minutes."):
                result = orchestrate_prepare(
                    episode_id=selected_episode,
                    data_root=DATA_ROOT,
                    force=False,  # Skip stages that already have artifacts
                    resume=True,  # Resume from last stage
                )

            if result.get("status") == "ok":
                st.success(f"✅ Full pipeline complete for {selected_episode}!")
                st.info("💡 **Next steps:**\n1. Curate facebank on CAST page\n2. Click **Cluster** button")
                workspace_logger.info(f"[UI] Full pipeline completed successfully | Episode={selected_episode}")
                safe_rerun()
            else:
                st.error(f"❌ Full pipeline failed: {result.get('error', 'Unknown error')}")
                workspace_logger.error(f"[UI] Full pipeline failed | Episode={selected_episode} | Error={result.get('error', 'Unknown')}")
                with st.expander("Error details"):
                    st.json(result)
                # Force rerun to show error state in progress bars
                workspace_logger.info(f"[UI] Forcing rerun to show error state")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Full pipeline failed: {str(e)}")
            workspace_logger.error(f"[UI] Full pipeline exception | Episode={selected_episode} | Error={str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())
            # Force rerun to show exception state
            workspace_logger.info(f"[UI] Forcing rerun to show exception state")
            st.rerun()

    # Handle Cluster button click
    if st.session_state.pop("_trigger_cluster", False):
        workspace_logger.info(f"[UI] Clicked 'Cluster' button | Episode={selected_episode}")

        st.info("🎯 **Starting Cluster pipeline...**")
        st.write("Grouping face tracks by identity using current facebank")
        st.caption("(Missing stages will auto-run: Detect → Track → Cluster)")

        # Mark pipeline as starting
        diagnostics_dir = DATA_ROOT / "harvest" / selected_episode / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        state_file = diagnostics_dir / "pipeline_state.json"

        with open(state_file, "w") as f:
            json.dump({
                "episode": selected_episode,
                "current_step": "Cluster",
                "step_index": 1,
                "total_steps": 1,
                "status": "running",
                "message": "Clustering tracks...",
            }, f, indent=2)

        try:
            from jobs.tasks.orchestrate import orchestrate_cluster_only

            with st.spinner("Running cluster pipeline (auto-running missing stages if needed)..."):
                result = orchestrate_cluster_only(
                    episode_id=selected_episode,
                    data_root=DATA_ROOT,
                )

            if result.get("status") == "ok":
                n_clusters = result.get("result", {}).get("n_clusters", 0)
                st.success(f"✅ Clustered into {n_clusters} clusters!")
                st.info("💡 **Next step:** Click **Analyze** to generate timeline & totals")
                safe_rerun()
            else:
                error = result.get("error", "Unknown error")
                st.error(f"❌ Cluster failed: {error}")
                with st.expander("Error details"):
                    st.json(result)

        except Exception as e:
            st.error(f"❌ Cluster failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # Handle Analyze button click
    if st.session_state.pop("_trigger_analyze", False):
        workspace_logger.info(f"[UI] Clicked 'Analytics' button | Episode={selected_episode}")

        st.info("📊 **Starting Analytics pipeline...**")
        st.write("Generating timeline & totals from final cluster labels")

        # Mark pipeline as starting
        diagnostics_dir = DATA_ROOT / "harvest" / selected_episode / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        state_file = diagnostics_dir / "pipeline_state.json"

        with open(state_file, "w") as f:
            json.dump({
                "episode": selected_episode,
                "current_step": "Analytics",
                "step_index": 1,
                "total_steps": 1,
                "status": "running",
                "message": "Generating analytics...",
            }, f, indent=2)

        try:
            from jobs.tasks.orchestrate import orchestrate_analytics_only

            with st.spinner("Generating analytics..."):
                result = orchestrate_analytics_only(
                    episode_id=selected_episode,
                    data_root=DATA_ROOT,
                )

            if result.get("status") == "ok":
                intervals = result.get("result", {}).get("intervals_created", 0)
                st.success(f"✅ Analytics complete! Generated {intervals} timeline intervals")
                st.info("💡 Check the **Review** tab to see timeline & totals")
                safe_rerun()
            else:
                st.error(f"❌ Analytics failed: {result.get('error', 'Unknown error')}")
                with st.expander("Error details"):
                    st.json(result)

        except Exception as e:
            st.error(f"❌ Analytics failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # Keep old cluster handler as fallback (for Enhance button compatibility)
    if st.session_state.pop("_trigger_cluster_full", False):
        st.info("🔄 Starting full pipeline: Detect/Embed → Track → Cluster → Generate Stills...")

    # Handle Enhance Clusters button click
    if st.session_state.pop("_trigger_enhance", False):
        st.info("✨ Starting enhance clusters: re-cluster with constraints + densify + analytics...")

        try:
            from jobs.tasks.recluster import recluster_task
            from jobs.tasks.densify_two_pass import densify_two_pass
            from jobs.tasks.analytics import analytics_task
            from app.lib.data import load_clusters
            import yaml

            # Load config to check if densify is enabled
            config_path = Path("configs/pipeline.yaml")
            config = {}
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f) or {}

            clustering_config = config.get('clustering', {})
            use_densify = clustering_config.get('use_densify_two_pass', False)

            with st.spinner("Step 1: Re-clustering with manual constraints..."):
                recluster_result = recluster_task(
                    "manual",
                    selected_episode,
                    show_id=show_id,
                    season_id=season_id,
                    sources=None,
                    use_constraints=True  # Always use constraints for enhance
                )
                st.toast(f"✅ Re-clustered into {recluster_result.get('n_clusters', 0)} clusters")

            # Optional densify step
            if use_densify:
                with st.spinner("Step 2: Densifying clusters (two-pass)..."):
                    densify_result = densify_two_pass("manual", selected_episode)
                    st.toast(f"✅ Densified {densify_result.get('tracks_promoted', 0)} tracks")

            with st.spinner(f"Step {3 if use_densify else 2}: Regenerating analytics..."):
                clusters_data = load_clusters(selected_episode, DATA_ROOT)
                cluster_assignments = {}
                if clusters_data:
                    for cluster in clusters_data.get("clusters", []):
                        if "name" in cluster:
                            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

                analytics_result = analytics_task("manual", selected_episode, cluster_assignments)
                st.toast(f"✅ Generated {analytics_result['stats']['intervals_created']} intervals")

            st.success(f"✅ Enhance clusters complete for {selected_episode}!")
            st.info("💡 Analytics are now fresh. Metrics have been recomputed with current constraints.")
            safe_rerun()

        except Exception as e:
            st.error(f"❌ Enhance clusters failed: {str(e)}")
            import traceback
            with st.expander("Error details"):
                st.code(traceback.format_exc())

    # Configure mutator
    mutator = configure_workspace_mutator(selected_episode, DATA_ROOT)

    if mutator is None:
        st.error(f"No cluster data found for {selected_episode}. Run clustering first.")
        return

    st.markdown("---")

    # Render tabs internally using radio
    tab = st.radio(
        "Workspace",
        ["Faces", "Clusters", "Tracks", "Review"],
        horizontal=True,
        label_visibility="collapsed",
        index=["Faces", "Clusters", "Tracks", "Review"].index(
            st.session_state.workspace_tab
        ),
        key="workspace_tab_radio",
    )

    # Update session state if tab changed
    if tab != st.session_state.workspace_tab:
        st.session_state.workspace_tab = tab

    # Render selected tab
    if tab == "Faces":
        render_faces_tab(mutator)
    elif tab == "Clusters":
        render_clusters_tab(mutator)
    elif tab == "Tracks":
        render_tracks_tab(mutator)
    else:  # Review
        render_review_tab(mutator)


if __name__ == "__main__":
    main()
