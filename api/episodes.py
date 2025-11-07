"""
Episode state API for querying registry and job status.

Part of Phase 3: Expose episode registry state for UI polling.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from api.jobs import job_manager
from episodes.runtime import get_all_active_jobs
from screentime.diagnostics.utils import safe_load_json

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


def get_episode_state(episode_id: str) -> dict:
    """
    Get complete episode state from registry and related jobs.

    Args:
        episode_id: Episode ID (any format)

    Returns:
        Dict with episode state information:
        {
            "exists": bool,
            "episode_key": str,
            "episode_id": str,
            "states": {...},
            "video_path": str,
            "registry_path": str,
            "related_jobs": [{job_id, mode, stages}, ...],
            "timestamps": {...}
        }
    """
    episode_key = job_manager.normalize_episode_key(episode_id)
    registry = job_manager.load_episode_registry(episode_key)

    if not registry:
        return {
            "exists": False,
            "episode_key": episode_key,
            "error": f"Episode registry not found for {episode_key}",
        }

    # Find related job envelopes
    jobs_dir = DATA_ROOT / "jobs"
    related_jobs = []

    if jobs_dir.exists():
        for job_dir in jobs_dir.iterdir():
            if job_dir.is_dir():
                envelope = job_manager.load_job_envelope(job_dir.name)
                if envelope and envelope.get("episode_key") == episode_key:
                    related_jobs.append({
                        "job_id": envelope.get("job_id"),
                        "mode": envelope.get("mode"),
                        "created_at": envelope.get("created_at"),
                        "stages": envelope.get("stages", {}),
                    })

    return {
        "exists": True,
        "episode_key": episode_key,
        "episode_id": registry.get("episode_id"),
        "show": registry.get("show"),
        "season": registry.get("season"),
        "episode": registry.get("episode"),
        "video_path": registry.get("video_path"),
        "registry_path": f"episodes/{episode_key}/state.json",
        "states": registry.get("states", {}),
        "paths": registry.get("paths", {}),
        "timestamps": registry.get("timestamps", {}),
        "related_jobs": related_jobs,
    }


def get_extraction_progress(episode_id: str) -> dict:
    """
    Get frame extraction progress for an episode.

    Checks if extraction is in progress, completed, or not started.

    Args:
        episode_id: Episode ID

    Returns:
        Dict with progress information
    """
    episode_key = job_manager.normalize_episode_key(episode_id)
    registry = job_manager.load_episode_registry(episode_key)

    if not registry:
        return {
            "status": "not_started",
            "message": "Episode not found",
        }

    states = registry.get("states", {})

    if states.get("extracted_frames"):
        return {
            "status": "completed",
            "message": "Frame extraction complete",
            "episode_key": episode_key,
        }

    # Check if there's an active extraction job
    jobs_dir = DATA_ROOT / "jobs"
    if jobs_dir.exists():
        for job_dir in jobs_dir.iterdir():
            if job_dir.is_dir() and ("auto_extract" in job_dir.name or "harvest" in job_dir.name):
                envelope = job_manager.load_job_envelope(job_dir.name)
                if envelope and envelope.get("episode_key") == episode_key:
                    # Check job status
                    stages = envelope.get("stages", {})
                    if any(stage.get("status") == "running" for stage in stages.values()):
                        return {
                            "status": "in_progress",
                            "message": "Frame extraction in progress",
                            "episode_key": episode_key,
                            "job_id": envelope.get("job_id"),
                        }

    if states.get("validated"):
        return {
            "status": "pending",
            "message": "Validated but not yet extracted",
            "episode_key": episode_key,
        }

    return {
        "status": "not_started",
        "message": "Not validated yet",
        "episode_key": episode_key,
    }


def list_all_episodes() -> list[dict]:
    """
    List all episodes from episode registry.

    Returns:
        List of episode state summaries
    """
    episodes_dir = DATA_ROOT / "episodes"

    if not episodes_dir.exists():
        return []

    episodes = []

    for episode_dir in episodes_dir.iterdir():
        if episode_dir.is_dir():
            registry = job_manager.load_episode_registry(episode_dir.name)
            if registry:
                episodes.append({
                    "episode_key": registry.get("episode_key"),
                    "episode_id": registry.get("episode_id"),
                    "show": registry.get("show"),
                    "season": registry.get("season"),
                    "episode": registry.get("episode"),
                    "states": registry.get("states", {}),
                    "timestamps": registry.get("timestamps", {}),
                })

    # Sort by created timestamp descending
    episodes.sort(
        key=lambda e: e.get("timestamps", {}).get("created", ""),
        reverse=True
    )

    return episodes


def _tail_log_file(path: Path, limit: int) -> List[str]:
    """Return the last `limit` lines from a log file."""
    if limit <= 0 or not path.exists():
        return []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as handle:
            lines = handle.readlines()
        return [line.rstrip("\n") for line in lines[-limit:]]
    except Exception as exc:  # pragma: no cover - best effort logging
        logger.warning(f"[STATUS] Failed to read log tail {path}: {exc}")
        return []


def _load_pipeline_state(episode_identifier: str) -> Dict[str, Any]:
    """Load pipeline_state.json for a given identifier (episode_id or key)."""
    diagnostics_path = DATA_ROOT / "harvest" / episode_identifier / "diagnostics" / "pipeline_state.json"
    if not diagnostics_path.exists():
        return {}

    try:
        result = safe_load_json(diagnostics_path)
    except Exception as exc:  # pragma: no cover
        logger.warning(f"[STATUS] Could not load pipeline_state for {episode_identifier}: {exc}")
        result = {}
    return result or {}


def get_status_snapshot(episode_id: str, log_limit: int = 200) -> dict:
    """
    Build a read-only status snapshot for the Workspace status panel.

    Returns a JSON-serializable dict with pipeline state, job envelope,
    summarized progress, and recent log tails.
    """
    episode_key = job_manager.normalize_episode_key(episode_id)

    # Try both the original ID and normalized key for pipeline state
    pipeline_state = _load_pipeline_state(episode_id)
    if not pipeline_state and episode_key != episode_id:
        pipeline_state = _load_pipeline_state(episode_key)

    # Determine active job envelope (if any)
    active_jobs = get_all_active_jobs(episode_key, DATA_ROOT)
    job_envelope: Dict[str, Any] = {}
    selected_job_id: Optional[str] = None
    if active_jobs:
        stage_preference = ["detect", "track", "cluster", "stills", "analytics", "full", "prepare"]
        for stage in stage_preference:
            if stage in active_jobs:
                selected_job_id = active_jobs[stage]
                break
        if not selected_job_id:
            selected_job_id = next(iter(active_jobs.values()))

        if selected_job_id:
            loaded_envelope = job_manager.load_job_envelope(selected_job_id) or {}
            if loaded_envelope:
                job_envelope = loaded_envelope
                job_envelope["_job_id"] = selected_job_id

    # Progress summary
    stage_name = pipeline_state.get("current_step", "Full Pipeline") if pipeline_state else "Full Pipeline"
    message = pipeline_state.get("message", "") if pipeline_state else ""
    pct_value = pipeline_state.get("pct", 0.0) if pipeline_state else 0.0
    try:
        pct_float = float(pct_value or 0.0)
    except (TypeError, ValueError):
        pct_float = 0.0

    pct_float = max(0.0, min(pct_float, 1.0))

    snapshot = {
        "episode_id": episode_id,
        "episode_key": episode_key,
        "pipeline_state": pipeline_state,
        "job_envelope": job_envelope,
        "active_jobs": active_jobs,
        "progress": {
            "stage": stage_name or "Full Pipeline",
            "pct": pct_float,
            "message": message or "",
        },
        "logs": {
            "workspace": _tail_log_file(Path("logs") / "workspace_debug.log", log_limit),
            "progress": _tail_log_file(Path("logs") / "pipeline_progress.log", log_limit),
        },
        "ts": datetime.utcnow().isoformat() + "Z",
    }

    return snapshot
