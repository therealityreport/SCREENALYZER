"""
Episode state API for querying registry and job status.

Part of Phase 3: Expose episode registry state for UI polling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from api.jobs import job_manager

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
