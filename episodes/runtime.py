"""Runtime job tracking for episodes.

Tracks active jobs per episode to enable resume after browser refresh,
prevent duplicate jobs, and provide job status information.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def _get_runtime_path(episode_key: str, data_root: Path = Path("data")) -> Path:
    """Get path to runtime.json for an episode."""
    episodes_dir = data_root / "episodes" / episode_key
    episodes_dir.mkdir(parents=True, exist_ok=True)
    return episodes_dir / "runtime.json"


def set_active_job(
    episode_key: str,
    stage: str,
    job_id: str,
    data_root: Path = Path("data"),
) -> None:
    """
    Mark a job as active for a given episode and stage.

    Args:
        episode_key: Episode key (e.g., "rhobh_s05_e03")
        stage: Stage name (e.g., "detect", "track", "cluster")
        job_id: Job identifier (e.g., "detect_RHOBH_S05_E03_20250107")
        data_root: Root data directory
    """
    runtime_path = _get_runtime_path(episode_key, data_root)

    # Load existing runtime data
    runtime = {}
    if runtime_path.exists():
        try:
            with open(runtime_path) as f:
                runtime = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load runtime.json for {episode_key}: {e}")

    # Update active jobs
    if "active_jobs" not in runtime:
        runtime["active_jobs"] = {}

    runtime["active_jobs"][stage] = job_id
    runtime["updated_at"] = datetime.utcnow().isoformat()

    # Write back
    try:
        with open(runtime_path, "w") as f:
            json.dump(runtime, f, indent=2)
        logger.info(f"[RUNTIME] {episode_key} set_active_job stage={stage} job_id={job_id}")
    except Exception as e:
        logger.error(f"[RUNTIME] {episode_key} Failed to write runtime.json: {e}")


def get_active_job(
    episode_key: str,
    stage: str,
    data_root: Path = Path("data"),
) -> Optional[str]:
    """
    Get the active job_id for a given episode and stage.

    Args:
        episode_key: Episode key
        stage: Stage name
        data_root: Root data directory

    Returns:
        Job ID if active, None otherwise
    """
    runtime_path = _get_runtime_path(episode_key, data_root)

    if not runtime_path.exists():
        return None

    try:
        with open(runtime_path) as f:
            runtime = json.load(f)

        active_jobs = runtime.get("active_jobs", {})
        job_id = active_jobs.get(stage)

        if job_id:
            logger.debug(f"[RUNTIME] {episode_key} get_active_job stage={stage} job_id={job_id}")

        return job_id
    except Exception as e:
        logger.warning(f"[RUNTIME] {episode_key} Could not read runtime.json: {e}")
        return None


def clear_active_job(
    episode_key: str,
    stage: str,
    data_root: Path = Path("data"),
) -> None:
    """
    Clear the active job for a given episode and stage.

    Args:
        episode_key: Episode key
        stage: Stage name
        data_root: Root data directory
    """
    runtime_path = _get_runtime_path(episode_key, data_root)

    if not runtime_path.exists():
        return

    try:
        with open(runtime_path) as f:
            runtime = json.load(f)

        # Remove active job entry
        active_jobs = runtime.get("active_jobs", {})
        if stage in active_jobs:
            del active_jobs[stage]
            runtime["updated_at"] = datetime.utcnow().isoformat()

            with open(runtime_path, "w") as f:
                json.dump(runtime, f, indent=2)

            logger.info(f"[RUNTIME] {episode_key} clear_active_job stage={stage}")
    except Exception as e:
        logger.warning(f"[RUNTIME] {episode_key} Could not clear active job: {e}")


def get_all_active_jobs(
    episode_key: str,
    data_root: Path = Path("data"),
) -> Dict[str, str]:
    """
    Get all active jobs for an episode.

    Args:
        episode_key: Episode key
        data_root: Root data directory

    Returns:
        Dict mapping stage names to job IDs
    """
    runtime_path = _get_runtime_path(episode_key, data_root)

    if not runtime_path.exists():
        return {}

    try:
        with open(runtime_path) as f:
            runtime = json.load(f)
        return runtime.get("active_jobs", {})
    except Exception as e:
        logger.warning(f"[RUNTIME] {episode_key} Could not read active jobs: {e}")
        return {}


def check_job_stalled(
    job_id: str,
    data_root: Path = Path("data"),
    stall_threshold_seconds: int = 30,
) -> bool:
    """
    Check if a job appears stalled (no heartbeat updates).

    Args:
        job_id: Job identifier
        data_root: Root data directory
        stall_threshold_seconds: Time without update to consider stalled

    Returns:
        True if job appears stalled
    """
    envelope_path = data_root / "jobs" / job_id / "meta.json"

    if not envelope_path.exists():
        return False

    try:
        with open(envelope_path) as f:
            envelope = json.load(f)

        # Check if job is running
        if envelope.get("status") != "running":
            return False

        # Check for detect stage status
        stages = envelope.get("stages", {})
        detect_stage = stages.get("detect", {})

        if detect_stage.get("status") != "running":
            return False

        # Check last update time
        result = detect_stage.get("result", {})
        updated_at_str = result.get("updated_at")

        if not updated_at_str:
            return True  # No heartbeat yet

        # Parse timestamp and check age
        try:
            from datetime import datetime, timezone
            updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            age_seconds = (now - updated_at).total_seconds()

            is_stalled = age_seconds > stall_threshold_seconds
            if is_stalled:
                logger.warning(f"[RUNTIME] {job_id} appears stalled (last update {age_seconds:.0f}s ago)")

            return is_stalled
        except Exception as e:
            logger.warning(f"[RUNTIME] {job_id} Could not parse updated_at: {e}")
            return False

    except Exception as e:
        logger.warning(f"[RUNTIME] {job_id} Could not check stall status: {e}")
        return False
