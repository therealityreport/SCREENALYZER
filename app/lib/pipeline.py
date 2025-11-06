"""
Pipeline helpers for checking artifact status and managing pipeline state.
"""

import json
from pathlib import Path
from typing import Any, Dict, Union

def check_artifacts_status(episode_id: str, data_root: Union[Path, str] = Path("data")) -> Dict[str, Any]:
    """
    Check which pipeline artifacts exist for an episode.

    Returns:
        Dict with keys:
        - prepared: bool - True if tracks.json exists
        - has_clusters: bool - True if clusters.json exists
        - has_analytics: bool - True if timeline.csv exists
        - missing_message: str - Human-readable message about what's missing
        - next_action: str - What action to take next ("prepare", "cluster", "analyze", or "ready")
    """
    data_root = Path(data_root)
    harvest_dir = data_root / "harvest" / episode_id

    tracks_file = harvest_dir / "tracks.json"
    clusters_file = harvest_dir / "clusters.json"
    timeline_file = harvest_dir / "timeline.csv"

    prepared = tracks_file.exists()
    has_clusters = clusters_file.exists()
    has_analytics = timeline_file.exists()

    if not prepared:
        return {
            "prepared": False,
            "has_clusters": False,
            "has_analytics": False,
            "missing_message": "Episode not prepared - no tracks found",
            "next_action": "prepare",
        }

    if not has_clusters:
        return {
            "prepared": True,
            "has_clusters": False,
            "has_analytics": False,
            "missing_message": "Tracks prepared but not clustered",
            "next_action": "cluster",
        }

    if not has_analytics:
        return {
            "prepared": True,
            "has_clusters": True,
            "has_analytics": False,
            "missing_message": "Clusters exist but analytics not generated",
            "next_action": "analyze",
        }

    return {
        "prepared": True,
        "has_clusters": True,
        "has_analytics": True,
        "missing_message": "",
        "next_action": "ready",
    }


def is_pipeline_running(episode_id: str, data_root: Union[Path, str] = Path("data")) -> bool:
    """
    Check if pipeline is currently running for an episode.

    Returns:
        True if pipeline_state.json exists and status is "running"
    """
    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    if not state_file.exists():
        return False

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        status = state.get("status", "")
        return status == "running"
    except Exception:
        return False


def get_progress_bars_config(episode_id: str, data_root: Union[Path, str] = Path("data")) -> Dict[str, Any]:
    """
    Get progress bar configuration from pipeline_state.json.

    Returns:
        Dict with keys:
        - active_stage: str - Current stage name
        - stages: list - List of stage dicts with name, status, pct, message
        - overall_pct: float - Overall progress (0.0 to 1.0)
        - status: str - "running", "ok", "error", etc.
    """
    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    if not state_file.exists():
        return {
            "active_stage": "",
            "stages": [],
            "overall_pct": 0.0,
            "status": "",
        }

    try:
        with open(state_file, "r") as f:
            state = json.load(f)
    except Exception:
        return {
            "active_stage": "",
            "stages": [],
            "overall_pct": 0.0,
            "status": "",
        }

    current_step = state.get("current_step", "")
    step_index = state.get("step_index", 0)
    total_steps = state.get("total_steps", 1)
    status = state.get("status", "")
    message = state.get("message", "")
    pct = state.get("pct")

    # Use stage names from state if provided (dynamic), otherwise use generic
    # Orchestrator can write "stage_names": ["Detect/Embed", "Track", "Generate Face Stills"]
    if "stage_names" in state:
        stage_names = state["stage_names"]
    else:
        # Fallback: generate generic stage names based on total_steps
        stage_names = [f"Stage {i}" for i in range(1, total_steps + 1)]
        # Try to use current_step if it's meaningful
        if current_step and step_index > 0 and step_index <= len(stage_names):
            stage_names[step_index - 1] = current_step

    stages = []
    for idx in range(1, total_steps + 1):
        stage_name = stage_names[idx - 1] if idx <= len(stage_names) else f"Stage {idx}"

        if idx < step_index:
            # Completed stages
            stages.append({
                "name": stage_name,
                "status": "ok",
                "pct": 1.0,
                "message": "Complete",
            })
        elif idx == step_index:
            # Current stage
            stages.append({
                "name": stage_name,
                "status": status,
                "pct": pct if pct is not None else 0.5,
                "message": message,
            })
        else:
            # Upcoming stages
            stages.append({
                "name": stage_name,
                "status": "pending",
                "pct": 0.0,
                "message": "Pending",
            })

    # Calculate overall progress
    if total_steps > 0 and step_index > 0:
        base_pct = (step_index - 1) / max(total_steps, 1)
        step_pct = (pct if pct is not None else 0.5) / max(total_steps, 1)
        overall_pct = base_pct + step_pct
    else:
        overall_pct = 0.0

    return {
        "active_stage": current_step,
        "stages": stages,
        "overall_pct": overall_pct,
        "status": status,
    }


def archive_pipeline_state(episode_id: str, data_root: Union[Path, str] = Path("data")) -> None:
    """
    Archive the current pipeline_state.json by renaming it with a timestamp.
    """
    from datetime import datetime

    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    if state_file.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_file = diagnostics_dir / f"pipeline_state_{timestamp}.json"
        state_file.rename(archive_file)


def read_pipeline_state(episode_id: str, data_root: Union[Path, str] = Path("data")) -> Dict[str, Any]:
    """
    Read the current pipeline state from pipeline_state.json.

    Returns:
        Dict with pipeline state, or empty dict if file doesn't exist.
        Includes maintenance_mode, active_op, active_op_progress, episode_hash keys.
    """
    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    if not state_file.exists():
        return {}

    try:
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def is_maintenance_mode(episode_id: str, data_root: Union[Path, str] = Path("data")) -> bool:
    """
    Check if episode is in maintenance mode (episode operation in progress).

    Returns:
        True if maintenance_mode is enabled for this episode
    """
    state = read_pipeline_state(episode_id, data_root)
    return state.get("maintenance_mode", False)


def get_maintenance_reason(episode_id: str, data_root: Union[Path, str] = Path("data")) -> str:
    """
    Get human-readable reason why episode is in maintenance mode.

    Returns:
        Description of active operation, or empty string if not in maintenance mode
    """
    state = read_pipeline_state(episode_id, data_root)
    if not state.get("maintenance_mode"):
        return ""

    active_op = state.get("active_op", {})
    op_type = active_op.get("type", "unknown")
    op_episode = active_op.get("episode_id", episode_id)

    progress = state.get("active_op_progress", {})
    stage = progress.get("stage", "")
    current = progress.get("current", 0)
    total = progress.get("total", 1)

    op_names = {
        "move": "Move",
        "remove": "Archive",
        "restore": "Restore",
        "rehash": "Rehash",
    }
    op_name = op_names.get(op_type, op_type.title())

    if stage:
        return f"{op_name} in progress: {stage} ({current}/{total})"
    return f"{op_name} operation in progress"


def check_pipeline_can_run(episode_id: str, data_root: Union[Path, str] = Path("data")) -> Dict[str, Any]:
    """
    Check if pipeline operations (Prepare/Cluster/Analyze) can run.

    Returns:
        Dict with keys:
        - can_run: bool - Whether pipeline can run
        - reason: str - Human-readable reason if can't run
        - maintenance_mode: bool - Whether in maintenance mode
        - pipeline_running: bool - Whether pipeline is already running
    """
    maintenance = is_maintenance_mode(episode_id, data_root)
    pipeline_running = is_pipeline_running(episode_id, data_root)

    if maintenance:
        reason = get_maintenance_reason(episode_id, data_root)
        return {
            "can_run": False,
            "reason": reason,
            "maintenance_mode": True,
            "pipeline_running": False,
        }

    if pipeline_running:
        return {
            "can_run": False,
            "reason": "Pipeline already running for this episode",
            "maintenance_mode": False,
            "pipeline_running": True,
        }

    return {
        "can_run": True,
        "reason": "",
        "maintenance_mode": False,
        "pipeline_running": False,
    }


def orchestrate_rehash_episode(episode_id: str, data_root: Union[Path, str] = Path("data")) -> Dict[str, Any]:
    """
    Orchestrate rehash operation for an episode.

    Validates files, recomputes hash, and updates registry.

    Returns:
        Dict with:
        - status: "ok" or "error"
        - episode_hash: New hash value
        - validated_files: Dict of file validation results
        - errors: List of error messages
    """
    try:
        from app.lib.episode_manager import rehash_episode

        result = rehash_episode(episode_id, actor="orchestrator", reason="Manual rehash")

        return {
            "status": "ok",
            "episode_id": result.episode_id,
            "episode_hash": result.new_hash,
            "old_hash": result.old_hash,
            "validated_files": result.validated_files,
            "errors": result.errors,
        }
    except Exception as e:
        return {
            "status": "error",
            "episode_id": episode_id,
            "error": str(e),
        }


# ============================================================================
# ETA Tracking System
# ============================================================================

def update_step_stats(
    episode_id: str,
    operation: str,
    step_name: str,
    duration_seconds: float,
    data_root: Union[Path, str] = Path("data"),
) -> None:
    """
    Update rolling median statistics for a pipeline step.

    Stores timing data in pipeline_state.json under stats key:
    {
        "stats": {
            "prepare": {
                "detect_embed": {"samples": [15.2, 16.1], "median_s": 15.65, "p90_s": 16.05},
                "track": {"samples": [8.5], "median_s": 8.5, "p90_s": 8.5}
            },
            "cluster": {...},
            "analyze": {...}
        }
    }

    Maintains max 10 samples per step (rolling window).

    Args:
        episode_id: Episode identifier
        operation: Operation name ("prepare", "cluster", "analyze")
        step_name: Step name within operation
        duration_seconds: Time taken for this step
        data_root: Root data directory
    """
    import statistics

    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    # Read existing state
    if state_file.exists():
        with open(state_file, "r") as f:
            state = json.load(f)
    else:
        state = {}

    # Initialize stats structure if needed
    if "stats" not in state:
        state["stats"] = {}
    if operation not in state["stats"]:
        state["stats"][operation] = {}
    if step_name not in state["stats"][operation]:
        state["stats"][operation][step_name] = {"samples": []}

    # Add new sample (keep max 10)
    samples = state["stats"][operation][step_name]["samples"]
    samples.append(duration_seconds)
    if len(samples) > 10:
        samples.pop(0)  # Remove oldest

    # Compute median and P90
    if len(samples) >= 1:
        state["stats"][operation][step_name]["median_s"] = statistics.median(samples)
        state["stats"][operation][step_name]["p90_s"] = (
            statistics.quantiles(samples, n=10)[8] if len(samples) >= 3 else max(samples)
        )

    # Write back
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def calculate_eta(
    episode_id: str,
    operation: str,
    current_step_name: str,
    steps_remaining: list,
    current_step_elapsed_s: float = 0.0,
    data_root: Union[Path, str] = Path("data"),
) -> Dict[str, Any]:
    """
    Calculate estimated time remaining based on rolling median stats.

    Args:
        episode_id: Episode identifier
        operation: Operation name ("prepare", "cluster", "analyze")
        current_step_name: Name of step currently executing
        steps_remaining: List of step names that will execute after current
        current_step_elapsed_s: Time already spent on current step
        data_root: Root data directory

    Returns:
        Dict with:
        - eta_seconds: float - Estimated seconds remaining (clamped >= 0)
        - confidence: str - "high" (>=3 samples), "low" (<3 samples), "none" (no data)
        - breakdown: Dict[str, float] - Estimated time per remaining step
    """
    state = read_pipeline_state(episode_id, data_root)
    stats = state.get("stats", {}).get(operation, {})

    # Estimate remaining time for current step
    current_step_stats = stats.get(current_step_name, {})
    current_step_median = current_step_stats.get("median_s", 0.0)
    current_step_remaining = max(0.0, current_step_median - current_step_elapsed_s)

    # Estimate time for remaining steps
    breakdown = {current_step_name: current_step_remaining}
    total_eta = current_step_remaining

    for step in steps_remaining:
        step_stats = stats.get(step, {})
        step_median = step_stats.get("median_s", 0.0)
        breakdown[step] = step_median
        total_eta += step_median

    # Determine confidence
    sample_counts = [len(stats.get(step, {}).get("samples", [])) for step in [current_step_name] + steps_remaining]
    min_samples = min(sample_counts) if sample_counts else 0

    if min_samples >= 3:
        confidence = "high"
    elif min_samples >= 1:
        confidence = "low"
    else:
        confidence = "none"

    return {
        "eta_seconds": max(0.0, total_eta),
        "confidence": confidence,
        "breakdown": breakdown,
    }


def format_eta(eta_seconds: float) -> str:
    """
    Format ETA in human-readable form.

    Args:
        eta_seconds: Seconds remaining

    Returns:
        Formatted string like "2m 30s" or "45s"
    """
    if eta_seconds < 0:
        return "0s"

    minutes = int(eta_seconds // 60)
    seconds = int(eta_seconds % 60)

    if minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def cancel_pipeline(episode_id: str, data_root: Union[Path, str] = Path("data")) -> bool:
    """
    Cancel a running pipeline by updating pipeline_state.json.

    Sets status to "cancelled" and clears maintenance mode.

    Args:
        episode_id: Episode whose pipeline to cancel
        data_root: Data root directory

    Returns:
        True if cancelled successfully, False if no pipeline was running
    """
    data_root = Path(data_root)
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    state_file = diagnostics_dir / "pipeline_state.json"

    if not state_file.exists():
        return False

    try:
        with open(state_file, "r") as f:
            state = json.load(f)

        status = state.get("status", "")
        if status != "running":
            return False

        # Update to cancelled state
        state["status"] = "cancelled"
        state["message"] = "Pipeline cancelled by user"
        state["maintenance_mode"] = False
        state.pop("active_op", None)
        state.pop("active_op_progress", None)

        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Cancelled pipeline for {episode_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to cancel pipeline for {episode_id}: {e}")
        return False
