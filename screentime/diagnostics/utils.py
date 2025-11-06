"""Diagnostics and progress tracking utilities."""

import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def json_safe(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Handles:
    - numpy integers → int
    - numpy floats → float
    - numpy arrays → list
    - dicts and lists recursively
    - datetime objects → ISO format strings
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def emit_progress(
    episode_id: str,
    step: str,
    step_index: int,
    total_steps: int,
    status: str = "running",
    pct: Optional[float] = None,
    message: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit pipeline progress for UI consumption.

    Writes to data/harvest/{episode}/diagnostics/pipeline_state.json

    Args:
        episode_id: Episode identifier
        step: Current step name (e.g., "Track", "Cluster")
        step_index: 1-indexed step number
        total_steps: Total number of steps
        status: "running", "ok", "error", "skipped"
        pct: Optional fine-grained progress percentage (0-1)
        message: Human-readable status message
        extra: Optional extra data (will be JSON-sanitized)
    """
    payload = {
        "episode": episode_id,
        "ts": dt.datetime.utcnow().isoformat(),
        "current_step": step,
        "step_index": step_index,
        "total_steps": total_steps,
        "status": status,
        "pct": float(pct) if pct is not None else None,
        "message": message,
        "extra": json_safe(extra) if extra is not None else None,
    }

    out_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(json_safe(payload), f, indent=2)


def write_pipeline_state(episode_id: str, state: Dict[str, Any]) -> None:
    """Write full pipeline state to diagnostics."""
    out_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        json.dump(json_safe(state), f, indent=2)


def read_pipeline_state(episode_id: str) -> Optional[Dict[str, Any]]:
    """Read current pipeline state."""
    state_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state.json"
    if not state_path.exists():
        return None

    with state_path.open() as f:
        return json.load(f)


def archive_pipeline_state(episode_id: str) -> None:
    """Archive completed pipeline state to prevent re-showing."""
    import datetime as dt
    state_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state.json"

    if not state_path.exists():
        return

    # Create archive directory
    archive_dir = state_path.parent / "archive"
    archive_dir.mkdir(exist_ok=True)

    # Move to archive with timestamp
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_path = archive_dir / f"{timestamp}_pipeline_state.json"

    state_path.rename(archive_path)
