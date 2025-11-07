"""Diagnostics and progress tracking utilities."""

import json
import logging
import os
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


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


def truncate_to_last_brace(path: Path) -> int:
    """
    Truncate corrupted JSON file to last valid closing brace.

    Attempts to recover from incomplete writes by finding the last '}'
    and truncating the file there.

    Args:
        path: Path to corrupted JSON file

    Returns:
        Number of bytes truncated, or -1 if recovery failed
    """
    try:
        with open(path, "r+b") as f:
            content = f.read()
            original_size = len(content)

            # Find last occurrence of '}'
            last_brace = content.rfind(b'}')

            if last_brace == -1:
                logger.error(f"[JSON-RECOVER] No closing brace found in {path}, cannot recover")
                return -1

            # Truncate to last brace + 1
            f.seek(0)
            f.write(content[:last_brace + 1])
            f.truncate()
            f.flush()
            os.fsync(f.fileno())

        chars_truncated = original_size - (last_brace + 1)
        logger.warning(f"[JSON-RECOVER] Healed {path} | chars_truncated={chars_truncated} | original_size={original_size} | final_size={last_brace + 1}")
        return chars_truncated
    except Exception as e:
        logger.error(f"[JSON-RECOVER] Failed to truncate {path}: {e}")
        return -1


def safe_load_json(path: Path) -> Dict[str, Any]:
    """
    Safely load JSON file with automatic recovery from corruption.

    If the file is corrupted (JSONDecodeError), attempts to recover by:
    1. Truncating to last valid closing brace
    2. Retrying load
    3. Returning empty dict if recovery fails

    Args:
        path: Path to JSON file

    Returns:
        Parsed JSON dict, or empty dict if file doesn't exist or is unrecoverable
    """
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"[JSON-ERROR] Malformed JSON in {path}: {e}")

        # Attempt recovery
        try:
            chars_truncated = truncate_to_last_brace(path)

            if chars_truncated >= 0:
                # Retry load
                with open(path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                logger.info(f"[JSON-RECOVER] Successfully recovered {path} after truncating {chars_truncated} bytes")
                return result
            else:
                logger.error(f"[JSON-RECOVER] Could not recover {path}, returning empty dict")
                return {}
        except Exception as recovery_error:
            logger.error(f"[JSON-RECOVER] Failed to recover {path}: {recovery_error}", exc_info=True)
            return {}
    except Exception as e:
        logger.error(f"[JSON-ERROR] Failed to load {path}: {e}", exc_info=True)
        return {}


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

    Uses atomic write with flush+fsync to prevent corruption.

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

    # Atomic write with flush+fsync to prevent corruption
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # CRITICAL: Append to pipeline progress telemetry log for QA debugging
    try:
        telemetry_dir = Path("logs")
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        telemetry_log = telemetry_dir / "pipeline_progress.log"

        # Extract metrics from extra
        metrics_str = ""
        if extra:
            frames_done = extra.get("frames_done")
            frames_total = extra.get("frames_total")
            faces_detected = extra.get("faces_detected")
            tracks_active = extra.get("tracks_active")
            clusters_built = extra.get("clusters_built")

            if frames_done is not None and frames_total is not None:
                metrics_str = f" frames={frames_done}/{frames_total}"
            if faces_detected is not None:
                metrics_str += f" faces={faces_detected}"
            if tracks_active is not None:
                metrics_str += f" tracks={tracks_active}"
            if clusters_built is not None:
                metrics_str += f" clusters={clusters_built}"

        # Format: [timestamp] episode stage pct% message metrics
        pct_str = f"{pct*100:.1f}%" if pct is not None else "N/A"
        log_line = f"[{dt.datetime.utcnow().isoformat()}] {episode_id} {step} {pct_str} {status} | {message}{metrics_str}\n"

        with telemetry_log.open("a", encoding="utf-8") as f:
            f.write(log_line)
            f.flush()
    except Exception as e:
        logger.warning(f"Failed to write pipeline telemetry: {e}")


def write_pipeline_state(episode_id: str, state: Dict[str, Any]) -> None:
    """Write full pipeline state to diagnostics with atomic write."""
    out_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state_full.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write with flush+fsync
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(state), f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def read_pipeline_state(episode_id: str) -> Optional[Dict[str, Any]]:
    """Read current pipeline state with safe JSON loading."""
    state_path = Path("data") / "harvest" / episode_id / "diagnostics" / "pipeline_state.json"
    if not state_path.exists():
        return None

    # Use safe_load_json to handle corruption
    result = safe_load_json(state_path)
    return result if result else None


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
