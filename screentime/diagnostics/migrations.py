"""
Audit logging for episode management operations.

Logs all episode lifecycle events (move, remove, restore, rehash) with
full before/after state, actor, timestamp, and reason.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

AUDIT_LOG_DIR = Path("data/audit")
AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def emit_episode_op_event(
    op_type: str,
    episode_id: str,
    before_state: Optional[Dict[str, Any]] = None,
    after_state: Optional[Dict[str, Any]] = None,
    actor: str = "system",
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit an audit event for an episode operation.

    Args:
        op_type: Operation type (move, remove, restore, rehash)
        episode_id: Episode identifier
        before_state: State before operation (paths, metadata, etc.)
        after_state: State after operation
        actor: Who performed the operation (user, system, etc.)
        reason: Why the operation was performed
        metadata: Additional operation-specific metadata
    """
    event = {
        "event_type": "episode_operation",
        "op_type": op_type,
        "episode_id": episode_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "actor": actor,
        "reason": reason or f"{op_type} operation",
        "before_state": before_state or {},
        "after_state": after_state or {},
        "metadata": metadata or {},
    }

    # Write to episode-specific audit log
    log_file = AUDIT_LOG_DIR / f"{episode_id}_audit.jsonl"
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")
        logger.info(f"[Audit] {op_type} for {episode_id} by {actor}")
    except Exception as e:
        logger.error(f"Failed to write audit log: {e}")

    # Also write to global audit log
    global_log = AUDIT_LOG_DIR / "global_audit.jsonl"
    try:
        with open(global_log, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        logger.error(f"Failed to write global audit log: {e}")


def get_audit_history(episode_id: str, limit: int = 50) -> list[Dict[str, Any]]:
    """
    Get audit history for an episode.

    Args:
        episode_id: Episode identifier
        limit: Maximum number of events to return (most recent first)

    Returns:
        List of audit events, newest first
    """
    log_file = AUDIT_LOG_DIR / f"{episode_id}_audit.jsonl"
    if not log_file.exists():
        return []

    try:
        with open(log_file) as f:
            events = [json.loads(line) for line in f]
        return events[-limit:][::-1]  # Most recent first
    except Exception as e:
        logger.error(f"Failed to read audit log: {e}")
        return []
