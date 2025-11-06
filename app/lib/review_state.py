"""
Review state management for Phase 1.6.

Handles undo/redo, autosave, and action history.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ReviewAction:
    """Single review action for undo/redo."""

    action_type: str  # merge, split, assign, skip
    timestamp: float
    data: dict
    user: str = "admin"


class ReviewStateManager:
    """Manages review session state with undo/autosave."""

    def __init__(self, episode_id: str, data_root: Path = Path("data"), max_undo: int = 10):
        """
        Initialize review state manager.

        Args:
            episode_id: Episode identifier
            data_root: Data root directory
            max_undo: Maximum number of undo operations
        """
        self.episode_id = episode_id
        self.data_root = data_root
        self.max_undo = max_undo

        # State
        self.undo_stack: deque[ReviewAction] = deque(maxlen=max_undo)
        self.last_autosave_time = time.time()
        self.autosave_interval_sec = 30

        # Paths
        self.harvest_dir = data_root / "harvest" / episode_id
        self.state_file = self.harvest_dir / "review_state.json"
        self.audit_file = self.harvest_dir / "diagnostics" / "audit.jsonl"

        # Load existing state
        self._load_state()

    def record_action(self, action_type: str, data: dict, user: str = "admin") -> None:
        """
        Record a review action.

        Args:
            action_type: Type of action (merge, split, assign, skip)
            data: Action data
            user: User who performed action
        """
        action = ReviewAction(
            action_type=action_type,
            timestamp=time.time(),
            data=data,
            user=user,
        )

        # Add to undo stack
        self.undo_stack.append(action)

        # Log to audit file
        self._log_audit(action)

        # Check autosave
        if time.time() - self.last_autosave_time > self.autosave_interval_sec:
            self.autosave()

    def undo(self) -> Optional[ReviewAction]:
        """
        Undo last action.

        Returns:
            ReviewAction that was undone, or None if stack empty
        """
        if not self.undo_stack:
            return None

        action = self.undo_stack.pop()

        # Log undo to audit
        self._log_audit(
            ReviewAction(
                action_type="undo",
                timestamp=time.time(),
                data={"original_action": action.action_type, "original_data": action.data},
                user=action.user,
            )
        )

        return action

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.undo_stack) > 0

    def get_undo_stack_size(self) -> int:
        """Get number of undoable actions."""
        return len(self.undo_stack)

    def autosave(self) -> None:
        """Save current state."""
        state_data = {
            "episode_id": self.episode_id,
            "last_saved": datetime.utcnow().isoformat(),
            "undo_stack_size": len(self.undo_stack),
        }

        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state_data, f, indent=2)

        self.last_autosave_time = time.time()

    def _load_state(self) -> None:
        """Load existing state from disk."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                state_data = json.load(f)
                # State loaded (undo stack not persisted for simplicity)

    def _log_audit(self, action: ReviewAction) -> None:
        """Log action to audit file."""
        self.audit_file.parent.mkdir(parents=True, exist_ok=True)

        audit_entry = {
            "timestamp": datetime.fromtimestamp(action.timestamp).isoformat(),
            "episode_id": self.episode_id,
            "action_type": action.action_type,
            "user": action.user,
            "data": action.data,
        }

        with open(self.audit_file, "a") as f:
            f.write(json.dumps(audit_entry) + "\n")
