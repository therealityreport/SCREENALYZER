"""
Telemetry tracking for Screenalyzer.

Events are logged to structured JSON for downstream aggregation and alerting.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class TelemetryEvent(str, Enum):
    """Telemetry event types."""

    # Upload events
    UPLOAD_STARTED = "upload_started"
    UPLOAD_CHUNK = "upload_chunk"
    UPLOAD_RESUMED = "upload_resumed"
    UPLOAD_COMPLETED = "upload_completed"
    UPLOAD_VALIDATED = "upload_validated"
    UPLOAD_FAILED = "upload_failed"

    # UI events
    UI_LOAD_TIME_MS = "ui_load_time_ms"
    UI_ACTION = "ui_action"

    # Job events
    JOB_STARTED = "job_started"
    JOB_STAGE_COMPLETE = "job_stage_complete"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"
    JOB_CANCELLED = "job_cancelled"

    # Processing events
    DETECTION_STARTED = "detection_started"
    DETECTION_COMPLETED = "detection_completed"
    TRACKING_STARTED = "tracking_started"
    TRACKING_COMPLETED = "tracking_completed"
    CLUSTERING_STARTED = "clustering_started"
    CLUSTERING_COMPLETED = "clustering_completed"

    # Review events
    REVIEW_MERGE = "review_merge"
    REVIEW_SPLIT = "review_split"
    REVIEW_ASSIGN = "review_assign"
    REVIEW_UNDO = "review_undo"


@dataclass
class TelemetryRecord:
    """Telemetry record structure."""

    event: TelemetryEvent
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = {
            "event": self.event.value,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
            "error": self.error,
        }
        return json.dumps(data)


class TelemetryLogger:
    """Structured telemetry logger."""

    def __init__(self, log_file: Optional[Path] = None):
        self.logger = logging.getLogger("screenalyzer.telemetry")
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(file_handler)

    def log(
        self,
        event: TelemetryEvent,
        duration_ms: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Log a telemetry event.

        Args:
            event: Event type
            duration_ms: Optional duration in milliseconds
            metadata: Optional event metadata
            error: Optional error message
        """
        record = TelemetryRecord(
            event=event,
            duration_ms=duration_ms,
            metadata=metadata or {},
            error=error,
        )
        self.logger.info(record.to_json())

    def log_upload_started(self, session_id: str, filename: str, size_bytes: int) -> None:
        """Log upload started event."""
        self.log(
            TelemetryEvent.UPLOAD_STARTED,
            metadata={
                "session_id": session_id,
                "filename": filename,
                "size_bytes": size_bytes,
            },
        )

    def log_upload_chunk(
        self, session_id: str, chunk_id: int, chunk_size: int, progress_pct: float
    ) -> None:
        """Log upload chunk event."""
        self.log(
            TelemetryEvent.UPLOAD_CHUNK,
            metadata={
                "session_id": session_id,
                "chunk_id": chunk_id,
                "chunk_size": chunk_size,
                "progress_pct": progress_pct,
            },
        )

    def log_upload_resumed(self, session_id: str, from_chunk: int, progress_pct: float) -> None:
        """Log upload resumed event."""
        self.log(
            TelemetryEvent.UPLOAD_RESUMED,
            metadata={
                "session_id": session_id,
                "from_chunk": from_chunk,
                "progress_pct": progress_pct,
            },
        )

    def log_upload_completed(self, session_id: str, duration_ms: float, size_bytes: int) -> None:
        """Log upload completed event."""
        self.log(
            TelemetryEvent.UPLOAD_COMPLETED,
            duration_ms=duration_ms,
            metadata={
                "session_id": session_id,
                "size_bytes": size_bytes,
            },
        )

    def log_upload_validated(
        self,
        session_id: str,
        is_valid: bool,
        duration_ms: float,
        errors: Optional[list[str]] = None,
    ) -> None:
        """Log upload validation event."""
        self.log(
            TelemetryEvent.UPLOAD_VALIDATED,
            duration_ms=duration_ms,
            metadata={
                "session_id": session_id,
                "is_valid": is_valid,
                "errors": errors or [],
            },
        )

    def log_upload_failed(self, session_id: str, error: str, stage: str) -> None:
        """Log upload failed event."""
        self.log(
            TelemetryEvent.UPLOAD_FAILED,
            metadata={
                "session_id": session_id,
                "stage": stage,
            },
            error=error,
        )

    def log_ui_load_time(self, page: str, duration_ms: float) -> None:
        """Log UI page load time."""
        self.log(
            TelemetryEvent.UI_LOAD_TIME_MS,
            duration_ms=duration_ms,
            metadata={"page": page},
        )


# Global telemetry logger instance
telemetry = TelemetryLogger(log_file=Path("data/diagnostics/telemetry.jsonl"))
