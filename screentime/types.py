"""
Data models and type definitions for Screenalyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class VideoCodec(str, Enum):
    """Supported video codecs."""

    H264 = "h264"
    H265 = "hevc"
    AV1 = "av1"


class ImageFormat(str, Enum):
    """Supported image formats."""

    JPEG = "jpeg"
    PNG = "png"


class JobStatus(str, Enum):
    """Job processing status."""

    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class UploadStatus(str, Enum):
    """Upload session status."""

    CREATED = "created"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class VideoMetadata:
    """Video file metadata."""

    path: Path
    codec: VideoCodec
    duration_sec: float
    fps: float
    width: int
    height: int
    file_size_bytes: int
    bitrate_kbps: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.path, str):
            self.path = Path(self.path)


@dataclass
class UploadChunk:
    """Upload chunk metadata."""

    chunk_id: int
    offset_bytes: int
    size_bytes: int
    checksum: str  # SHA256
    uploaded_at: Optional[datetime] = None


@dataclass
class UploadSession:
    """Upload session tracking."""

    session_id: str
    filename: str
    total_size_bytes: int
    chunk_size_bytes: int
    total_chunks: int
    uploaded_chunks: list[int] = field(default_factory=list)
    status: UploadStatus = UploadStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None

    @property
    def progress_pct(self) -> float:
        """Calculate upload progress percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (len(self.uploaded_chunks) / self.total_chunks) * 100.0

    @property
    def is_complete(self) -> bool:
        """Check if upload is complete."""
        return len(self.uploaded_chunks) == self.total_chunks

    @property
    def next_chunk_id(self) -> int:
        """Get the next chunk ID to upload."""
        for i in range(self.total_chunks):
            if i not in self.uploaded_chunks:
                return i
        return self.total_chunks


@dataclass
class ValidationResult:
    """Video validation result."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: Optional[VideoMetadata] = None


@dataclass
class ProcessingJob:
    """Processing job metadata."""

    job_id: str
    episode_id: str
    video_path: Path
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_pct: float = 0.0
    current_stage: Optional[str] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if isinstance(self.video_path, str):
            self.video_path = Path(self.video_path)


@dataclass
class CastImage:
    """External cast reference image."""

    person_name: str
    image_path: Path
    width: int
    height: int
    format: ImageFormat
    uploaded_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if isinstance(self.image_path, str):
            self.image_path = Path(self.image_path)
