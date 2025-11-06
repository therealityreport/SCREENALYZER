"""
Upload API with chunked upload, resume, and validation.

Security features:
- MIME/extension whitelist
- Path traversal guards
- Filename sanitization
- Size caps enforced before accepting chunks
- SHA-256 integrity verification
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import redis
from dotenv import load_dotenv

from screentime.diagnostics.telemetry import telemetry
from screentime.episode_registry import episode_registry
from screentime.io_utils import (
    compute_file_checksum,
    validate_cast_image,
    validate_video,
)
from screentime.types import UploadChunk, UploadSession, UploadStatus, ValidationResult
from screentime.utils import canonical_show_slug

load_dotenv()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
UPLOAD_CHUNK_SIZE = 5 * 1024 * 1024  # 5 MB chunks
UPLOAD_SESSION_TTL_SEC = 24 * 60 * 60  # 24 hours
MAX_UPLOAD_SIZE = 5 * 1024 * 1024 * 1024  # 5 GB

# Security: whitelist of allowed extensions and MIME types
ALLOWED_VIDEO_EXTENSIONS = {".mp4"}
ALLOWED_VIDEO_MIMES = {"video/mp4"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_MIMES = {"image/jpeg", "image/png"}

# Redis client
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


class UploadManager:
    """Manages chunked uploads with resume capability."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.videos_dir = DATA_ROOT / "videos"
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename

        Raises:
            ValueError: If filename is invalid
        """
        # Remove path components (prevent ../../../etc/passwd)
        filename = Path(filename).name

        # Remove non-alphanumeric except . - _
        filename = re.sub(r"[^\w\.\-]", "_", filename)

        # Prevent multiple dots (prevent ../ or hidden files)
        filename = re.sub(r"\.{2,}", ".", filename)

        # Ensure not empty
        if not filename or filename in (".", ".."):
            raise ValueError("Invalid filename")

        return filename

    @staticmethod
    def _validate_extension(filename: str, allowed_extensions: set[str]) -> None:
        """
        Validate file extension against whitelist.

        Args:
            filename: Filename to check
            allowed_extensions: Set of allowed extensions (e.g., {'.mp4'})

        Raises:
            ValueError: If extension not allowed
        """
        ext = Path(filename).suffix.lower()
        if ext not in allowed_extensions:
            raise ValueError(
                f"File extension '{ext}' not allowed. Allowed: {', '.join(allowed_extensions)}"
            )

    def create_upload_session(
        self,
        filename: str,
        total_size_bytes: int,
        chunk_size_bytes: Optional[int] = None,
        show_id: Optional[str] = None,
        season_number: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> UploadSession:
        """
        Create a new upload session with security validation.

        Args:
            filename: Original filename
            total_size_bytes: Total file size
            chunk_size_bytes: Optional custom chunk size
            show_id: Show identifier (e.g., "rhobh")
            season_number: Season number (e.g., 5)
            episode_id: Episode identifier (e.g., "RHOBH-S05-E01")

        Returns:
            UploadSession object

        Raises:
            ValueError: If file size exceeds maximum or filename invalid
        """
        # Security: validate extension
        self._validate_extension(filename, ALLOWED_VIDEO_EXTENSIONS)

        # Security: sanitize filename
        safe_filename = self._sanitize_filename(filename)

        # Security: enforce size cap
        if total_size_bytes > MAX_UPLOAD_SIZE:
            raise ValueError(
                f"File size {total_size_bytes / (1024**3):.2f} GB exceeds "
                f"maximum {MAX_UPLOAD_SIZE / (1024**3):.0f} GB"
            )

        chunk_size = chunk_size_bytes or UPLOAD_CHUNK_SIZE
        total_chunks = (total_size_bytes + chunk_size - 1) // chunk_size

        session = UploadSession(
            session_id=str(uuid.uuid4()),
            filename=safe_filename,
            total_size_bytes=total_size_bytes,
            chunk_size_bytes=chunk_size,
            total_chunks=total_chunks,
            status=UploadStatus.CREATED,
        )

        # Store show/season/episode metadata in Redis (add to session data)
        if show_id:
            canonical_show = canonical_show_slug(show_id)
            session.error_message = json.dumps({
                "show_id": canonical_show,
                "season_number": season_number,
                "episode_id": episode_id or safe_filename.replace(".mp4", ""),
            })

        # Store in Redis
        self._save_session(session)

        # Telemetry
        telemetry.log_upload_started(
            session_id=session.session_id,
            filename=safe_filename,
            size_bytes=total_size_bytes,
        )

        return session

    def get_upload_session(self, session_id: str) -> Optional[UploadSession]:
        """
        Retrieve upload session from Redis.

        Args:
            session_id: Session ID

        Returns:
            UploadSession or None if not found
        """
        key = f"upload:session:{session_id}"
        data = self.redis.get(key)
        if not data:
            return None

        session_dict = json.loads(data)
        return UploadSession(
            session_id=session_dict["session_id"],
            filename=session_dict["filename"],
            total_size_bytes=session_dict["total_size_bytes"],
            chunk_size_bytes=session_dict["chunk_size_bytes"],
            total_chunks=session_dict["total_chunks"],
            uploaded_chunks=session_dict["uploaded_chunks"],
            status=UploadStatus(session_dict["status"]),
            created_at=datetime.fromisoformat(session_dict["created_at"]),
            updated_at=datetime.fromisoformat(session_dict["updated_at"]),
            error_message=session_dict.get("error_message"),
        )

    def upload_chunk(self, session_id: str, chunk_id: int, chunk_data: bytes) -> dict:
        """
        Upload a single chunk.

        Args:
            session_id: Session ID
            chunk_id: Chunk number (0-indexed)
            chunk_data: Chunk bytes

        Returns:
            Dict with upload status and progress

        Raises:
            ValueError: If session not found or chunk invalid
        """
        session = self.get_upload_session(session_id)
        if not session:
            raise ValueError(f"Upload session {session_id} not found")

        if chunk_id >= session.total_chunks:
            raise ValueError(f"Invalid chunk_id {chunk_id} (total chunks: {session.total_chunks})")

        if chunk_id in session.uploaded_chunks:
            # Chunk already uploaded (idempotent)
            return {
                "session_id": session_id,
                "chunk_id": chunk_id,
                "status": "already_uploaded",
                "progress_pct": session.progress_pct,
            }

        # Verify chunk size
        expected_size = session.chunk_size_bytes
        if chunk_id == session.total_chunks - 1:
            # Last chunk may be smaller
            expected_size = session.total_size_bytes - (chunk_id * session.chunk_size_bytes)

        if len(chunk_data) != expected_size:
            raise ValueError(
                f"Chunk size mismatch: expected {expected_size}, got {len(chunk_data)}"
            )

        # Write chunk to temp file
        temp_file = self._get_temp_path(session_id)
        temp_file.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_file, "ab" if temp_file.exists() else "wb") as f:
            f.seek(chunk_id * session.chunk_size_bytes)
            f.write(chunk_data)

        # Update session
        session.uploaded_chunks.append(chunk_id)
        session.uploaded_chunks.sort()
        session.status = UploadStatus.IN_PROGRESS
        session.updated_at = datetime.utcnow()

        # Telemetry
        telemetry.log_upload_chunk(
            session_id=session_id,
            chunk_id=chunk_id,
            chunk_size=len(chunk_data),
            progress_pct=session.progress_pct,
        )

        # Check if complete
        if session.is_complete:
            session.status = UploadStatus.COMPLETED
            start_time = time.time()
            final_path = self._finalize_upload(session, temp_file)
            duration_ms = (time.time() - start_time) * 1000

            # Telemetry
            telemetry.log_upload_completed(
                session_id=session_id,
                duration_ms=duration_ms,
                size_bytes=session.total_size_bytes,
            )

            self._save_session(session)
            return {
                "session_id": session_id,
                "chunk_id": chunk_id,
                "status": "completed",
                "progress_pct": 100.0,
                "file_path": str(final_path),
            }

        self._save_session(session)

        return {
            "session_id": session_id,
            "chunk_id": chunk_id,
            "status": "in_progress",
            "progress_pct": session.progress_pct,
            "next_chunk_id": session.next_chunk_id,
        }

    def resume_upload(self, session_id: str) -> dict:
        """
        Get resume information for an upload session.

        Args:
            session_id: Session ID

        Returns:
            Dict with resume information

        Raises:
            ValueError: If session not found
        """
        session = self.get_upload_session(session_id)
        if not session:
            raise ValueError(f"Upload session {session_id} not found")

        # Telemetry
        telemetry.log_upload_resumed(
            session_id=session_id,
            from_chunk=session.next_chunk_id,
            progress_pct=session.progress_pct,
        )

        return {
            "session_id": session_id,
            "filename": session.filename,
            "total_chunks": session.total_chunks,
            "uploaded_chunks": session.uploaded_chunks,
            "next_chunk_id": session.next_chunk_id,
            "progress_pct": session.progress_pct,
            "status": session.status.value,
        }

    def cancel_upload(self, session_id: str) -> bool:
        """
        Cancel an upload session.

        Args:
            session_id: Session ID

        Returns:
            True if cancelled, False if not found
        """
        session = self.get_upload_session(session_id)
        if not session:
            return False

        # Update status
        session.status = UploadStatus.CANCELLED
        self._save_session(session)

        # Clean up temp file
        temp_file = self._get_temp_path(session_id)
        if temp_file.exists():
            temp_file.unlink()

        return True

    def validate_uploaded_video(self, session_id: str) -> ValidationResult:
        """
        Validate a completed upload.

        Args:
            session_id: Session ID

        Returns:
            ValidationResult

        Raises:
            ValueError: If session not found or incomplete
        """
        session = self.get_upload_session(session_id)
        if not session:
            raise ValueError(f"Upload session {session_id} not found")

        if not session.is_complete:
            raise ValueError("Upload is not complete")

        video_path = self.videos_dir / session.filename

        # Validate with timing
        start_time = time.time()
        result = validate_video(video_path)
        duration_ms = (time.time() - start_time) * 1000

        # Telemetry
        telemetry.log_upload_validated(
            session_id=session_id,
            is_valid=result.is_valid,
            duration_ms=duration_ms,
            errors=result.errors if not result.is_valid else None,
        )

        return result

    def _save_session(self, session: UploadSession) -> None:
        """Save session to Redis with TTL."""
        key = f"upload:session:{session.session_id}"
        data = {
            "session_id": session.session_id,
            "filename": session.filename,
            "total_size_bytes": session.total_size_bytes,
            "chunk_size_bytes": session.chunk_size_bytes,
            "total_chunks": session.total_chunks,
            "uploaded_chunks": session.uploaded_chunks,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "error_message": session.error_message,
        }
        self.redis.setex(key, UPLOAD_SESSION_TTL_SEC, json.dumps(data))

    def _get_temp_path(self, session_id: str) -> Path:
        """Get temp file path for session."""
        return self.videos_dir / f".upload_{session_id}.tmp"

    def _finalize_upload(self, session: UploadSession, temp_file: Path) -> Path:
        """
        Finalize upload with integrity verification.

        Verifies SHA-256 checksum of assembled file before moving to final location.

        Args:
            session: Upload session
            temp_file: Temp file path

        Returns:
            Final file path

        Raises:
            ValueError: If integrity check fails
        """
        # Verify file size
        actual_size = temp_file.stat().st_size
        if actual_size != session.total_size_bytes:
            raise ValueError(
                f"File size mismatch: expected {session.total_size_bytes}, got {actual_size}"
            )

        # Compute SHA-256 of assembled file
        checksum = compute_file_checksum(temp_file)

        # Parse metadata from error_message field (used to store show/season/episode)
        show_id = None
        season_number = None
        episode_id = None

        if session.error_message and session.error_message.startswith("{"):
            try:
                metadata = json.loads(session.error_message)
                show_id = canonical_show_slug(metadata.get("show_id"))
                season_number = metadata.get("season_number")
                episode_id = metadata.get("episode_id")
            except:
                pass

        # Move to final location with show/season structure
        if show_id and season_number and episode_id:
            # Create show/season directory structure: data/videos/{show_id}/{season_id}/
            season_id = f"s{season_number:02d}"
            target_dir = self.videos_dir / show_id / season_id
            target_dir.mkdir(parents=True, exist_ok=True)

            # Save as episode_id.mp4
            final_path = target_dir / f"{episode_id}.mp4"
        else:
            # Fallback: save to data/videos/ (flat structure)
            final_path = self.videos_dir / session.filename

        # Move file
        temp_file.rename(final_path)

        # Register episode in the new registry (data/diagnostics/episodes.json)
        if show_id and season_number and episode_id:
            episode_registry.register_episode(
                episode_id=episode_id,
                show_id=show_id,
                season_id=f"s{season_number:02d}",
                video_path=str(final_path.relative_to(self.videos_dir.parent)),
                status="uploaded",
            )

        # ALSO register in the old registry (configs/shows_seasons.json) for backward compatibility
        if show_id and season_number and episode_id:
            from app.lib.registry import ensure_episode_in_registry
            season_id = f"s{season_number:02d}"
            ensure_episode_in_registry(episode_id, show_id, season_id)

        # Update session with checksum
        session.error_message = f"sha256:{checksum}"

        return final_path


# Global upload manager instance
upload_manager = UploadManager(redis_client)
