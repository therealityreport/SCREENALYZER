"""
I/O utilities for video and image validation.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from screentime.types import (
    CastImage,
    ImageFormat,
    ValidationResult,
    VideoCodec,
    VideoMetadata,
)

# Validation constants
MAX_VIDEO_DURATION_SEC = 90 * 60  # 90 minutes
MAX_VIDEO_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
MIN_CAST_IMAGE_SIZE_PX = 200
SUPPORTED_VIDEO_CODECS = {VideoCodec.H264, VideoCodec.H265}
SUPPORTED_IMAGE_FORMATS = {ImageFormat.JPEG, ImageFormat.PNG}


def validate_video(video_path: Path, timeout_sec: float = 5.0) -> ValidationResult:
    """
    Validate video file meets requirements.

    Args:
        video_path: Path to video file
        timeout_sec: Maximum time for validation

    Returns:
        ValidationResult with validation status and metadata
    """
    start_time = time.time()
    errors = []
    warnings = []
    metadata = None

    try:
        # Check file exists
        if not video_path.exists():
            errors.append(f"File not found: {video_path}")
            return ValidationResult(is_valid=False, errors=errors)

        # Check file size
        file_size = video_path.stat().st_size
        if file_size > MAX_VIDEO_SIZE_BYTES:
            errors.append(
                f"File size {file_size / (1024**3):.2f} GB exceeds maximum "
                f"{MAX_VIDEO_SIZE_BYTES / (1024**3):.0f} GB"
            )

        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            errors.append("Cannot open video file - may be corrupted or unsupported format")
            return ValidationResult(is_valid=False, errors=errors)

        try:
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

            # Calculate duration
            duration_sec = frame_count / fps if fps > 0 else 0

            # Check timeout
            if time.time() - start_time > timeout_sec:
                warnings.append(f"Validation timeout ({timeout_sec}s) - using partial metadata")

            # Validate duration
            if duration_sec > MAX_VIDEO_DURATION_SEC:
                errors.append(
                    f"Duration {duration_sec / 60:.1f} min exceeds maximum "
                    f"{MAX_VIDEO_DURATION_SEC / 60:.0f} min"
                )

            # Detect codec
            codec = _fourcc_to_codec(fourcc)
            if codec is None:
                errors.append(f"Unsupported codec: {_fourcc_to_string(fourcc)}")
            elif codec not in SUPPORTED_VIDEO_CODECS:
                errors.append(
                    f"Codec {codec.value} not in supported list: {SUPPORTED_VIDEO_CODECS}"
                )

            # Validate resolution
            if width < 640 or height < 480:
                warnings.append(f"Low resolution {width}x{height} may affect detection quality")

            # Create metadata
            metadata = VideoMetadata(
                path=video_path,
                codec=codec or VideoCodec.H264,  # fallback
                duration_sec=duration_sec,
                fps=fps,
                width=width,
                height=height,
                file_size_bytes=file_size,
            )

        finally:
            cap.release()

    except Exception as e:
        errors.append(f"Validation error: {str(e)}")

    is_valid = len(errors) == 0
    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, metadata=metadata)


def validate_cast_image(
    image_path: Path, person_name: Optional[str] = None
) -> tuple[bool, list[str], Optional[CastImage]]:
    """
    Validate cast reference image.

    Args:
        image_path: Path to image file
        person_name: Optional person name for the image

    Returns:
        Tuple of (is_valid, errors, CastImage metadata)
    """
    errors = []
    cast_image = None

    try:
        # Check file exists
        if not image_path.exists():
            errors.append(f"File not found: {image_path}")
            return False, errors, None

        # Open with PIL
        img = Image.open(image_path)
        width, height = img.size
        img_format = img.format.lower() if img.format else None

        # Validate format
        if img_format not in ["jpeg", "png"]:
            errors.append(f"Unsupported format: {img_format}. Only JPEG and PNG are supported.")

        # Validate dimensions
        if width < MIN_CAST_IMAGE_SIZE_PX or height < MIN_CAST_IMAGE_SIZE_PX:
            errors.append(
                f"Image size {width}x{height} is too small. "
                f"Minimum {MIN_CAST_IMAGE_SIZE_PX}x{MIN_CAST_IMAGE_SIZE_PX} required."
            )

        # Create metadata if valid
        if not errors:
            cast_image = CastImage(
                person_name=person_name or image_path.stem,
                image_path=image_path,
                width=width,
                height=height,
                format=ImageFormat.JPEG if img_format == "jpeg" else ImageFormat.PNG,
            )

    except Exception as e:
        errors.append(f"Image validation error: {str(e)}")

    is_valid = len(errors) == 0
    return is_valid, errors, cast_image


def compute_file_checksum(file_path: Path, chunk_size: int = 8192) -> str:
    """
    Compute SHA256 checksum of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of SHA256 checksum
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            sha256.update(chunk)
    return sha256.hexdigest()


def _fourcc_to_codec(fourcc: int) -> Optional[VideoCodec]:
    """Convert OpenCV FourCC to VideoCodec enum."""
    fourcc_str = _fourcc_to_string(fourcc).lower()

    # H.264 variants
    if fourcc_str in ["avc1", "h264", "x264"]:
        return VideoCodec.H264

    # H.265/HEVC variants
    if fourcc_str in ["hvc1", "hevc", "h265", "x265"]:
        return VideoCodec.H265

    # AV1
    if fourcc_str in ["av01", "av1"]:
        return VideoCodec.AV1

    return None


def _fourcc_to_string(fourcc: int) -> str:
    """Convert OpenCV FourCC integer to string."""
    return "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
