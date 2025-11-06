"""
Utility functions for Screenalyzer.
"""

from __future__ import annotations

import re
from pathlib import Path

SHOW_SLUG_ALIASES = {
    "rhobh": "rhobh",
    "real_housewives_of_beverly_hills": "rhobh",
    "realhousewivesofbeverlyhills": "rhobh",
    "real-housewives-of-beverly-hills": "rhobh",
}

SHOW_DISPLAY_NAMES = {
    "rhobh": "Real Housewives of Beverly Hills",
}


def normalize_episode_id(episode_id: str) -> str:
    """
    Normalize episode ID to consistent format.

    Rules:
    - Preserve hyphens
    - Replace spaces and underscores with hyphens
    - Remove special characters except hyphens
    - Convert to uppercase
    - Collapse multiple hyphens

    Args:
        episode_id: Raw episode identifier

    Returns:
        Normalized episode ID

    Examples:
        >>> normalize_episode_id("RHOBH TEST 10 28")
        'RHOBH-TEST-10-28'
        >>> normalize_episode_id("rhobh_test_10_28")
        'RHOBH-TEST-10-28'
        >>> normalize_episode_id("RHOBH--TEST--10--28")
        'RHOBH-TEST-10-28'
    """
    # Convert to uppercase
    normalized = episode_id.upper()

    # Replace spaces and underscores with hyphens
    normalized = normalized.replace(" ", "-").replace("_", "-")

    # Remove special characters except hyphens and alphanumeric
    normalized = re.sub(r"[^A-Z0-9\-]", "", normalized)

    # Collapse multiple hyphens
    normalized = re.sub(r"-+", "-", normalized)

    # Strip leading/trailing hyphens
    normalized = normalized.strip("-")

    return normalized


def validate_episode_id(episode_id: str) -> tuple[bool, list[str]]:
    """
    Validate episode ID format.

    Args:
        episode_id: Episode identifier to validate

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []

    if not episode_id:
        errors.append("Episode ID cannot be empty")
        return False, errors

    if len(episode_id) < 3:
        errors.append("Episode ID must be at least 3 characters")

    if len(episode_id) > 100:
        errors.append("Episode ID must be less than 100 characters")

    # Check for invalid characters
    if not re.match(r"^[A-Z0-9\-]+$", episode_id):
        errors.append("Episode ID must contain only uppercase letters, numbers, and hyphens")

    # Check for consecutive hyphens
    if "--" in episode_id:
        errors.append("Episode ID cannot contain consecutive hyphens")

    # Check for leading/trailing hyphens
    if episode_id.startswith("-") or episode_id.endswith("-"):
        errors.append("Episode ID cannot start or end with a hyphen")

    return len(errors) == 0, errors


def get_video_path(episode_id: str, data_root: Path = Path("data")) -> Path:
    """
    Get video path for episode.

    Args:
        episode_id: Episode identifier
        data_root: Data root directory

    Returns:
        Path to video file
    """
    videos_dir = data_root / "videos"

    # Try common extensions in multiple locations
    for ext in [".mp4", ".MP4"]:
        # 1. Try flat structure: data/videos/{episode_id}.mp4
        video_path = videos_dir / f"{episode_id}{ext}"
        if video_path.exists():
            return video_path

        # 2. Try show/season structure: data/videos/**/{episode_id}.mp4
        # Search recursively for the video file
        for video_file in videos_dir.rglob(f"{episode_id}{ext}"):
            if video_file.is_file():
                return video_file

    # Fallback: return expected path even if doesn't exist
    return videos_dir / f"{episode_id}.mp4"


def canonical_show_slug(value: str | None) -> str:
    """Normalize show identifier to canonical slug."""
    if not value:
        return ""

    normalized = value.strip().lower()
    normalized = normalized.replace(" ", "_").replace("-", "_")
    return SHOW_SLUG_ALIASES.get(normalized, normalized)


def show_display_name(slug: str | None) -> str:
    """Return human-friendly display name for a show slug."""
    canonical = canonical_show_slug(slug or "")
    if not canonical:
        return ""
    return SHOW_DISPLAY_NAMES.get(canonical, canonical.upper())
