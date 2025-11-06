"""
Auto-extract frames after upload validation.

Part of Phase 3 P1 automation: eliminate manual "Prepare Tracks & Stills" step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from api.jobs import job_manager
from jobs.tasks.harvest import harvest_task

logger = logging.getLogger(__name__)


def trigger_auto_extraction(
    episode_id: str,
    video_path: str | Path,
    data_root: str | Path = Path("data"),
) -> dict:
    """
    Automatically trigger frame extraction after upload validation.

    This is called immediately after a video is uploaded and validated,
    eliminating the need for the manual "Prepare Tracks & Stills" button.

    Workflow:
    1. Ensure episode registry exists
    2. Set validated=true in registry
    3. Run harvest_task to extract frames
    4. Set extracted_frames=true in registry
    5. Return result

    Args:
        episode_id: Episode ID (e.g., 'RHOBH_S05_E03_11062025')
        video_path: Path to uploaded video file
        data_root: Data root directory

    Returns:
        Dict with extraction results and episode_key

    Raises:
        ValueError: If video file not found or extraction fails
    """
    logger.info(f"[auto_extract] Starting auto-extraction for {episode_id}")

    video_path = Path(video_path)
    data_root = Path(data_root)

    # Verify video exists
    if not video_path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    # Ensure episode registry exists and get canonical key
    episode_key = job_manager.ensure_episode_registry(
        episode_id=episode_id,
        video_path=str(video_path),
    )
    logger.info(f"[auto_extract] Episode registry ensured: {episode_key}")

    # Mark as validated in registry
    job_manager.update_registry_state(episode_key, "validated", True)
    logger.info(f"[auto_extract] Set validated=true for {episode_key}")

    # Run harvest to extract frames
    logger.info(f"[auto_extract] Starting frame extraction...")

    try:
        # Use job_id="manual" to skip auto-enqueue of detect_embed
        # Auto-extraction should ONLY extract frames, not trigger detection
        harvest_result = harvest_task(
            job_id="manual",  # Prevents harvest from auto-enqueueing detect_embed
            episode_id=episode_id,
            video_path=str(video_path),
            resume_from=None,
        )

        logger.info(f"[auto_extract] Frame extraction complete")

        # Mark as extracted_frames in registry
        job_manager.update_registry_state(episode_key, "extracted_frames", True)
        logger.info(f"[auto_extract] Set extracted_frames=true for {episode_key}")

        return {
            "success": True,
            "episode_id": episode_id,
            "episode_key": episode_key,
            "harvest_result": harvest_result,
        }

    except Exception as e:
        logger.error(f"[auto_extract] Frame extraction failed: {e}")

        # Keep validated=true even if extraction fails
        # User can retry later from Workspace

        return {
            "success": False,
            "episode_id": episode_id,
            "episode_key": episode_key,
            "error": str(e),
        }


def check_extraction_status(episode_id: str) -> dict:
    """
    Check the extraction status for an episode.

    Args:
        episode_id: Episode ID

    Returns:
        Dict with status information
    """
    episode_key = job_manager.normalize_episode_key(episode_id)
    registry = job_manager.load_episode_registry(episode_key)

    if not registry:
        return {
            "exists": False,
            "validated": False,
            "extracted_frames": False,
        }

    states = registry.get("states", {})

    return {
        "exists": True,
        "episode_key": episode_key,
        "validated": states.get("validated", False),
        "extracted_frames": states.get("extracted_frames", False),
        "detected": states.get("detected", False),
        "tracked": states.get("tracked", False),
    }
