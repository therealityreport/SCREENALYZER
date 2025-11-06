"""
Scene boundary detection using frame difference analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_scene_boundaries(
    manifest_path: Path,
    video_path: str,
    threshold: float = 30.0,
    min_scene_duration_ms: int = 1000,
) -> list[int]:
    """
    Detect scene boundaries (camera cuts) using frame difference analysis.

    Args:
        manifest_path: Path to manifest.parquet
        video_path: Path to video file
        threshold: Frame difference threshold for detecting cuts (0-100)
        min_scene_duration_ms: Minimum scene duration to avoid false positives

    Returns:
        List of frame indices where scene boundaries occur
    """
    logger.info(f"Detecting scene boundaries with threshold={threshold}")

    # Load manifest
    manifest = pd.read_parquet(manifest_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        boundaries = []
        prev_frame = None
        prev_boundary_ms = -min_scene_duration_ms

        for _, row in manifest.iterrows():
            frame_id = int(row["frame_id"])
            ts_ms = int(row["ts_ms"])

            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if not ret or frame is None:
                continue

            # Convert to grayscale and resize for faster comparison
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (160, 90))  # 16:9 aspect ratio

            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(small, prev_frame)
                mean_diff = diff.mean()

                # Detect cut
                if mean_diff > threshold:
                    # Check minimum scene duration
                    if ts_ms - prev_boundary_ms >= min_scene_duration_ms:
                        boundaries.append(frame_id)
                        prev_boundary_ms = ts_ms
                        logger.debug(
                            f"Scene boundary at frame {frame_id} (ts={ts_ms}ms, diff={mean_diff:.1f})"
                        )

            prev_frame = small

        logger.info(f"Detected {len(boundaries)} scene boundaries")
        return boundaries

    finally:
        cap.release()
