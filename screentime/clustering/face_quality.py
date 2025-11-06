"""
Face Quality Gating - Identity-Agnostic.

Filters samples to faces-only feed for clustering and gallery display.
Prevents non-face frames (glasses, arms, background) from contaminating clusters.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FaceQualityFilter:
    """
    Face quality filter configuration.

    All identities use same criteria - no per-person tuning.
    """
    min_face_conf: float = 0.65      # Minimum detector confidence
    min_face_px: int = 72            # Minimum face size (px)
    max_co_face_iou: float = 0.10    # Max IoU with other faces (reject co-face crops)
    min_sharpness: float = 0.0       # Minimum sharpness score (0 = disabled)
    require_embedding: bool = True   # Must have valid embedding


def compute_bbox_iou(bbox1: list, bbox2: list) -> float:
    """
    Compute IoU between two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value (0.0 to 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0  # No overlap

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Union
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def check_co_face_overlap(
    bbox: list,
    frame_detections: list,
    max_iou: float = 0.10
) -> bool:
    """
    Check if this face crop overlaps with other faces in same frame.

    Args:
        bbox: Target face bbox [x1, y1, x2, y2]
        frame_detections: List of all detections in frame [{bbox, ...}, ...]
        max_iou: Maximum allowed IoU with other faces

    Returns:
        True if co-face overlap detected (should reject), False if clean
    """
    for other_det in frame_detections:
        other_bbox = other_det.get("bbox")
        if other_bbox is None:
            continue

        # Skip if same bbox
        if bbox == other_bbox:
            continue

        iou = compute_bbox_iou(bbox, other_bbox)
        if iou > max_iou:
            return True  # Co-face overlap detected

    return False  # No significant overlap


def compute_face_size(bbox: list) -> int:
    """Compute face size (max of width/height) from bbox."""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    return max(width, height)


def filter_face_samples(
    embeddings_df: pd.DataFrame,
    tracks_data: dict,
    filter_config: FaceQualityFilter,
    detections_per_frame: Optional[dict] = None
) -> pd.DataFrame:
    """
    Filter embeddings to faces-only (remove non-face chips).

    Args:
        embeddings_df: Raw embeddings DataFrame (from embeddings.parquet)
        tracks_data: Loaded tracks.json
        filter_config: Quality filter configuration
        detections_per_frame: Optional dict {frame_id: [detections]} for co-face check

    Returns:
        Filtered DataFrame with faces-only samples
    """
    logger.info(f"Filtering face samples (before: {len(embeddings_df)} samples)")

    # Copy to avoid modifying original
    df = embeddings_df.copy()

    # 1. Require valid embedding
    if filter_config.require_embedding:
        df = df[df['embedding'].notna()]
        logger.info(f"  After embedding filter: {len(df)} samples")

    # 2. Face confidence threshold
    if 'confidence' in df.columns:
        df = df[df['confidence'] >= filter_config.min_face_conf]
        logger.info(f"  After confidence ≥{filter_config.min_face_conf}: {len(df)} samples")

    # 3. Face size threshold
    if 'face_size' in df.columns:
        df = df[df['face_size'] >= filter_config.min_face_px]
        logger.info(f"  After face_size ≥{filter_config.min_face_px}px: {len(df)} samples")
    elif 'bbox' in df.columns:
        # Compute face size from bbox if not pre-computed
        df['face_size'] = df['bbox'].apply(lambda b: compute_face_size(b) if b else 0)
        df = df[df['face_size'] >= filter_config.min_face_px]
        logger.info(f"  After face_size ≥{filter_config.min_face_px}px (computed): {len(df)} samples")

    # 4. Co-face overlap check (if detections provided)
    if detections_per_frame is not None and 'frame_id' in df.columns and 'bbox' in df.columns:
        co_face_mask = []
        for _, row in df.iterrows():
            frame_id = row['frame_id']
            bbox = row['bbox']

            if frame_id not in detections_per_frame:
                co_face_mask.append(False)  # No other detections, assume clean
                continue

            frame_dets = detections_per_frame[frame_id]
            has_co_face = check_co_face_overlap(bbox, frame_dets, filter_config.max_co_face_iou)
            co_face_mask.append(has_co_face)

        df = df[~pd.Series(co_face_mask, index=df.index)]
        logger.info(f"  After co-face filter (IoU ≤{filter_config.max_co_face_iou}): {len(df)} samples")

    # 5. Sharpness filter (if enabled and column exists)
    if filter_config.min_sharpness > 0 and 'sharpness' in df.columns:
        df = df[df['sharpness'] >= filter_config.min_sharpness]
        logger.info(f"  After sharpness ≥{filter_config.min_sharpness}: {len(df)} samples")

    logger.info(f"Face filtering complete: {len(df)} / {len(embeddings_df)} samples retained ({len(df)/len(embeddings_df)*100:.1f}%)")

    return df


def pick_top_k_per_track(
    embeddings_df: pd.DataFrame,
    k: int = 10,
    quality_weights: Optional[dict] = None
) -> pd.DataFrame:
    """
    Pick top-K highest-quality embeddings per track for centroid computation.

    Args:
        embeddings_df: Filtered faces-only DataFrame
        k: Number of top samples to keep per track
        quality_weights: Weights for quality score (e.g., {"confidence": 0.6, "face_size": 0.4})

    Returns:
        DataFrame with top-K samples per track
    """
    if quality_weights is None:
        quality_weights = {
            "confidence": 0.6,
            "face_size": 0.3,
            "sharpness": 0.1
        }

    logger.info(f"Picking top-{k} samples per track")

    # Compute quality score
    quality_scores = []
    for _, row in embeddings_df.iterrows():
        score = 0.0

        # Confidence component (normalize to 0-1)
        if 'confidence' in embeddings_df.columns:
            conf_norm = (row['confidence'] - 0.5) / 0.5  # 0.5-1.0 → 0-1
            score += quality_weights.get("confidence", 0) * max(0, conf_norm)

        # Face size component (normalize to 0-1, assuming 72-300px range)
        if 'face_size' in embeddings_df.columns:
            size_norm = (row['face_size'] - 72) / 228  # 72-300 → 0-1
            score += quality_weights.get("face_size", 0) * np.clip(size_norm, 0, 1)

        # Sharpness component (if available)
        if 'sharpness' in embeddings_df.columns and quality_weights.get("sharpness", 0) > 0:
            score += quality_weights.get("sharpness", 0) * row['sharpness']

        quality_scores.append(score)

    embeddings_df = embeddings_df.copy()
    embeddings_df['quality_score'] = quality_scores

    # Pick top-K per track
    picked_samples = []

    for track_id in embeddings_df['track_id'].unique():
        track_samples = embeddings_df[embeddings_df['track_id'] == track_id]
        track_samples = track_samples.sort_values('quality_score', ascending=False).head(k)
        picked_samples.append(track_samples)

    result_df = pd.concat(picked_samples, ignore_index=True)

    logger.info(f"Picked samples: {len(result_df)} from {len(embeddings_df)} ({len(result_df)/len(embeddings_df)*100:.1f}%)")

    return result_df


def save_picked_samples(episode_id: str, data_root: Path, picked_df: pd.DataFrame):
    """Save picked samples (faces-only, top-K per track) to parquet."""
    output_path = data_root / "harvest" / episode_id / "picked_samples.parquet"
    picked_df.to_parquet(output_path)
    logger.info(f"Saved {len(picked_df)} picked samples to {output_path}")


def load_picked_samples(episode_id: str, data_root: Path) -> Optional[pd.DataFrame]:
    """Load picked samples from parquet if available."""
    picked_path = data_root / "harvest" / episode_id / "picked_samples.parquet"

    if not picked_path.exists():
        return None

    return pd.read_parquet(picked_path)
