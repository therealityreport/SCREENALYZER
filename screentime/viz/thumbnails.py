"""
Thumbnail extraction and caching for face clusters.

Generates face thumbnails from video frames for UI display.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from screentime.viz.frame_index import FrameIndex, load_frames_index

logger = logging.getLogger(__name__)


def _crop_to_ratio(img: Image.Image, ratio: float = 4/5) -> Image.Image:
    """
    Crop PIL Image to target aspect ratio (width/height).

    Args:
        img: PIL Image to crop
        ratio: Target width/height ratio (default 4/5 for 4:5 aspect)

    Returns:
        Cropped PIL Image with exact aspect ratio
    """
    w, h = img.size
    current_ratio = w / float(h)

    # Already correct ratio
    if abs(current_ratio - ratio) < 1e-3:
        return img

    if current_ratio > ratio:
        # Too wide → crop left/right
        new_w = int(h * ratio)
        x_offset = (w - new_w) // 2
        return img.crop((x_offset, 0, x_offset + new_w, h))
    else:
        # Too tall → crop top/bottom
        new_h = int(w / ratio)
        y_offset = (h - new_h) // 2
        return img.crop((0, y_offset, w, y_offset + new_h))


class ThumbnailGenerator:
    """Generate and cache face thumbnails."""

    def __init__(
        self,
        cache_dir: Path,
        thumbnail_size: tuple[int, int] = (160, 200),
        cache_enabled: bool = True,
        data_root: Optional[Path] = None,
    ):
        """
        Initialize thumbnail generator.

        Args:
            cache_dir: Directory to cache thumbnails
            thumbnail_size: Target thumbnail size (width, height) - default 160×200 for 4:5
            cache_enabled: Whether to use cache
        """
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir
        self.thumbnail_size = thumbnail_size
        self.cache_enabled = cache_enabled
        if data_root is None:
            try:
                self.data_root = cache_dir.parents[1]
            except IndexError:
                self.data_root = Path("data")
        else:
            self.data_root = Path(data_root)

        if cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def generate_thumbnail(
        self,
        video_path: Path,
        frame_id: int,
        bbox: list[int],
        episode_id: str,
        cluster_id: int,
        *,
        frame_index: FrameIndex | None = None,
    ) -> Optional[Path]:
        """
        Generate thumbnail for a face detection.

        Args:
            video_path: Path to video file
            frame_id: Frame number
            bbox: Bounding box [x1, y1, x2, y2]
            episode_id: Episode identifier
            cluster_id: Cluster ID

        Returns:
            Path to thumbnail image, or None if failed
        """
        # Generate cache key
        cache_key = self._get_cache_key(episode_id, cluster_id, frame_id, bbox)
        cache_path = self.cache_dir / f"{cache_key}.jpg"

        # Check cache
        if self.cache_enabled and cache_path.exists():
            return cache_path

        frame_img = None
        source_index: Optional[FrameIndex] = frame_index or load_frames_index(episode_id, self.data_root)
        if source_index:
            frame_path = source_index.resolve_path(frame_id)
            if frame_path:
                frame_img = cv2.imread(str(frame_path))
                if frame_img is None:
                    logger.warning("Failed to read frame asset %s; falling back to video", frame_path)

        if frame_img is None:
            cap = cv2.VideoCapture(str(video_path))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame_img = cap.read()
            cap.release()

            if not ret or frame_img is None:
                logger.warning(f"Failed to extract frame {frame_id} from {video_path}")
                return None

        # Crop face with padding
        x1, y1, x2, y2 = bbox
        padding = 0.2  # 20% padding
        width = x2 - x1
        height = y2 - y1
        pad_w = int(width * padding)
        pad_h = int(height * padding)

        # Apply padding with bounds check
        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(frame_img.shape[1], x2 + pad_w)
        y2_pad = min(frame_img.shape[0], y2 + pad_h)

        face_crop = frame_img[y1_pad:y2_pad, x1_pad:x2_pad]

        if face_crop.size == 0:
            logger.warning(f"Empty crop for frame {frame_id}, bbox {bbox}")
            return None

        # Convert to PIL and ensure 4:5 aspect ratio
        face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_crop_rgb)

        # Crop to 4:5 ratio first
        face_pil = _crop_to_ratio(face_pil, ratio=4/5)

        # Resize to exact target size (160×200)
        face_pil = face_pil.resize(self.thumbnail_size, Image.Resampling.LANCZOS)

        # Save to cache
        if self.cache_enabled:
            face_pil.save(cache_path, "JPEG", quality=85)

        return cache_path

    def generate_frame_thumbnail(
        self,
        video_path: Path,
        frame_id: int,
        bbox: list[int],
        episode_id: str,
        track_id: int,
        *,
        frame_index: FrameIndex | None = None,
    ) -> Optional[Path]:
        """
        Generate thumbnail for a single frame/track.

        Args:
            video_path: Path to video file
            frame_id: Frame number
            bbox: Bounding box [x1, y1, x2, y2]
            episode_id: Episode identifier
            track_id: Track ID (used for cache key)

        Returns:
            Path to thumbnail image, or None if failed
        """
        return self.generate_thumbnail(
            video_path,
            frame_id,
            bbox,
            episode_id,
            track_id,
            frame_index=frame_index,
        )

    def generate_cluster_thumbnail(
        self,
        video_path: Path,
        cluster: dict,
        tracks_data: dict,
        episode_id: str,
        max_samples: int = 5,
    ) -> list[Path]:
        """
        Generate thumbnails for a cluster (multiple samples).

        Args:
            video_path: Path to video file
            cluster: Cluster dict with track_ids
            tracks_data: Tracks data from load_tracks()
            episode_id: Episode identifier
            max_samples: Maximum number of thumbnails to generate

        Returns:
            List of thumbnail paths
        """
        cluster_id = cluster["cluster_id"]
        track_ids = cluster["track_ids"][:max_samples]

        thumbnails = []
        index = load_frames_index(episode_id, self.data_root)

        for track_id in track_ids:
            # Find track
            track = next(
                (t for t in tracks_data.get("tracks", []) if t["track_id"] == track_id), None
            )
            if not track:
                continue

            # Get first frame reference
            frame_refs = track.get("frame_refs", [])
            if not frame_refs:
                continue

            ref = frame_refs[0]
            frame_id = ref["frame_id"]
            bbox = ref["bbox"]

            # Generate thumbnail
            thumb_path = self.generate_thumbnail(
                video_path,
                frame_id,
                bbox,
                episode_id,
                cluster_id,
                frame_index=index,
            )
            if thumb_path:
                thumbnails.append(thumb_path)

            if len(thumbnails) >= max_samples:
                break

        return thumbnails

    def _get_cache_key(
        self, episode_id: str, cluster_id: int, frame_id: int, bbox: list[int]
    ) -> str:
        """Generate cache key for thumbnail."""
        key_str = f"{episode_id}_{cluster_id}_{frame_id}_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def clear_cache(self, episode_id: Optional[str] = None) -> int:
        """
        Clear thumbnail cache.

        Args:
            episode_id: If provided, only clear thumbnails for this episode

        Returns:
            Number of files deleted
        """
        if not self.cache_enabled or not self.cache_dir.exists():
            return 0

        deleted = 0
        for thumb_file in self.cache_dir.glob("*.jpg"):
            if episode_id is None or episode_id in thumb_file.stem:
                thumb_file.unlink()
                deleted += 1

        return deleted
