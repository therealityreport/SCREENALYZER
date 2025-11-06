"""
Multi-format image ingestion with normalization and deduplication.

Supports: jpg, jpeg, png, avif, webp, heic
Normalizes to: PNG (RGB, 8-bit, EXIF-stripped, orientation-corrected)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
from PIL import Image

# Conditional import for HEIC support
try:
    import pillow_heif
    HEIC_SUPPORT = True
except ImportError:
    HEIC_SUPPORT = False

# Conditional import for AVIF support
try:
    import pillow_avif
    AVIF_SUPPORT = True
except ImportError:
    AVIF_SUPPORT = False

SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.avif', '.webp', '.heic'}


@dataclass
class ImageMetadata:
    """Metadata for normalized image."""
    original_path: str
    original_format: str
    original_size: Tuple[int, int]  # (width, height)
    original_filesize_bytes: int
    normalized_path: str
    normalized_format: str = "PNG"
    normalized_size: Tuple[int, int] = None
    normalized_filesize_bytes: int = 0
    normalized_hash: str = ""  # SHA-256 of normalized bytes
    has_exif: bool = False
    orientation_corrected: bool = False
    processed_at: str = ""


class ImageNormalizer:
    """
    Normalize multi-format images to analysis-ready format.

    - Accepts: jpg/jpeg/png/avif/webp/heic
    - Outputs: PNG (RGB, 8-bit, EXIF-stripped, orientation-corrected)
    - Tracks: original format, size, hash for deduplication
    """

    def __init__(self):
        """Initialize image normalizer."""
        if HEIC_SUPPORT:
            # Register HEIF opener
            pillow_heif.register_heif_opener()

    @staticmethod
    def is_supported_format(file_path: Path) -> bool:
        """Check if file format is supported."""
        return file_path.suffix.lower() in SUPPORTED_FORMATS

    @staticmethod
    def compute_image_hash(image_bytes: bytes) -> str:
        """Compute SHA-256 hash of image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    def normalize_image(
        self,
        input_path: Path,
        output_path: Path,
        save_format: str = "png"
    ) -> ImageMetadata:
        """
        Normalize image to analysis-ready format.

        Args:
            input_path: Original image path
            output_path: Where to save normalized image
            save_format: "png" or "webp" (lossless)

        Returns:
            ImageMetadata with normalization details

        Raises:
            ValueError: If format unsupported or image cannot be processed
            FileNotFoundError: If input_path doesn't exist
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Image not found: {input_path}")

        if not self.is_supported_format(input_path):
            raise ValueError(
                f"Unsupported format: {input_path.suffix}. "
                f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )

        # Check HEIC support
        if input_path.suffix.lower() == '.heic' and not HEIC_SUPPORT:
            raise ValueError(
                "HEIC format requires pillow-heif package. "
                "Install with: pip install pillow-heif"
            )

        # Check AVIF support
        if input_path.suffix.lower() == '.avif' and not AVIF_SUPPORT:
            raise ValueError(
                "AVIF format requires pillow-avif-plugin package. "
                "Install with: pip install pillow-avif-plugin"
            )

        # Open image
        img = Image.open(input_path)

        # Capture original metadata
        original_format = img.format or input_path.suffix.lstrip('.').upper()
        original_size = img.size
        original_filesize = input_path.stat().st_size

        # Check for EXIF data
        has_exif = hasattr(img, '_getexif') and img._getexif() is not None
        orientation_corrected = False

        # Apply EXIF orientation (transpose to correct rotation)
        if has_exif:
            try:
                from PIL import ImageOps
                img_corrected = ImageOps.exif_transpose(img)
                if img_corrected is not img:
                    img = img_corrected
                    orientation_corrected = True
            except Exception:
                # If orientation correction fails, continue with original
                pass

        # Convert to RGB (strip alpha, handle CMYK, grayscale, etc.)
        if img.mode != 'RGB':
            # Handle transparency by compositing on white background
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            else:
                img = img.convert('RGB')

        # Ensure 8-bit depth (RGB mode is always 8-bit in PIL)
        # PIL RGB mode is already 8-bit per channel

        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save normalized version
        save_format_lower = save_format.lower()
        if save_format_lower == "png":
            # Save as PNG with no compression for reproducibility
            img.save(output_path, format='PNG', optimize=False, compress_level=0)
        elif save_format_lower == "webp":
            # Save as lossless WebP
            img.save(output_path, format='WEBP', lossless=True, quality=100)
        else:
            raise ValueError(f"Unsupported save format: {save_format}. Use 'png' or 'webp'.")

        # Compute hash of normalized image bytes
        with open(output_path, 'rb') as f:
            normalized_bytes = f.read()
            normalized_hash = self.compute_image_hash(normalized_bytes)

        normalized_filesize = output_path.stat().st_size

        # Build metadata
        metadata = ImageMetadata(
            original_path=str(input_path.resolve()),
            original_format=original_format,
            original_size=original_size,
            original_filesize_bytes=original_filesize,
            normalized_path=str(output_path.resolve()),
            normalized_format=save_format.upper(),
            normalized_size=img.size,
            normalized_filesize_bytes=normalized_filesize,
            normalized_hash=normalized_hash,
            has_exif=has_exif,
            orientation_corrected=orientation_corrected,
            processed_at=datetime.utcnow().isoformat() + 'Z'
        )

        return metadata

    def save_metadata(self, metadata: ImageMetadata, metadata_path: Path) -> None:
        """
        Save image metadata to JSON sidecar.

        Args:
            metadata: ImageMetadata object
            metadata_path: Where to save metadata JSON
        """
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)


class ImageDeduplicator:
    """
    Deduplicate normalized images by hash and optional cosine similarity.
    """

    def __init__(self, cosine_threshold: Optional[float] = None):
        """
        Initialize deduplicator.

        Args:
            cosine_threshold: Optional cosine similarity threshold (0-1).
                            If provided, also check visual similarity.
        """
        self.cosine_threshold = cosine_threshold
        self.seen_hashes = set()
        self.seen_embeddings = []  # For cosine dedup

    def is_duplicate(
        self,
        image_hash: str,
        image_embedding: Optional[np.ndarray] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if image is a duplicate.

        Args:
            image_hash: SHA-256 hash of normalized image
            image_embedding: Optional embedding vector for cosine dedup

        Returns:
            (is_duplicate, reason) where reason is "hash" or "cosine" or None
        """
        # Check hash first (exact duplicate)
        if image_hash in self.seen_hashes:
            return True, "hash"

        # Check cosine similarity if threshold provided
        if self.cosine_threshold is not None and image_embedding is not None:
            for existing_emb in self.seen_embeddings:
                # Compute cosine similarity
                sim = np.dot(image_embedding, existing_emb) / (
                    np.linalg.norm(image_embedding) * np.linalg.norm(existing_emb)
                )
                if sim >= self.cosine_threshold:
                    return True, "cosine"

        return False, None

    def add(self, image_hash: str, image_embedding: Optional[np.ndarray] = None) -> None:
        """
        Add image to seen set.

        Args:
            image_hash: SHA-256 hash of normalized image
            image_embedding: Optional embedding vector for cosine dedup
        """
        self.seen_hashes.add(image_hash)
        if image_embedding is not None and self.cosine_threshold is not None:
            self.seen_embeddings.append(image_embedding)


def normalize_cast_images_batch(
    input_paths: List[Path],
    output_dir: Path,
    deduplicate: bool = True,
    cosine_threshold: Optional[float] = 0.98,
    save_format: str = "png"
) -> Tuple[List[ImageMetadata], List[Tuple[Path, str]]]:
    """
    Normalize a batch of cast images with deduplication.

    Args:
        input_paths: List of input image paths
        output_dir: Directory to save normalized images
        deduplicate: Whether to deduplicate by hash/cosine
        cosine_threshold: Cosine similarity threshold for dedup (None to disable)
        save_format: "png" or "webp"

    Returns:
        (valid_metadata, rejected) where:
            valid_metadata: List of ImageMetadata for accepted images
            rejected: List of (input_path, reason) for rejected images
    """
    normalizer = ImageNormalizer()
    deduplicator = ImageDeduplicator(cosine_threshold=cosine_threshold if deduplicate else None)

    valid_metadata = []
    rejected = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, input_path in enumerate(input_paths):
        try:
            # Generate output filename
            output_filename = f"seed_{idx + 1:03d}.{save_format}"
            output_path = output_dir / output_filename

            # Normalize
            metadata = normalizer.normalize_image(input_path, output_path, save_format)

            # Check for duplicates
            if deduplicate:
                is_dup, dup_reason = deduplicator.is_duplicate(metadata.normalized_hash)
                if is_dup:
                    rejected.append((input_path, f"Duplicate ({dup_reason})"))
                    # Remove the output file
                    output_path.unlink(missing_ok=True)
                    continue

                # Add to deduplicator
                deduplicator.add(metadata.normalized_hash)

            # Save metadata sidecar
            metadata_path = output_dir / f"seed_{idx + 1:03d}_metadata.json"
            normalizer.save_metadata(metadata, metadata_path)

            valid_metadata.append(metadata)

        except Exception as e:
            rejected.append((input_path, str(e)))

    return valid_metadata, rejected
