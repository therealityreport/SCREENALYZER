"""
SER-FIQ: Stochastic Embedding Robustness Face Image Quality Assessment

Estimates face image quality by measuring embedding robustness under stochastic perturbations.
Higher quality faces produce more stable embeddings across augmentations.

References:
- Terhorst et al., "SER-FIQ: Unsupervised Estimation of Face Image Quality" (2020)
- Quality = 1 / (1 + σ) where σ is embedding std across stochastic passes
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SERFIQScorer:
    """
    SER-FIQ quality scorer using stochastic forward passes.

    Applies lightweight augmentations to input face chips and measures
    embedding variance. More robust (higher quality) faces have lower variance.
    """

    def __init__(
        self,
        num_passes: int = 5,
        noise_std: float = 1.5,
        brightness_range: float = 0.03,
        contrast_range: float = 0.03,
        rotation_deg: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize SER-FIQ scorer.

        Args:
            num_passes: Number of stochastic forward passes (default 5)
            noise_std: Gaussian noise std (0-255 scale, default 3.0)
            brightness_range: Random brightness adjustment range (±fraction)
            contrast_range: Random contrast adjustment range (±fraction)
            rotation_deg: Random rotation range in degrees (±deg)
            seed: Random seed for reproducibility
        """
        self.num_passes = num_passes
        self.noise_std = noise_std
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.rotation_deg = rotation_deg
        self.rng = np.random.RandomState(seed)

    def augment_chip(self, chip: np.ndarray) -> np.ndarray:
        """
        Apply stochastic augmentation to face chip.

        Args:
            chip: Face chip (H x W x 3, BGR, 0-255)

        Returns:
            Augmented chip (same shape/dtype)
        """
        # Convert to float for processing
        aug = chip.astype(np.float32)

        # 1. Gaussian noise
        noise = self.rng.normal(0, self.noise_std, chip.shape)
        aug = aug + noise

        # 2. Brightness adjustment
        brightness_delta = self.rng.uniform(-self.brightness_range, self.brightness_range)
        aug = aug * (1.0 + brightness_delta)

        # 3. Contrast adjustment
        contrast_delta = self.rng.uniform(-self.contrast_range, self.contrast_range)
        mean = np.mean(aug)
        aug = (aug - mean) * (1.0 + contrast_delta) + mean

        # 4. Small rotation
        if self.rotation_deg > 0:
            angle = self.rng.uniform(-self.rotation_deg, self.rotation_deg)
            h, w = chip.shape[:2]
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            aug = cv2.warpAffine(aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # Clip and convert back to uint8
        aug = np.clip(aug, 0, 255).astype(np.uint8)

        return aug

    def score_chip(
        self,
        chip: np.ndarray,
        embedder,
    ) -> tuple[float, dict]:
        """
        Compute SER-FIQ quality score for a face chip.

        Args:
            chip: Face chip (H x W x 3, BGR, 0-255)
            embedder: ArcFaceEmbedder instance

        Returns:
            (quality_score, metadata) where:
                quality_score: 0-1 quality score (higher = better)
                metadata: dict with {mean_embedding, std, num_valid_passes}
        """
        if chip is None or chip.size == 0:
            return 0.0, {"std": float('inf'), "num_valid_passes": 0}

        embeddings = []

        # Run multiple stochastic forward passes
        for i in range(self.num_passes):
            # Augment chip
            aug_chip = self.augment_chip(chip)

            # Get embedding
            try:
                faces = embedder.app.get(aug_chip)

                if faces and len(faces) > 0:
                    emb = faces[0].normed_embedding

                    if emb is not None and not np.isnan(emb).any():
                        embeddings.append(emb)
            except Exception as e:
                logger.debug(f"SER-FIQ pass {i} failed: {e}")
                continue

        # Check if we got enough valid embeddings
        if len(embeddings) < 2:
            # Not enough data - return low quality score
            return 0.0, {"std": float('inf'), "num_valid_passes": len(embeddings)}

        # Compute statistics
        embeddings_arr = np.array(embeddings)  # shape: (num_passes, 512)
        mean_emb = np.mean(embeddings_arr, axis=0)
        std_emb = np.std(embeddings_arr, axis=0)

        # Aggregate std across dimensions (L2 norm of std vector)
        std_scalar = float(np.linalg.norm(std_emb))

        # Convert std to quality score
        # Lower std = more robust = higher quality
        # Formula: Q = 1 / (1 + k*σ)
        # Typical std range: 0.01-0.10 for good faces, >0.15 for bad faces
        # Using k=10 to map: σ=0.01→Q=0.91, σ=0.05→Q=0.67, σ=0.10→Q=0.50
        quality_score = 1.0 / (1.0 + 10.0 * std_scalar)

        metadata = {
            "mean_embedding": mean_emb,
            "std": std_scalar,
            "num_valid_passes": len(embeddings),
        }

        return quality_score, metadata


    def score_with_bbox(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        embedder,
        margin_scale: float = 1.5,
    ) -> tuple[float, dict]:
        """
        Compute SER-FIQ quality score from full frame + bbox.

        This extracts a context crop around the face (with margin) to help
        the detector find the face more reliably during stochastic passes.

        Args:
            frame: Full frame (H x W x 3, BGR, 0-255)
            bbox: Face bbox (x, y, w, h)
            embedder: ArcFaceEmbedder instance
            margin_scale: Expansion factor for context crop (default 1.5)

        Returns:
            (quality_score, metadata) where:
                quality_score: 0-1 quality score (higher = better)
                metadata: dict with {mean_embedding, std, num_valid_passes}
        """
        if frame is None or frame.size == 0:
            return 0.0, {"std": float('inf'), "num_valid_passes": 0}

        # Extract context crop with margin
        x, y, w, h = bbox
        frame_h, frame_w = frame.shape[:2]

        # Expand bbox by margin_scale
        cx = x + w / 2.0
        cy = y + h / 2.0
        size = max(w, h) * margin_scale

        x1 = int(max(0, cx - size / 2))
        y1 = int(max(0, cy - size / 2))
        x2 = int(min(frame_w, cx + size / 2))
        y2 = int(min(frame_h, cy + size / 2))

        if x2 <= x1 or y2 <= y1:
            return 0.0, {"std": float('inf'), "num_valid_passes": 0}

        context_crop = frame[y1:y2, x1:x2]

        if context_crop.size == 0:
            return 0.0, {"std": float('inf'), "num_valid_passes": 0}

        # Use the context crop for SER-FIQ scoring
        return self.score_chip(context_crop, embedder)

    def score_from_embedding_list(
        self,
        embeddings: list[np.ndarray],
    ) -> tuple[float, dict]:
        """
        Compute quality score from pre-computed embeddings (for testing/debugging).

        Args:
            embeddings: List of normalized embeddings from stochastic passes

        Returns:
            (quality_score, metadata)
        """
        if len(embeddings) < 2:
            return 0.0, {"std": float('inf'), "num_valid_passes": len(embeddings)}

        embeddings_arr = np.array(embeddings)
        mean_emb = np.mean(embeddings_arr, axis=0)
        std_emb = np.std(embeddings_arr, axis=0)
        std_scalar = float(np.linalg.norm(std_emb))

        quality_score = 1.0 / (1.0 + 10.0 * std_scalar)

        metadata = {
            "mean_embedding": mean_emb,
            "std": std_scalar,
            "num_valid_passes": len(embeddings),
        }

        return quality_score, metadata


def create_serfiq_scorer(
    num_passes: int = 5,
    seed: Optional[int] = None,
) -> SERFIQScorer:
    """
    Factory function to create SER-FIQ scorer with default parameters.

    Args:
        num_passes: Number of stochastic forward passes (default 5)
        seed: Random seed for reproducibility

    Returns:
        SERFIQScorer instance
    """
    return SERFIQScorer(
        num_passes=num_passes,
        noise_std=1.5,
        brightness_range=0.03,
        contrast_range=0.03,
        rotation_deg=0.0,
        seed=seed,
    )
