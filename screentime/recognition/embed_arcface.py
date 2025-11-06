"""
ArcFace embedding generator using InsightFace with ONNX Runtime.

Provider fallback: CoreML → CPU

UPDATED: Bypasses re-detection, aligns directly from detector outputs (keypoints or bbox).
"""

from __future__ import annotations

import logging
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from insightface.app import FaceAnalysis
import skimage.transform as trans

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embed_from_detection with metadata."""
    embedding: Optional[np.ndarray]
    chip_px: int
    used_kps: bool
    tries: int
    success: bool


class ArcFaceEmbedder:
    """ArcFace embedder with provider fallback and direct alignment mode."""

    # ArcFace standard 5-point template (112x112 chip)
    ARCFACE_DST = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041]
    ], dtype=np.float32)

    def __init__(
        self,
        provider_order: list[str] | None = None,
        pad_ratio: float = 0.12,
        skip_redetect: bool = False,
        align_priority: str = "kps_then_bbox",
        margin_scale: float = 1.25,
        min_chip_px: int = 112,
        fallback_scales: list[float] | None = None,
    ):
        """
        Initialize embedder.

        Args:
            provider_order: ONNX Runtime providers
            pad_ratio: Padding for old embed() method (kept for compatibility)
            skip_redetect: If True, bypass re-detection in crops (NEW)
            align_priority: "kps_then_bbox" or "bbox_only" (NEW)
            margin_scale: Box expansion factor for bbox alignment (NEW)
            min_chip_px: Target chip size (112 for ArcFace) (NEW)
            fallback_scales: Retry scales if embedding fails [1.0, 1.2, 1.4] (NEW)
        """
        self.provider_order = provider_order or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.actual_provider = None
        self.pad_ratio = max(0.0, min(pad_ratio, 0.25))

        # New alignment config
        self.skip_redetect = skip_redetect
        self.align_priority = align_priority
        self.margin_scale = margin_scale
        self.min_chip_px = min_chip_px
        self.fallback_scales = fallback_scales or [1.0, 1.2, 1.4]

        # Initialize InsightFace
        self.app = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize face analysis with provider fallback."""
        for provider in self.provider_order:
            try:
                logger.info(f"Attempting to initialize embedder with provider: {provider}")

                # Map friendly names to ONNX Runtime provider names
                providers = []
                if "coreml" in provider.lower():
                    providers.append("CoreMLExecutionProvider")
                elif "cuda" in provider.lower():
                    providers.append("CUDAExecutionProvider")
                else:
                    providers.append("CPUExecutionProvider")

                # Initialize
                self.app = FaceAnalysis(
                    name="buffalo_l",  # InsightFace model with ArcFace
                    providers=providers,
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))

                self.actual_provider = provider
                logger.info(f"Successfully initialized embedder with provider: {provider}")
                return

            except Exception as e:
                logger.warning(f"Failed to initialize embedder with {provider}: {e}")
                continue

        # Fallback to CPU if all else fails
        if self.app is None:
            logger.warning("All providers failed, falling back to CPU for embedder")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.actual_provider = "cpu"

    def align_from_kps(self, frame: np.ndarray, kps: np.ndarray, target: int = 112) -> Optional[np.ndarray]:
        """
        Align face chip using 5-point keypoints (similarity transform).

        Args:
            frame: Full frame (BGR)
            kps: 5x2 keypoints array [[x,y], ...]
            target: Target chip size (default 112 for ArcFace)

        Returns:
            Aligned chip (target x target x 3) or None if alignment fails
        """
        try:
            if kps is None or len(kps) != 5:
                return None

            # Compute similarity transform from kps to ArcFace template
            kps = np.array(kps, dtype=np.float32)

            # Scale template to target size
            scale = target / 112.0
            dst = self.ARCFACE_DST * scale

            # Estimate similarity transform
            tform = trans.SimilarityTransform()
            tform.estimate(kps, dst)

            # Warp
            M = tform.params[0:2, :]
            chip = cv2.warpAffine(frame, M, (target, target), borderValue=0.0)

            return chip

        except Exception as e:
            logger.debug(f"Failed to align from keypoints: {e}")
            return None

    def chip_from_bbox(
        self,
        frame: np.ndarray,
        bbox: list[int],
        margin_scale: float = 1.25,
        target: int = 112
    ) -> Optional[np.ndarray]:
        """
        Create face chip from bounding box (expand + resize).

        Args:
            frame: Full frame (BGR)
            bbox: [x1, y1, x2, y2]
            margin_scale: Expansion factor (e.g. 1.25 = 25% margin)
            target: Target chip size

        Returns:
            Face chip (target x target x 3) or None if invalid
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]

            # Compute center and size
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            size = max(x2 - x1, y2 - y1) * margin_scale

            # Expand to square with margin
            x1_exp = int(max(0, cx - size / 2))
            y1_exp = int(max(0, cy - size / 2))
            x2_exp = int(min(w, cx + size / 2))
            y2_exp = int(min(h, cy + size / 2))

            if x2_exp <= x1_exp or y2_exp <= y1_exp:
                return None

            # Crop and resize
            crop = frame[y1_exp:y2_exp, x1_exp:x2_exp]
            chip = cv2.resize(crop, (target, target))

            return chip

        except Exception as e:
            logger.debug(f"Failed to create bbox chip: {e}")
            return None

    def embed_from_detection(
        self,
        frame: np.ndarray,
        bbox: list[int],
        kps: Optional[np.ndarray] = None,
    ) -> EmbeddingResult:
        """
        Generate embedding directly from detector outputs (no re-detection).

        This is the NEW primary method that bypasses the re-detection bug.

        Args:
            frame: Full frame (BGR)
            bbox: Detection bbox [x1, y1, x2, y2]
            kps: Optional 5-point keypoints array (5x2)

        Returns:
            EmbeddingResult with embedding (or None) and metadata
        """
        if self.app is None:
            raise RuntimeError("Embedder not initialized")

        tries = 0
        embedding = None
        used_kps = False
        chip_px = self.min_chip_px

        # Try 1: Align from keypoints if available and priority is kps_then_bbox
        if self.align_priority == "kps_then_bbox" and kps is not None:
            tries += 1
            chip = self.align_from_kps(frame, kps, target=self.min_chip_px)

            if chip is not None:
                # Run recognition model directly on aligned chip
                try:
                    # Get embedding from recognition model
                    # InsightFace expects BGR, chip is already BGR
                    faces = self.app.get(chip)

                    if faces and len(faces) > 0:
                        embedding = faces[0].normed_embedding
                        used_kps = True

                        if embedding is not None and not np.isnan(embedding).any():
                            return EmbeddingResult(
                                embedding=embedding,
                                chip_px=chip_px,
                                used_kps=True,
                                tries=tries,
                                success=True
                            )
                except Exception as e:
                    logger.debug(f"Failed to embed from kps chip: {e}")

        # Try 2: Fallback to bbox alignment with multiple scales
        for scale in self.fallback_scales:
            tries += 1
            chip = self.chip_from_bbox(
                frame,
                bbox,
                margin_scale=self.margin_scale * scale,
                target=self.min_chip_px
            )

            if chip is None:
                continue

            try:
                faces = self.app.get(chip)

                if faces and len(faces) > 0:
                    embedding = faces[0].normed_embedding

                    if embedding is not None and not np.isnan(embedding).any():
                        return EmbeddingResult(
                            embedding=embedding,
                            chip_px=chip_px,
                            used_kps=False,
                            tries=tries,
                            success=True
                        )
            except Exception as e:
                logger.debug(f"Failed to embed from bbox chip (scale={scale:.2f}): {e}")
                continue

        # All attempts failed
        return EmbeddingResult(
            embedding=None,
            chip_px=chip_px,
            used_kps=used_kps,
            tries=tries,
            success=False
        )

    def embed(self, image: np.ndarray, bbox: list[int]) -> Optional[np.ndarray]:
        """
        Generate embedding for face in image (LEGACY METHOD - kept for compatibility).

        NOTE: This method uses the old re-detection approach and may fail on multi-face frames.
        Use embed_from_detection() for better reliability.

        Args:
            image: Input image (BGR)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            512-d embedding vector or None if failed
        """
        if self.app is None:
            raise RuntimeError("Embedder not initialized")

        # If skip_redetect is enabled, delegate to new method
        if self.skip_redetect:
            result = self.embed_from_detection(image, bbox, kps=None)
            return result.embedding

        # Old method (with re-detection bug)
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]

            pad = int(max(x2 - x1, y2 - y1) * self.pad_ratio)
            x1 = int(max(0, x1 - pad))
            y1 = int(max(0, y1 - pad))
            x2 = int(min(w, x2 + pad))
            y2 = int(min(h, y2 + pad))

            if x2 <= x1 or y2 <= y1:
                return None

            face_crop = image[y1:y2, x1:x2]

            faces = self.app.get(face_crop)
            if not faces:
                return None

            # Select detection whose bbox is closest to crop centre (co-face suppression)
            cx = (face_crop.shape[1] - 1) / 2
            cy = (face_crop.shape[0] - 1) / 2

            best_face = None
            best_distance = float("inf")
            crop_area = face_crop.shape[0] * face_crop.shape[1]

            for face in faces:
                fx1, fy1, fx2, fy2 = face.bbox
                fc_x = (fx1 + fx2) / 2.0
                fc_y = (fy1 + fy2) / 2.0
                distance = (fc_x - cx) ** 2 + (fc_y - cy) ** 2
                face_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
                coverage = face_area / crop_area

                # Skip stray detections that barely cover the crop
                if coverage < 0.25:
                    continue

                if distance < best_distance:
                    best_distance = distance
                    best_face = face

            if best_face is None:
                return None

            embedding = best_face.normed_embedding
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None

    def embed_batch(self, image: np.ndarray, bboxes: list[list[int]]) -> list[Optional[np.ndarray]]:
        """
        Generate embeddings for multiple faces.

        Args:
            image: Input image (BGR)
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]

        Returns:
            List of 512-d embedding vectors (None for failures)
        """
        embeddings = []
        for bbox in bboxes:
            embedding = self.embed(image, bbox)
            embeddings.append(embedding)
        return embeddings

    def get_provider_info(self) -> dict:
        """Get information about active provider."""
        return {
            "requested_providers": self.provider_order,
            "actual_provider": self.actual_provider,
        }
