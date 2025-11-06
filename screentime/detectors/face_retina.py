"""
RetinaFace face detector using InsightFace with ONNX Runtime.

Provider fallback: CoreML → CPU
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


class RetinaFaceDetector:
    """RetinaFace detector with provider fallback."""

    def __init__(
        self,
        min_face_px: int = 80,
        min_confidence: float = 0.7,
        provider_order: list[str] | None = None,
    ):
        """
        Initialize detector.

        Args:
            min_face_px: Minimum face size in pixels
            min_confidence: Minimum detection confidence
            provider_order: ONNX Runtime providers (e.g., ["CoreMLExecutionProvider", "CPUExecutionProvider"])
        """
        self.min_face_px = min_face_px
        self.min_confidence = min_confidence
        self.provider_order = provider_order or ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        self.actual_provider = None

        # Initialize InsightFace
        self.app = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialize face analysis with provider fallback."""
        for provider in self.provider_order:
            try:
                logger.info(f"Attempting to initialize with provider: {provider}")

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
                    name="buffalo_l",  # InsightFace model
                    providers=providers,
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))

                self.actual_provider = provider
                logger.info(f"Successfully initialized with provider: {provider}")
                return

            except Exception as e:
                logger.warning(f"Failed to initialize with {provider}: {e}")
                continue

        # Fallback to CPU if all else fails
        if self.app is None:
            logger.warning("All providers failed, falling back to CPU")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.actual_provider = "cpu"

    def detect(self, image: np.ndarray) -> list[dict]:
        """
        Detect faces in image.

        Args:
            image: Input image (BGR)

        Returns:
            List of detections with bbox, landmarks, confidence, embedding
        """
        if self.app is None:
            raise RuntimeError("Detector not initialized")

        # Detect faces
        faces = self.app.get(image)

        # Filter by size and confidence
        detections = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox

            # Calculate face size
            face_width = x2 - x1
            face_height = y2 - y1
            face_size = min(face_width, face_height)

            # Filter by minimum size
            if face_size < self.min_face_px:
                continue

            # Filter by confidence
            if face.det_score < self.min_confidence:
                continue

            detections.append(
                {
                    "bbox": bbox.tolist(),
                    "landmarks": (
                        face.landmark_2d_106.tolist() if hasattr(face, "landmark_2d_106") else None
                    ),
                    "confidence": float(face.det_score),
                    "face_size": face_size,
                }
            )

        return detections

    def get_provider_info(self) -> dict:
        """Get information about active provider."""
        return {
            "requested_providers": self.provider_order,
            "actual_provider": self.actual_provider,
        }
