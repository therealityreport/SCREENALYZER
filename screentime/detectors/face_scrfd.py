"""
SCRFD face detector using InsightFace with ONNX Runtime.

SCRFD (Sample and Computation Redistribution for Efficient Face Detection)
is optimized for small faces and profile views.

Provider fallback: CoreML → CPU
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from screentime.detectors.registry import Detection, FaceDetector

logger = logging.getLogger(__name__)


class SCRFDDetector(FaceDetector):
    """SCRFD detector with provider fallback."""

    def __init__(
        self,
        min_face_px: int = 80,
        min_confidence: float = 0.7,
        provider_order: list[str] | None = None,
        model_name: str = "scrfd_10g_bnkps",
        **kwargs
    ):
        """
        Initialize SCRFD detector.

        Args:
            min_face_px: Minimum face size in pixels
            min_confidence: Minimum detection confidence
            provider_order: ONNX Runtime providers (e.g., ["coreml", "cpu"])
            model_name: SCRFD model variant ("scrfd_10g_bnkps" or "scrfd_2.5g_bnkps")
            **kwargs: Additional detector-specific parameters
        """
        super().__init__(min_face_px, min_confidence, provider_order, **kwargs)

        self.model_name = model_name
        self.actual_provider = None
        self.app = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize face analysis with provider fallback."""
        provider_order = self.provider_order or ["coreml", "cpu"]

        for provider in provider_order:
            try:
                logger.info(f"Attempting to initialize SCRFD with provider: {provider}")

                # Map friendly names to ONNX Runtime provider names
                providers = []
                if "coreml" in provider.lower():
                    providers.append("CoreMLExecutionProvider")
                elif "cuda" in provider.lower():
                    providers.append("CUDAExecutionProvider")
                else:
                    providers.append("CPUExecutionProvider")

                # Initialize InsightFace with SCRFD model
                # Note: InsightFace will auto-download the model if not present
                self.app = FaceAnalysis(
                    name="buffalo_sc",  # SCRFD models are in buffalo_sc package
                    providers=providers,
                    allowed_modules=['detection']  # Only load detection module
                )
                self.app.prepare(ctx_id=0, det_size=(640, 640))

                self.actual_provider = provider
                logger.info(f"Successfully initialized SCRFD with provider: {provider}")
                return

            except Exception as e:
                logger.warning(f"Failed to initialize SCRFD with {provider}: {e}")
                continue

        # Fallback to CPU if all else fails
        if self.app is None:
            logger.warning("All providers failed, falling back to CPU")
            self.app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CPUExecutionProvider"],
                allowed_modules=['detection']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self.actual_provider = "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect faces in frame.

        Args:
            frame: Input frame (BGR format, H x W x 3)

        Returns:
            List of Detection objects
        """
        if self.app is None:
            raise RuntimeError("SCRFD detector not initialized")

        # Detect faces using InsightFace
        faces = self.app.get(frame)

        # Convert to standardized Detection format
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

            # Extract keypoints if available (SCRFD provides 5-point landmarks)
            kps = None
            if hasattr(face, 'kps') and face.kps is not None:
                kps = face.kps.astype(np.float32)

            detection = Detection(
                bbox=bbox.tolist(),
                confidence=float(face.det_score),
                kps=kps,
                face_size=face_size
            )
            detections.append(detection)

        return detections

    @property
    def name(self) -> str:
        """Detector name for reporting."""
        return "scrfd"

    @property
    def version(self) -> str:
        """Detector version/model identifier."""
        return self.model_name

    def get_provider_info(self) -> dict:
        """Get information about active provider."""
        return {
            "requested_providers": self.provider_order,
            "actual_provider": self.actual_provider,
            "model": self.model_name
        }


# Register SCRFD detector
from screentime.detectors.registry import DetectorRegistry
DetectorRegistry.register("scrfd", SCRFDDetector)
