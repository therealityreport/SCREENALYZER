"""
RetinaFace detector wrapper implementing the common FaceDetector interface.

Wraps the existing RetinaFaceDetector to work with the detector registry.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from screentime.detectors.face_retina import RetinaFaceDetector as OriginalRetinaFace
from screentime.detectors.registry import Detection, FaceDetector

logger = logging.getLogger(__name__)


class RetinaFaceDetectorWrapper(FaceDetector):
    """RetinaFace detector implementing FaceDetector interface."""

    def __init__(
        self,
        min_face_px: int = 80,
        min_confidence: float = 0.7,
        provider_order: list[str] | None = None,
        **kwargs
    ):
        """
        Initialize RetinaFace detector wrapper.

        Args:
            min_face_px: Minimum face size in pixels
            min_confidence: Minimum detection confidence
            provider_order: ONNX Runtime providers (e.g., ["coreml", "cpu"])
            **kwargs: Additional detector-specific parameters
        """
        super().__init__(min_face_px, min_confidence, provider_order, **kwargs)

        # Initialize the original RetinaFace detector
        self.detector = OriginalRetinaFace(
            min_face_px=min_face_px,
            min_confidence=min_confidence,
            provider_order=provider_order or ["coreml", "cpu"]
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect faces in frame.

        Args:
            frame: Input frame (BGR format, H x W x 3)

        Returns:
            List of Detection objects
        """
        # Call original detector
        raw_detections = self.detector.detect(frame)

        # Convert to standardized Detection format
        detections = []
        for det in raw_detections:
            # Extract keypoints (5-point landmarks)
            kps = None
            if 'landmarks' in det and det['landmarks'] is not None:
                # Original format may have 106-point landmarks, extract 5-point
                landmarks = np.array(det['landmarks'])
                if len(landmarks) == 106:
                    # Extract eyes, nose, mouth corners (standard 5-point)
                    # This is a simplified extraction - adjust indices based on actual format
                    kps = landmarks[[38, 88, 56, 72, 93]].astype(np.float32)
                else:
                    kps = landmarks.astype(np.float32) if len(landmarks) == 5 else None

            detection = Detection(
                bbox=det['bbox'],
                confidence=det['confidence'],
                kps=kps,
                face_size=det.get('face_size')
            )
            detections.append(detection)

        return detections

    @property
    def name(self) -> str:
        """Detector name for reporting."""
        return "retinaface"

    @property
    def version(self) -> str:
        """Detector version/model identifier."""
        return "det_10g"  # InsightFace buffalo_l uses det_10g model

    def get_provider_info(self) -> dict:
        """Get information about active provider."""
        return self.detector.get_provider_info()


# Register RetinaFace detector
from screentime.detectors.registry import DetectorRegistry
DetectorRegistry.register("retinaface", RetinaFaceDetectorWrapper)
