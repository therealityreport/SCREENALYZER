"""
Face detector registry and abstract interface.

Provides a pluggable detector system for A/B testing and comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class Detection:
    """Standardized face detection result."""

    bbox: list[int]  # [x1, y1, x2, y2]
    confidence: float
    kps: Optional[np.ndarray] = None  # 5-point facial landmarks (5, 2)
    face_size: Optional[int] = None  # min(width, height)

    def __post_init__(self):
        """Compute face_size from bbox if not provided."""
        if self.face_size is None:
            x1, y1, x2, y2 = self.bbox
            w = x2 - x1
            h = y2 - y1
            self.face_size = min(w, h)


class FaceDetector(ABC):
    """
    Abstract base class for face detectors.

    All detectors must implement this interface to participate in A/B testing.
    """

    def __init__(
        self,
        min_face_px: int = 80,
        min_confidence: float = 0.7,
        provider_order: list[str] = None,
        **kwargs
    ):
        """
        Initialize detector with common parameters.

        Args:
            min_face_px: Minimum face size in pixels
            min_confidence: Minimum detection confidence
            provider_order: ONNX provider order (e.g., ["coreml", "cpu"])
            **kwargs: Detector-specific parameters
        """
        self.min_face_px = min_face_px
        self.min_confidence = min_confidence
        self.provider_order = provider_order or ["cpu"]
        self.kwargs = kwargs

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Detect faces in a frame.

        Args:
            frame: RGB or BGR frame (H, W, 3)

        Returns:
            List of Detection objects
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Detector name for reporting (e.g., 'retinaface', 'scrfd')."""
        pass

    @property
    @abstractmethod
    def version(self) -> str:
        """Detector version/model identifier."""
        pass

    def get_metadata(self) -> dict[str, Any]:
        """Get detector metadata for reporting."""
        return {
            "name": self.name,
            "version": self.version,
            "min_face_px": self.min_face_px,
            "min_confidence": self.min_confidence,
            "provider_order": self.provider_order,
            **self.kwargs
        }


class DetectorRegistry:
    """
    Global registry for face detectors.

    Allows lookup and instantiation of detectors by name.
    """

    _detectors: dict[str, type[FaceDetector]] = {}

    @classmethod
    def register(cls, name: str, detector_class: type[FaceDetector]):
        """Register a detector class."""
        cls._detectors[name] = detector_class

    @classmethod
    def get(cls, name: str) -> type[FaceDetector]:
        """Get a registered detector class by name."""
        if name not in cls._detectors:
            raise ValueError(f"Unknown detector: {name}. Available: {list(cls._detectors.keys())}")
        return cls._detectors[name]

    @classmethod
    def create(cls, name: str, **kwargs) -> FaceDetector:
        """Create a detector instance by name."""
        detector_class = cls.get(name)
        return detector_class(**kwargs)

    @classmethod
    def list_detectors(cls) -> list[str]:
        """List all registered detector names."""
        return list(cls._detectors.keys())
