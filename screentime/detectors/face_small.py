"""
High-recall face detector tuned for Local Densify windows.

Wraps InsightFace RetinaFace with multi-scale decoding and optional tiling
to recover small / low-confidence faces without changing the global detector.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """Detected face metadata."""

    bbox: List[int]
    confidence: float
    face_size: int
    scale: float


class SmallFaceRetinaDetector:
    """
    Multi-scale RetinaFace detector for densify spans.

    Runs InsightFace at several upscales to recover faces that were filtered out
    by the baseline detector. This class intentionally keeps the interface small
    so Local Densify can decide how to post-process detections.
    """

    def __init__(
        self,
        min_face_px: int = 36,
        min_confidence: float = 0.55,
        scales: Sequence[float] | None = None,
        tile_size: Tuple[int, int] | None = None,
        provider_order: Sequence[str] | None = None,
    ) -> None:
        self.min_face_px = min_face_px
        self.min_confidence = min_confidence
        self.scales = tuple(scales or (1.0, 1.25, 1.5, 2.0))
        self.tile_size = tile_size
        self.provider_order = tuple(provider_order or ("coreml", "cpu"))

        self._face_app: FaceAnalysis | None = None
        self._initialize()

    def _initialize(self) -> None:
        """Initialise InsightFace with provider fallback."""
        for provider in self.provider_order:
            providers = []
            if provider.lower() == "coreml":
                providers.append("CoreMLExecutionProvider")
            elif provider.lower() == "cuda":
                providers.append("CUDAExecutionProvider")
            else:
                providers.append("CPUExecutionProvider")

            try:
                app = FaceAnalysis(name="buffalo_l", providers=providers)
                app.prepare(ctx_id=0, det_size=(640, 640))
                self._face_app = app
                logger.info("SmallFaceRetinaDetector initialised with provider=%s", provider)
                return
            except Exception as exc:  # pragma: no cover - provider fallback
                logger.warning("Failed to initialise InsightFace with provider %s: %s", provider, exc)

        if self._face_app is None:
            logger.warning("Falling back to CPUExecutionProvider for SmallFaceRetinaDetector")
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_app = app

    def detect(self, image: np.ndarray) -> List[DetectedFace]:
        """
        Detect faces using multi-scale inference.

        Args:
            image: BGR frame

        Returns:
            List of DetectedFace entries sorted by confidence (descending)
        """
        if self._face_app is None:
            raise RuntimeError("SmallFaceRetinaDetector not initialised")

        detections: List[DetectedFace] = []
        image_h, image_w = image.shape[:2]

        def add_detection(raw_bbox: np.ndarray, score: float, scale: float) -> None:
            x1, y1, x2, y2 = raw_bbox.astype(float)
            x1 = np.clip(x1, 0, image_w - 1)
            y1 = np.clip(y1, 0, image_h - 1)
            x2 = np.clip(x2, 0, image_w - 1)
            y2 = np.clip(y2, 0, image_h - 1)

            face_w = x2 - x1
            face_h = y2 - y1
            face_size = int(min(face_w, face_h))

            if face_size < self.min_face_px or score < self.min_confidence:
                return

            detections.append(
                DetectedFace(
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    confidence=float(score),
                    face_size=face_size,
                    scale=scale,
                )
            )

        for scale in self.scales:
            if scale == 1.0:
                scaled = image
                scale_factor = 1.0
            else:
                scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale_factor = scale

            faces = self._face_app.get(scaled)
            for face in faces:
                bbox = face.bbox / scale_factor
                add_detection(bbox, float(face.det_score), scale)

        # De-duplicate overlapping detections – keep the highest confidence per IoU cluster
        pruned: List[DetectedFace] = []
        for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if not any(self._iou(det.bbox, kept.bbox) > 0.5 for kept in pruned):
                pruned.append(det)

        return pruned

    @staticmethod
    def _iou(a: Sequence[int], b: Sequence[int]) -> float:
        """Intersection over Union helper for duplication removal."""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, (ax2 - ax1)) * max(0, (ay2 - ay1))
        area_b = max(0, (bx2 - bx1)) * max(0, (by2 - by1))

        denom = area_a + area_b - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom
