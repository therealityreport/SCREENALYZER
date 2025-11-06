"""FAISS-backed vector index with centroid tracking for persistent re-identification."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - exercised via fallback
    faiss = None

logger = logging.getLogger(__name__)


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a vector."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


@dataclass
class LabelStats:
    """Running centroid and spread statistics for a label."""

    centroid: np.ndarray
    count: int = 0
    mean_dist: float = 0.0
    m2: float = 0.0

    def observe(self, vec: np.ndarray, alpha: Optional[float] = None) -> None:
        """
        Update centroid/spread statistics with a new observation.

        Args:
            vec: L2-normalized embedding
            alpha: Optional EMA weight. Defaults to 1/count (running mean).
        """
        vec = vec.astype(np.float32)
        if self.count == 0:
            self.centroid = vec.copy()
            self.count = 1
            self.mean_dist = 0.0
            self.m2 = 0.0
            return

        dist = float(np.linalg.norm(vec - self.centroid))
        self.count += 1
        delta = dist - self.mean_dist
        self.mean_dist += delta / self.count
        self.m2 += delta * (dist - self.mean_dist)

        weight = alpha if alpha is not None else 1.0 / max(self.count, 1)
        new_centroid = (1.0 - weight) * self.centroid + weight * vec
        norm = np.linalg.norm(new_centroid)
        if norm > 0:
            new_centroid = new_centroid / norm
        self.centroid = new_centroid.astype(np.float32)

    @property
    def sigma(self) -> float:
        """Return the standard deviation of distances to the centroid."""
        if self.count <= 1:
            return 0.0
        variance = self.m2 / (self.count - 1)
        return float(np.sqrt(max(variance, 0.0)))

    def to_dict(self) -> Dict[str, object]:
        """Serialize stats for persistence."""
        return {
            "centroid": self.centroid.tolist(),
            "count": self.count,
            "mean_dist": self.mean_dist,
            "m2": self.m2,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "LabelStats":
        """Deserialize stats from metadata."""
        centroid = np.array(data.get("centroid", []), dtype=np.float32)
        count = int(data.get("count", 0))
        mean_dist = float(data.get("mean_dist", 0.0))
        m2 = float(data.get("m2", 0.0))
        return cls(centroid=centroid, count=count, mean_dist=mean_dist, m2=m2)


class _NumpyIndex:
    """Minimal approximate replacement when FAISS is unavailable."""

    def __init__(self, metric: str):
        self.metric = metric
        self._vectors: list[np.ndarray] = []
        self._ids: list[int] = []

    def add(self, vector: np.ndarray, idx: int) -> None:
        self._vectors.append(vector.astype(np.float32))
        self._ids.append(int(idx))

    def search(self, vector: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if not self._vectors:
            return (
                np.full((1, k), -1, dtype=np.int64),
                np.full((1, k), -np.inf, dtype=np.float32),
            )

        mat = np.stack(self._vectors)
        if self.metric == "ip":
            scores = mat @ vector
        elif self.metric == "l2":
            scores = -np.linalg.norm(mat - vector, axis=1)
        else:
            raise ValueError(f"Unsupported metric for numpy fallback: {self.metric}")

        order = np.argsort(-scores)[:k]
        ids = np.array([self._ids[i] for i in order], dtype=np.int64)
        dists = scores[order].astype(np.float32)
        return ids.reshape(1, -1), dists.reshape(1, -1)

    def reset(self) -> None:
        self._vectors.clear()
        self._ids.clear()


class FaceIndex:
    """FAISS-backed index with centroid maintenance."""

    def __init__(self, metric: str = "ip", normalize: bool = True):
        if metric not in {"ip", "l2"}:
            raise ValueError("metric must be 'ip' (inner product) or 'l2'")
        self.metric = metric
        self.normalize = normalize

        self._dim: Optional[int] = None
        self._index = None
        self._label_stats: Dict[str, LabelStats] = {}
        self._label_to_ids: Dict[str, set[int]] = {}
        self._id_to_label: Dict[int, str] = {}
        self._vectors_by_id: Dict[int, np.ndarray] = {}
        self._next_id: int = 0

        self._faiss_available = faiss is not None

    # ------------------------------------------------------------------ helpers
    def _prepare_vector(self, vec: np.ndarray) -> np.ndarray:
        arr = np.asarray(vec, dtype=np.float32).reshape(-1)
        if self.normalize:
            arr = _l2_normalize(arr)
        if self._dim is None:
            self._dim = arr.shape[0]
            self._index = self._build_index()
        elif arr.shape[0] != self._dim:
            raise ValueError(f"Vector dimension {arr.shape[0]} != index dim {self._dim}")
        return arr

    def _build_index(self):
        if self._faiss_available:
            if self.metric == "ip":
                base = faiss.IndexFlatIP(self._dim)
            else:
                base = faiss.IndexFlatL2(self._dim)
            return faiss.IndexIDMap(base)
        return _NumpyIndex(self.metric)

    def _ensure_index(self) -> None:
        if self._index is None and self._dim is not None:
            self._index = self._build_index()

    # ------------------------------------------------------------------ API
    def load(self, faiss_path: str, centroids_path: str) -> None:
        """Load index data and centroids from disk."""
        meta_path = Path(centroids_path)
        if not meta_path.exists():
            raise FileNotFoundError(f"Centroids metadata not found: {centroids_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.metric = metadata.get("metric", self.metric)
        self.normalize = metadata.get("normalize", self.normalize)
        self._dim = metadata.get("dim", self._dim)
        self._next_id = metadata.get("next_id", 0)

        # Rehydrate mappings
        self._id_to_label = {int(k): v for k, v in metadata.get("id_to_label", {}).items()}
        self._label_to_ids = {}
        for idx, label in self._id_to_label.items():
            self._label_to_ids.setdefault(label, set()).add(int(idx))

        # Rehydrate vectors (used for fallback + centroid bootstrap)
        vectors_meta = metadata.get("vectors", {})
        self._vectors_by_id = {
            int(k): np.array(v, dtype=np.float32) for k, v in vectors_meta.items()
        }

        # Rehydrate label stats
        self._label_stats = {
            label: LabelStats.from_dict(stats) for label, stats in metadata.get("label_stats", {}).items()
        }

        self._ensure_index()

        index_path = Path(faiss_path)
        if self._faiss_available and index_path.exists():
            try:
                self._index = faiss.read_index(str(index_path))
            except Exception as exc:  # pragma: no cover - fallback path
                logger.warning("Failed to read FAISS index (%s), falling back to numpy: %s", faiss_path, exc)
                self._faiss_available = False
                self._index = self._build_index()

        if not self._faiss_available:
            # Populate numpy fallback
            if isinstance(self._index, _NumpyIndex):
                for idx, vec in self._vectors_by_id.items():
                    self._index.add(vec, idx)

    def save(self, faiss_path: str, centroids_path: str) -> None:
        """Persist FAISS index and centroid metadata to disk."""
        Path(faiss_path).parent.mkdir(parents=True, exist_ok=True)
        Path(centroids_path).parent.mkdir(parents=True, exist_ok=True)

        if self._faiss_available and self._index is not None and hasattr(self._index, "ntotal"):
            faiss.write_index(self._index, str(faiss_path))
        else:
            # Store as numpy archive for fallback use
            ids = np.array(list(self._vectors_by_id.keys()), dtype=np.int64)
            if ids.size:
                order = np.argsort(ids)
                ids = ids[order]
                vectors = np.stack([self._vectors_by_id[int(i)] for i in ids])
            else:
                vectors = np.zeros((0, self._dim or 0), dtype=np.float32)
            np.savez_compressed(faiss_path, embeddings=vectors, ids=ids)

        meta = {
            "metric": self.metric,
            "normalize": self.normalize,
            "dim": self._dim,
            "next_id": self._next_id,
            "id_to_label": {str(k): v for k, v in self._id_to_label.items()},
            "label_stats": {label: stats.to_dict() for label, stats in self._label_stats.items()},
            "vectors": {str(k): v.tolist() for k, v in self._vectors_by_id.items()},
        }

        with open(centroids_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def add(self, label: str, vec: np.ndarray) -> None:
        """Add a new embedding under a label."""
        vector = self._prepare_vector(vec)
        self._ensure_index()

        idx = self._next_id
        self._next_id += 1

        if self._faiss_available and hasattr(self._index, "add_with_ids"):
            self._index.add_with_ids(
                np.ascontiguousarray(vector[None, :]), np.array([idx], dtype=np.int64)
            )
        elif isinstance(self._index, _NumpyIndex):
            self._index.add(vector, idx)
        else:  # pragma: no cover - defensive
            raise RuntimeError("Index backend is not initialized")

        self._id_to_label[idx] = label
        self._label_to_ids.setdefault(label, set()).add(idx)
        self._vectors_by_id[idx] = vector.copy()

        stats = self._label_stats.get(label)
        if stats is None:
            self._label_stats[label] = LabelStats(centroid=vector.copy(), count=1)
        else:
            stats.observe(vector)

    def search(self, vec: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Search nearest labels for the given embedding."""
        if not self._id_to_label:
            return []

        vector = self._prepare_vector(vec)
        fetch = min(k * 4, max(len(self._id_to_label), 1))
        if self._faiss_available and hasattr(self._index, "search"):
            distances, indices = self._index.search(
                np.ascontiguousarray(vector[None, :]), fetch
            )
        elif isinstance(self._index, _NumpyIndex):
            indices, distances = self._index.search(vector, fetch)
        else:  # pragma: no cover - defensive
            raise RuntimeError("Index backend is not initialized")

        flat_indices = indices[0].tolist()
        flat_scores = distances[0].tolist()

        results: List[Tuple[str, float]] = []
        seen: set[str] = set()
        for idx, score in zip(flat_indices, flat_scores):
            if idx < 0:
                continue
            label = self._id_to_label.get(int(idx))
            if not label or label in seen:
                continue
            seen.add(label)
            results.append((label, float(score)))
            if len(results) >= k:
                break
        return results

    def update_centroid(self, label: str, vec: np.ndarray, alpha: float = 0.01) -> None:
        """Update centroid statistics for a label using EMA smoothing."""
        vector = self._prepare_vector(vec)
        stats = self._label_stats.get(label)
        if stats is None:
            self._label_stats[label] = LabelStats(centroid=vector.copy(), count=1)
        else:
            stats.observe(vector, alpha=alpha)

    @property
    def labels(self) -> List[str]:
        """Return list of labels known to the index."""
        return sorted(self._label_stats.keys())

    @property
    def centroids(self) -> Dict[str, np.ndarray]:
        """Return centroid vectors per label."""
        return {label: stats.centroid for label, stats in self._label_stats.items()}

    def get_label_stats(self, label: str) -> Optional[LabelStats]:
        """Access raw label statistics (sigma, counts)."""
        return self._label_stats.get(label)

