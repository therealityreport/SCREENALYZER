"""Facebank curation utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _load_embedding_file(path: Path) -> Optional[Tuple[np.ndarray, float]]:
    """
    Load an embedding (and optional quality score) from disk.

    Supports:
        - .npy       → 1D array embedding
        - .npz       → expects key "embedding"; optional "quality"

    Returns:
        Tuple of (embedding, quality) or None if unsupported.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            arr = np.load(path)
            if isinstance(arr, np.ndarray):
                return arr.astype(np.float32).reshape(-1), 1.0
        elif suffix == ".npz":
            data = np.load(path)
            keys = list(data.keys())
            emb_key = "embedding" if "embedding" in data else (keys[0] if keys else None)
            if emb_key:
                emb = np.array(data[emb_key], dtype=np.float32).reshape(-1)
                quality = float(data.get("quality", 1.0))
                return emb, quality
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        logger.warning("Failed to load embedding from %s: %s", path, exc)
    return None


def curate_facebank(
    bank_dir: str,
    out_embeddings: str,
    out_meta: str,
    min_quality: float = 0.6,
    max_per_member: int = 40,
) -> None:
    """
    Curate a facebank from seed embeddings.

    The current implementation expects each cast member to have a subdirectory
    under ``bank_dir`` containing pre-computed embeddings (.npy / .npz). Image
    chips are skipped unless embeddings are already present. The routine:

        1. Loads embeddings and optional quality scores.
        2. Filters by ``min_quality``.
        3. Removes outliers via z-score on Euclidean distance from the cluster centroid.
        4. Limits each member to ``max_per_member`` exemplars (highest quality).
        5. Saves consolidated embeddings to ``out_embeddings`` (npz) and metadata JSON.
    """
    bank_path = Path(bank_dir)
    if not bank_path.exists():
        raise FileNotFoundError(f"Facebank directory not found: {bank_dir}")

    records: List[Dict[str, object]] = []
    meta: Dict[str, Dict[str, object]] = {}

    member_dirs = sorted([p for p in bank_path.iterdir() if p.is_dir()])

    for member_dir in member_dirs:
        member = member_dir.name
        embeddings: List[np.ndarray] = []
        qualities: List[float] = []
        sources: List[str] = []
        dropped_low_quality = 0
        dropped_outliers = 0

        files = sorted(member_dir.rglob("*"))
        for file_path in files:
            if file_path.is_dir():
                continue

            if file_path.suffix.lower() in IMAGE_SUFFIXES:
                logger.debug("Skipping raw image file without embedding: %s", file_path)
                continue

            loaded = _load_embedding_file(file_path)
            if not loaded:
                continue

            emb, quality = loaded
            if quality < min_quality:
                dropped_low_quality += 1
                continue

            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm

            embeddings.append(emb.astype(np.float32))
            qualities.append(float(quality))
            sources.append(str(file_path.relative_to(bank_path)))

        if not embeddings:
            meta[member] = {
                "kept": 0,
                "dropped_low_quality": dropped_low_quality,
                "dropped_outliers": 0,
                "sources": [],
            }
            continue

        emb_matrix = np.stack(embeddings)
        centroid = np.mean(emb_matrix, axis=0)
        dists = np.linalg.norm(emb_matrix - centroid, axis=1)
        mean_dist = float(np.mean(dists))
        std_dist = float(np.std(dists))

        if std_dist > 0:
            zscores = (dists - mean_dist) / (std_dist + 1e-6)
        else:
            zscores = np.zeros_like(dists)

        mask = np.abs(zscores) <= 2.5
        dropped_outliers = int(np.size(zscores) - int(np.count_nonzero(mask)))

        emb_matrix = emb_matrix[mask]
        qualities = [q for q, keep in zip(qualities, mask) if keep]
        sources = [s for s, keep in zip(sources, mask) if keep]

        if emb_matrix.size == 0:
            meta[member] = {
                "kept": 0,
                "dropped_low_quality": dropped_low_quality,
                "dropped_outliers": dropped_outliers,
                "sources": [],
            }
            continue

        order = np.argsort(-np.array(qualities))
        emb_matrix = emb_matrix[order]
        qualities = [qualities[i] for i in order]
        sources = [sources[i] for i in order]

        if len(emb_matrix) > max_per_member:
            emb_matrix = emb_matrix[:max_per_member]
            qualities = qualities[:max_per_member]
            sources = sources[:max_per_member]

        for emb_vec, quality, src in zip(emb_matrix, qualities, sources):
            records.append(
                {
                    "label": member,
                    "embedding": emb_vec.astype(np.float32),
                    "quality": float(quality),
                    "source": src,
                }
            )

        meta[member] = {
            "kept": len(emb_matrix),
            "dropped_low_quality": dropped_low_quality,
            "dropped_outliers": dropped_outliers,
            "mean_quality": float(np.mean(qualities)) if qualities else 0.0,
            "sources": sources,
        }

    embeddings_out = Path(out_embeddings)
    embeddings_out.parent.mkdir(parents=True, exist_ok=True)

    if records:
        emb_stack = np.stack([r["embedding"] for r in records])
        labels = np.array([r["label"] for r in records])
        qualities = np.array([r["quality"] for r in records], dtype=np.float32)
        sources = np.array([r["source"] for r in records])
    else:
        emb_stack = np.zeros((0, 0), dtype=np.float32)
        labels = np.array([], dtype=object)
        qualities = np.array([], dtype=np.float32)
        sources = np.array([], dtype=object)

    np.savez_compressed(
        embeddings_out,
        embeddings=emb_stack,
        labels=labels,
        qualities=qualities,
        sources=sources,
    )

    meta_out = Path(out_meta)
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
