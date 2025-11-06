"""
Helpers for loading frame asset manifest (frames_index.json).

The manifest defines the contract between the frame writer and UI/asset server.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

logger = logging.getLogger(__name__)

FRAME_INDEX_FILENAME = "frames_index.json"
DEFAULT_FRAME_PAD = 6

# Cache of {(manifest_path, data_root): (mtime, FrameIndex | None)}
_INDEX_CACHE: Dict[tuple[Path, Path], tuple[float, "FrameIndex | None"]] = {}


@dataclass
class FrameIndex:
    """Parsed frame index manifest for an episode."""

    episode_id: str
    asset_base: str
    frame_ext: str
    frame_pad: int
    fps: Optional[float]
    entries: Dict[int, Dict]
    manifest_path: Path
    harvest_dir: Path
    data_root: Path

    def resolve_path(self, frame_id: int) -> Optional[Path]:
        """
        Return absolute filesystem path for a given frame_id if available.

        Prefers explicit manifest paths and falls back to asset_base/frame_pad.
        """
        candidates: list[Path] = []

        entry = self.entries.get(frame_id)
        if entry:
            explicit = entry.get("path") or entry.get("rel_path")
            if explicit:
                candidates.append(Path(explicit))

        # Fallback: derive relative path from asset_base + padded id
        padded = str(frame_id).zfill(self.frame_pad or DEFAULT_FRAME_PAD)
        if self.asset_base:
            candidates.append(Path(self.asset_base) / f"{padded}{self.frame_ext or '.jpg'}")

        for candidate in candidates:
            resolved = self._to_absolute(candidate)
            if resolved is not None and resolved.exists():
                return resolved

        return None

    def iter_entries(self) -> Iterable[tuple[int, Dict]]:
        """Iterate (frame_id, entry) pairs in the manifest."""
        return self.entries.items()

    def _to_absolute(self, candidate: Path) -> Optional[Path]:
        """Convert relative manifest path to absolute filesystem path."""
        if candidate.is_absolute():
            return candidate

        harvest_candidate = (self.harvest_dir / candidate).resolve()
        if harvest_candidate.exists():
            return harvest_candidate

        root_candidate = (self.data_root / candidate).resolve()
        if root_candidate.exists():
            return root_candidate

        return None


def load_frames_index(episode_id: str, data_root: Path = Path("data")) -> Optional[FrameIndex]:
    """
    Load frames_index.json for an episode with caching keyed by file mtime.

    Returns:
        FrameIndex instance or None if manifest is missing/invalid.
    """
    harvest_dir = data_root / "harvest" / episode_id
    manifest_path = harvest_dir / "frames" / FRAME_INDEX_FILENAME

    if not manifest_path.exists():
        logger.debug("Frame index not found for %s (%s)", episode_id, manifest_path)
        return None

    try:
        mtime = manifest_path.stat().st_mtime
    except OSError:
        logger.warning("Unable to stat frame index %s", manifest_path)
        return None

    cache_key = (manifest_path, data_root)
    cached = _INDEX_CACHE.get(cache_key)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in frame index %s: %s", manifest_path, exc)
        _INDEX_CACHE[cache_key] = (mtime, None)
        return None
    except OSError as exc:
        logger.error("Failed to read frame index %s: %s", manifest_path, exc)
        _INDEX_CACHE[cache_key] = (mtime, None)
        return None

    entries = _build_path_map(payload)
    if not entries:
        logger.warning("Frame index %s has no path entries", manifest_path)

    frame_index = FrameIndex(
        episode_id=episode_id,
        asset_base=str(payload.get("asset_base") or ""),
        frame_ext=str(payload.get("frame_ext") or ".jpg"),
        frame_pad=int(payload.get("frame_pad") or DEFAULT_FRAME_PAD),
        fps=payload.get("fps"),
        entries=entries,
        manifest_path=manifest_path,
        harvest_dir=harvest_dir,
        data_root=data_root,
    )

    _INDEX_CACHE[cache_key] = (mtime, frame_index)
    return frame_index


def resolve_frame_path(
    episode_id: str, frame_id: int, data_root: Path = Path("data")
) -> Optional[Path]:
    """Convenience wrapper to resolve a single frame path."""
    index = load_frames_index(episode_id, data_root=data_root)
    if not index:
        return None
    return index.resolve_path(frame_id)


def _build_path_map(payload: dict) -> Dict[int, Dict]:
    """Normalise manifest \"paths\" field into {frame_id: entry} map."""
    path_map: Dict[int, Dict] = {}

    if not isinstance(payload, dict):
        return path_map

    raw_paths = payload.get("paths")
    if isinstance(raw_paths, dict):
        for key, value in raw_paths.items():
            try:
                frame_id = int(key)
            except (TypeError, ValueError):
                continue

            if isinstance(value, dict):
                entry = value
            else:
                entry = {"path": value}

            path_map[frame_id] = entry
    elif isinstance(raw_paths, list):
        for idx, item in enumerate(raw_paths):
            if isinstance(item, dict):
                frame_id = item.get("frame_id", idx)
                try:
                    frame_id = int(frame_id)
                except (TypeError, ValueError):
                    frame_id = idx
                path_map[frame_id] = item
            elif isinstance(item, str):
                path_map[idx] = {"path": item}
    else:
        logger.debug("Frame index missing 'paths' collection")

    # Legacy fallback: allow "frames" list to seed entries if needed
    if not path_map and isinstance(payload.get("frames"), list):
        for item in payload["frames"]:
            if not isinstance(item, dict):
                continue
            frame_id = item.get("frame_id")
            if frame_id is None:
                continue
            try:
                frame_id = int(frame_id)
            except (TypeError, ValueError):
                continue
            path_map[frame_id] = item

    return path_map
