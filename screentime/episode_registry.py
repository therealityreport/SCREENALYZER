"""Episode registry for tracking uploaded episodes."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from screentime.utils import canonical_show_slug


class EpisodeRegistry:
    """Manages episode registry for tracking uploaded episodes."""

    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        self.registry_path = data_root / "diagnostics" / "episodes.json"
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_registry(self) -> dict:
        """Load registry from disk."""
        if not self.registry_path.exists():
            return {"episodes": []}

        try:
            with open(self.registry_path, "r") as f:
                return json.load(f)
        except:
            return {"episodes": []}

    def _save_registry(self, registry: dict) -> None:
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def register_episode(
        self,
        episode_id: str,
        show_id: str,
        season_id: str,
        video_path: str,
        status: str = "uploaded",
    ) -> None:
        """
        Register a new episode in the registry.

        Args:
            episode_id: Episode identifier
            show_id: Show identifier (e.g., "rhobh")
            season_id: Season identifier (e.g., "s05")
            video_path: Path to video file
            status: Episode status (uploaded, harvested, clustered, etc.)
        """
        show_id = canonical_show_slug(show_id)
        season_id = season_id.lower()

        registry = self._load_registry()

        # Check if episode already exists
        for episode in registry["episodes"]:
            if episode["episode_id"] == episode_id:
                # Update existing episode
                episode["show_id"] = show_id
                episode["season_id"] = season_id
                episode["video_path"] = video_path
                episode["status"] = status
                episode["updated_at"] = datetime.now().isoformat()
                self._save_registry(registry)
                return

        # Add new episode
        registry["episodes"].append({
            "episode_id": episode_id,
            "show_id": show_id,
            "season_id": season_id,
            "video_path": video_path,
            "status": status,
            "uploaded_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })

        self._save_registry(registry)

    def get_episode(self, episode_id: str) -> Optional[dict]:
        """Get episode info by ID."""
        registry = self._load_registry()
        for episode in registry["episodes"]:
            if episode["episode_id"] == episode_id:
                return episode
        return None

    def list_episodes(
        self,
        show_id: Optional[str] = None,
        season_id: Optional[str] = None,
    ) -> List[dict]:
        """
        List all episodes, optionally filtered by show/season.

        Args:
            show_id: Optional show ID filter
            season_id: Optional season ID filter

        Returns:
            List of episode dicts
        """
        registry = self._load_registry()
        episodes = registry["episodes"]

        if show_id:
            canonical_show = canonical_show_slug(show_id)
            episodes = [e for e in episodes if canonical_show_slug(e["show_id"]) == canonical_show]

        if season_id:
            episodes = [e for e in episodes if e["season_id"] == season_id]

        # Sort by uploaded_at descending (newest first)
        episodes.sort(key=lambda e: e.get("uploaded_at", ""), reverse=True)

        return episodes

    def update_status(self, episode_id: str, status: str) -> bool:
        """
        Update episode status.

        Args:
            episode_id: Episode identifier
            status: New status

        Returns:
            True if updated, False if not found
        """
        registry = self._load_registry()

        for episode in registry["episodes"]:
            if episode["episode_id"] == episode_id:
                episode["status"] = status
                episode["updated_at"] = datetime.now().isoformat()
                self._save_registry(registry)
                return True

        return False

    def scan_videos_directory(self) -> None:
        """
        Scan videos directory and register any unregistered episodes.

        Looks for videos in both:
        - data/videos/*.mp4 (flat structure)
        - data/videos/{show_id}/{season_id}/*.mp4 (hierarchical structure)
        """
        videos_dir = self.data_root / "videos"
        if not videos_dir.exists():
            return

        registry = self._load_registry()
        registered_ids = {e["episode_id"] for e in registry["episodes"]}

        # Scan flat structure
        for video_file in videos_dir.glob("*.mp4"):
            episode_id = video_file.stem
            if episode_id not in registered_ids:
                self.register_episode(
                    episode_id=episode_id,
                    show_id="unknown",
                    season_id="unknown",
                    video_path=str(video_file.relative_to(self.data_root)),
                    status="uploaded",
                )

        # Scan hierarchical structure: show_id/season_id/*.mp4
        for video_file in videos_dir.rglob("*/*.mp4"):
            episode_id = video_file.stem
            if episode_id not in registered_ids:
                # Extract show_id and season_id from path
                parts = video_file.relative_to(videos_dir).parts
                if len(parts) >= 2:
                    show_id = canonical_show_slug(parts[0])
                    season_id = parts[1]
                else:
                    show_id = "unknown"
                    season_id = "unknown"

                self.register_episode(
                    episode_id=episode_id,
                    show_id=show_id,
                    season_id=season_id,
                    video_path=str(video_file.relative_to(self.data_root)),
                    status="uploaded",
                )


# Global registry instance
episode_registry = EpisodeRegistry()
