"""
Show/Season registry management (file-based, no database).

Registry: configs/shows_seasons.json
Directory structure: data/{videos,facebank,harvest,outputs}/<SHOW>/<SEASON>/
"""

from __future__ import annotations

import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

REGISTRY_PATH = Path("configs/shows_seasons.json")


@dataclass
class CastMember:
    """Cast member for a season."""
    name: str
    seed_count: int = 0
    valid_seeds: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')


@dataclass
class Episode:
    """Episode within a season."""
    episode_id: str
    video_path: str
    status: str = "uploaded"  # uploaded, processing, completed, failed
    duration_sec: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')


@dataclass
class Season:
    """Season within a show."""
    season_id: str
    season_label: str
    season_number: int
    cast: List[CastMember] = field(default_factory=list)
    episodes: List[Episode] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')


@dataclass
class Show:
    """TV show."""
    show_id: str
    show_name: str
    seasons: List[Season] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + 'Z')


class ShowSeasonRegistry:
    """
    Manage show/season registry with atomic writes and schema validation.

    Registry stored in: configs/shows_seasons.json
    """

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        """
        Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = registry_path
        self.shows: List[Show] = []
        self.load()

    def load(self) -> None:
        """Load registry from JSON."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)

                # Validate schema
                if not self._validate_schema(data):
                    raise ValueError("Invalid registry schema")

                # Convert to dataclasses
                self.shows = [self._dict_to_show(s) for s in data.get('shows', [])]
            except (json.JSONDecodeError, ValueError) as e:
                # Backup corrupt file
                backup_path = self.registry_path.with_suffix('.json.backup')
                if self.registry_path.exists():
                    shutil.copy(self.registry_path, backup_path)
                raise ValueError(f"Failed to load registry: {e}. Backup saved to {backup_path}")
        else:
            # Initialize empty registry
            self.shows = []

    def save(self) -> None:
        """Save registry to JSON with atomic write."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict
        data = {
            'version': '1.0',
            'shows': [asdict(s) for s in self.shows]
        }

        # Atomic write: tmp → move
        with tempfile.NamedTemporaryFile(
            mode='w',
            dir=self.registry_path.parent,
            delete=False,
            suffix='.tmp'
        ) as tmp_file:
            json.dump(data, tmp_file, indent=2)
            tmp_path = Path(tmp_file.name)

        # Move atomically
        tmp_path.replace(self.registry_path)

    def _validate_schema(self, data: Dict[str, Any]) -> bool:
        """
        Validate registry schema.

        Args:
            data: Registry data dict

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False

        if 'shows' not in data:
            return False

        if not isinstance(data['shows'], list):
            return False

        # Validate each show has required fields
        for show in data['shows']:
            if not isinstance(show, dict):
                return False
            if 'show_id' not in show or 'show_name' not in show:
                return False
            if not isinstance(show.get('seasons', []), list):
                return False

        return True

    def _dict_to_show(self, data: dict) -> Show:
        """Convert dict to Show dataclass."""
        seasons = []
        for season_data in data.get('seasons', []):
            cast = [CastMember(**c) for c in season_data.get('cast', [])]
            episodes = [Episode(**e) for e in season_data.get('episodes', [])]
            seasons.append(Season(
                season_id=season_data['season_id'],
                season_label=season_data['season_label'],
                season_number=season_data['season_number'],
                cast=cast,
                episodes=episodes,
                created_at=season_data.get('created_at', datetime.utcnow().isoformat() + 'Z')
            ))

        return Show(
            show_id=data['show_id'],
            show_name=data['show_name'],
            seasons=seasons,
            created_at=data.get('created_at', datetime.utcnow().isoformat() + 'Z')
        )

    def create_show(self, show_id: str, show_name: str) -> Show:
        """
        Create new show.

        Args:
            show_id: Show ID (e.g., "rhobh")
            show_name: Display name (e.g., "Real Housewives of Beverly Hills")

        Returns:
            Created Show object

        Raises:
            ValueError: If show_id already exists
        """
        if self.get_show(show_id):
            raise ValueError(f"Show '{show_id}' already exists")

        show = Show(show_id=show_id, show_name=show_name)
        self.shows.append(show)
        self.save()
        return show

    def create_season(
        self,
        show_id: str,
        season_number: int,
        season_label: Optional[str] = None
    ) -> Season:
        """
        Create new season.

        Args:
            show_id: Show ID
            season_number: Season number (e.g., 5)
            season_label: Display label (defaults to "Season {number}")

        Returns:
            Created Season object

        Raises:
            ValueError: If show not found or season already exists
        """
        show = self.get_show(show_id)
        if not show:
            raise ValueError(f"Show '{show_id}' not found")

        season_id = f"s{season_number:02d}"

        # Check if season already exists
        if self.get_season(show_id, season_id):
            raise ValueError(f"Season '{season_id}' already exists for show '{show_id}'")

        if season_label is None:
            season_label = f"Season {season_number}"

        season = Season(
            season_id=season_id,
            season_label=season_label,
            season_number=season_number
        )
        show.seasons.append(season)
        self.save()
        return season

    def get_show(self, show_id: str) -> Optional[Show]:
        """Get show by ID."""
        return next((s for s in self.shows if s.show_id == show_id), None)

    def get_season(self, show_id: str, season_id: str) -> Optional[Season]:
        """Get season by ID."""
        show = self.get_show(show_id)
        if not show:
            return None
        return next((s for s in show.seasons if s.season_id == season_id), None)

    def get_or_create_show(self, show_id: str, show_name: str) -> Show:
        """Get existing show or create if not exists."""
        show = self.get_show(show_id)
        if not show:
            show = self.create_show(show_id, show_name)
        return show

    def get_or_create_season(
        self,
        show_id: str,
        season_number: int,
        season_label: Optional[str] = None
    ) -> Season:
        """Get existing season or create if not exists."""
        season_id = f"s{season_number:02d}"
        season = self.get_season(show_id, season_id)
        if not season:
            season = self.create_season(show_id, season_number, season_label)
        return season

    def add_cast_member(
        self,
        show_id: str,
        season_id: str,
        name: str,
        seed_count: int = 0,
        valid_seeds: int = 0
    ) -> CastMember:
        """
        Add or update cast member.

        Args:
            show_id: Show ID
            season_id: Season ID
            name: Cast member name
            seed_count: Total seeds uploaded
            valid_seeds: Valid seeds after validation

        Returns:
            CastMember object

        Raises:
            ValueError: If show/season not found
        """
        season = self.get_season(show_id, season_id)
        if not season:
            raise ValueError(f"Season '{season_id}' not found in show '{show_id}'")

        # Check if cast member already exists
        cast_member = next((c for c in season.cast if c.name == name), None)

        if cast_member:
            # Update existing
            cast_member.seed_count = seed_count
            cast_member.valid_seeds = valid_seeds
        else:
            # Create new
            cast_member = CastMember(
                name=name,
                seed_count=seed_count,
                valid_seeds=valid_seeds
            )
            season.cast.append(cast_member)

        self.save()
        return cast_member

    def add_episode(
        self,
        show_id: str,
        season_id: str,
        episode_id: str,
        video_path: str,
        status: str = "uploaded",
        duration_sec: Optional[float] = None
    ) -> Episode:
        """
        Add episode to season.

        Args:
            show_id: Show ID
            season_id: Season ID
            episode_id: Episode ID
            video_path: Path to video file
            status: Episode status
            duration_sec: Optional duration

        Returns:
            Episode object

        Raises:
            ValueError: If show/season not found or episode already exists
        """
        season = self.get_season(show_id, season_id)
        if not season:
            raise ValueError(f"Season '{season_id}' not found in show '{show_id}'")

        # Check if episode already exists
        if any(e.episode_id == episode_id for e in season.episodes):
            raise ValueError(f"Episode '{episode_id}' already exists")

        episode = Episode(
            episode_id=episode_id,
            video_path=video_path,
            status=status,
            duration_sec=duration_sec
        )
        season.episodes.append(episode)
        self.save()
        return episode

    def get_facebank_dir(self, show_id: str, season_id: str) -> Path:
        """Get facebank directory for show/season."""
        return Path(f"data/facebank/{show_id}/{season_id}")

    def get_cast_dir(self, show_id: str, season_id: str, cast_name: str) -> Path:
        """Get cast directory for seed images."""
        return self.get_facebank_dir(show_id, season_id) / cast_name

    def get_season_bank_path(self, show_id: str, season_id: str) -> Path:
        """Get season bank JSON path."""
        return self.get_facebank_dir(show_id, season_id) / "multi_prototypes.json"

    def get_video_dir(self, show_id: str, season_id: str) -> Path:
        """Get video directory for show/season."""
        return Path(f"data/videos/{show_id}/{season_id}")

    def get_harvest_dir(self, show_id: str, season_id: str, episode_id: str) -> Path:
        """Get harvest directory for episode."""
        return Path(f"data/harvest/{show_id}/{season_id}/{episode_id}")

    def get_outputs_dir(self, show_id: str, season_id: str, episode_id: str) -> Path:
        """Get outputs directory for episode."""
        return Path(f"data/outputs/{show_id}/{season_id}/{episode_id}")


# Global registry instance
_registry: Optional[ShowSeasonRegistry] = None


def get_registry() -> ShowSeasonRegistry:
    """Get global registry instance."""
    global _registry
    if _registry is None:
        _registry = ShowSeasonRegistry()
    return _registry
