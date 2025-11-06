"""
Metadata management for shows, seasons, and cast members.

Uses JSON files for simple, filesystem-based storage consistent with existing architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

from screentime.utils import canonical_show_slug


@dataclass
class Show:
    """Show metadata."""
    name: str  # "RHOBH"
    display_name: str  # "Real Housewives of Beverly Hills"
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Season:
    """Season metadata."""
    show_name: str
    season_number: int
    label: str  # "Season 5" or "S05"
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CastMember:
    """Cast member metadata."""
    show_name: str
    season_number: int
    name: str  # "KIM"
    seed_count: int
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)


class MetadataManager:
    """Manages show/season/cast metadata using JSON files."""

    def __init__(self, data_root: Path = Path("data")):
        self.data_root = data_root
        self.metadata_dir = data_root / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.shows_file = self.metadata_dir / "shows.json"
        self.seasons_file = self.metadata_dir / "seasons.json"
        self.cast_file = self.metadata_dir / "cast_members.json"

        # Initialize files if they don't exist
        self._ensure_files_exist()

    def _ensure_files_exist(self) -> None:
        """Create metadata files if they don't exist."""
        if not self.shows_file.exists():
            self._save_json(self.shows_file, {"shows": []})

        if not self.seasons_file.exists():
            self._save_json(self.seasons_file, {"seasons": []})

        if not self.cast_file.exists():
            self._save_json(self.cast_file, {"cast_members": []})

    def _load_json(self, path: Path) -> dict:
        """Load JSON file."""
        with open(path) as f:
            return json.load(f)

    def _save_json(self, path: Path, data: dict) -> None:
        """Save JSON file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ========== Show Methods ==========

    def list_shows(self) -> List[Show]:
        """List all shows."""
        data = self._load_json(self.shows_file)
        return [Show(**show) for show in data["shows"]]

    def get_show(self, show_name: str) -> Optional[Show]:
        """Get show by name."""
        show_name = canonical_show_slug(show_name)
        shows = self.list_shows()
        for show in shows:
            if show.name == show_name:
                return show
        return None

    def create_show(self, name: str, display_name: str) -> Show:
        """
        Create new show.

        Args:
            name: Short name (e.g., "RHOBH")
            display_name: Full name (e.g., "Real Housewives of Beverly Hills")

        Returns:
            Show object

        Raises:
            ValueError: If show already exists
        """
        name = canonical_show_slug(name)

        if self.get_show(name):
            raise ValueError(f"Show {name} already exists")

        show = Show(
            name=name,
            display_name=display_name,
            created_at=datetime.utcnow().isoformat(),
        )

        data = self._load_json(self.shows_file)
        data["shows"].append(show.to_dict())
        self._save_json(self.shows_file, data)

        return show

    # ========== Season Methods ==========

    def list_seasons(self, show_name: str) -> List[Season]:
        """List all seasons for a show."""
        show_name = canonical_show_slug(show_name)
        data = self._load_json(self.seasons_file)
        return [
            Season(**season)
            for season in data["seasons"]
            if season["show_name"] == show_name
        ]

    def get_season(self, show_name: str, season_number: int) -> Optional[Season]:
        """Get season by show and number."""
        seasons = self.list_seasons(show_name)
        for season in seasons:
            if season.season_number == season_number:
                return season
        return None

    def create_season(self, show_name: str, season_number: int, label: str = None) -> Season:
        """
        Create new season.

        Args:
            show_name: Show name
            season_number: Season number (e.g., 5)
            label: Optional label (e.g., "Season 5"). If not provided, defaults to "S{season_number:02d}"

        Returns:
            Season object

        Raises:
            ValueError: If season already exists or show doesn't exist
        """
        # Check show exists
        show_name = canonical_show_slug(show_name)

        if not self.get_show(show_name):
            raise ValueError(f"Show {show_name} does not exist")

        # Check season doesn't exist
        if self.get_season(show_name, season_number):
            raise ValueError(f"Season {season_number} already exists for {show_name}")

        # Default label
        if label is None:
            label = f"S{season_number:02d}"

        season = Season(
            show_name=show_name,
            season_number=season_number,
            label=label,
            created_at=datetime.utcnow().isoformat(),
        )

        data = self._load_json(self.seasons_file)
        data["seasons"].append(season.to_dict())
        self._save_json(self.seasons_file, data)

        return season

    # ========== Cast Member Methods ==========

    def list_cast_members(self, show_name: str, season_number: int) -> List[CastMember]:
        """List all cast members for a show/season."""
        show_name = canonical_show_slug(show_name)
        data = self._load_json(self.cast_file)
        return [
            CastMember(**cast)
            for cast in data["cast_members"]
            if cast["show_name"] == show_name and cast["season_number"] == season_number
        ]

    def get_cast_member(
        self, show_name: str, season_number: int, cast_name: str
    ) -> Optional[CastMember]:
        """Get cast member by show/season/name."""
        cast_members = self.list_cast_members(show_name, season_number)
        for cast in cast_members:
            if cast.name == cast_name:
                return cast
        return None

    def create_cast_member(
        self, show_name: str, season_number: int, cast_name: str
    ) -> CastMember:
        """
        Create new cast member.

        Args:
            show_name: Show name
            season_number: Season number
            cast_name: Cast member name (e.g., "KIM")

        Returns:
            CastMember object

        Raises:
            ValueError: If cast member already exists or season doesn't exist
        """
        canonical_show = canonical_show_slug(show_name)

        # Check season exists
        if not self.get_season(canonical_show, season_number):
            raise ValueError(f"Season {season_number} does not exist for {canonical_show}")

        # Check cast member doesn't exist
        if self.get_cast_member(canonical_show, season_number, cast_name):
            raise ValueError(
                f"Cast member {cast_name} already exists for {canonical_show} S{season_number}"
            )

        cast = CastMember(
            show_name=canonical_show,
            season_number=season_number,
            name=cast_name,
            seed_count=0,
            created_at=datetime.utcnow().isoformat(),
        )

        data = self._load_json(self.cast_file)
        data["cast_members"].append(cast.to_dict())
        self._save_json(self.cast_file, data)

        return cast

    def update_cast_seed_count(
        self, show_name: str, season_number: int, cast_name: str, seed_count: int
    ) -> None:
        """Update seed count for cast member."""
        data = self._load_json(self.cast_file)
        canonical_show = canonical_show_slug(show_name)

        for cast in data["cast_members"]:
            if (
                cast["show_name"] == canonical_show
                and cast["season_number"] == season_number
                and cast["name"] == cast_name
            ):
                cast["seed_count"] = seed_count
                self._save_json(self.cast_file, data)
                return

        raise ValueError(f"Cast member {cast_name} not found")

    # ========== Facebank Methods ==========

    def get_facebank_path(self, show_name: str, season_number: int, cast_name: str) -> Path:
        """Get facebank directory path for a cast member."""
        canonical_show = canonical_show_slug(show_name)
        return self.data_root / "facebank" / canonical_show / f"S{season_number:02d}" / cast_name

    def save_cast_seeds(
        self,
        show_name: str,
        season_number: int,
        cast_name: str,
        seeds: List[Dict],
    ) -> Path:
        """
        Save seed metadata for cast member.

        Args:
            show_name: Show name
            season_number: Season number
            cast_name: Cast name
            seeds: List of seed dicts with keys: filename, confidence, face_size, embedding

        Returns:
            Path to seeds.json file
        """
        facebank_dir = self.get_facebank_path(show_name, season_number, cast_name)
        facebank_dir.mkdir(parents=True, exist_ok=True)

        seeds_file = facebank_dir / "seeds.json"

        canonical_show = canonical_show_slug(show_name)

        seeds_data = {
            "show_name": canonical_show,
            "season_number": season_number,
            "cast_name": cast_name,
            "seed_count": len(seeds),
            "created_at": datetime.utcnow().isoformat(),
            "seeds": seeds,
        }

        with open(seeds_file, "w") as f:
            json.dump(seeds_data, f, indent=2)

        # Update cast seed count
        self.update_cast_seed_count(canonical_show, season_number, cast_name, len(seeds))

        return seeds_file

    def load_cast_seeds(
        self, show_name: str, season_number: int, cast_name: str
    ) -> Optional[Dict]:
        """Load seed metadata for cast member."""
        facebank_dir = self.get_facebank_path(show_name, season_number, cast_name)
        seeds_file = facebank_dir / "seeds.json"

        if not seeds_file.exists():
            return None

        with open(seeds_file) as f:
            return json.load(f)
