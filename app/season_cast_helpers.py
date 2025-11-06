"""
Helper functions for managing season cast members.
"""

from pathlib import Path
from typing import List


def get_season_cast_names(show_id: str, season_id: str, data_root: Path) -> List[str]:
    """
    Get list of cast member names from season facebank.

    Args:
        show_id: Show ID (e.g., 'rhobh')
        season_id: Season ID (e.g., 's05')
        data_root: Data root path

    Returns:
        List of cast member names
    """
    facebank_dir = data_root / "facebank" / show_id / season_id

    if not facebank_dir.exists():
        return []

    # Get all subdirectories (each is a cast member)
    cast_names = []
    for subdir in facebank_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            # Check if it has seed images
            seed_files = list(subdir.glob("seed_*.png"))
            if seed_files:
                cast_names.append(subdir.name)

    return sorted(cast_names)


def get_season_cast_dropdown_options(show_id: str, season_id: str, data_root: Path) -> List[str]:
    """
    Get dropdown options for season cast (names + special options).

    Args:
        show_id: Show ID
        season_id: Season ID
        data_root: Data root path

    Returns:
        List of options: cast names + "Unknown" + "Add a Cast Member..."
    """
    cast_names = get_season_cast_names(show_id, season_id, data_root)

    options = cast_names.copy()
    options.append("Unknown")
    options.append("Add a Cast Member...")

    return options
