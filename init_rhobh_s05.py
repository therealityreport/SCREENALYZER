"""
Initialize registry with RHOBH Season 5.

Canonical IDs (locked):
- show_id: rhobh
- show_name: Real Housewives of Beverly Hills
- season_id: s05
- season_number: 5
- season_label: Season 5
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from screentime.models import get_registry

def init_rhobh_s05():
    """Initialize RHOBH Season 5 in registry."""
    registry = get_registry()

    print("Initializing RHOBH Season 5...")

    # Create or get show
    show = registry.get_or_create_show(
        show_id="rhobh",
        show_name="Real Housewives of Beverly Hills"
    )
    print(f"✓ Show: {show.show_name} ({show.show_id})")

    # Create or get season
    season = registry.get_or_create_season(
        show_id="rhobh",
        season_number=5,
        season_label="Season 5"
    )
    print(f"✓ Season: {season.season_label} ({season.season_id})")

    # Create directories
    facebank_dir = registry.get_facebank_dir("rhobh", "s05")
    video_dir = registry.get_video_dir("rhobh", "s05")

    facebank_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    print(f"✓ Created facebank: {facebank_dir}")
    print(f"✓ Created videos: {video_dir}")

    # Display registry
    print("\nRegistry contents:")
    print(f"  Show: {show.show_name}")
    print(f"  Seasons: {len(show.seasons)}")
    for s in show.seasons:
        print(f"    - {s.season_label} (cast: {len(s.cast)}, episodes: {len(s.episodes)})")

    print(f"\n✅ Registry saved to: {registry.registry_path}")

if __name__ == "__main__":
    init_rhobh_s05()
