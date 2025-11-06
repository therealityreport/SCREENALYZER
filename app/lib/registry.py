"""
Canonical registry loader for all pages.
"""

from pathlib import Path
import json
import hashlib
import streamlit as st

from screentime.utils import canonical_show_slug, show_display_name


REG = Path("configs/shows_seasons.json")
DATA_ROOT = Path("data")


def get_episode_hash(episode_id: str, data_root: Path | str = DATA_ROOT) -> str:
    """
    Return a stable, content-derived hash that invalidates caches when
    tracks.json, clusters.json, stills/track_stills.jsonl, or pipeline_state.json changes.
    
    Hashes the tuple of present artifact mtimes/sizes. Identity-agnostic.
    Falls back to "none" if no artifacts exist.
    
    Args:
        episode_id: Episode identifier
        data_root: Root data directory (default: DATA_ROOT)
        
    Returns:
        12-character MD5 hash prefix or "none"
    """
    data_root = Path(data_root)
    harvest_dir = data_root / "harvest" / episode_id
    
    # Key artifacts to track
    artifact_paths = [
        harvest_dir / "tracks.json",
        harvest_dir / "clusters.json",
        harvest_dir / "stills" / "track_stills.jsonl",
        harvest_dir / "diagnostics" / "pipeline_state.json",
    ]
    
    # Build hash input from existing artifacts
    hash_parts = [episode_id]
    
    for path in artifact_paths:
        if path.exists():
            try:
                stat = path.stat()
                # Include mtime and size for cache busting
                hash_parts.append(f"{path.name}:{stat.st_mtime}:{stat.st_size}")
            except OSError:
                # Skip if we can't stat
                pass
    
    if len(hash_parts) == 1:
        # No artifacts found, return default
        return "none"
    
    # Compute MD5 hash
    hash_input = "::".join(hash_parts)
    hash_full = hashlib.md5(hash_input.encode()).hexdigest()
    
    return hash_full[:12]



def load_registry() -> dict:
    """
    Load registry from configs/shows_seasons.json.

    Always reads fresh from disk to avoid stale cache.
    Returns empty structure if file doesn't exist or is invalid.
    """
    try:
        if REG.exists():
            return json.loads(REG.read_text())
        return {"shows": []}
    except Exception as e:
        st.warning(f"Registry read error: {e}")
        return {"shows": []}


def save_registry(reg: dict):
    """
    Save registry atomically to configs/shows_seasons.json.

    Marks registry as dirty in session state to force reload.
    """
    REG.parent.mkdir(parents=True, exist_ok=True)
    tmp = REG.with_suffix(".tmp")
    tmp.write_text(json.dumps(reg, indent=2))
    tmp.replace(REG)
    
    # Mark as dirty in session state (if streamlit is available)
    try:
        st.session_state["registry_dirty"] = True
    except (RuntimeError, AttributeError):
        # Streamlit not available or session state not initialized - that's okay
        pass


def get_all_episodes(reg: dict) -> list[tuple[str, str, str]]:
    """
    Get all episodes from registry.

    Returns:
        List of (show_id, season_id, episode_id) tuples
    """
    episodes = []
    for show in reg.get("shows", []):
        show_id = canonical_show_slug(show.get("show_id", ""))
        for season in show.get("seasons", []):
            season_id = season.get("season_id", "")
            for episode in season.get("episodes", []):
                episode_id = episode.get("episode_id", "")
                if episode_id:
                    episodes.append((show_id, season_id, episode_id))
    return episodes


def recover_episodes_from_fs() -> list[str]:
    """
    Scan data/harvest for episodes with clusters.json.

    Returns:
        List of episode IDs found on disk
    """
    harvest_dir = DATA_ROOT / "harvest"
    if not harvest_dir.exists():
        return []

    eps = []
    for ep_dir in sorted(harvest_dir.glob("*")):
        if ep_dir.is_dir() and (ep_dir / "clusters.json").exists():
            eps.append(ep_dir.name)
    return eps


def ensure_episode_in_registry(ep_id: str, show_id: str = "rhobh", season_id: str = "s05"):
    """
    Add episode to registry if it doesn't exist.

    Creates show and season if needed.
    Saves atomically and marks registry dirty.
    """
    show_id = canonical_show_slug(show_id)
    season_id = season_id.lower()

    reg = load_registry()

    # Find or create show
    show = next((s for s in reg["shows"] if canonical_show_slug(s.get("show_id", "")) == show_id), None)
    if not show:
        show = {
            "show_id": show_id,
            "show_name": show_display_name(show_id),
            "seasons": []
        }
        reg["shows"].append(show)
    else:
        show["show_id"] = show_id
        show["show_name"] = show_display_name(show_id)

    # Find or create season
    season = next((s for s in show["seasons"] if s["season_id"].lower() == season_id), None)
    if not season:
        # Extract season number from season_id (e.g., "s05" -> 5)
        season_num = int(season_id.lstrip("s")) if season_id.startswith("s") else 1
        season = {
            "season_id": season_id,
            "season_label": f"Season {season_num}",
            "season_number": season_num,
            "cast": [],
            "episodes": []
        }
        show["seasons"].append(season)
    else:
        season["season_id"] = season_id

    # Add episode if not exists
    if not any(e["episode_id"] == ep_id for e in season["episodes"]):
        season["episodes"].append({
            "episode_id": ep_id,
            "video_path": str(DATA_ROOT / "videos" / show_id / season_id / f"{ep_id}.mp4"),
            "status": "processing"
        })

        save_registry(reg)
        st.toast(f"Recovered episode {ep_id}", icon="✅")


def load_episodes_json() -> dict:
    """
    Load episodes from diagnostics/episodes.json.

    Returns:
        Dict with 'episodes' list, or empty structure if file doesn't exist
    """
    episodes_path = DATA_ROOT / "diagnostics" / "episodes.json"

    if not episodes_path.exists():
        return {"episodes": []}

    try:
        with open(episodes_path, "r") as f:
            return json.load(f)
    except Exception:
        return {"episodes": []}


def get_newest_episode(episodes_data: dict | None = None) -> str | None:
    """
    Get the most recently uploaded episode from diagnostics/episodes.json.

    Args:
        episodes_data: Optional pre-loaded episodes data. If None, loads fresh.

    Returns:
        Episode ID of newest episode, or None if no episodes exist
    """
    if episodes_data is None:
        episodes_data = load_episodes_json()

    episodes = episodes_data.get("episodes", [])
    if not episodes:
        return None

    # Sort by uploaded_at descending
    sorted_eps = sorted(
        episodes,
        key=lambda e: e.get("uploaded_at", ""),
        reverse=True
    )

    return sorted_eps[0].get("episode_id") if sorted_eps else None


def get_default_episode(available_episodes: list[str]) -> str | None:
    """
    Get the default episode to select in dropdowns.

    Prefers:
    1. Current session state episode_id if still valid
    2. Newest episode from diagnostics/episodes.json
    3. First available episode from list

    Args:
        available_episodes: List of episode IDs that are valid choices

    Returns:
        Episode ID to select, or None if no episodes available
    """
    if not available_episodes:
        return None

    # Check if current selection is still valid
    current = st.session_state.get("episode_id")
    if current and current in available_episodes:
        return current

    # Try newest from diagnostics
    newest = get_newest_episode()
    if newest and newest in available_episodes:
        return newest

    # Fallback to first available
    return available_episodes[0]
