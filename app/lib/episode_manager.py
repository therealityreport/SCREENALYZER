"""
Episode Management: Move, Remove, Restore, Rehash operations.

Provides safe episode lifecycle management with audit logging, maintenance mode
gating, and data preservation guarantees.
"""

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.lib.registry import load_registry, save_registry
from screentime.diagnostics.migrations import emit_episode_op_event
from screentime.utils import canonical_show_slug

logger = logging.getLogger(__name__)

DATA_ROOT = Path("data")


@dataclass
class EpisodeMeta:
    """Episode metadata."""
    episode_id: str
    show_id: str
    season_id: str
    season_number: int
    harvest_path: Path
    archived: bool = False
    episode_hash: Optional[str] = None


def _get_episode_paths(episode_id: str) -> Dict[str, Path]:
    """Get all paths for an episode."""
    harvest_dir = DATA_ROOT / "harvest" / episode_id
    return {
        "harvest": harvest_dir,
        "diagnostics": harvest_dir / "diagnostics",
        "manifest": harvest_dir / "manifest.parquet",
        "embeddings": harvest_dir / "embeddings.parquet",
        "tracks": harvest_dir / "tracks.json",
        "clusters": harvest_dir / "clusters.json",
        "stills": harvest_dir / "stills",
        "outputs": DATA_ROOT / "outputs" / episode_id,
    }


def _compute_episode_hash(episode_id: str) -> str:
    """Compute hash of episode's key files for cache busting."""
    paths = _get_episode_paths(episode_id)
    hash_input = f"{episode_id}::"

    # Hash key files if they exist
    for key in ["manifest", "tracks", "clusters"]:
        path = paths[key]
        if path.exists():
            with open(path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:8]
            hash_input += f"{key}={file_hash}::"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]


def _set_maintenance_mode(
    episode_id: str,
    op_type: str,
    enabled: bool,
    progress: Optional[Dict[str, Any]] = None,
) -> None:
    """Set or clear maintenance mode for an episode."""
    paths = _get_episode_paths(episode_id)
    state_file = paths["diagnostics"] / "pipeline_state.json"
    paths["diagnostics"].mkdir(parents=True, exist_ok=True)

    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    if enabled:
        state["maintenance_mode"] = True
        state["active_op"] = {"type": op_type, "episode_id": episode_id}
        state["active_op_progress"] = progress or {"current": 0, "total": 1, "stage": "starting"}
    else:
        state["maintenance_mode"] = False
        state.pop("active_op", None)
        state.pop("active_op_progress", None)

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def list_episodes(
    show: Optional[str] = None,
    season: Optional[int] = None,
    include_archived: bool = False,
) -> List[EpisodeMeta]:
    """
    List all episodes, optionally filtered by show/season.

    Args:
        show: Filter by show ID (canonical slug)
        season: Filter by season number
        include_archived: Include soft-removed episodes

    Returns:
        List of EpisodeMeta objects
    """
    reg = load_registry()
    episodes = []

    for show_data in reg.get("shows", []):
        show_id = canonical_show_slug(show_data.get("show_id", ""))
        if show and show_id != canonical_show_slug(show):
            continue

        for season_data in show_data.get("seasons", []):
            season_num = season_data.get("season_number", 0)
            if season is not None and season_num != season:
                continue

            season_id = season_data.get("season_id", "")

            for ep_data in season_data.get("episodes", []):
                episode_id = ep_data.get("episode_id", "")
                if not episode_id:
                    continue

                archived = ep_data.get("archived", False)
                if archived and not include_archived:
                    continue

                harvest_path = DATA_ROOT / "harvest" / episode_id
                episode_hash = ep_data.get("episode_hash")

                episodes.append(EpisodeMeta(
                    episode_id=episode_id,
                    show_id=show_id,
                    season_id=season_id,
                    season_number=season_num,
                    harvest_path=harvest_path,
                    archived=archived,
                    episode_hash=episode_hash,
                ))

    return episodes


def move_episode(
    episode_id: str,
    dst_show: str,
    dst_season: int,
    actor: str = "user",
    reason: str = "",
) -> None:
    """
    Move episode to a different show/season.

    Preserves all data (manifests, clusters, stills). Updates registry and
    emits audit event. Sets maintenance mode during operation.

    Args:
        episode_id: Episode to move
        dst_show: Destination show ID (canonical slug)
        dst_season: Destination season number
        actor: Who is performing the move
        reason: Why the move is being performed

    Raises:
        ValueError: If episode not found or destination invalid
        RuntimeError: If move operation fails
    """
    dst_show = canonical_show_slug(dst_show)

    # Get current state
    reg = load_registry()
    src_show_id = None
    src_season_num = None
    ep_data = None

    for show in reg.get("shows", []):
        for season in show.get("seasons", []):
            for ep in season.get("episodes", []):
                if ep.get("episode_id") == episode_id:
                    src_show_id = canonical_show_slug(show.get("show_id", ""))
                    src_season_num = season.get("season_number", 0)
                    ep_data = ep
                    break

    if not ep_data:
        raise ValueError(f"Episode {episode_id} not found in registry")

    if src_show_id == dst_show and src_season_num == dst_season:
        raise ValueError("Source and destination are the same")

    # Set maintenance mode
    _set_maintenance_mode(episode_id, "move", True, {
        "current": 0,
        "total": 3,
        "stage": "updating_registry"
    })

    try:
        # Capture before state
        before_state = {
            "show": src_show_id,
            "season": src_season_num,
            "paths": {k: str(v) for k, v in _get_episode_paths(episode_id).items()},
        }

        # Remove from source
        for show in reg["shows"]:
            if canonical_show_slug(show.get("show_id", "")) == src_show_id:
                for season in show["seasons"]:
                    if season.get("season_number") == src_season_num:
                        season["episodes"] = [
                            e for e in season.get("episodes", [])
                            if e.get("episode_id") != episode_id
                        ]

        # Find or create destination show
        dst_show_data = None
        for show in reg["shows"]:
            if canonical_show_slug(show.get("show_id", "")) == dst_show:
                dst_show_data = show
                break

        if not dst_show_data:
            raise ValueError(f"Destination show {dst_show} not found in registry")

        # Find or create destination season
        dst_season_data = None
        for season in dst_show_data["seasons"]:
            if season.get("season_number") == dst_season:
                dst_season_data = season
                break

        if not dst_season_data:
            raise ValueError(f"Season {dst_season} not found in show {dst_show}")

        # Add to destination
        if "episodes" not in dst_season_data:
            dst_season_data["episodes"] = []
        dst_season_data["episodes"].append(ep_data)

        # Save registry
        save_registry(reg)

        # Capture after state
        after_state = {
            "show": dst_show,
            "season": dst_season,
            "paths": {k: str(v) for k, v in _get_episode_paths(episode_id).items()},
        }

        # Emit audit event
        # Invalidate caches after operation

        from app.workspace.common import invalidate_episode_caches

        cache_counts = invalidate_episode_caches(episode_id)


        emit_episode_op_event(
            op_type="move",
            episode_id=episode_id,
            before_state=before_state,
            after_state=after_state,
            actor=actor,
            reason=reason or f"Moved from {src_show_id}/S{src_season_num:02d} to {dst_show}/S{dst_season:02d}",
            metadata={
                "cache_invalidations": sum(cache_counts.values()),
                **cache_counts,
            },
        )

        logger.info(f"Moved episode {episode_id} from {src_show_id}/S{src_season_num:02d} to {dst_show}/S{dst_season:02d}")

    finally:
        # Clear maintenance mode
        _set_maintenance_mode(episode_id, "move", False)


def remove_episode(
    episode_id: str,
    soft: bool = True,
    actor: str = "user",
    reason: str = "",
) -> None:
    """
    Remove episode (soft archive only in Phase 1).

    Marks episode as archived in registry but keeps all data on disk.

    Args:
        episode_id: Episode to remove
        soft: Must be True in Phase 1 (hard delete not implemented)
        actor: Who is performing the removal
        reason: Why the removal is being performed

    Raises:
        ValueError: If episode not found or soft=False
        RuntimeError: If removal fails
    """
    if not soft:
        raise ValueError("Hard delete not supported. Use soft=True for archival.")

    reg = load_registry()
    found = False

    # Set maintenance mode
    _set_maintenance_mode(episode_id, "remove", True, {
        "current": 0,
        "total": 1,
        "stage": "archiving"
    })

    try:
        # Capture before state
        before_state = {"archived": False}

        # Mark as archived
        for show in reg.get("shows", []):
            for season in show.get("seasons", []):
                for ep in season.get("episodes", []):
                    if ep.get("episode_id") == episode_id:
                        ep["archived"] = True
                        ep["archived_at"] = datetime.utcnow().isoformat() + "Z"
                        found = True
                        break

        if not found:
            raise ValueError(f"Episode {episode_id} not found in registry")

        # Save registry
        save_registry(reg)

        # Capture after state
        after_state = {"archived": True}

        # Emit audit event
        # Invalidate caches after operation

        from app.workspace.common import invalidate_episode_caches

        cache_counts = invalidate_episode_caches(episode_id)


        emit_episode_op_event(
            op_type="remove",
            episode_id=episode_id,
            before_state=before_state,
            after_state=after_state,
            actor=actor,
            reason=reason or "Soft archive",
            metadata={
                "cache_invalidations": sum(cache_counts.values()),
                **cache_counts,
            },
        )

        logger.info(f"Archived episode {episode_id}")

    finally:
        # Clear maintenance mode
        _set_maintenance_mode(episode_id, "remove", False)


def restore_episode(
    episode_id: str,
    actor: str = "user",
    reason: str = "",
) -> None:
    """
    Restore a soft-removed episode.

    Clears archived flag and makes episode visible in listings again.

    Args:
        episode_id: Episode to restore
        actor: Who is performing the restore
        reason: Why the restore is being performed

    Raises:
        ValueError: If episode not found
        RuntimeError: If restore fails
    """
    reg = load_registry()
    found = False

    # Set maintenance mode
    _set_maintenance_mode(episode_id, "restore", True, {
        "current": 0,
        "total": 1,
        "stage": "restoring"
    })

    try:
        # Capture before state
        before_state = {"archived": True}

        # Clear archived flag
        for show in reg.get("shows", []):
            for season in show.get("seasons", []):
                for ep in season.get("episodes", []):
                    if ep.get("episode_id") == episode_id:
                        ep["archived"] = False
                        ep.pop("archived_at", None)
                        found = True
                        break

        if not found:
            raise ValueError(f"Episode {episode_id} not found in registry")

        # Save registry
        save_registry(reg)

        # Capture after state
        after_state = {"archived": False}

        # Emit audit event
        # Invalidate caches after operation

        from app.workspace.common import invalidate_episode_caches

        cache_counts = invalidate_episode_caches(episode_id)


        emit_episode_op_event(
            op_type="restore",
            episode_id=episode_id,
            before_state=before_state,
            after_state=after_state,
            actor=actor,
            reason=reason or "Restore from archive",
            metadata={
                "cache_invalidations": sum(cache_counts.values()),
                **cache_counts,
            },
        )

        logger.info(f"Restored episode {episode_id}")

    finally:
        # Clear maintenance mode
        _set_maintenance_mode(episode_id, "restore", False)


@dataclass
class EpisodeHash:
    """Result of rehash operation."""
    episode_id: str
    old_hash: Optional[str]
    new_hash: str
    validated_files: Dict[str, bool]
    errors: List[str]


def rehash_episode(
    episode_id: str,
    actor: str = "user",
    reason: str = "",
) -> EpisodeHash:
    """
    Recompute episode hash and validate files.

    Validates that key files (manifest, tracks, clusters, stills) exist and
    are readable. Recomputes episode_hash for cache busting.

    Args:
        episode_id: Episode to rehash
        actor: Who is performing the rehash
        reason: Why the rehash is being performed

    Returns:
        EpisodeHash with validation results

    Raises:
        ValueError: If episode not found
    """
    reg = load_registry()
    ep_data = None

    # Find episode
    for show in reg.get("shows", []):
        for season in show.get("seasons", []):
            for ep in season.get("episodes", []):
                if ep.get("episode_id") == episode_id:
                    ep_data = ep
                    break

    if not ep_data:
        raise ValueError(f"Episode {episode_id} not found in registry")

    # Set maintenance mode
    _set_maintenance_mode(episode_id, "rehash", True, {
        "current": 0,
        "total": 3,
        "stage": "validating"
    })

    try:
        old_hash = ep_data.get("episode_hash")
        paths = _get_episode_paths(episode_id)
        validated = {}
        errors = []

        # Validate key files
        for key in ["manifest", "embeddings", "tracks", "clusters"]:
            path = paths[key]
            if path.exists():
                try:
                    # Try to read file
                    with open(path, "rb") as f:
                        f.read(1024)  # Read first 1KB to verify
                    validated[key] = True
                except Exception as e:
                    validated[key] = False
                    errors.append(f"{key}: {str(e)}")
            else:
                validated[key] = False

        # Validate stills directory
        stills_dir = paths["stills"]
        if stills_dir.exists():
            try:
                file_count = len(list(stills_dir.glob("**/*")))
                validated["stills"] = file_count > 0
                if file_count == 0:
                    errors.append("stills: directory empty")
            except Exception as e:
                validated["stills"] = False
                errors.append(f"stills: {str(e)}")
        else:
            validated["stills"] = False

        # Compute new hash
        new_hash = _compute_episode_hash(episode_id)

        # Update registry
        ep_data["episode_hash"] = new_hash
        ep_data["last_rehash"] = datetime.utcnow().isoformat() + "Z"
        save_registry(reg)

        # Invalidate caches after successful rehash
        from app.workspace.common import invalidate_episode_caches
        cache_counts = invalidate_episode_caches(episode_id)

        # Emit audit event with cache invalidation metadata
        emit_episode_op_event(
            op_type="rehash",
            episode_id=episode_id,
            before_state={"hash": old_hash, "validated": {}},
            after_state={"hash": new_hash, "validated": validated},
            actor=actor,
            reason=reason or "Recompute hash and validate files",
            metadata={
                "errors": errors,
                "cache_invalidations": sum(cache_counts.values()),
                **cache_counts,
            },
        )

        logger.info(f"Rehashed episode {episode_id}: {old_hash} -> {new_hash}")

        return EpisodeHash(
            episode_id=episode_id,
            old_hash=old_hash,
            new_hash=new_hash,
            validated_files=validated,
            errors=errors,
        )

    finally:
        # Clear maintenance mode
        _set_maintenance_mode(episode_id, "rehash", False)


def purge_all_episodes(
    archive_videos: bool = False,
    actor: str = "user",
    reason: str = "",
) -> Dict[str, Any]:
    """
    Purge all episodes from diagnostics/episodes.json and archive their data.

    Moves all harvest and output data to archive with timestamp. Does NOT touch
    facebank or shows_seasons.json.

    Args:
        archive_videos: Whether to also move video files to archive
        actor: Who is performing the purge
        reason: Why the purge is being performed

    Returns:
        Dict with purge statistics:
            - episodes_purged: Number of episodes archived
            - harvest_archived: List of archived harvest paths
            - outputs_archived: List of archived output paths
            - videos_archived: List of archived video paths (if archive_videos=True)
            - errors: List of error messages

    Raises:
        RuntimeError: If purge operation fails critically
    """
    from datetime import datetime
    import json
    import shutil
    from pathlib import Path

    episodes_json = Path("data/diagnostics/episodes.json")
    archive_root = Path("data/archive/episodes")
    archive_root.mkdir(parents=True, exist_ok=True)

    # Load current episodes
    if not episodes_json.exists():
        return {
            "episodes_purged": 0,
            "harvest_archived": [],
            "outputs_archived": [],
            "videos_archived": [],
            "errors": ["diagnostics/episodes.json not found"],
        }

    try:
        with open(episodes_json, "r") as f:
            episodes_data = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read episodes.json: {e}")

    episodes = episodes_data.get("episodes", [])
    if not episodes:
        return {
            "episodes_purged": 0,
            "harvest_archived": [],
            "outputs_archived": [],
            "videos_archived": [],
            "errors": [],
        }

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stats = {
        "episodes_purged": 0,
        "harvest_archived": [],
        "outputs_archived": [],
        "videos_archived": [],
        "errors": [],
    }

    # Archive each episode
    for ep_data in episodes:
        episode_id = ep_data.get("episode_id")
        if not episode_id:
            continue

        try:
            # Create episode archive directory
            ep_archive_dir = archive_root / f"{episode_id}_{timestamp}"
            ep_archive_dir.mkdir(parents=True, exist_ok=True)

            # Archive harvest data
            harvest_src = DATA_ROOT / "harvest" / episode_id
            if harvest_src.exists():
                harvest_dst = ep_archive_dir / "harvest"
                shutil.move(str(harvest_src), str(harvest_dst))
                stats["harvest_archived"].append(str(harvest_dst))
                logger.info(f"Archived harvest: {harvest_src} -> {harvest_dst}")

            # Archive outputs
            outputs_src = DATA_ROOT / "outputs" / episode_id
            if outputs_src.exists():
                outputs_dst = ep_archive_dir / "outputs"
                shutil.move(str(outputs_src), str(outputs_dst))
                stats["outputs_archived"].append(str(outputs_dst))
                logger.info(f"Archived outputs: {outputs_src} -> {outputs_dst}")

            # Optionally archive videos
            if archive_videos:
                video_path = ep_data.get("video_path")
                if video_path:
                    video_src = Path(video_path)
                    if video_src.exists():
                        video_dst = ep_archive_dir / "videos" / video_src.name
                        video_dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(video_src), str(video_dst))
                        stats["videos_archived"].append(str(video_dst))
                        logger.info(f"Archived video: {video_src} -> {video_dst}")

            stats["episodes_purged"] += 1

            # Emit individual audit event
            emit_episode_op_event(
                op_type="purge",
                episode_id=episode_id,
                before_state={"episode_data": ep_data},
                after_state={"archived_to": str(ep_archive_dir)},
                actor=actor,
                reason=reason or f"Purged to archive at {timestamp}",
                metadata={"timestamp": timestamp, "archive_videos": archive_videos},
            )

        except Exception as e:
            error_msg = f"Failed to archive {episode_id}: {str(e)}"
            stats["errors"].append(error_msg)
            logger.error(error_msg)

    # Clear episodes.json (keep structure but empty episodes list)
    try:
        with open(episodes_json, "w") as f:
            json.dump({"episodes": []}, f, indent=2)
        logger.info("Cleared diagnostics/episodes.json")
    except Exception as e:
        error_msg = f"Failed to clear episodes.json: {str(e)}"
        stats["errors"].append(error_msg)
        logger.error(error_msg)

    # CRITICAL: Also clear episodes from configs/shows_seasons.json (the registry)
    # This is what the Workspace dropdown uses - if we don't clear this, episodes still show!
    try:
        from app.lib.registry import load_registry, save_registry

        reg = load_registry()

        # Clear episodes from all shows/seasons
        for show in reg.get("shows", []):
            for season in show.get("seasons", []):
                season["episodes"] = []

        save_registry(reg)
        logger.info("Cleared episodes from configs/shows_seasons.json registry")
    except Exception as e:
        error_msg = f"Failed to clear shows_seasons.json registry: {str(e)}"
        stats["errors"].append(error_msg)
        logger.error(error_msg)

    # Also clear Phase 2 episode registry files (episodes/*/state.json)
    try:
        episodes_registry_dir = DATA_ROOT / "episodes"
        if episodes_registry_dir.exists():
            import shutil
            # Move to archive instead of deleting
            archive_episodes_dir = archive_root / f"episodes_registry_{timestamp}"
            shutil.move(str(episodes_registry_dir), str(archive_episodes_dir))
            logger.info(f"Archived episode registry: {episodes_registry_dir} -> {archive_episodes_dir}")
    except Exception as e:
        error_msg = f"Failed to archive episodes registry: {str(e)}"
        stats["errors"].append(error_msg)
        logger.error(error_msg)

    # Emit summary audit event
    emit_episode_op_event(
        op_type="purge_all",
        episode_id="*",
        before_state={"episode_count": len(episodes)},
        after_state={"episode_count": 0},
        actor=actor,
        reason=reason or f"Purged all {stats['episodes_purged']} episodes to archive",
        metadata=stats,
    )

    logger.info(f"Purge complete: {stats['episodes_purged']} episodes archived")
    return stats


def delete_video_file(
    episode_id: str,
    actor: str = "user",
    reason: str = "",
) -> Dict[str, Any]:
    """
    Permanently delete the video file for an episode.

    Does NOT require pipeline to be stopped. Simply deletes the video file
    from disk and updates diagnostics.

    Args:
        episode_id: Episode whose video to delete
        actor: Who is performing the deletion
        reason: Why the deletion is being performed

    Returns:
        Dict with deletion results:
            - deleted: True if file was deleted
            - video_path: Path that was deleted (or would have been)
            - file_existed: True if file existed before deletion
            - error: Error message if deletion failed
    """
    from datetime import datetime
    import json
    from pathlib import Path

    # Load episodes.json to find video path
    episodes_json = Path("data/diagnostics/episodes.json")
    video_path = None
    file_existed = False
    deleted = False
    error = None

    if episodes_json.exists():
        try:
            with open(episodes_json, "r") as f:
                episodes_data = json.load(f)

            for ep_data in episodes_data.get("episodes", []):
                if ep_data.get("episode_id") == episode_id:
                    video_path = ep_data.get("video_path")
                    break
        except Exception as e:
            error = f"Failed to read episodes.json: {e}"

    if not video_path:
        error = f"Video path not found for episode {episode_id}"

    # Try to delete the video file
    if video_path:
        video_file = Path(video_path)
        file_existed = video_file.exists()

        if file_existed:
            try:
                video_file.unlink()
                deleted = True
                logger.info(f"Deleted video file: {video_path}")
            except Exception as e:
                error = f"Failed to delete video file: {e}"
                logger.error(error)

    # Emit audit event
    try:
        emit_episode_op_event(
            op_type="delete_video",
            episode_id=episode_id,
            before_state={"video_path": str(video_path) if video_path else None, "existed": file_existed},
            after_state={"deleted": deleted},
            actor=actor,
            reason=reason or "Deleted video file",
            metadata={"error": error} if error else {},
        )
    except Exception:
        pass  # Don't fail if audit logging fails

    return {
        "deleted": deleted,
        "video_path": str(video_path) if video_path else None,
        "file_existed": file_existed,
        "error": error,
    }
