"""
Analytics dirty flag management.

Tracks when analytics need to be rebuilt due to clustering changes,
manual edits, or suppression updates.
"""

from pathlib import Path
from datetime import datetime
import json


def mark_analytics_dirty(episode_id: str, data_root: Path, reason: str = "unknown"):
    """
    Mark analytics as dirty (needing rebuild).

    Args:
        episode_id: Episode identifier
        data_root: Data root path
        reason: Reason for marking dirty (for debugging)
    """
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    flag_path = diagnostics_dir / "needs_analytics.flag"

    # Write timestamp and reason
    with open(flag_path, 'w') as f:
        json.dump({
            'marked_dirty_at': datetime.utcnow().isoformat() + 'Z',
            'reason': reason
        }, f, indent=2)


def clear_analytics_dirty(episode_id: str, data_root: Path):
    """
    Clear analytics dirty flag (after successful rebuild).

    Args:
        episode_id: Episode identifier
        data_root: Data root path
    """
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    flag_path = diagnostics_dir / "needs_analytics.flag"

    if flag_path.exists():
        flag_path.unlink()


def is_analytics_dirty(episode_id: str, data_root: Path) -> tuple[bool, str]:
    """
    Check if analytics are dirty (need rebuild).

    Args:
        episode_id: Episode identifier
        data_root: Data root path

    Returns:
        Tuple of (is_dirty, reason)
    """
    diagnostics_dir = data_root / "harvest" / episode_id / "diagnostics"
    flag_path = diagnostics_dir / "needs_analytics.flag"

    if not flag_path.exists():
        return False, ""

    try:
        with open(flag_path) as f:
            data = json.load(f)
            return True, data.get('reason', 'unknown')
    except:
        return True, "flag exists but unreadable"


def get_analytics_freshness(episode_id: str, data_root: Path) -> dict:
    """
    Get analytics freshness status with timestamps.

    Args:
        episode_id: Episode identifier
        data_root: Data root path

    Returns:
        Dict with:
            - is_dirty: bool
            - reason: str
            - clusters_mtime: timestamp (if exists)
            - totals_mtime: timestamp (if exists)
            - is_fresh: bool (totals newer than clusters and not dirty)
    """
    harvest_dir = data_root / "harvest" / episode_id
    outputs_dir = data_root / "outputs" / episode_id

    clusters_path = harvest_dir / "clusters.json"
    totals_csv_path = outputs_dir / "totals.csv"

    is_dirty, reason = is_analytics_dirty(episode_id, data_root)

    result = {
        'is_dirty': is_dirty,
        'reason': reason,
        'clusters_exists': clusters_path.exists(),
        'totals_exists': totals_csv_path.exists(),
        'is_fresh': False
    }

    if clusters_path.exists():
        result['clusters_mtime'] = datetime.fromtimestamp(clusters_path.stat().st_mtime).isoformat()

    if totals_csv_path.exists():
        result['totals_mtime'] = datetime.fromtimestamp(totals_csv_path.stat().st_mtime).isoformat()

    # Check freshness: totals newer than clusters and not dirty
    if not is_dirty and clusters_path.exists() and totals_csv_path.exists():
        totals_mtime = totals_csv_path.stat().st_mtime
        clusters_mtime = clusters_path.stat().st_mtime

        result['is_fresh'] = totals_mtime > clusters_mtime

    return result
