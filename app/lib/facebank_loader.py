"""
Facebank loader - loads cast member data from facebank directories.

This module provides functions to load cast member metadata directly from
the facebank, independent of clustering results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from screentime.utils import canonical_show_slug


logger = logging.getLogger(__name__)

def load_facebank_identities(show_id: str, season_id: str, data_root: Path = Path("data")) -> list[dict]:
    """
    Load all seeded identities from facebank for a show/season.

    Returns list of dicts with:
    - person: str (cast member name)
    - seed_count: int (number of seed images)
    - featured_seed: str (filename of featured seed)
    - featured_seed_path: str (full path to featured seed image)
    - bank_conf_median_p25: float (from person_meta, default 0.0)
    - bank_contam_rate: float (from person_meta, default 0.0)
    - n_clusters: int (populated separately, default 0)
    - n_tracks: int (populated separately, default 0)

    Args:
        show_id: Show identifier (e.g., "rhobh")
        season_id: Season identifier (e.g., "s05")
        data_root: Root data directory

    Returns:
        List of person dicts, one per seeded identity
    """
    canonical_slug = canonical_show_slug(show_id)
    facebank_dir = _resolve_facebank_dir(canonical_slug, season_id, data_root)

    if not facebank_dir.exists():
        return []

    identities = []

    # Iterate through all directories in the facebank
    for person_dir in facebank_dir.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name

        # Count seed images
        seed_files = list(person_dir.glob("seed_*.png")) + list(person_dir.glob("seed_*.jpg"))
        seed_count = len(seed_files)

        if seed_count == 0:
            continue  # Skip if no seeds

        # Load person_meta.json if it exists
        person_meta_path = person_dir / "person_meta.json"
        featured_seed = None
        featured_seed_path = None
        bank_conf_median_p25 = 0.0
        bank_contam_rate = 0.0

        if person_meta_path.exists():
            try:
                with open(person_meta_path) as f:
                    person_meta = json.load(f)
                    featured_seed = person_meta.get("featured_seed")
                    featured_seed_path = person_meta.get("featured_seed_path")
                    bank_conf_median_p25 = person_meta.get("bank_conf_median_p25", 0.0)
                    bank_contam_rate = person_meta.get("bank_contam_rate", 0.0)
            except Exception as e:
                # If meta doesn't exist or is malformed, continue with defaults
                pass

        # If no featured seed, pick the first seed alphabetically
        if not featured_seed and seed_files:
            featured_seed = seed_files[0].name
            featured_seed_path = str(seed_files[0])

        identities.append({
            "person": person_name,
            "seed_count": seed_count,
            "featured_seed": featured_seed,
            "featured_seed_path": featured_seed_path,
            "bank_conf_median_p25": bank_conf_median_p25,
            "bank_contam_rate": bank_contam_rate,
            "n_clusters": 0,  # Will be populated from clusters.json
            "n_tracks": 0,  # Will be populated from clusters.json
        })

    # Sort by person name
    identities.sort(key=lambda x: x["person"])

    return identities


def _resolve_facebank_dir(show_slug: str, season_id: str, data_root: Path) -> Path:
    _migrate_facebank_aliases(show_slug, data_root)
    return data_root / "facebank" / show_slug / season_id


def _migrate_facebank_aliases(show_slug: str, data_root: Path) -> None:
    if show_slug != "rhobh":
        return

    alias_slug = "real_housewives_of_beverly_hills"
    alias_dir = data_root / "facebank" / alias_slug
    canonical_dir = data_root / "facebank" / show_slug

    if not alias_dir.exists():
        return

    if not canonical_dir.exists():
        logger.info("Migrating facebank directory %s → %s", alias_dir, canonical_dir)
        alias_dir.rename(canonical_dir)
    else:
        # Preserve existing alias directory by renaming to legacy backup before linking
        if not alias_dir.is_symlink():
            backup_dir = alias_dir.with_name(f"{alias_slug}_legacy")
            logger.warning("Renaming legacy facebank dir %s to %s", alias_dir, backup_dir)
            alias_dir.rename(backup_dir)

    alias_path = data_root / "facebank" / alias_slug
    if alias_path.exists():
        if alias_path.is_symlink():
            return
        # If backup rename happened but alias still present, skip to avoid overwriting manual setup
        return

    try:
        alias_path.symlink_to(canonical_dir, target_is_directory=True)
        logger.info("Created facebank alias symlink %s → %s", alias_path, canonical_dir)
    except OSError as exc:
        logger.warning("Unable to create facebank alias symlink %s: %s", alias_path, exc)


def merge_with_cluster_metrics(
    facebank_identities: list[dict],
    person_metrics: list[dict]
) -> list[dict]:
    """
    Merge facebank identities with cluster metrics.

    Args:
        facebank_identities: List from load_facebank_identities()
        person_metrics: List from clusters.json person_metrics

    Returns:
        Merged list with cluster data added to facebank records
    """
    # Create lookup by person name
    metrics_by_person = {m["person"]: m for m in person_metrics}

    result = []
    for fb_identity in facebank_identities:
        person_name = fb_identity["person"]

        # Start with facebank data
        merged = fb_identity.copy()

        # Overlay cluster metrics if available
        if person_name in metrics_by_person:
            cluster_data = metrics_by_person[person_name]
            merged["n_clusters"] = cluster_data.get("n_clusters", 0)
            merged["n_tracks"] = cluster_data.get("n_tracks", 0)
            # Use cluster metrics if they exist, otherwise keep facebank defaults
            if "bank_conf_median_p25" in cluster_data:
                merged["bank_conf_median_p25"] = cluster_data["bank_conf_median_p25"]
            if "bank_contam_rate" in cluster_data:
                merged["bank_contam_rate"] = cluster_data["bank_contam_rate"]

        result.append(merged)

    return result
