"""Materialize 160×200 thumbnails from SER-FIQ crops."""

import json
import logging
from pathlib import Path
from typing import Dict, Any

from PIL import Image

logger = logging.getLogger(__name__)


def materialize_thumbs(episode_id: str, data_root: Path = Path("data")) -> Dict[str, Any]:
    """
    Ensure all tracks have 160×200 thumbnails.

    Reads track_stills.jsonl and generates thumbs from crops where missing.
    Updates manifest with thumb_path for newly created thumbnails.

    Args:
        episode_id: Episode identifier
        data_root: Root data directory

    Returns:
        Dict with stats: {created, skipped, failed, total}
    """
    logger.info(f"Materializing thumbs for {episode_id}")

    harvest_dir = data_root / "harvest" / episode_id
    manifest_path = harvest_dir / "stills" / "track_stills.jsonl"
    thumbs_dir = harvest_dir / "stills" / "thumbs"

    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return {"created": 0, "skipped": 0, "failed": 0, "total": 0}

    thumbs_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest entries
    entries = []
    with open(manifest_path) as f:
        for line in f:
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    created = 0
    skipped = 0
    failed = 0
    updated_entries = []

    for entry in entries:
        track_id = entry["track_id"]
        thumb_path_str = entry.get("thumb_path")
        crop_path_str = entry.get("crop_path")

        # Check if thumb already exists
        if thumb_path_str:
            thumb_path = Path(thumb_path_str)
            if thumb_path.exists():
                skipped += 1
                updated_entries.append(entry)
                continue

        # Try to create thumb from crop
        if not crop_path_str:
            failed += 1
            logger.warning(f"No crop_path for track {track_id}, cannot create thumb")
            updated_entries.append(entry)
            continue

        crop_path = Path(crop_path_str)
        if not crop_path.exists():
            failed += 1
            logger.warning(f"Crop not found: {crop_path}")
            updated_entries.append(entry)
            continue

        # Generate thumb
        try:
            thumb_path = thumbs_dir / f"{track_id}.jpg"

            with Image.open(crop_path) as img:
                # Resize to 160×200 maintaining aspect ratio with cover
                img_resized = img.resize((160, 200), Image.Resampling.LANCZOS)
                img_resized.save(thumb_path, "JPEG", quality=90)

            # Update entry
            entry["thumb_path"] = str(thumb_path)
            updated_entries.append(entry)
            created += 1

            if created % 50 == 0:
                logger.info(f"Created {created} thumbs...")

        except Exception as exc:
            failed += 1
            logger.error(f"Failed to create thumb for {track_id}: {exc}")
            updated_entries.append(entry)

    # Rewrite manifest with updated thumb_path entries
    if created > 0:
        logger.info(f"Rewriting manifest with {created} new thumb_path entries")
        with open(manifest_path, "w") as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + "\n")

    stats = {
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "total": len(entries),
    }

    logger.info(
        f"Thumb materialization complete: "
        f"{created} created, {skipped} skipped, {failed} failed (total: {len(entries)})"
    )

    return stats
