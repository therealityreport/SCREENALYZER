from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RetentionPolicy:
    thumbnails_ttl_days: int = 14
    debug_ttl_days: int = 7
    max_artifact_bytes: int = 8_000_000_000
    enforce_single_tracks_json: bool = True


def _purge_older_than(root: Path, ttl_days: int) -> list[Path]:
    now = time.time()
    cutoff = now - ttl_days * 86400
    deleted = []
    for p in root.rglob("*"):
        try:
            if p.is_file() and p.stat().st_mtime < cutoff:
                p.unlink(missing_ok=True)
                deleted.append(p)
        except Exception:
            pass
    return deleted


def enforce_single_tracks_json(harvest_dir: Path) -> list[Path]:
    """Keep only canonical tracks.json, delete tracks_fix*.json variants."""
    deleted = []
    for p in harvest_dir.rglob("tracks_*.json"):
        if p.name != "tracks.json":
            try:
                p.unlink(missing_ok=True)
                deleted.append(p)
            except Exception:
                pass
    return deleted


def sweep(data_root: Path, policy: RetentionPolicy) -> dict:
    report = {"deleted": []}
    thumbs = data_root / "harvest"
    debug = data_root / "harvest"
    if thumbs.exists():
        report["deleted"] += [
            str(p) for p in _purge_older_than(thumbs / ".thumbnails", policy.thumbnails_ttl_days)
        ]
    if debug.exists():
        report["deleted"] += [
            str(p) for p in _purge_older_than(debug / "diag", policy.debug_ttl_days)
        ]
    if policy.enforce_single_tracks_json:
        for ep in (data_root / "harvest").glob("*"):
            if ep.is_dir():
                report["deleted"] += [str(p) for p in enforce_single_tracks_json(ep)]
    return report
