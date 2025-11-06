# Artifact Retention Policy

## Purpose

This document describes the lifecycle and retention policy for Screenalyzer artifacts. The goal is to prevent sprawl while maintaining necessary data for analytics and auditing.

## What We Keep

### Permanent Artifacts
- `manifest.(json|parquet)` — episode frame metadata
- `selected_samples.csv` — representative crops for each track
- `totals.(csv|parquet)` — final screen-time summaries
- `timeline.csv` — detailed appearance intervals (optional)
- Tracks: **tracks.json** (single canonical file per harvest); no tracks_fix* variants
- `diagnostics/audit.jsonl` — edit history (who/what/when)
- Compact diagnostics and reports

### Ephemeral Artifacts (TTL)
- **Thumbnails** (`.thumbnails/`): TTL = 14 days (configurable via `pipeline.yaml`)
- **Debug artifacts** (`diag/`): TTL = 7 days (configurable via `pipeline.yaml`)
- **Variant tracks files** (`tracks_fix*.json`): Deleted automatically; only canonical `tracks.json` is kept

## Configuration

Retention settings are controlled via [configs/pipeline.yaml](../../configs/pipeline.yaml):

```yaml
retention:
  thumbnails_ttl_days: 14
  debug_ttl_days: 7
  max_artifact_bytes: 8_000_000_000  # 8 GB soft cap
  enforce_single_tracks_json: true
```

## How It Works

The retention sweep is implemented in [retention.py](retention.py) and includes:

1. **TTL enforcement**: Files older than the configured TTL are automatically deleted
2. **Single tracks.json rule**: Only the canonical `tracks.json` is kept; all `tracks_fix*.json` variants are removed
3. **Size caps**: Total artifact size is monitored and reported (soft cap enforcement)
4. **De-duplication**: Prevents duplicate artifacts from accumulating

## Running the Sweep

The retention agent runs automatically on a schedule (configured in the worker/scheduler). Manual invocation:

```python
from screentime.diagnostics.retention import sweep, RetentionPolicy
from pathlib import Path

policy = RetentionPolicy(
    thumbnails_ttl_days=14,
    debug_ttl_days=7,
    max_artifact_bytes=8_000_000_000,
    enforce_single_tracks_json=True
)

report = sweep(Path("data"), policy)
print(f"Deleted {len(report['deleted'])} files")
```

## Tuning

- **Increase TTL** if reviewers need more time to inspect debug artifacts
- **Decrease TTL** to save disk space (especially for thumbnails on large episodes)
- **Adjust size caps** based on available storage and episode volume
- **Disable single-tracks rule** temporarily if debugging tracking issues (not recommended for production)

## Anti-Sprawl Rules

1. UI never writes files directly — all mutations go through API → queue → workers
2. No `tracks_fix*` variants committed or retained
3. No backups/temps in the data tree (use external backup systems)
4. Compact diagnostics only — verbose debug logs go to `.diag/` with TTL
