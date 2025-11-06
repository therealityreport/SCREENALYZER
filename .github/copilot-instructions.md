# Copilot Instructions for Screenalyzer

## Project Overview
Screenalyzer is an AI-assisted video analysis tool that automates cast screen-time measurement for episodic TV shows. It combines face detection, tracking, and clustering with a reviewer-first admin UI.

## Architecture Essentials

### Core Pipeline Flow
1. **Harvest** (`jobs/tasks/harvest.py`) → video sampling at 10fps baseline + scene detection
2. **Detect/Embed** (`jobs/tasks/detect_embed.py`) → RetinaFace detection + ArcFace embeddings  
3. **Track** (`jobs/tasks/track.py`) → ByteTrack tracklets with re-ID stitching
4. **Cluster** (`jobs/tasks/cluster.py`) → DBSCAN clustering with quality scoring
5. **Suggestions** (`jobs/tasks/suggestions.py`) → merge recommendations + low-confidence queues
6. **Analytics** (`jobs/tasks/analytics.py`) → interval-merge algorithm for definitive screen-time

### Key Data Contracts
- **Episode structure**: `data/harvest/{episode_id}/` contains all processing artifacts
- **Core artifacts**: `manifest.parquet`, `embeddings.parquet`, `tracks.json`, `clusters.json`
- **Single source policy**: One `tracks.json` per episode (no variants), enforced by retention
- **Parquet facebank**: Columnar storage for embeddings/clusters optimized for speed

### UI Architecture (Streamlit)
- **Entry point**: `app/Home.py` (not `labeler.py` which is the review interface)
- **Review workflows**: All Faces → Pairwise Review → Analytics (thumbnail-first design)
- **Navigation system**: Uses `st.session_state` routing for nested gallery views
- **Mutator API**: `app/lib/mutator_api.py` provides cached workspace state management

## Critical Patterns

### Configuration Hierarchy
```yaml
configs/pipeline.yaml     # Main config with nested sections (detection, clustering, timeline)
configs/bytetrack.yaml    # ByteTrack-specific tracking parameters  
configs/presets/          # Episode-specific overrides
```
Always load via `screentime.config` module, never direct YAML parsing.

### Async Job System
- **Queue separation**: `harvest.q`, `inference.q`, `tracking.q`, `cluster.q`, `assist.q`, `analytics.q`
- **Checkpoint/resume**: Jobs save progress to Redis, resumable on crash
- **Maintenance mode**: Episode-level locking during operations (see `app/lib/episode_manager.py`)

### Testing Infrastructure
```bash
# Unit tests
pytest tests/unit/

# Integration tests (require test data setup)
pytest tests/test_episode_manager_integration.py

# E2E validation (full pipeline)
bash scripts/run_e2e_validation.sh
```

### Local Densify (Gap Recovery)
- **Purpose**: High-recall 30fps decode only for gaps ≤3.2s in existing tracks
- **Triggered**: Via `local_densify.enabled=true` in pipeline config
- **Scope**: Identity-specific, leaves stable cast (90%+ confidence) untouched

## Development Workflows

### Starting the System
```bash
# Environment setup
source .venv/bin/activate
pip install -r requirements.txt

# Start Redis (required for async jobs)
redis-server

# Launch UI
streamlit run app/Home.py

# Worker processes (separate terminals)
python jobs/worker.py harvest.q
python jobs/worker.py inference.q
```

### Code Standards
- **Linting**: `black . && isort . && pyright` (configs in `pyproject.toml`)
- **Imports**: Relative imports within packages, absolute from project root
- **Logging**: Structured JSON via `screentime.diagnostics.telemetry`

### Debugging Pipeline Issues
1. Check `data/harvest/{episode}/diagnostics/pipeline_state.json` for stage status
2. Review queue-specific logs in `logs/` directory  
3. Use `python scripts/apply_season_aware_fixes.py` for infrastructure validation
4. Analytics debug info in `data/outputs/{episode}/analytics_debug.json`

## File Modification Guidelines

### When editing UI components:
- **Review pages**: Use `app/review_pages.py` for dedicated gallery views
- **State management**: Leverage `WorkspaceMutator` for cached data access
- **Navigation**: Set `st.session_state.navigation_page` for routing

### When editing pipeline tasks:
- **Job IDs**: Always include in logging for traceability
- **Progress updates**: Use `update_progress(stage, percent)` pattern
- **Error handling**: Distinguish transient (retry) vs permanent failures

### When editing clustering logic:
- **Constraints**: Use `screentime.clustering.constraints` for must-link/cannot-link
- **Quality scoring**: Leverage existing confidence thresholds in `configs/pipeline.yaml`
- **Cluster mutations**: Always use `ClusterMutator` API to maintain state consistency

## Common Gotchas

1. **Thumbnail fallback**: UI prefers `assets/thumbnails_map.json` over generated thumbnails
2. **Episode ID normalization**: Always use `screentime.utils.normalize_episode_id()`
3. **Path handling**: Use `pathlib.Path` consistently, avoid string concatenation
4. **Streamlit state**: Key UI state with episode-specific prefixes to avoid collisions
5. **Redis job data**: Serialize complex objects as JSON, not pickle

## Integration Points

### External dependencies:
- **ONNX providers**: CoreML → CPU fallback for embeddings/detection
- **Video codecs**: ffmpeg via opencv-python for frame extraction
- **Scene detection**: Optional PySceneDetect integration

### Export formats:
- **CSV/XLSX**: 3-sheet format (Summary/Timeline/Metadata) via `jobs/tasks/analytics.py`
- **Parquet**: Columnar storage for embeddings and cluster metadata
- **JSONL**: Audit logs and telemetry events

This project emphasizes **reviewer-first UX**, **anti-sprawl policies**, and **async resilience**. When in doubt, prioritize data consistency and user workflow efficiency over raw performance.