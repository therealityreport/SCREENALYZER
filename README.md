# Screenalyzer

Automated cast screen-time measurement for episodic video with AI-assisted review workflow.

## Overview

Screenalyzer automates cast screen-time tracking by combining face detection, tracking, and clustering with a reviewer-first admin UI. It replaces manual splitting/labeling with an async pipeline that achieves:

- Processing time: ≤30 minutes per 30-minute episode
- Clustering accuracy: ≥90% by episode 3 (auto), ≥98% after review
- Reviewer throughput: ≤25 minutes median by episode 5

## Start Here

**Documentation**

- [PRD (Ultimate Requirements)](docs/PRD.md) — Complete product requirements and feature specifications
- [Solution Architecture](docs/SOLUTION_ARCHITECTURE.md) — System design, components, and data contracts
- [Solution Architecture (Condensed)](docs/SOLUTION_ARCHITECTURE_NOTION.md) — Quick reference version
- [Directory Structure](docs/DIRECTORY_STRUCTURE.md) — Repository layout and anti-sprawl rules
- [Master TODO](docs/MASTER_TODO.md) — Project plan with phases, exit criteria, and tracking
- [Acceptance Matrix](docs/ACCEPTANCE_MATRIX.csv) — Functional/non-functional requirements mapped to tests

**Architecture Diagrams**

- [C4 Context & Containers (Mermaid)](docs/C4_Screenalyzer.mmd)
- [C4 Context & Containers (PlantUML)](docs/C4_Screenalyzer.puml)
- PNG exports available in `docs/`

**Agent Registry**

- [AGENTS/agents.yml](AGENTS/agents.yml) — Async worker agents with queues, SLAs, and I/O contracts

## Quick Start

### Prerequisites

- Python 3.11+
- Redis (for queue/workers)
- Virtual environment tool (venv, conda, etc.)

### Setup

1. Create and activate virtual environment:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your settings (Redis URL, secrets, etc.)
   ```

4. Start Redis (if not already running):
   ```bash
   redis-server
   ```

5. Run linters (optional):
   ```bash
   pip install black==24.4.2 isort==5.13.2 pyright==1.1.379
   black .
   isort .
   pyright
   ```

6. Launch Streamlit UI (when implemented in Phase 1):
   ```bash
   streamlit run app/Home.py
   ```

## Project Structure

```
screenalyzer/
├─ app/                   # Streamlit admin UI
├─ api/                   # API layer (uploads, jobs, auth)
├─ jobs/                  # Async workers & task orchestration
├─ screentime/            # Core CV/ML library
│  ├─ detectors/          # Face detection (RetinaFace)
│  ├─ recognition/        # Embeddings & clustering (ArcFace, DBSCAN)
│  ├─ tracking/           # ByteTrack integration
│  ├─ attribution/        # Screen-time analytics (interval-merge)
│  └─ diagnostics/        # Retention, audit, validation
├─ configs/               # Pipeline & tracking configuration
├─ data/                  # Videos, harvest, facebank, outputs (gitignored)
├─ AGENTS/                # Agent playbooks & registry
├─ docs/                  # Architecture & requirements docs
└─ tests/                 # Unit, E2E, and performance tests
```

See [docs/DIRECTORY_STRUCTURE.md](docs/DIRECTORY_STRUCTURE.md) for complete layout.

## Key Features

**Pipeline**
- Async queue/workers (Redis + RQ)
- Face detection (RetinaFace), embeddings (ArcFace), tracking (ByteTrack)
- DBSCAN clustering with quality scoring
- Checkpoint/resume for long-running jobs

**Review UI**
- All Faces grid with filters & search
- Pairwise Review (merge/split/skip)
- Autosave & Undo (last 10 ops)
- Low-confidence queue & merge suggestions
- Manual Add (draw box on frame)
- Job status/ETA/resume/cancel

**Analytics & Exports**
- Definitive screen-time via interval-merge algorithm
- Co-appearance credit for concurrent appearances
- CSV/XLSX exports (3 sheets: Summary / Timeline / Metadata)

**Retention & Anti-Sprawl**
- TTL for thumbnails/debug artifacts
- Single `tracks.json` policy (no `tracks_fix*` variants)
- Size caps and de-duplication
- See [screentime/diagnostics/RETENTION.md](screentime/diagnostics/RETENTION.md)

## Development Status

**Phase 0 (Foundation)** — Complete
- Repository structure scaffolded
- Configuration and environment files in place
- Anti-sprawl policies defined
- CI/lint configured

**Phase 1 (v1 Core)** — In Progress
- Upload & validation
- Async pipeline & workers
- Detection, tracking, clustering
- Review UI & analytics
- Security & telemetry

**Phase 2+ (Enhancements)** — Planned
- Overlay player
- REST API
- Multi-admin RBAC
- pgvector ANN search
- Model fine-tuning

See [docs/MASTER_TODO.md](docs/MASTER_TODO.md) for detailed plan.

## Testing

Run the test suite (when implemented):
```bash
pytest tests/
```

Performance benchmarks:
```bash
pytest tests/perf/ -v
```

## Configuration

**Pipeline settings:** [configs/pipeline.yaml](configs/pipeline.yaml)
- Video sampling, detection thresholds, tracking parameters
- Clustering method & quality thresholds
- Retention TTLs & artifact limits

**Tracking settings:** [configs/bytetrack.yaml](configs/bytetrack.yaml)
- ByteTrack IoU thresholds, match confidence, buffer size

**Environment:** [.env.example](.env.example)
- Redis connection, secrets, paths, ports

## Contributing

1. Create a feature branch from `main`
2. Make changes following the coding standards (black, isort, pyright)
3. Add/update tests as needed
4. Ensure CI passes: `black --check . && isort --check-only . && pyright`
5. Submit PR with clear description

## License

[Add license information here]

## Support

For questions or issues, see [docs/PRD.md](docs/PRD.md) or contact the team.
