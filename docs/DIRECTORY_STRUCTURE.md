# SCREENALYZER — Repository Structure (Proposed)

> **Goal:** lean, async, reviewer-first repo mapped 1:1 to the ULTIMATE PRD; anti-sprawl; friendly to Codex & your team.

---

## Top-Level Layout

```text
screenalyzer/
├─ app/                         # Streamlit admin UI (review-first)
│  ├─ components/               # Reusable UI widgets (overlay etc.)
│  ├─ lib/                      # UI data helpers (caching, loaders)
│  ├─ labeler.py                # Main Streamlit app
│  └─ analytics.py              # Simple trends/summary views
│
├─ api/                         # Thin API layer (UI <-> queue/workers)
│  ├─ uploads.py                # Chunked upload + resume endpoints
│  ├─ jobs.py                   # Enqueue/cancel/resume/status/ETA
│  ├─ auth.py                   # Login, roles (Admin/Analyst), tokens
│  └─ rest.py                   # (Phase 2) Read-only analytics API
│
├─ jobs/                        # Async orchestrators (Redis + RQ)
│  ├─ worker.py                 # Worker bootstrap & routing
│  └─ tasks/                    # Stage-specific tasks
│     ├─ harvest.py             # Ingest, scene sample, frame IO
│     ├─ detect_embed.py        # RetinaFace + ArcFace (ONNX EPs)
│     ├─ track.py               # ByteTrack + stitching thresholds
│     ├─ cluster.py             # DBSCAN + quality ranking
│     ├─ suggestions.py         # Centroid/ANN merge candidates
│     └─ analytics.py           # Interval-merge timelines & totals
│
├─ screentime/                  # Core CV/ML & analytics library
│  ├─ detectors/
│  │  └─ face_retina.py         # InsightFace RetinaFace wrapper (ONNX)
│  ├─ recognition/
│  │  ├─ embed_arcface.py       # ArcFace embeddings (ONNX)
│  │  ├─ cluster_dbscan.py      # DBSCAN + utilities
│  │  ├─ suggestions.py         # Similarity/centroid logic
│  │  └─ facebank.py            # Parquet Facebank + meta DB
│  ├─ tracking/
│  │  └─ bytetrack_wrap.py      # ByteTrack integration
│  ├─ attribution/
│  │  └─ timeline.py            # Interval merge (definitive screen-time)
│  ├─ diagnostics/
│  │  ├─ audit.py               # JSONL audit trail (who/what/when)
│  │  ├─ retention.py           # TTL/limits; anti-sprawl lifecycle
│  │  └─ validation/            # Sanity checks & QC reports
│  ├─ viz/
│  │  └─ overlay.py             # (Phase 2) overlay JSON/frames
│  ├─ io_utils.py               # I/O helpers (video, parquet, csv)
│  └─ types.py                  # Typed models, enums, config schemas
│
├─ configs/
│  ├─ pipeline.yaml             # Source of truth for thresholds/paths
│  └─ bytetrack.yaml            # Tracking params (IoU, match, etc.)
│
├─ data/                        # Standardized data roots (gitignored)
│  ├─ videos/                   # Uploaded sources (or S3 paths)
│  ├─ harvest/                  # Stage artifacts (manifest, samples)
│  ├─ facebank/                 # Parquet + images (curated)
│  └─ outputs/                  # totals.(csv|parquet), timeline.csv
│
├─ diagnostics/
│  ├─ reports/                  # Auto-generated QC summaries
│  └─ perf/                     # Stage timings & perf snapshots
│
├─ models/
│  └─ weights/                  # ONNX weights (RetinaFace/ArcFace)
│
├─ scripts/
│  ├─ export_totals.py          # CLI export (CSV/XLSX)
│  └─ make_overlays.py          # Overlay video generator (optional)
│
├─ tests/
│  ├─ unit/                     # Fast unit tests
│  ├─ e2e/                      # Upload→review→export happy-path
│  └─ perf/                     # SLA checks (≤20m, ≤10s, ≤2s, ≤5s)
│
├─ AGENTS/                      # Agent definitions (playbooks & registry; see AGENTS/agents.yml)
│  ├─ agents.yml                # Agent registry (names, queues, SLAs)
│  ├─ harvest_agent.md
│  ├─ detect_embed_agent.md
│  ├─ track_agent.md
│  ├─ cluster_agent.md
│  ├─ suggestions_agent.md
│  ├─ manual_add_helper.md
│  ├─ analytics_export_agent.md
│  ├─ retention_agent.md
│  ├─ telemetry_agent.md
│  └─ overlay_agent.md
│
├─ docs/
│  ├─ DIRECTORY_STRUCTURE.md    # (this file)
│  ├─ PRD.md                    # Ultimate PRD
│  ├─ TECHNOLOGY_STACK.md
│  └─ ACCEPTANCE_MATRIX.csv
│
├─ .env.example                 # Minimal runtime config template
├─ pyproject.toml               # Build & tooling config
├─ requirements.txt             # Pinned runtime deps
├─ Makefile                     # Convenience only (not core flows)
└─ README.md
```

---

## Folder Notes (brief)

- **app/**: UI never writes raw files; mutating ops go through **api/**. Lazy thumbnails, autosave, undo (last 10), pairwise review, low-conf queue, status/ETA/resume/cancel.  
- **api/**: Thin façade; validates, enqueues, returns status/ETAs; holds auth (Admin/Analyst).  
- **jobs/**: Only place long-running work happens; **checkpoint/resume** every ~5 min; idempotent outputs.  
- **screentime/**: Single source of truth for CV/ML/analytics; **timeline.py** defines the interval-merge rules.  
- **configs/**: One **pipeline.yaml**; no backups/temps committed.  
- **data/**: Lean artifacts: manifest, selected_samples, totals, optional timeline; thumbnails/debug with TTL.  
- **AGENTS/**: Agent playbooks (below).

---

## AGENTS/ — Suggested Agents (playbooks)

> Source of truth for agent contracts: `AGENTS/agents.yml`.

> Each agent = small, single-purpose worker/scheduler with **inputs, outputs, queue, SLA**.

1) **harvest_agent** — ingest → optional scene segmentation → sampling  
   - **In:** `data/videos/<ep>.mp4`, `configs/pipeline.yaml`  
   - **Out:** `data/harvest/<ep>/manifest.(json|parquet)`, `selected_samples.csv`  
   - **Queue:** `harvest.q` • **SLA:** 30-min ep ≤ 6–8 min sampling

2) **detect_embed_agent** — RetinaFace detect + ArcFace embeddings (ONNX EPs, CoreML→CPU fallback)  
   - **In:** manifest frames • **Out:** embeddings parquet, det stats  
   - **Queue:** `inference.q` • **SLA:** ≥90% frontal recall on test set

3) **track_agent** — ByteTrack tracklets; cross-shot stitching  
   - **Out:** stable tracklets + diverse crops • **Queue:** `tracking.q` • **SLA:** switch rate under configured cap

4) **cluster_agent** — DBSCAN + quality scoring; flag high intra-variance clusters  
   - **Out:** cluster map, quality report • **Queue:** `cluster.q` • **SLA:** 500 faces < 30s

5) **suggestions_agent** — centroid/ANN **merge suggestions** + low-confidence queue  
   - **Out:** prioritized pairs & problem sets • **Queue:** `assist.q` • **SLA:** ≥15 pairs/ep; 8–12s decision target

6) **manual_add_helper** — draw box → embed → insert; update timeline  
   - **Queue:** `assist.q` • **SLA:** ≤2s round-trip feedback

7) **analytics_export_agent** — interval-merge, totals, CSV/XLSX (3 sheets)  
   - **Out:** `data/outputs/<ep>/totals.(csv|parquet)`, `timeline.csv`, `.xlsx`  
   - **Queue:** `analytics.q` • **SLA:** export ≤5s

8) **retention_agent** — enforce TTL, de-dupe, size caps; single `tracks.json`  
   - **Queue:** `ops.q` • **SLA:** daily pass completes; zero orphan artifacts

9) **telemetry_agent** — aggregate per-stage timings, FPS, counts, reviewer actions/100, false-merge rate; alerts  
   - **Queue:** `ops.q` • **SLA:** metrics for every job; alert on SLA breach

10) **overlay_agent** *(Phase 2)* — produce overlay JSON/frames; toggles & bookmarks  
   - **Queue:** `viz.q` • **SLA:** segment-ready overlays < 30s

### Example `AGENTS/agents.yml`

```yaml
agents:
  - name: harvest_agent
    queue: harvest.q
    inputs: [data/videos/<ep>.mp4, configs/pipeline.yaml]
    outputs: [data/harvest/<ep>/manifest.parquet, data/harvest/<ep>/selected_samples.csv]
    sla: "30min episode: ≤ 6–8 minutes sampling"
    retries: 3
    backoff: exponential

  - name: suggestions_agent
    queue: assist.q
    inputs: [facebank.parquet, clusters.json]
    outputs: [assist/merge_suggestions.parquet, assist/lowconf_queue.parquet]
    sla: "≥15 plausible pairs per episode; 8–12s decision target"
    retries: 3
    backoff: jitter
```
