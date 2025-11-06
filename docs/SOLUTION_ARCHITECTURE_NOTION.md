# Screenalyzer — Solution Architecture (Notion)

**Goals**
- Reviewer time ≤ 30m per 30m episode; automatic grouping ≥ 90% by episode 3 (≥ 98% after review).
- Exports: CSV/XLSX; async, resumable, lean artifacts.

**Core Ideas**
- UI never writes files directly. All mutations go API → Queue → Workers.
- Redis keeps job state/ETA/checkpoints. Workers are idempotent and resumable.
- Parquet Facebank for embeddings/clusters; small DB for audit/RBAC/jobs.
- Minimal artifacts + TTL for thumbnails/debug.

**Key Pieces**
- **UI (Streamlit):** Upload, Review (All Faces, Pairwise), Low-Conf Queue, Merge Suggestions, Manual Add, Autosave/Undo, Status/ETA/Resume/Cancel, Analytics.
- **API:** uploads (chunked/resume), jobs (enqueue/status/cancel/resume), auth (Admin/Analyst), (Phase 2) read-only REST analytics.
- **Workers:** harvest → detect+embed → track → cluster → suggestions → analytics (interval-merge). Checkpoint every ~5 min.
- **Storage:** Parquet Facebank + SQLite/Postgres meta; data roots under `data/*` (videos/harvest/facebank/outputs).

**Data Contracts (examples)**
- `manifest.parquet/json` — episode frames/ts/scene_id (optional).
- `selected_samples.csv` — track_id, ts_ms, crop_path, quality, reason.
- `embeddings.parquet` — embedding[512], bbox, conf.
- `tracks.json` — track_id, start_ms, end_ms, count, stitch_score.
- `clusters.parquet` — cluster_id, name, member_count, intra_variance, rep_crop, low_conf.
- `assist/merge_suggestions.parquet` — cluster_a, cluster_b, similarity, rank.
- `timeline.csv` — person_name, start_ms, end_ms, source, confidence.
- `totals.(csv|parquet)` — person_name, total_ms, appearances, %, first_ms, last_ms.
- `diagnostics/audit.jsonl` — append-only edit log.

**Screen-Time Rules**
1) Fuse detections < 2s apart into a continuous interval.  
2) Total = sum of intervals.  
3) Co-appearance credits all present persons concurrently.  
4) Exclude blips < 0.5s; include partials if conf ≥ 0.7 and duration ≥ 1s.  
5) Recompute analytics incrementally after merges/splits/assign/manual add.

**Scaling & Operations**
- Queues: harvest.q, inference.q, tracking.q, cluster.q, assist.q, analytics.q, ops.q.
- Start with 4 workers; add capacity to inference/cluster heavy stages.
- Metrics: per-stage times, FPS, counts, cluster variance, reviewer actions/100, false-merge rate, queue depth, ETA accuracy.
- Alerts: stalled jobs, SLA breaches, low storage, excessive low-confidence.

**Security & Privacy**
- TLS; hashed passwords; Admin vs Analyst roles; signed URLs; retention TTLs; explicit delete path that purges images/manifests/derived analytics.
