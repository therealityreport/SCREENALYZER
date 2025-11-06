# SCREENALYZER — Solution Architecture

> **Purpose:** Describe how the Screenalyzer tool is designed, how components interact, and how we meet our product & operational goals (≤30m processing, ≥90% grouping accuracy by ep3, reviewer-first UX). This file complements the PRD and the Acceptance Matrix.

---

## 1) Architectural Goals & SLOs

**Product goals**
- Reviewer time ≤ 30 minutes per 30-minute episode (post-automation).
- ≥ 90% automatic grouping accuracy by episode 3; ≥ 98% after review.
- Simple, repeatable exports (CSV/XLSX) for stakeholders.

**Operational SLOs**
- 30-minute episode processing: ≤ 20 minutes on target hardware.
- Analytics calculation: ≤ 10 seconds; export: ≤ 5 seconds.
- UI page loads: ≤ 2 seconds; keyboard actions: ≤ 500 ms.
- Uptime target: ≥ 99%; safe resume after crash.

**Design principles**
- Lean artifacts (anti-sprawl), async orchestration, idempotent & resumable stages.
- Capability-driven requirements (no vendor lock-in); simple first, extensible later.

---

## 2) High-Level Architecture

```
+-------------------+       +------------------+       +----------------------+
|   Streamlit UI    | <---> |      API         | <---> |     Redis (Queue)    |
|   (app/*)         |       |  (api/*)         |       |  jobs, status, ETA   |
|  - upload/review  |       | - uploads/jobs   |       +----------------------+
|  - pairwise/undo  |       | - auth (RBAC)    |                   |
|  - status/ETA     |       | - (REST Phase 2) |                   v
+---------^---------+       +---------^--------+       +----------------------+
          |                             |              |   Worker Pool        |
          |  read-only analytics        |              |  (jobs/* + screentime)|
          |   + audit logs              |              | - harvest/detect      |
          |                             |              | - track/cluster       |
          |                             |              | - suggestions/analytics|
          |                             |              +----------+-----------+
          |                             |                         |
          v                             v                         v
+-------------------+       +------------------+       +----------------------+
|  Parquet Facebank |       |  Meta DB (lite)  |       |   Data (videos,      |
|  (embeddings,     |       | SQLite/Postgres  |       |   harvest, outputs)  |
|   clusters, etc.) |       | (audit, jobs)    |       |   (data/*, S3 opt.)  |
+-------------------+       +------------------+       +----------------------+
```

**Key ideas**
- **UI never writes files directly**. All mutations go through the API → queue → workers.
- **Redis** holds job states, ETAs, checkpoints; workers are stateless aside from staged outputs.
- **Parquet Facebank** stores embeddings & cluster metadata; a light DB stores audit, RBAC, job records.
- **Lean artifact set** with TTL for heavy debug/thumbnail data.

---

## 3) Components

### 3.1 UI (`app/*`)
- Streamlit admin: Upload, Review (All Faces, Pairwise), Low-Confidence Queue, Merge Suggestions, Manual Add (draw box), Autosave & Undo, Job Status/ETA/Resume/Cancel, Analytics view.
- Lazy-loading thumbnails; keyboard shortcuts; no direct FS writes.

### 3.2 API (`api/*`)
- **uploads.py**: chunked upload + resume, codec/size validation.
- **jobs.py**: enqueue/cancel/resume/status/ETA; publishes progress ticks.
- **auth.py**: Auth + simple RBAC (Admin/Analyst).
- **rest.py** (Phase 2): read-only analytics endpoints for dashboards.

### 3.3 Workers & Tasks (`jobs/*`)
- **worker.py** boots a worker; long-running tasks run here.
- **tasks/harvest.py**: ingest → (optional) SceneDetect → frame sampling → manifest + selected_samples.
- **tasks/detect_embed.py**: RetinaFace detection → ArcFace embeddings (ONNX EPs w/ CoreML→CPU fallback).
- **tasks/track.py**: ByteTrack tracklets + stitching thresholds.
- **tasks/cluster.py**: DBSCAN clustering, quality ranking; flag high intra-variance clusters.
- **tasks/suggestions.py**: centroid/ANN merge suggestions, low-confidence/problem queues.
- **tasks/analytics.py**: interval-merge → totals & timeline; CSV/XLSX export.

### 3.4 Core Library (`screentime/*`)
- **detectors/**: RetinaFace wrapper (InsightFace/ONNX providers).
- **recognition/**: ArcFace embeddings, DBSCAN clustering, suggestions, Facebank API.
- **tracking/**: ByteTrack integration + config thresholds.
- **attribution/**: `timeline.py` — definitive interval-merge screen-time algorithm (rules below).
- **diagnostics/**: audit JSONL, retention TTL/de-dupe, validation checks.
- **viz/**: overlay JSON/frames (Phase 2).

### 3.5 Storage
- **Parquet Facebank**: embeddings, clusters, representative crops metadata; columnar for speed.
- **Meta DB**: SQLite (local) or Postgres (cloud) for audit events, user accounts/roles, job registry.
- **Data root**: `data/videos`, `data/harvest`, `data/facebank`, `data/outputs`; S3/GCS optional.

### 3.6 Local Densify (Gap-Focused High-Recall)
- The global sampling stride stays at **10 fps** (100 ms) for stability; a 30 fps decode is triggered only inside gap windows ≤ 3.2 s (± 300 ms pad) for specific identities.
- Gap spans are decoded at **30 fps**, run through a high-recall detection stack (min confidence ~0.55–0.58, min face 45–60 px, multi-scale 1.0×/1.3×/1.6×/2.0, optional tiling + blur/cut heuristics), and verified with ArcFace thresholds (sim ≥ 0.82, margin ≥ 0.08).
- Verified detections birth ≥ 3–4 frame tracklets that feed the existing per-identity re-ID + timeline merge logic, ensuring `gap_merge_ms_max` and freeze guards remain intact.
- Outputs (embeddings, tracklets, recall stats) update analytics artifacts (`timeline.csv`, `totals.*`) in place, closing under-count gaps while leaving stable cast (KIM/KYLE/LVP) untouched; Local Densify remains optional and targeted to resolve residual misses.

---

## 4) Data Contracts (Core Artifacts)

**manifest.parquet/json (per episode)**
- `episode_id`, `video_path`, `frame_id`, `ts_ms`, `scene_id` (optional), `frame_path`.
- Derivable: sampling stride, stage status.

**selected_samples.csv**
- `track_id`, `ts_ms`, `crop_path`, `quality_score`, `reason` (focus/profile/diversity).

**embeddings.parquet**
- `track_id`, `face_id`, `ts_ms`, `bbox`, `embedding[512]`, `conf`.

**tracks.json**
- `track_id`, `start_ms`, `end_ms`, `count`, `stitch_score`.

**clusters.parquet**
- `cluster_id`, `name` (nullable), `member_count`, `intra_variance`, `rep_crop`, `low_conf`(bool).

**assist/merge_suggestions.parquet**
- `cluster_a`, `cluster_b`, `similarity`, `rank`.

**timeline.csv**
- `person_name`, `start_ms`, `end_ms`, `source` (auto/manual), `confidence`.

**totals.(csv|parquet)**
- `person_name`, `total_ms`, `appearances`, `%_of_episode`, `first_ms`, `last_ms`.

**diagnostics/audit.jsonl**
```json
{"ts":"2025-10-28T12:03:11Z","user":"admin","op":"merge","a":"c12","b":"c34","result":"c12"}
{"ts":"2025-10-28T12:05:02Z","user":"admin","op":"assign","cluster":"c12","name":"LISA"}
```

---

## 5) Definitive Screen-Time Rules

1) **Aggregate detections** into contiguous **appearance intervals**: detections < 2s apart are fused.  
2) **Interval duration** = `end - start`.  
3) **Total** screen time per person = sum of intervals (co-appearance credited to **every** person present).  
4) **Exclusions**: blips < 0.5s; **includes** partial profiles if `conf ≥ 0.7` and duration ≥ 1s.  
5) **Recompute** totals/timeline **incrementally** after merges/splits/assign/manual add.

---

## 6) Workflow Sequences

### 6.1 Upload & Validation
1. UI calls `api/uploads` to create an upload session.
2. Client sends chunks (PATCH) with resume token; API writes to staging; progress/ETA updated in Redis.
3. On completion, job enqueued (`jobs.enqueue(process_ep)`), UI navigates to status page.

### 6.2 Processing Pipeline
1. `harvest_agent`: scene segmentation (optional) + sampling → manifest + selected_samples.
2. `detect_embed_agent`: RetinaFace + ArcFace; provider fallback.
3. `track_agent`: ByteTrack + stitching.
4. `cluster_agent`: DBSCAN + quality; mark low-confidence/problem clusters.
5. `suggestions_agent`: fill Merge Suggestions & Low-Confidence queues.

### 6.3 Review Loop
1. UI All Faces: filter by low-confidence; open Pairwise Review.
2. Reviewer **Merge / Not same / Skip**; Autosave + Undo available.
3. Names assigned; **Manual Add** to fix misses (draw box → embed → insert).

### 6.4 Analytics & Export
1. `analytics_export_agent`: interval-merge → timeline & totals.
2. UI shows summary/trends; user downloads CSV/XLSX (3 sheets).

### 6.5 Retention
1. `retention_agent`: TTL purge of thumbnails/debug; enforce single `tracks.json`.
2. Size caps; de-dupe; archive reports by date.

---

## 7) Configuration & Feature Flags

- `pipeline.yaml`: stride, detection thresholds, tracker parameters, min face size, ANN on/off (Phase 2), sampling policy.
- `configs/presets/RHOBH-TEST-10-28.yaml`: sets `video.sampling_stride_ms = 100` (10 fps baseline) and references Local Densify for identity-specific gap recovery.
- Flags: `use_scene_detect`, `min_confidence`, `include_profiles`, `auto_lowconf_threshold`, `manual_add_enabled`, `retention_ttl_days`.

---

## 8) Scaling & Capacity Planning

- **Concurrency** = `workers × tasks per worker`. Start with W=4 workers; concurrency=1–2 per worker.
- **Queue separation**: `harvest.q`, `inference.q`, `tracking.q`, `cluster.q`, `assist.q`, `analytics.q`, `ops.q` (avoid priority inversion).
- **Horizontal scale**: add workers for inference/cluster-heavy stages; keep Redis single-node (HA if needed).
- **Storage**: local SSD for temp; back up Facebank/outputs; object storage optional for videos.
- **Cost levers**: adjust sampling stride, ANN on/off, thumbnail TTL, export scope.

---

## 9) Observability

- **Metrics**: per-stage times, FPS, faces/track counts, cluster variance, reviewer actions per 100 detections, false-merge rate, queue depth, job ETA accuracy, export times.
- **Logs**: structured JSON; correlate job_id across UI/API/worker.
- **Alerts**: stalled jobs, ETA runaway, high low-confidence %, perf regressions, storage low.

---

## 10) Security & Privacy

- TLS everywhere; passwords hashed (bcrypt/Argon2). Minimal PII (names/images only).
- RBAC: Admin (edit), Analyst (read-only). Token expiration, signed image URLs.
- **Retention**: enforce TTL on heavy artifacts; explicit delete path purges images/manifests/derived analytics.
- Secrets: env-injected; no secrets in logs; rotate regularly.

---

## 11) Failure Modes & Recovery

- **Resumable uploads** (chunk token) and **checkpoint/resume** (every ~5 min).
- Backoff/retry on transient failures (GPU OOM, provider fallback to CPU, DB reconnect).
- Disk-full pause with alert; safe cancel that leaves consistent state.
- Idempotent tasks (based on episode/job IDs and stage markers).

---

## 12) Environments & Deployment

- **Local dev**: single-node Redis + workers; Facebank/outputs on local disk.
- **Single-node prod**: UI+API + Redis + workers on one box; nightly backups.
- **Cloud**: UI/API container(s), Redis (managed), worker autoscaling; object storage for videos/outputs.
- **Migrations**: Phase 2 can add Postgres+pgvector for ANN search without breaking Parquet Facebank.

---

## 13) Roadmap & Extensibility (Phase 2+)

- **Overlay Player**: browser component reads overlay JSON; toggles layers; bookmarks suspect frames.
- **REST API**: read-only analytics for external dashboards.
- **ANN Search (pgvector)**: faster, scalable merge suggestions; lower reviewer ops.
- **Fine-tuned models**: train on curated Facebank to raise auto accuracy; keep ArcFace backbone.

---

## 14) Glossary

- **Facebank**: curated collection of embeddings and image metadata for cast identities.
- **Low-confidence cluster**: cluster flagged by confidence/variance heuristics for priority review.
- **Manual Add**: UI flow to draw a bounding box on a frame to insert a missed face into a cluster.
- **Interval-merge**: algorithm that merges close detections into continuous presence intervals.
- **TTL**: time-to-live (retention policy for heavy artifacts).
