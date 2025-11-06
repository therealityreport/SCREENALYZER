ULTIMATE PRD — Screenalyzer v1 (for your team + Codex)

Checklist (what we will do)
	•	Define a lean, async pipeline (queue/workers) using InsightFace (RetinaFace + ArcFace), ByteTrack, optional YOLOv8, with checkpoint/resume and chunked uploads.
	•	Ship an admin‑first Streamlit UI with autosave, undo, pairwise merges, merge suggestions, low‑confidence queue, Manual Add, status/ETA/resume/cancel.
	•	Implement precise screen‑time math (interval merge) with incremental recompute after edits.
	•	Maintain a Parquet Facebank (initially) + lightweight metadata DB; define a clean artifact lifecycle.
	•	Deliver analytics + CSV/XLSX exports; log telemetry; enforce security/privacy; include acceptance tests and benchmarks.

⸻

1) Overview

Screenalyzer automates cast screen‑time measurement for episodic video. It replaces manual splitting/labeling with an AI‑assisted pipeline and a review‑first admin UI that brings reviewer time ≤30 minutes per episode while achieving ≥90% grouping accuracy by episode 3.

2) Objectives & Success Metrics
	•	Processing time: ≤30 min / 30‑min episode (P0).
	•	Clustering accuracy: ≥85% ep1; ≥90% by ep3 (auto), ≥98% post‑review (P0).
	•	Reviewer throughput: full pass ≤25 min median; <15 manual ops by ep5 (P1).
	•	Uptime: ≥99% monthly (P1).
	•	Exports: CSV/XLSX within 5s for single episode (P0).

3) Scope

In v1: upload (chunked/resume), processing (scene‑aware sampling optional), face detect/embed, track (ByteTrack), cluster (DBSCAN), merge suggestions, Manual Add, pairwise merge/split, assign names, incremental recompute, analytics UI, CSV/XLSX exports, telemetry, basic auth.
Future: Overlay Player, REST API, multi‑admin RBAC + approval workflow, pgvector ANN search, fine‑tuned models, multi‑tenancy, VAD (speaking time).

4) Roles
	•	Admin: full review/edit/export; sees diagnostics.
	•	Analyst (read‑only): view analytics/exports only.
(RBAC models for multi‑admin workflows are Phase 2.)

5) Features & Requirements

5.1 Input & Upload

FR‑INP‑1 Accept MP4 H.264/H.265 up to 90 min / 5 GB.
FR‑INP‑2 Chunked uploads with resume & retry; progress + ETA.
FR‑INP‑3 External cast images (JPEG/PNG ≥200×200).
NFR‑INP‑1 Validation ≤5s; progress updates ≤500ms.
ACCEPT: interrupted upload resumes from last chunk; rejects unsupported codecs with clear error.

5.2 Processing Pipeline

Architecture: Streamlit UI → API thin layer → Queue (Redis + RQ) → Workers. Status polled or pushed (WebSocket).
Stages
	•	Scene/shot segmentation (optional) via SceneDetect to reduce frames on long static scenes.
	•	Face detection/embeddings: RetinaFace (InsightFace) → ArcFace embeddings (ONNX Runtime CoreML/CPU; optional CUDA); min face size policy (e.g., ≥80 px tall or ≥64 px inter‑ocular).
	•	Tracking: ByteTrack to create stable tracklets; sample diverse crops per track (frontal/profile/temporal).
	•	Clustering: DBSCAN/hierarchical on embeddings; auto merge suggestions using centroid/ANN similarity; flag ambiguous clusters (high intra‑variance).
	•	Manual Add: reviewer can draw a box on a frame to insert a missed face → embed → add to cluster.
	•	Checkpoint/resume: save stage progress every 5 min; idempotent re‑runs.
NFR: 30‑min episode processed in ≤15–20 min on target HW; worker safe to resume after crash.

5.3 Facebank & Storage

FR‑FB‑1 Parquet Facebank for embeddings/crops metadata; thin SQLite (local) or Postgres (cloud) for metadata and audit logs.
FR‑FB‑2 Incremental updates; audit JSONL (who/what/when).
FR‑FB‑3 Artifact lifecycle policy (see §9): keep manifest.parquet/json, selected_samples.csv, totals.(csv|parquet), optional timeline.csv, compact diagnostics; TTL for thumbnails/debug; de‑dupe/expire tracks_*.
NFR‑FB‑1 Cluster 500 new faces in ≤30s; query clusters in <200ms.

5.4 Admin Dashboard (Streamlit)

FR‑UI‑1 All Faces grid (cluster cards: name, size, confidence; filters/search/low‑conf first).
FR‑UI‑2 Drill‑down (multi‑select; move/delete/assign; quality overlays).
FR‑UI‑3 Pairwise Review (side‑by‑side; Merge / Not same; queue progress).
FR‑UI‑4 Autosave, Undo (last 10 ops), Keyboard shortcuts (M/S/D/←/→).
FR‑UI‑5 Merge suggestions queue & Problem Cases queue (low‑conf, high variance, tiny cluster size, or conflict).
FR‑UI‑6 Manual Add tool (frame scrub; draw box → embed → insert).
FR‑UI‑7 Job status: progress bars per stage, accurate ETA, Resume/Cancel.
NFR‑UI‑1 Page loads <2s; ops apply <500ms; grids lazy‑load thumbnails.
ACCEPT: 12‑face move in <10s; undo restores prior state; review of 15 merge suggestions in <4 min.

5.5 Analytics & Exports

Screen‑time algorithm (definitive):
	•	Group detections per cast into appearance intervals: detections <2s apart ⇒ same interval.
	•	Interval duration = end − start; sum over intervals ⇒ total screen time.
	•	Co‑appearance: time credited to all present cast concurrently.
	•	Exclusions: <0.5s blips; include partial profiles if confidence ≥0.7 and duration ≥1s.
FR‑AN‑1 Episode summary table (total secs, %, appearances, first/last, confidence).
FR‑AN‑2 Trend view across episodes; bar/line charts (keep v1 lean).
FR‑AN‑3 CSV/XLSX exports (3 sheets: Summary / Timeline / Metadata).
NFR‑AN‑1 Compute analytics in ≤10s; export ≤5s.

5.6 Security, Privacy, Compliance

FR‑SEC‑1 TLS enforced; hashed passwords (bcrypt/Argon2); minimal PII (names, images).
FR‑SEC‑2 Role‑gated access (Admin vs Analyst); signed URLs for images; basic retention policy.
FR‑SEC‑3 Audit trail for all edits/access; dependency scanning; backups (daily) + restore test.
(Extended RBAC, tenants in Phase 2.)

⸻

6) Telemetry & QA
	•	Metrics: per‑stage times, FPS, face counts, cluster counts/variance, reviewer actions per 100 detections, false‑merge rate, re‑open rate, errors/retries.
	•	Dashboards & Alerts: stalled jobs, unidentified >20%, low confidence dominant, SLA breaches.
	•	Release gates: block deploy if P0 tests fail or SLAs regress >10%.

7) Tests & Benchmarks (P0 unless noted)
	•	Upload 2.5GB / 60‑min MP4: chunked resume OK; processing auto‑starts.
	•	E2E auto run (no manual): ≥85% correct grouping on test corpus.
	•	Merge two clusters → totals update correctly; Undo restores.
	•	Split multi‑identity cluster → recompute both groups.
	•	Resume after crash at 60%: identical results.
	•	Analytics export: CSV/XLSX valid; Excel/Sheets open cleanly (durations, %).
	•	Performance: 30‑min ep processed ≤20m; analytics ≤10s; admin pages load <2s.
	•	Accuracy after review: ≥98% cluster correctness (sampled audit).
	•	(P1) Reviewer time full pass ≤25m median; <15 manual ops by ep5.

8) Non‑Functional Targets
	•	Throughput: N parallel episodes via workers; configure W workers & per‑worker concurrency.
	•	Reliability: ≥99% uptime; automatic retry with backoff; checkpoint every 5 min.
	•	Portability: ONNX providers (CoreML/CPU; optional CUDA) for detectors/recognizer.

9) Artifact Lifecycle (anti‑sprawl)
	•	Keep: manifest.(json|parquet), selected_samples.csv, totals.(csv|parquet), optional timeline.csv, compact diagnostics, audit logs.
	•	Ephemeral: thumbnails/debug (TTL 7–14 days) with size caps; single tracks.json only (no tracks_fix*).
	•	No UI direct FS writes: go through API/service; apply optimistic locking.

10) Roadmap

Phase 2: Overlay Player (browser component using pre‑encoded overlay JSON), REST API, multi‑admin RBAC & approvals, pgvector ANN search, model fine‑tuning from curated data.
Phase 3: Multi‑tenancy, cloud auto‑scaling, VAD (speaking time).

⸻

Implementation Notes (mapping to your stack)
	•	Detectors/Embeddings: insightface (RetinaFace + ArcFace, ONNX EPs).
	•	Tracking: ByteTrack wrapper.
	•	Clustering: scikit‑learn DBSCAN; similarity via cosine/ANN (faiss/pgvector later).
	•	Queue: Redis + RQ; workers run pipeline stages; checkpoint to Redis/FS.
	•	Storage: Parquet Facebank + SQLite/Postgres metadata; S3/GCS optional for videos/artifacts.
	•	UI: Streamlit + keyboard bridges; progressive/lazy thumbnails.
	•	Exports: Pandas to CSV/XLSX (openpyxl).
	•	Testing: pytest + synthetic corpora; performance harness with golden baselines.

⸻

If you want this converted into a checklist acceptance matrix (✓/✗ per FR/NFR/AC mapped to modules), say the word and I’ll include it.
