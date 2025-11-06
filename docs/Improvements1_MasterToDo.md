Improvements1_MasterToDo.md (Revised v1.1 — Uniform Pipeline + Detector A/B)

Goal: Ship a scene-agnostic, uniform screen-time pipeline that runs the same steps for every identity, at a stable 10 fps baseline, with local 30 fps densify only in flagged windows; never drops faces in the embedder; optionally fuses face+body under guardrails; and pinpoints bottlenecks via a freeze-tracking diagnostic.
New in v1.1: A/B RetinaFace vs SCRFD (InsightFace) for face detection; no “freeze”/skip per identity; standardize “seconds recovered” counting across all reports.
Acceptance bar: ≤ 4.5 s absolute error per cast member without overrides, or documented detector-limit/off-screen proofs.

⸻

0) Success Criteria
	•	Accuracy: Every principal ≤ 4.5 s abs. error per episode without overrides; if not, attach detector-limit/off-screen proofs (boundary & full-gap identity verification).
	•	Uniformity: No per-identity “freeze/skip” logic; the same entrance recovery and densify policy applies to all cast.
	•	Stability: Timeline total ≤ runtime, 0 overlaps; co-appearance credited; no count-based autolabeling.
	•	Coverage: Entrance Recovery & Local Densify enabled where first-seen is late or gaps exist.
	•	Diagnostics: Freeze-tracking report + Unknown-Face/Unknown-Body minutes emitted per episode.
	•	UX: “Confirm & Stitch Entrance” one-click in labeler; uniform tiles + per-image delete (shipped).

⸻

1) Face Detection @10 fps + Local 30 fps Densify (A/B RetinaFace vs SCRFD)

Objective: Improve face recall (small/profile/occluded) without destabilizing global tracks.
	•	A/B harness for face detectors
	•	Add screentime/detectors/registry.py with a common FaceDetector interface.
	•	Implement RetinaFace (face_retina.py) and SCRFD (face_scrfd.py).
	•	New task jobs/tasks/detector_ab.py: run both detectors on the same decoded frames (10 fps baseline + any densify windows), emit parallel artifacts:
	•	embeddings__retina.parquet, tracks__retina.json, timeline__retina.csv, totals__retina.csv
	•	embeddings__scrfd.parquet, tracks__scrfd.json, timeline__scrfd.csv, totals__scrfd.csv
	•	diagnostics/reports/detector_ab_report.json & detector_ab_summary.md with:
	•	small-face bins (≤80 px, 80–120 px, >120 px), track birth rate, ID quality (sim to facebank), total abs. error by cast, runtime.
	•	Decision rule (configurable)
	•	Winner = minimum total abs error across cast; tie-break → higher small-face recall; final tie-break → lower runtime.
	•	Persist winner in episode and optional per-show preset for future runs.
	•	Keep YOLOv8-person for bodies (unchanged here).

Initial detector config (tune after A/B)

detection_ab:
  enabled: true
  detectors: ["retinaface","scrfd"]
  common:
    min_confidence: 0.70
    min_face_px: 72
    nms_iou: 0.50
  densify_common:
    min_confidence: 0.55
    min_face_px: 40
    nms_iou: 0.40
    scales: [1.0, 1.25, 1.5, 2.0]
    min_consecutive: 4

Acceptance
	•	Both detectors produce valid artifacts on the same frames; no overlaps; totals ≤ runtime.
	•	A/B report present; winner stored in preset; the same policy runs for all identities.

⸻

2) Embedding & Identity (ArcFace-compatible, InsightFace-managed)

Objective: Robust identity across pose/scale/illumination; never drop faces at the embedder.
	•	Make embedder fix default (DONE; verify it’s universal)
	•	embedding.skip_redetect=true: align from detector bbox/5-pt kps; fallback scales [1.0,1.2,1.4].
	•	Persist has_embedding; keep detections even if embedding fails; add retry task if needed.
	•	Multi-face frames embed and flow end-to-end (sampler→tracker→timeline).
	•	Multi-Prototype Identity Bank (pose/scale aware)
	•	New facebank.MultiProtoIdentityBank with pose bins (frontal/¾/profile) & scale bins (small/med/large).
	•	Re-ID scoring → set-to-set (config: max(sim) or mean(topK)); integrate with reid.py.
	•	Assimilate accepted entrance seeds as additional prototypes automatically (no manual patching).

Acceptance
	•	Early-angle looks no longer mismatch later references; bridges succeed when evidence supports it.
	•	Presence is never lost due to embedding failures.

⸻

3) Entrance Recovery (generic & default for every identity)

Objective: Ensure first on-screen seconds are always found.
	•	Run for each identity with first_interval_start_ms>0: pre-first window [first−800, first+800].
	•	If local_densify.enabled, decode at ~30 fps; otherwise 10 fps okay.
	•	Candidates = all detections; embed (fixed path).
	•	Cluster (DBSCAN/HDBSCAN), validate temporal consistency (≥600 ms span & ≥4 frames).
	•	Negative gating: sim_to_seed ≥ 0.72 AND (sim_to_seed − best_other) ≥ 0.06.
	•	Bridge using Top-K set similarity + adjacency ≤ 1000 ms; if rejected, keep entrance segment assigned to that identity (no override).

Acceptance
	•	diagnostics/reports/entrance_audit.json lists seeds/bridges & reasons for every identity with late first-seen.

⸻

4) Local 30 fps Densify (identity/window-scoped)

Objective: Recover seconds only where faces actually exist.
	•	Global baseline remains 10 fps.
	•	For gap windows & entrances of any identity, run small-face recall (multi-scale; optional tiling; optional person-ROI) with strict verify:

local_densify:
  min_confidence: 0.55
  min_face_px: 40
  scales: [1.0, 1.25, 1.5, 2.0]
  tile_size: 512
  min_consecutive: 4
verify:
  min_similarity: 0.86
  second_best_margin: 0.12

	•	Same densify policy applies to all cast; no per-identity skip/freeze.

Acceptance
	•	Seconds recovered in flagged windows without drift; numbers comparable across detectors in A/B.

⸻

5) Timeline Hardening & Policy (applies to everyone)

Objective: Accurate unions; visibility-aware merges; consistent edge handling.
	•	Per-identity gap caps & visibility checks in timeline.py (to prevent over-merge through occlusions).
	•	edge_epsilon_ms default (e.g., 150 ms), overridable per show preset (not ad-hoc per person).
	•	Co-appearance credit independent of prototype purity.

Acceptance
	•	No inflation at cuts; 0.2–0.4 s edge misses reduced consistently across identities.

⸻

6) Freeze-Tracking Diagnostic (QA)

Objective: Separate detector vs. tracker vs. recognizer bottlenecks.
	•	Add jobs/tasks/freeze_tracking.py:
	•	Tracking-only baseline totals (unlabeled).
	•	Recognition-applied totals (labeled).
	•	Emit: detector coverage, ID-switches/min, avg track length, % Unknown, body-only minutes & % later face-confirmed.

Acceptance
	•	Report present per episode; used to choose tuning focus before threshold changes.

⸻

7) Face+Body Fusion (feature-flag; same rules for all)

Objective: Survive occlusions without identity drift.
	•	Link face→body when IoU ≥ T for K consecutive frames (e.g., T=0.5, K=5 @24 fps).
	•	Grace 1–2 s carry after last face; reset at hard cuts; re-verify after long occlusion.
	•	Never invent IDs: body without face → Unknown-Body; always reported.

Acceptance
	•	Recover 1–4 s in back-turn scenes without drift; Unknown-Body minutes logged and reviewable.

⸻

8) Standardize “seconds recovered” across all reports

Objective: One definition so numbers always match.
	•	Utility used by both entrance_audit.json writer and *_status.md:
	1.	Quantize to native fps frame period;
	2.	Union accepted frames → intervals;
	3.	Clamp to window;
	4.	Subtract overlap with existing intervals;
	5.	Sum and round to nearest frame.

Acceptance
	•	entrance_audit.json.seconds_recovered == *_status.md.seconds_recovered.

⸻

9) Config & Flags
	•	configs/pipeline.yaml sections: detection_ab, embedding, entrance, local_densify, per_identity.*, timeline.*, sampling.max_faces_per_frame, fusion.*.
	•	Per-show/episode presets (e.g., configs/presets/RHOBH-S6.yaml) reflect chosen detector winner and timeline thresholds; do not silently change global defaults.
	•	Feature flags: detection_ab.enabled, entrance.enabled, local_densify.enabled, fusion.enabled, freeze_tracking.enabled.

⸻

10) Testing & Rollout

Unit/Integration
	•	Embedder: multi-face crops → embeddings (no re-detect).
	•	Entrance: seed→bridge on synthetic A/B dialog clip.
	•	Small-face recall: accept ≤100 px faces with strict verify.
	•	Fusion: IoU+K-frame link & timeout; cut resets.
	•	Detector A/B: parallel outputs on the same frames, identical decode.

Canary Episodes
	•	RHOBH-TEST-10-28 + two additional episodes (different lighting/shot styles).
	•	Verify ≤ 4.5 s per identity or attach proofs; compare detector winners per episode.

Rollout
	•	Enable Detector A/B, Entrance Recovery, Local Densify, Freeze-Tracking by default.
	•	Enable Fusion per show behind a flag; monitor drift metrics.

Back-out
	•	Flags exist to disable densify/fusion/A-B quickly if drift or FPR spikes.

⸻

11) Ownership, Effort & Order (suggested)

Workstream	Owner	Effort	Depends
Detector A/B (RetinaFace & SCRFD + report)	Claude	2–3d	—
Embedder default (skip-redetect, retry)	Claude	0.5d	—
Local Densify (identity/window-scoped)	Claude	1–2d	Detector
Entrance Recovery (default for all)	Claude	1d	Embedder
Multi-Prototype Identity Bank	Claude	2–3d	Embedder
Freeze-Tracking diagnostic	Claude	0.5–1d	—
Timeline hardening (caps/visibility/epsilon)	Claude	0.5–1d	—
Fusion (flagged)	Claude	2–4d	YOLO-person
Standardize seconds-recovered function	Claude	0.5d	—
Labeler: Confirm & Stitch Entrance	Claude	0.5d	Entrance


⸻

12) Acceptance Checklist (per episode)
	•	≤ 4.5 s abs. error for each cast or detector-limit/off-screen proofs attached.
	•	A/B report shows detector choice and rationale; chosen detector persisted in preset.
	•	Unknown-Face/Unknown-Body minutes reported; freeze-tracking panel present.
	•	Telemetry healthy (ID-switches/min, overlaps=0, totals ≤ runtime).
	•	Reports agree on recovered seconds (standard function used).

⸻

Appendix — Practical Starting Values
	•	RetinaFace/SCRFD: min_conf=0.70, min_face_px=72, nms_iou=0.50.
	•	Local Densify: min_conf=0.55, min_face_px=40, scales [1.0,1.25,1.5,2.0], min_consecutive=4, nms_iou=0.40.
	•	Timeline: edge_epsilon_ms=150 (can 180–200 in presets if needed), visibility checks (min_interval_frames, min_visible_frac) per show.
	•	Re-ID/Facebank: set-to-set scoring; negative gating (sim_to_seed − best_other)≥0.06; topK=5.

⸻

Note: This file supersedes the prior version; it removes all “freeze/skip” language and adds the RetinaFace vs SCRFD A/B plan. Use this as the canonical runbook for the next implementation sprint.