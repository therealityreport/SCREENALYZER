# Suggestions

| Item | Impact | Effort | Priority | Owner | Notes |
| --- | --- | --- | --- | --- | --- |
| Wire `bootstrap_labels_task` into the post-cluster pipeline (or manual trigger) and persist `label_source` markers | High | Medium | P0 | Claude | No caller currently enqueues the task; labels in `clusters.json` lack `label_source`, so we can’t audit bootstrap assignments yet. |
| Align merge suggestions schema between producer and UI consumer | Medium | Low | P0 | Codex | `jobs/tasks/cluster.py:172-180` writes `cluster_a_id/cluster_b_id`, but the emitted parquet exposes `from_cluster_id/to_cluster_id`; harmonising avoids future UI breakage once suggestions populate. |
| Regenerate validation summaries as part of analytics export | Medium | Medium | P1 | Claude | `validation_summary.txt` still claims merge suggestions are missing; hook the generator to the current artifact checks so QA docs stay authoritative. |
| Document SKIP handling and unlabeled clusters in reviewer guide | Medium | Low | P1 | Codex | Cluster `4` stays unlabeled (SKIP) while analytics filters it out; add reviewer guidance so future analysts recognise the workflow. |
| Track densify assumptions in ENV setup (ffmpeg/pyav/onnxruntime flags) | Medium | Medium | P2 | Claude | Needed before implementing the local densify plan described for YOLANDA gaps on Apple Silicon laptops. |
