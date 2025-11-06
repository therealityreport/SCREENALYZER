# Changes Applied — 2025-10-28

```diff
diff --git a/docs/SOLUTION_ARCHITECTURE.md b/docs/SOLUTION_ARCHITECTURE.md
@@
-**tracks.parquet**
-- `track_id`, `start_ms`, `end_ms`, `count`, `stitch_score`.
+**tracks.json**
+- `track_id`, `start_ms`, `end_ms`, `count`, `stitch_score`.
```

```diff
diff --git a/docs/SOLUTION_ARCHITECTURE_NOTION.md b/docs/SOLUTION_ARCHITECTURE_NOTION.md
@@
-- `tracks.parquet` — track_id, start_ms, end_ms, count, stitch_score.
+- `tracks.json` — track_id, start_ms, end_ms, count, stitch_score.
```

```diff
diff --git a/docs/PRD.md b/docs/PRD.md
@@
-Architecture: Streamlit UI → API thin layer → Queue (Redis + Celery/RQ) → Workers.
+Architecture: Streamlit UI → API thin layer → Queue (Redis + RQ) → Workers.
@@
-• Queue: Redis + Celery/RQ; workers run pipeline stages; checkpoint to Redis/FS.
+• Queue: Redis + RQ; workers run pipeline stages; checkpoint to Redis/FS.
```

```diff
diff --git a/docs/DIRECTORY_STRUCTURE.md b/docs/DIRECTORY_STRUCTURE.md
@@
-├─ jobs/                        # Async orchestrators (Redis + RQ/Celery)
+├─ jobs/                        # Async orchestrators (Redis + RQ)
@@
-├─ AGENTS/                      # Agent definitions (playbooks & registry)
+├─ AGENTS/                      # Agent definitions (playbooks & registry; see AGENTS/agents.yml)
@@
-## AGENTS/ — Suggested Agents (playbooks)
+## AGENTS/ — Suggested Agents (playbooks)
+
+> Source of truth for agent contracts: `AGENTS/agents.yml`.
@@
-    inputs: [facebank.parquet, clusters.parquet]
+    inputs: [facebank.parquet, clusters.json]
```

```diff
diff --git a/AGENTS/agents.yml b/AGENTS/agents.yml
@@
-  - name: track_agent
-    outputs: [data/harvest/<ep>/tracks.parquet]
+  - name: track_agent
+    outputs: [data/harvest/<ep>/tracks.json]
@@
-  - name: cluster_agent
-    inputs: [data/harvest/<ep>/tracks.parquet, data/facebank]
-    outputs: [data/harvest/<ep>/clusters.parquet, data/harvest/<ep>/quality_report.json]
+  - name: cluster_agent
+    inputs: [data/harvest/<ep>/tracks.json, data/facebank]
+    outputs: [data/harvest/<ep>/clusters.json, data/harvest/<ep>/quality_report.json]
@@
-  - name: suggestions_agent
-    inputs: [data/facebank.parquet, data/harvest/<ep>/clusters.parquet]
+  - name: suggestions_agent
+    inputs: [data/facebank.parquet, data/harvest/<ep>/clusters.json]
@@
-  - name: manual_add_helper
-    inputs: [frame_path, bbox, cluster_id]
-    outputs: [updated embeddings.parquet, updated clusters.parquet]
+  - name: manual_add_helper
+    inputs: [data/harvest/<ep>/manifest.parquet, frame_path, bbox, cluster_id]
+    outputs: [data/harvest/<ep>/clusters.json, data/outputs/<ep>/timeline.csv]
@@
-  - name: analytics_export_agent
-    inputs: [data/harvest/<ep>/clusters.parquet, data/harvest/<ep>/timeline.csv]
+  - name: analytics_export_agent
+    inputs: [data/harvest/<ep>/clusters.json, data/harvest/<ep>/tracks.json]
@@
-  - name: overlay_agent
-    inputs: [data/harvest/<ep>/clusters.parquet, data/videos/<ep>.mp4]
+  - name: overlay_agent
+    inputs: [data/harvest/<ep>/clusters.json, data/videos/<ep>.mp4]
```

```diff
diff --git a/README.md b/README.md
@@
-- [C4 Context & Containers (PlantUML)](docs/C4_Screanalyzer.puml)
+- [C4 Context & Containers (PlantUML)](docs/C4_Screenalyzer.puml)
```

```diff
diff --git a/screentime/diagnostics/RETENTION.md b/screentime/diagnostics/RETENTION.md
@@
-`timeline.csv` — detailed appearance intervals (optional)
-`diagnostics/audit.jsonl` — edit history (who/what/when)
+`timeline.csv` — detailed appearance intervals (optional)
+Tracks: **tracks.json** (single canonical file per harvest); no tracks_fix* variants
+`diagnostics/audit.jsonl` — edit history (who/what/when)
```

```diff
diff --git a/.github/workflows/ci.yml b/.github/workflows/ci.yml
@@
-      - run: pyright
+      - run: pyright
+      - name: Link check (docs)
+        uses: lycheeverse/lychee-action@v1
+        with:
+          args: --no-progress --max-redirects 5 --require-valid-ssl docs/**/*.md README.md
+        env:
+          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

```diff
diff --git a/docs/MASTER_TODO.md b/docs/MASTER_TODO.md
@@
-> Purpose: end‑to‑end plan to deliver Screenalyzer v1, aligned to the Ultimate PRD and Acceptance Matrix. Use this as the single source of truth for scope, owners, sequencing, dependencies, and exit criteria.
+> Purpose: end‑to‑end plan to deliver Screenalyzer v1, aligned to the Ultimate PRD and Acceptance Matrix. Use this as the single source of truth for scope, owners, sequencing, dependencies, and exit criteria.
+
+## Changelog
+
+- Phase-0 doc alignment applied (tracks.json, agents single-source, RQ).
```

```diff
diff --git a/docs/agents.yml b/docs/agents.yml
deleted file mode 100644
```
