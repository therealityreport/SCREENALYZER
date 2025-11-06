# Phase 1 E2E Validation Checklist

**Status:** Implementation Complete - Ready for Validation
**Date:** 2025-10-28

## Pre-Validation Setup

### 1. Environment Verification
- [ ] Redis is running (`redis-cli ping` returns PONG)
- [ ] Virtual environment activated (`source .venv/bin/activate`)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Test video exists: `/Volumes/HardDrive/SCREENALYZER/data/videos/RHOBH-TEST-10-28.mp4`

### 2. Start RQ Workers

Open 4 terminal windows and run:

```bash
# Terminal 1: Harvest worker
source .venv/bin/activate
python jobs/worker.py harvest.q

# Terminal 2: Inference worker
source .venv/bin/activate
python jobs/worker.py inference.q

# Terminal 3: Tracking worker
source .venv/bin/activate
python jobs/worker.py tracking.q

# Terminal 4: Cluster worker
source .venv/bin/activate
python jobs/worker.py cluster.q
```

## Pipeline Validation

### 3. Run E2E Test Script

```bash
source .venv/bin/activate
python scripts/test_e2e.py
```

**Expected:** Script enqueues job, monitors progress, reports completion

### 4. Verify Pipeline Artifacts

Check that all files exist and have reasonable sizes:

#### Core Outputs (`data/harvest/RHOBH-TEST-10-28/`)
- [ ] `tracks.json` - Track metadata with frame references
- [ ] `clusters.json` - Cluster metadata with quality scores
- [ ] `assist/merge_suggestions.parquet` - Ranked merge suggestions
- [ ] `assist/lowconf_queue.parquet` - Low-confidence clusters

#### Statistics (`data/harvest/RHOBH-TEST-10-28/diagnostics/reports/`)
- [ ] `det_stats.json` - Detection stats with confidence histogram
- [ ] `track_stats.json` - Tracking stats with switch rate
- [ ] `cluster_stats.json` - Clustering stats with variance metrics

### 5. Verify Telemetry

Check `data/diagnostics/telemetry.jsonl` for events:
- [ ] `job_enqueued`, `job_started`, `job_stage_complete`
- [ ] `faces_detected`, `det_conf_hist`, `embeddings_computed`
- [ ] `tracks_built`, `track_switches`, `track_duration_hist`, `stage_time_ms_tracking`
- [ ] `clusters_built`, `cluster_variance`, `suggestions_enqueued`, `lowconf_enqueued`, `stage_time_ms_clustering`

## UI Validation

### 6. Start Streamlit UI

```bash
source .venv/bin/activate
streamlit run app/labeler.py
```

### 7. Review Tab - All Faces Grid
- [ ] Navigate to Review tab
- [ ] Select episode: `RHOBH-TEST-10-28`
- [ ] **Real thumbnails display** (not placeholders)
- [ ] Filters work (All / Low Confidence / High Quality)
- [ ] Search by cluster ID works
- [ ] Pagination works (<50 clusters per page)
- [ ] Assign Name action opens input and saves

### 8. Review Tab - Pairwise Review
- [ ] Switch to "Pairwise Review" mode
- [ ] Suggestions load from `merge_suggestions.parquet`
- [ ] Cluster pairs display side-by-side
- [ ] "Merge" button **performs atomic merge**
- [ ] **Suggestions regenerate** with real centroids (not zeros)
- [ ] "Not Same" and "Skip" buttons work
- [ ] Queue progress indicator shows N of total

### 9. Review Tab - Low-Confidence Queue
- [ ] Switch to "Low-Confidence Queue" mode
- [ ] Clusters from `lowconf_queue.parquet` display
- [ ] Sorted by quality score (lowest first)
- [ ] "Mark as Good" and "Split Cluster" actions record in audit
- [ ] Batch actions available

### 10. Review Tab - Autosave & Undo
- [ ] Perform a merge or assign action
- [ ] Wait ≤30 seconds - autosave should tick
- [ ] Check `review_state.json` exists and updates
- [ ] Click "Undo" button - **action reverses**
- [ ] Undo stack shows up to 10 operations
- [ ] Audit log (`diagnostics/audit.jsonl`) records all actions

### 11. Review Tab - Status Panel
- [ ] Episode Status expander shows metrics
- [ ] Job status displays last job ID and completion %
- [ ] Metrics show: Faces Detected, Tracks Built, Clusters, Merge Suggestions

## Analytics Validation

### 12. Generate Analytics

In Analytics tab:
- [ ] Select episode: `RHOBH-TEST-10-28`
- [ ] **Note:** Must assign names to clusters first for meaningful output
- [ ] Click "Generate Analytics" button
- [ ] Wait ≤10 seconds for completion

### 13. Verify Analytics Outputs (`data/outputs/RHOBH-TEST-10-28/`)
- [ ] `timeline.csv` - Timeline with person_name, start_ms, end_ms, duration_ms, source, confidence
- [ ] `totals.csv` - Per-person totals with total_ms, total_sec, appearances, percent, first_ms, last_ms
- [ ] `totals.parquet` - Same as CSV in Parquet format
- [ ] `totals.xlsx` - 3 sheets: Summary, Timeline, Metadata

### 14. Analytics UI
- [ ] Summary metrics display (People Detected, Total Screen Time, etc.)
- [ ] Top Cast Members table shows data
- [ ] Bar chart renders
- [ ] Timeline preview shows sample intervals
- [ ] **CSV download** button works
- [ ] **Excel download** button works

### 15. Verify Analytics Stats
- [ ] `diagnostics/reports/analytics_stats.json` exists
- [ ] Contains: `intervals_created`, `people_detected`, `episode_duration_ms`, `stage_time_ms_analytics`

## Acceptance Criteria Verification

### Phase 1.4 - Tracking (FR-TRK-1)
- [ ] Stable track IDs across shots
- [ ] Track switch rate within configured cap
- [ ] No negative durations in tracks.json
- [ ] Configurable thresholds in `configs/bytetrack.yaml`

### Phase 1.5 - Clustering (FR-CLS-1, FR-CLS-2)
- [ ] 500 faces cluster in <30 seconds
- [ ] ≥15 merge suggestions generated
- [ ] Top suggestions exceed similarity threshold (0.75)
- [ ] Low-confidence clusters flagged (quality < 0.6)
- [ ] Decision time 8-12s per pair in UI

### Phase 1.6 - Review UI (FR-UI-1 through FR-UI-5)
- [ ] 50 clusters render in <3 seconds
- [ ] Filter/search <500ms response
- [ ] Actions apply instantly (<500ms)
- [ ] Autosave cadence ≤30 seconds
- [ ] Undo stack supports 10 operations
- [ ] Real thumbnails display from video frames
- [ ] Merge/split updates clusters.json atomically

### Phase 1.7 - Analytics (FR-AN-1, FR-AN-2, FR-EXP-1)
- [ ] Analytics generation <10 seconds
- [ ] Export generation <5 seconds
- [ ] Excel/Sheets open cleanly (test by opening files)
- [ ] Interval-merge uses 2-second threshold
- [ ] Co-appearance detection functional
- [ ] Timeline accuracy within ±2s of ground truth (if GT available)

## Final Sign-Off

Once all checkboxes are complete:

1. Update `docs/ACCEPTANCE_MATRIX.csv`:
   - Set FR-TRK-1, FR-CLS-1, FR-CLS-2, FR-UI-1 through FR-UI-5, FR-AN-1, FR-AN-2, FR-EXP-1
   - Change Test column from "Pending" to "✓"

2. Update `docs/MASTER_TODO.md`:
   - Add "Phase 1 E2E validated" to changelog
   - Mark exit criteria as complete for 1.4, 1.5, 1.6, 1.7

3. **Phase 1 Complete!** Ready to proceed to Phase 2 or production hardening.

## Troubleshooting

### Workers not processing jobs
- Check Redis is running: `redis-cli ping`
- Check worker logs in logs/ directory
- Verify queue names match in worker.py and job enqueue calls

### Thumbnails not displaying
- Verify video file exists at expected path
- Check thumbnail cache directory: `data/cache/thumbnails/`
- Check browser console for errors

### Merge/split not working
- Check `diagnostics/audit.jsonl` for action logs
- Verify `clusters.json` has backup (.bak) file
- Check that embeddings.parquet exists for centroid computation

### Analytics fails to generate
- Ensure clusters have assigned names
- Check `diagnostics/reports/analytics_stats.json` for error messages
- Verify tracks.json and clusters.json are valid JSON
