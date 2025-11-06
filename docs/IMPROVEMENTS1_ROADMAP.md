# Implementation Roadmap - Next Session (5-6 Hours)

**Objective**: Achieve 7/7 identities PASS (≤4.5s absolute error) with NO overrides, same pipeline for all cast, RetinaFace locked, 10fps baseline maintained.

**Current Status**: 2/7 PASS (YOLANDA 0.00s, KIM +1.50s)

**Target Status**: 7/7 PASS (all ≤4.5s)

**Session Budget**: 5-6 hours total

---

## Phase 1: Streamlit Key Fix (30 min)

**Problem**: DuplicateWidgetID errors in "View All Tracks" gallery

**Solution**: Apply `wkey()` pattern to all widgets in gallery loops

**Reference**: [docs/STREAMLIT_KEYS_FIX.md](STREAMLIT_KEYS_FIX.md)

### Tasks:

1. **Apply wkey() to Images** (10 min)
   - [ ] File: `app/labeler.py` (line ~1180)
   - [ ] Find: `st.image(img)`
   - [ ] Replace: `st.image(img, key=wkey("img", episode_id, cluster_id, track_id, frame_id, ts_ms, idx))`

2. **Apply wkey() to Trash Buttons** (10 min)
   - [ ] File: `app/labeler.py` (line ~1190)
   - [ ] Find: `key = f"delete_frame_c{cluster_id}_t{track_id}_f{frame_id}"`
   - [ ] Replace: `key = wkey("del", episode_id, cluster_id, track_id, frame_id, ts_ms, tile_idx)`

3. **Apply wkey() to Move/Split Buttons** (5 min)
   - [ ] File: `app/labeler.py` (lines ~1240, ~1260)
   - [ ] Add episode_id and timestamp to key seeds

4. **Test Gallery** (5 min)
   - [ ] Navigate to "View All Tracks"
   - [ ] Verify NO DuplicateWidgetID errors in console
   - [ ] Verify trash button deletes correct frame
   - [ ] Verify all tiles render uniformly

**Acceptance**:
- ✅ NO DuplicateWidgetID errors in Streamlit console
- ✅ All gallery tiles uniform 160×160 px
- ✅ Per-image trash deletes single frame, triggers prototype rebuild

**Time**: 30 minutes

---

## Phase 2: Densify Threshold Tuning - 2 Pass (60 min)

**Problem**: min_face_px=72 filters all small faces before tracking → 5-7s deficits for RINNA, EILEEN, BRANDI

**Solution**: 2-pass densify with conservative then aggressive thresholds

**Reference**: [docs/CONFIG_UPDATES_DENSIFY.md](CONFIG_UPDATES_DENSIFY.md)

### Tasks:

1. **Update Config** (10 min)
   - [ ] File: `configs/pipeline.yaml`
   - [ ] Add `local_densify` section (conservative pass)
   - [ ] Add `local_densify_pass2` section (aggressive pass)
   - [ ] Set pass 1: min_conf=0.58, min_face_px=44
   - [ ] Set pass 2: min_conf=0.50, min_face_px=36 (conditional on still >4.5s)

2. **Implement Pass 1 Logic** (25 min)
   - [ ] File: `screentime/pipeline/local_densify.py` (line ~50)
   - [ ] Load pass 1 config
   - [ ] Run densify with conservative thresholds
   - [ ] Generate tracklets, update tracks.json
   - [ ] Log `densify_audit_pass1.json` with `seconds_recovered` per identity

3. **Implement Pass 2 Gate** (15 min)
   - [ ] After pass 1, compute delta table
   - [ ] For identities still >4.5s error, run pass 2 with aggressive thresholds
   - [ ] Log `densify_audit_pass2.json`
   - [ ] Combine pass 1 + pass 2 contributions

4. **Run on RHOBH-TEST-10-28** (5 min)
   - [ ] Command: `python jobs/tasks/run_densify_two_pass.py RHOBH-TEST-10-28`
   - [ ] Verify tracklets created
   - [ ] Check delta table improvement

5. **Re-run Analytics** (5 min)
   - [ ] Re-run timeline merge
   - [ ] Re-generate delta_table.csv
   - [ ] Verify expected 2-4s recovery across RINNA, EILEEN, BRANDI

**Acceptance**:
- ✅ Pass 1 recovers 1-2s across 3-4 identities
- ✅ Pass 2 (if triggered) recovers additional 1-2s
- ✅ Total densify recovery: 2-4s
- ✅ Delta table shows 4-5/7 PASS after densify

**Expected Impact**:
- RINNA: 30.08s → ~26s (pass 1: -2s, pass 2: -1.5s)
- EILEEN: 14.42s → ~11s (pass 1: -1.5s, pass 2: -1.5s)
- BRANDI: 6.59s → ~9s (pass 1: +1.5s, pass 2: +0.8s) - still undercount, needs more
- LVP: 3.17s → ~2.2s (pass 1: -0.5s, pass 2: -0.3s)

**Time**: 60 minutes

---

## Phase 3: Identity-Guided Recall (120 min)

**Problem**: Residual gaps remain after densify (e.g., BRANDI -3.43s undercount)

**Solution**: Use facebank prototypes to target-search specific identities in gap windows

**Reference**: Entrance recovery architecture (adapt for mid-gaps)

### Tasks:

1. **Create Identity-Guided Recall Module** (60 min)
   - [ ] File: `jobs/tasks/identity_guided_recall.py` (new, 400 lines)
   - [ ] Load facebank prototypes per identity
   - [ ] Identify residual gaps >2s from timeline
   - [ ] Sample gap windows at 30fps (densify)
   - [ ] Detect + embed + verify against target identity prototype
   - [ ] Create tracklets for accepted candidates
   - [ ] Log `identity_recall_audit.json` with `seconds_recovered`

2. **Implement Verification Logic** (20 min)
   - [ ] Minimum similarity: 0.82-0.84 (identity-dependent)
   - [ ] Margin over second-best: 0.08-0.10
   - [ ] Temporal consistency: require ≥4 consecutive frames
   - [ ] Negative gating: reject if any other identity scores within 0.06

3. **Run on Residual Gaps** (20 min)
   - [ ] Target identities: RINNA, EILEEN, BRANDI (residual deficits)
   - [ ] Scan gaps >2s that remain after densify
   - [ ] Generate tracklets
   - [ ] Inject into tracks.json

4. **Re-run Analytics** (10 min)
   - [ ] Re-merge timeline
   - [ ] Re-generate delta_table.csv
   - [ ] Verify final results

5. **Validate Prototypes** (10 min)
   - [ ] Ensure BRANDI prototype includes diverse poses (not just frontal)
   - [ ] Check RINNA prototype quality (may need multi-proto bank if still failing)
   - [ ] Verify LVP prototype exists (only 2s GT, may be sparse)

**Acceptance**:
- ✅ Identity-guided recall recovers 1-3s across 3 identities
- ✅ BRANDI moves from -3.43s to ≤4.5s (expected ~+2.5s recovery)
- ✅ RINNA moves from +5.07s to ≤4.5s (expected -1.5s correction via better verification)
- ✅ EILEEN moves from +4.42s to ≤4.5s (expected -1s correction)

**Expected Impact**:
- BRANDI: 6.59s → ~9.5s (+2.5s from targeted recall)
- RINNA: ~26s → ~24s (-1.5s from stricter verification preventing false adds)
- EILEEN: ~11s → ~10s (-1s from stricter verification)

**Time**: 120 minutes (2 hours)

---

## Phase 4: Minimal Multi-Prototype Bank (180 min)

**Problem**: Entrance bridging fails (0/6 success) due to pose/lighting variation between entrance and later appearance

**Solution**: Multi-prototype bank with pose × scale bins, set-to-set matching

**Reference**: [docs/MULTI_PROTO_FACEBANK_DESIGN.md](MULTI_PROTO_FACEBANK_DESIGN.md)

### Tasks:

1. **Create MultiProtoIdentityBank Class** (60 min)
   - [ ] File: `screentime/recognition/multi_proto_bank.py` (new, 200 lines)
   - [ ] Implement PrototypeBin dataclass
   - [ ] Implement add_prototype() method
   - [ ] Implement match() method (max/mean topK)
   - [ ] Implement export_to_parquet() / load_from_parquet()

2. **Implement Pose Estimation** (30 min)
   - [ ] Add `estimate_pose_bin()` using 5-point landmarks
   - [ ] Add `estimate_scale_bin()` using face_size
   - [ ] Unit tests for pose classification

3. **Populate Bank from Clusters** (40 min)
   - [ ] File: `jobs/tasks/build_multi_proto_bank.py` (new, 150 lines)
   - [ ] Load all embeddings per identity
   - [ ] Bin by pose × scale
   - [ ] Average within bins to create prototypes
   - [ ] Save to `data/facebank/RHOBH-TEST-10-28/multi_proto_bank.parquet`

4. **Integrate Set-to-Set Bridging** (40 min)
   - [ ] File: `jobs/tasks/entrance_recovery.py` (line ~700, modify bridge logic)
   - [ ] Implement `set_to_set_similarity()` function
   - [ ] Replace single medoid comparison with top-5 set comparison
   - [ ] Accept bridge if sim ≥ 0.70

5. **Re-run Entrance Recovery** (10 min)
   - [ ] Run with updated bridge logic
   - [ ] Verify bridge success rate improves (expect 4-5/6)
   - [ ] Check Track 307 → Track 42 bridge (expect sim ~0.72-0.78)

**Acceptance**:
- ✅ Bank contains 4-6 prototypes per identity
- ✅ Set-to-set bridge sim ≥ 0.70 for YOLANDA Track 307
- ✅ Bridge success rate ≥ 60% (4+/6 entrance tracks merge)
- ✅ Delta table shows additional +0.5-1.5s per identity from successful bridges

**Expected Impact**:
- KIM: +1.50s → +0.5s (bridge entrance → downstream, +1s from merge)
- KYLE: +2.73s → +1.8s (bridge entrance → downstream, +0.9s from merge)
- EILEEN: ~10s → ~9s (bridge entrance → downstream, +1s from merge)
- YOLANDA: 0.00s (perfect, bridge not critical but nice to have)

**Time**: 180 minutes (3 hours)

---

## Phase 5: Analytics Page Implementation (60-90 min)

**Problem**: No unified post-pipeline report, hard to visualize improvements

**Solution**: Comprehensive analytics page with accuracy table, recovery panel, QA metrics

**Reference**: [docs/ANALYTICS_PAGE_SPEC.md](ANALYTICS_PAGE_SPEC.md)

### Tasks:

1. **Create Analytics View Module** (40 min)
   - [ ] File: `app/lib/analytics_view.py` (new, 300 lines)
   - [ ] Implement `render_analytics_page()` entry point
   - [ ] Implement `render_pipeline_config()` block
   - [ ] Implement `render_accuracy_table()` with color coding
   - [ ] Implement `render_recovery_panel()` (entrance + densify)
   - [ ] Implement `render_tracking_qa()` (freeze-tracking, coverage)
   - [ ] Implement `render_downloads()` section

2. **Add Analytics Tab** (10 min)
   - [ ] File: `app/labeler.py` (line ~100)
   - [ ] Add "📊 Analytics" tab
   - [ ] Call `render_analytics_page(episode_id, data_root, config)`

3. **Standardize Audit JSON Schema** (20 min)
   - [ ] Update entrance_audit.json to use `seconds_recovered` field
   - [ ] Update densify_audit.json (once implemented) to match schema
   - [ ] Document schema in comments

4. **Test Analytics Page** (20 min)
   - [ ] Load Streamlit UI
   - [ ] Navigate to Analytics tab
   - [ ] Verify all 6 sections render correctly
   - [ ] Test downloads
   - [ ] Verify color coding (✅ green, ⚠️ yellow, ❌ red)

**Acceptance**:
- ✅ Analytics page renders all 6 sections
- ✅ Accuracy table shows 7/7 identities with color-coded status
- ✅ Recovery panel shows entrance + densify contributions
- ✅ Downloads work for all available files
- ✅ Page loads in <2s with caching

**Time**: 60-90 minutes

---

## Phase 6: Final Analytics & Export (30 min)

**Problem**: Need final delta table, timeline.csv, and audit reports for validation

**Solution**: Re-run full analytics pipeline with all improvements

### Tasks:

1. **Re-run Full Analytics** (15 min)
   - [ ] Command: `python jobs/tasks/run_analytics.py RHOBH-TEST-10-28`
   - [ ] Generate timeline.csv
   - [ ] Generate delta_table.csv
   - [ ] Combine all audit JSON files

2. **Validate Results** (10 min)
   - [ ] Check delta_table.csv: expect 7/7 PASS (all ≤4.5s)
   - [ ] Check timeline.csv: totals ≤ runtime, no overlaps
   - [ ] Check entrance_audit.json: 6 identities with `seconds_recovered`
   - [ ] Check densify_audit.json: 3-4 identities with `seconds_recovered`
   - [ ] Check identity_recall_audit.json: 2-3 identities with `seconds_recovered`

3. **Export Final Artifacts** (5 min)
   - [ ] Copy delta_table.csv to root for easy access
   - [ ] Copy timeline.csv to root
   - [ ] Generate summary report (1-page markdown)

**Acceptance**:
- ✅ Delta table shows 7/7 PASS (≤4.5s absolute error)
- ✅ Timeline totals ≤ runtime
- ✅ No overlaps (co-appearance credit applied)
- ✅ All audit files standardized with `seconds_recovered`

**Time**: 30 minutes

---

## Contingency Plans

### If Time Runs Short:

**Priority 1** (Must Do):
1. Streamlit key fix (30 min)
2. Densify pass 1 only (40 min)
3. Re-run analytics (15 min)
**Total**: 85 minutes → Expect 4-5/7 PASS

**Priority 2** (If 3 Hours Available):
- Add densify pass 2 (20 min)
- Add identity-guided recall (120 min)
**Total**: 3h 25m → Expect 6/7 PASS

**Priority 3** (If Full 6 Hours):
- Add multi-prototype bank (180 min)
- Add analytics page (90 min)
**Total**: 6h 35m → Expect 7/7 PASS with comprehensive reporting

### If 7/7 Not Achieved:

**Fallback**: Document residual issues and defer to next episode:
- BRANDI undercount may require person detector (catch sitting shots)
- LVP sparse GT (2s total) may be edge case
- Adjust acceptance criteria to 6/7 PASS with documented exceptions

---

## Files Summary

### New Files (8 total):
1. `screentime/recognition/multi_proto_bank.py` (200 lines)
2. `jobs/tasks/build_multi_proto_bank.py` (150 lines)
3. `jobs/tasks/identity_guided_recall.py` (400 lines)
4. `jobs/tasks/run_densify_two_pass.py` (200 lines)
5. `app/lib/analytics_view.py` (300 lines)
6. `tests/test_multi_proto_bank.py` (100 lines)
7. `data/facebank/RHOBH-TEST-10-28/multi_proto_bank.parquet` (data file)
8. `docs/IMPROVEMENTS1_SUMMARY.md` (final summary report)

### Modified Files (5 total):
1. `configs/pipeline.yaml` (add local_densify, local_densify_pass2, detection_ab sections)
2. `app/labeler.py` (line ~100: add Analytics tab, line ~1180-1300: apply wkey())
3. `jobs/tasks/entrance_recovery.py` (line ~700: set-to-set bridging)
4. `screentime/pipeline/local_densify.py` (line ~50: 2-pass logic)
5. `screentime/pipeline/clustering.py` (line ~400: multi-proto bank integration)

---

## Expected Final Results

| Identity | Current Δ | Expected Δ | Status |
|----------|-----------|------------|--------|
| YOLANDA  | 0.00s     | 0.00s      | ✅ PASS |
| KIM      | +1.50s    | +0.50s     | ✅ PASS |
| KYLE     | +2.73s    | +1.80s     | ✅ PASS |
| RINNA    | +5.07s    | +2.50s     | ✅ PASS |
| EILEEN   | +4.42s    | +3.50s     | ✅ PASS |
| BRANDI   | -3.43s    | -1.50s     | ✅ PASS |
| LVP      | +1.15s    | +0.80s     | ✅ PASS |

**Pass Rate**: 7/7 (100%) ← up from 2/7 (29%)

**Recovery Breakdown**:
- Entrance: +6.25s (6 identities)
- Densify pass 1: +4.5s (4 identities)
- Densify pass 2: +3.0s (3 identities)
- Identity-guided recall: +4.0s (3 identities)
- Multi-proto bridges: +3.5s (5 identities)
**Total**: +21.25s improvement

---

## Session Checklist

**Before Starting**:
- [ ] Confirm RetinaFace locked in config
- [ ] Confirm 10fps baseline maintained
- [ ] Backup current tracks.json
- [ ] Clear thumbnail cache

**During Session**:
- [ ] Track time per phase
- [ ] Test after each phase
- [ ] Commit after each major milestone
- [ ] Log all parameter choices in audit files

**After Session**:
- [ ] Verify 7/7 PASS in delta_table.csv
- [ ] Export all audit files
- [ ] Generate final summary report
- [ ] Document any deviations from plan

---

**Status**: Roadmap complete, ready for 5-6 hour implementation session
**ETA**: 7/7 PASS (≤4.5s absolute error) with comprehensive analytics
**Confidence**: High (80%+ probability of success given spot-check insights)
