# Analytics Page Specification

**Purpose**: Comprehensive post-pipeline report showing accuracy, coverage quality, and diagnostic metrics in standardized format.

**Location**: Streamlit UI - New tab "Analytics" (or existing Overview page enhancement)

**Data Sources**:
- `data/outputs/EPISODEID/delta_table.csv`
- `data/outputs/EPISODEID/timeline.csv`
- `data/harvest/EPISODEID/diagnostics/reports/entrance_audit.json`
- `data/harvest/EPISODEID/diagnostics/reports/densify_audit.json`
- `data/harvest/EPISODEID/tracks.json`

---

## 1. Page Layout

```
┌─────────────────────────────────────────────────────────┐
│ 📊 Analytics - RHOBH-TEST-10-28                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🔧 PIPELINE CONFIG                                      │
│ ├─ Detector: RetinaFace (buffalo_l det_10g)            │
│ ├─ Baseline: 10fps (100ms stride)                      │
│ ├─ Thresholds: conf≥0.70, face≥72px                    │
│ └─ Entrance Recovery: ✓ Enabled (all identities)       │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📈 ACCURACY SUMMARY (vs Ground Truth)                  │
│                                                         │
│  Person    │ Auto (s) │ GT (s) │ Δ (s) │ Error % │ ✓  │
│  ──────────┼──────────┼────────┼───────┼─────────┼────│
│  YOLANDA   │   16.00  │ 16.00  │  0.00 │   0.0%  │ ✅ │
│  KIM       │   49.50  │ 48.00  │ +1.50 │  +3.1%  │ ✅ │
│  KYLE      │   23.75  │ 21.02  │ +2.73 │ +13.0%  │ ⚠️ │
│  RINNA     │   30.08  │ 25.02  │ +5.07 │ +20.3%  │ ❌ │
│  EILEEN    │   14.42  │ 10.00  │ +4.42 │ +44.1%  │ ❌ │
│  BRANDI    │    6.59  │ 10.01  │ -3.43 │ -34.2%  │ ❌ │
│  LVP       │    3.17  │  2.02  │ +1.15 │ +56.9%  │ ❌ │
│                                                         │
│  TOTALS    │  143.51  │ 132.07 │+11.44 │  +8.7%  │    │
│                                                         │
│  Pass Rate: 2/7 (29%)    [Target: 7/7 ≤4.5s]           │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🎯 ENTRANCE & DENSIFY RECOVERY                          │
│                                                         │
│  Identity  │ Entrance Δ │ Densify Δ │ Total Δ │ Bridge│
│  ──────────┼────────────┼───────────┼─────────┼───────│
│  YOLANDA   │   +2.08s   │    ---    │  +2.08s │  ❌   │
│  KIM       │   +1.50s   │    ---    │  +1.50s │  ❌   │
│  KYLE      │   +0.50s   │    ---    │  +0.50s │  ❌   │
│  EILEEN    │   +0.75s   │    ---    │  +0.75s │  ❌   │
│  BRANDI    │   +0.67s   │    ---    │  +0.67s │  ❌   │
│  LVP       │   +0.75s   │    ---    │  +0.75s │  ❌   │
│  RINNA     │     ---    │    ---    │    ---  │  ---  │
│                                                         │
│  Total Recovered: 6.25s across 6 identities            │
│  Bridge Success: 0/6 (0%) - Multi-proto bank needed    │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🎬 COVERAGE & TRACKING QA                               │
│                                                         │
│  Total Tracks: 307 (baseline) + 6 (entrance)           │
│  Total Intervals: 142 (post-merge)                     │
│  Total Screentime: 143.51s (108.6% of GT)              │
│  Overlap Budget: 0.00s (co-appearance credit applied)  │
│                                                         │
│  Freeze-Tracking Metrics:                              │
│  ├─ Frozen Identities: KIM, KYLE, LVP (3/7)            │
│  ├─ Active Identities: YOLANDA, RINNA, BRANDI, EILEEN  │
│  └─ Regression Check: ✅ No frozen identity changed     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 🔍 DETECTOR COMPARISON (if A/B enabled)                 │
│                                                         │
│  Metric           │ RetinaFace │ SCRFD      │ Winner  │
│  ─────────────────┼────────────┼────────────┼─────────│
│  Total Detections │   12,450   │   11,890   │ Retina  │
│  Small (≤80px)    │      0     │      0     │   Tie   │
│  Avg Face Size    │    124px   │    127px   │  SCRFD  │
│  Final Accuracy   │   2/7 PASS │     ---    │   ---   │
│                                                         │
│  Spot-Check Gate: ✗ FAILED (0% lift < 30% threshold)   │
│  Decision: RetinaFace locked, skip full A/B             │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│ 📥 DOWNLOADS                                            │
│                                                         │
│  [⬇️ delta_table.csv]      Accuracy metrics             │
│  [⬇️ timeline.csv]         Per-identity intervals       │
│  [⬇️ entrance_audit.json]  Entrance recovery details    │
│  [⬇️ densify_audit.json]   Densify scan results         │
│  [⬇️ tracks.json]          Full track data              │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Component Specifications

### A. Pipeline Config Block

**Purpose**: Document exact pipeline settings for reproducibility

**Data Sources**:
- `configs/pipeline.yaml` (read at render time)
- Detector metadata from harvest

**Fields**:
```python
def render_pipeline_config(config: dict):
    st.subheader("🔧 Pipeline Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Detector", "RetinaFace")
        st.caption("Model: buffalo_l det_10g")

        st.metric("Baseline Sampling", "10fps")
        st.caption("Stride: 100ms (every 3rd frame @ 30fps)")

    with col2:
        st.metric("Detection Thresholds", "conf≥0.70, face≥72px")

        entrance_enabled = config.get("entrance", {}).get("enabled", False)
        st.metric("Entrance Recovery", "✓ Enabled" if entrance_enabled else "✗ Disabled")
        st.caption("All identities" if entrance_enabled else "N/A")
```

---

### B. Accuracy Summary Table

**Purpose**: Primary accuracy report with color-coded status

**Data Source**: `data/outputs/EPISODEID/delta_table.csv`

**Format**:
```python
def render_accuracy_table(delta_df: pd.DataFrame):
    st.subheader("📈 Accuracy Summary (vs Ground Truth)")

    # Add status column
    def get_status(delta_s: float) -> str:
        abs_delta = abs(delta_s)
        if abs_delta <= 4.5:
            return "✅"
        elif abs_delta <= 6.0:
            return "⚠️"
        else:
            return "❌"

    delta_df['Status'] = delta_df['Delta (s)'].apply(get_status)

    # Display with color coding
    st.dataframe(
        delta_df[['Person', 'Auto (ms)', 'GT (ms)', 'Delta (s)', 'Error %', 'Status']],
        use_container_width=True,
        hide_index=True
    )

    # Summary metrics
    pass_count = (delta_df['Status'] == '✅').sum()
    total_count = len(delta_df)
    pass_rate = pass_count / total_count * 100

    st.metric("Pass Rate", f"{pass_count}/{total_count} ({pass_rate:.1f}%)")
    st.caption("Target: 7/7 identities ≤4.5s absolute error")
```

**Color Coding**:
- ✅ Green: |Δ| ≤ 4.5s (PASS)
- ⚠️ Yellow: 4.5s < |Δ| ≤ 6.0s (WARN)
- ❌ Red: |Δ| > 6.0s (FAIL)

---

### C. Entrance & Densify Recovery Panel

**Purpose**: Show recovery contributions from entrance and densify modules

**Data Sources**:
- `data/harvest/EPISODEID/diagnostics/reports/entrance_audit.json`
- `data/harvest/EPISODEID/diagnostics/reports/densify_audit.json` (when implemented)

**Format**:
```python
def render_recovery_panel(entrance_audit: dict, densify_audit: dict):
    st.subheader("🎯 Entrance & Densify Recovery")

    recovery_data = []

    # Parse entrance audit
    for identity, stats in entrance_audit.get("per_identity", {}).items():
        entrance_s = stats.get("seconds_recovered", 0.0)
        bridge_success = stats.get("bridge_success", False)

        # Parse densify audit (when available)
        densify_s = densify_audit.get(identity, {}).get("seconds_recovered", 0.0)

        recovery_data.append({
            "Identity": identity,
            "Entrance Δ": f"+{entrance_s:.2f}s" if entrance_s > 0 else "---",
            "Densify Δ": f"+{densify_s:.2f}s" if densify_s > 0 else "---",
            "Total Δ": f"+{entrance_s + densify_s:.2f}s" if (entrance_s + densify_s) > 0 else "---",
            "Bridge": "✅" if bridge_success else "❌"
        })

    df = pd.DataFrame(recovery_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Summary
    total_recovered = sum(r.get("entrance_s", 0) + r.get("densify_s", 0) for r in recovery_data)
    bridge_count = sum(1 for r in recovery_data if r["Bridge"] == "✅")

    st.metric("Total Recovered", f"{total_recovered:.2f}s across {len(recovery_data)} identities")
    st.caption(f"Bridge Success: {bridge_count}/{len(recovery_data)} ({bridge_count/len(recovery_data)*100:.0f}%)")
```

**CRITICAL**: Standardize on `seconds_recovered` field in all audit JSON files:
```json
{
  "episode_id": "RHOBH-TEST-10-28",
  "per_identity": {
    "YOLANDA": {
      "seconds_recovered": 2.08,  // ← Standardized field name
      "frames_added": 26,
      "bridge_success": false
    }
  }
}
```

---

### D. Coverage & Tracking QA

**Purpose**: High-level QA metrics for freeze-tracking and coverage

**Data Sources**:
- `data/harvest/EPISODEID/tracks.json`
- `data/outputs/EPISODEID/timeline.csv`
- `configs/pipeline.yaml` (for freeze list)

**Format**:
```python
def render_tracking_qa(tracks: dict, timeline_df: pd.DataFrame, config: dict):
    st.subheader("🎬 Coverage & Tracking QA")

    # Track counts
    baseline_tracks = [t for t in tracks['tracks'] if t.get('source') != 'entrance_recovery']
    entrance_tracks = [t for t in tracks['tracks'] if t.get('source') == 'entrance_recovery']

    st.metric("Total Tracks", f"{len(baseline_tracks)} (baseline) + {len(entrance_tracks)} (entrance)")

    # Timeline stats
    total_intervals = len(timeline_df)
    total_screentime = timeline_df['duration_ms'].sum() / 1000

    st.metric("Total Intervals", total_intervals)
    st.metric("Total Screentime", f"{total_screentime:.2f}s")

    # Freeze-tracking
    frozen_identities = []
    for identity, overrides in config.get("timeline", {}).get("per_identity", {}).items():
        if overrides.get("freeze", False):
            frozen_identities.append(identity)

    st.caption(f"Frozen Identities: {', '.join(frozen_identities)} ({len(frozen_identities)}/7)")

    # Regression check (compare current timeline to baseline)
    regression_detected = check_frozen_regression(timeline_df, frozen_identities)
    if not regression_detected:
        st.success("✅ No frozen identity changed")
    else:
        st.error("⚠️ Frozen identity regression detected!")
```

---

### E. Detector Comparison Block (Conditional)

**Purpose**: Show A/B results when detector comparison was run

**Data Source**: `data/harvest/EPISODEID/diagnostics/reports/detector_comparison.json`

**Visibility**: Only show if A/B was performed (file exists)

**Format**:
```python
def render_detector_comparison(comparison_data: dict):
    if not comparison_data:
        return  # Skip if no A/B performed

    st.subheader("🔍 Detector Comparison")

    metrics = [
        {"Metric": "Total Detections",
         "RetinaFace": comparison_data["retinaface"]["total_detections"],
         "SCRFD": comparison_data["scrfd"]["total_detections"]},
        {"Metric": "Small Faces (≤80px)",
         "RetinaFace": comparison_data["retinaface"]["small_faces"],
         "SCRFD": comparison_data["scrfd"]["small_faces"]},
        {"Metric": "Avg Face Size",
         "RetinaFace": f"{comparison_data['retinaface']['avg_face_size']}px",
         "SCRFD": f"{comparison_data['scrfd']['avg_face_size']}px"},
    ]

    df = pd.DataFrame(metrics)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Decision summary
    decision = comparison_data.get("decision", {})
    if decision.get("gate_passed", False):
        st.success(f"✓ Gate PASSED: {decision['lift_percent']:.1f}% lift ≥ 30% threshold")
        st.info(f"Winner: {decision['winner']}")
    else:
        st.warning(f"✗ Gate FAILED: {decision['lift_percent']:.1f}% lift < 30% threshold")
        st.info("Decision: RetinaFace locked, skip full A/B")
```

---

### F. Downloads Section

**Purpose**: One-click access to all diagnostic files

**Format**:
```python
def render_downloads(episode_id: str, data_root: Path):
    st.subheader("📥 Downloads")

    files = [
        ("delta_table.csv", "Accuracy metrics", data_root / "outputs" / episode_id / "delta_table.csv"),
        ("timeline.csv", "Per-identity intervals", data_root / "outputs" / episode_id / "timeline.csv"),
        ("entrance_audit.json", "Entrance recovery details",
         data_root / "harvest" / episode_id / "diagnostics" / "reports" / "entrance_audit.json"),
        ("densify_audit.json", "Densify scan results",
         data_root / "harvest" / episode_id / "diagnostics" / "reports" / "densify_audit.json"),
        ("tracks.json", "Full track data", data_root / "harvest" / episode_id / "tracks.json"),
    ]

    for filename, description, path in files:
        if path.exists():
            with open(path, "rb") as f:
                st.download_button(
                    label=f"⬇️ {filename}",
                    data=f,
                    file_name=filename,
                    mime="text/csv" if filename.endswith(".csv") else "application/json",
                    help=description
                )
        else:
            st.caption(f"⬇️ {filename} - Not available")
```

---

## 3. Integration into Streamlit App

### Option A: New "Analytics" Tab

**File**: `app/labeler.py` (line ~100, tab creation)

```python
tabs = st.tabs(["🏠 Overview", "📋 Review", "📊 Analytics", "🔧 Settings"])

with tabs[2]:  # Analytics tab
    render_analytics_page(episode_id, data_root, config)
```

---

### Option B: Enhance Existing Overview

**File**: `app/labeler.py` (line ~300, overview rendering)

Add analytics blocks below existing overview content:
```python
def render_overview_page(episode_id: str, data_root: Path):
    # ... existing overview content ...

    st.divider()

    # Analytics blocks
    render_pipeline_config(config)
    render_accuracy_table(delta_df)
    render_recovery_panel(entrance_audit, densify_audit)
    # ... etc
```

**Recommendation**: Option A (new tab) to avoid cluttering overview

---

## 4. Acceptance Criteria

✅ Analytics page renders all 6 sections correctly
✅ Accuracy table shows color-coded status (✅/⚠️/❌)
✅ Recovery panel uses standardized `seconds_recovered` field
✅ Freeze-tracking regression check implemented
✅ Detector comparison block conditional (only if A/B performed)
✅ Downloads work for all available files
✅ Page loads in <2s with caching

---

## 5. Implementation Checklist

**Estimated Time**: 60-90 minutes

### Files to Modify:
- [ ] `app/labeler.py` (line ~100) - Add "Analytics" tab
- [ ] `app/lib/analytics_view.py` (NEW, 300 lines) - Analytics rendering logic

### Functions to Create:
- [ ] `render_analytics_page()` - Main entry point
- [ ] `render_pipeline_config()` - Config block
- [ ] `render_accuracy_table()` - Delta table with color coding
- [ ] `render_recovery_panel()` - Entrance + densify metrics
- [ ] `render_tracking_qa()` - Coverage & freeze-tracking
- [ ] `render_detector_comparison()` - A/B results (conditional)
- [ ] `render_downloads()` - File download buttons
- [ ] `check_frozen_regression()` - Compare frozen identities to baseline

### Audit JSON Schema Updates:
- [ ] Standardize on `seconds_recovered` field in entrance_audit.json
- [ ] Ensure densify_audit.json follows same schema
- [ ] Document schema in comments

---

**Status**: Specification complete, ready for 60-90 minute implementation
**File**: `app/lib/analytics_view.py` (300 lines)
**Integration Point**: `app/labeler.py` line ~100 (new tab)
