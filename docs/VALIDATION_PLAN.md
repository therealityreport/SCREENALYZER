# Validation Plan

## Phase-1 Closure via Local Densify
- Baseline 10 fps run complete; bootstrap labels frozen in analytics artifacts before densify.
- For each failing cast: compute gap windows, execute Local Densify (30 fps decode), verify identity (ArcFace sim ≥ 0.82, margin ≥ 0.08), birth tracklets with ≥ 3 consecutive frames, run re-ID + adaptive merge, regenerate analytics exports.
- Required telemetry: `recall_stats.json` (segments scanned, recall faces/tracks, accept vs reject counts), `cut_flushes`, `reid_overrides_used{person}`, `merge_clamped{person}`, `co_face_samples_dropped`, `gaps_merged` and `gaps_blocked`.
- Pass criteria: every cast ≤ 4 s absolute error, totals ≤ video duration, no cross-speaker fuses detected in diagnostics.
