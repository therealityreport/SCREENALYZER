# Local Densify — Gap-Focused High-Recall Runbook

## Purpose
- Recover missed detections only inside per-identity gap segments while preserving the stabilized 10 fps global baseline.
- Target RHOBH-TEST-10-28 undercounts (YOLANDA, RINNA, BRANDI) without regressing frozen cast (KIM, KYLE, LVP).
- Produce deterministic telemetry and artifacts so QA can confirm ≤ 4 s absolute error per identity.

## Window Selection
- Input: `timeline.csv`, `totals.csv`, optional `recall_gaps.json` and SceneDetect/motion features.
- Select inter-interval gaps ≤ 3.2 s for a single person; pad each side by ± 300 ms to capture boundary frames.
- Optional flags: require low-scale/blur/cut signals (SceneDetect + simple motion/variance heuristics) before scheduling; allow manual annotations to force inclusion/exclusion.
- Deduplicate overlapping windows and respect per-identity freeze settings before enqueuing decode work.

## Segment Decode & Detect
- Decode only the selected spans at 30 fps (`sampling_stride_ms ≈ 33`) using PyAV or ffmpeg.
- Run high-recall face detection per frame batch with:
  - `min_confidence: 0.55–0.58` (episode default 0.56).
  - `min_face_px: 45–60`.
  - Multi-scale passes at 1.0× / 1.3× / 1.6× / 2.0; optional 2×2 tiling for wide shots.
  - Optional person-ROI gating when baseline tracks provide a bounding region.
- Emit detections with embeddings so downstream verification has per-frame context.

## Identity Verification
- Use ArcFace similarity ≥ 0.82 with a top-minus-second margin ≥ 0.08.
- Fall back to per-cluster prototypes when no frozen embedding exists; log every relax override.
- Reject detections that violate scene-level guards (e.g., conflicting freeze, co-speaker overlap).

## Tracklets
- Require ≥ 3–4 consecutive frames (config default 3) before birthing a tracklet.
- Feed verified detections into per-identity re-ID + adaptive merge respecting `gap_merge_ms_max` and conflict guardrails.
- Track merges must record source window, recovered duration, and whether gap was fully or partially closed.

## Outputs
- Update `embeddings.parquet`, `tracks.json`, and `clusters.json` with recovered segments.
- Regenerate analytics exports (`timeline.csv`, `totals.csv`, `totals.parquet`, `totals.xlsx`) in-place after merges.
- Write `data/diagnostics/reports/<ep>/recall_stats.json` with per-window stats (scanned frames, accepted tracklets, rejects).
- Preserve a densify run manifest that ties windows → tracklets → analytics for QA replay.

## Acceptance
- Failing cast must land at absolute error ≤ 4 s after densify.
- No regressions for KIM/KYLE/LVP; totals for every identity remain ≤ episode duration.
- Telemetry shows no KIM/KYLE/LVP gap merges and logs any manual overrides.

## Proposed function signatures (reference for Claude)
```python
identify_gap_windows(episode_id, person_name, timeline, pad_ms=300, max_gap_ms=3200) -> List[TimeSpan]
extract_frames_30fps(video_path, spans, stride_ms=33) -> Iterator[FrameBatch]
detect_verify_embeddings(frames, min_conf, min_face_px, scales) -> List[VerifiedFace]
build_tracklets_and_merge(verified_faces, reid_cfg, timeline_cfg) -> TrackletStats
```

## Commands (future)
- Example invocation:
  ```
  python scripts/local_densify.py --episode RHOBH-TEST-10-28 --people YOLANDA RINNA BRANDI --write-analytics
  ```
- Expected outputs:
  - `data/outputs/RHOBH-TEST-10-28/timeline.csv` regenerated with recovered intervals.
  - `data/diagnostics/reports/RHOBH-TEST-10-28/recall_stats.json` summarising segments scanned, accepted tracklets, rejects.
  - `data/outputs/RHOBH-TEST-10-28/totals.csv|parquet|xlsx` updated with ≤ 4 s absolute error per identity.
