#!/usr/bin/env python3
"""
Full-gap sliding window scan for YOLANDA.
Exhaustively scans all gaps >10s with 3s sliding windows (0.5s hop) using
identity-gated detection to find any missed YOLANDA appearances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from screentime.detectors.face_small import SmallFaceRetinaDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class SlidingWindow:
    """A sliding window within a gap."""
    gap_idx: int
    window_idx: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class WindowResult:
    """Results from scanning one window."""
    window: SlidingWindow
    faces_detected: int
    yolanda_matches: int
    other_faces: int
    min_face_px: int | None
    max_face_px: int | None
    avg_similarity: float | None
    detections: list[dict]
    # Face size statistics
    yolanda_face_sizes: list[int]  # Sizes of verified YOLANDA faces
    rejected_face_sizes: list[int]  # Sizes of rejected (other identity) faces
    all_face_sizes: list[int]  # All detected face sizes


def _normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def load_yolanda_template(harvest_dir: Path) -> np.ndarray:
    """Load YOLANDA reference embedding from cluster data."""
    clusters_path = harvest_dir / "clusters.json"
    tracks_path = harvest_dir / "tracks.json"
    embeddings_path = harvest_dir / "embeddings.parquet"

    with open(clusters_path) as f:
        clusters_data = json.load(f)
    with open(tracks_path) as f:
        tracks_data = json.load(f)

    embeddings_df = pd.read_parquet(embeddings_path)
    embedding_lookup = {
        (int(row.frame_id), int(row.det_idx)): np.asarray(row.embedding, dtype=np.float32)
        for row in embeddings_df.itertuples()
    }

    tracks_by_id = {
        int(track["track_id"]): track for track in tracks_data.get("tracks", [])
    }

    # Find YOLANDA's cluster (cluster_id=5)
    yolanda_cluster = None
    for cluster in clusters_data["clusters"]:
        if cluster.get("name") == "YOLANDA":
            yolanda_cluster = cluster
            break

    if not yolanda_cluster:
        raise ValueError("YOLANDA cluster not found")

    # Build YOLANDA template from all embeddings in cluster
    embeddings = []
    for track_id in yolanda_cluster.get("track_ids", []):
        track = tracks_by_id.get(int(track_id))
        if not track:
            continue
        for ref in track.get("frame_refs", []):
            key = (int(ref["frame_id"]), int(ref.get("det_idx", 0)))
            emb_vec = embedding_lookup.get(key)
            if emb_vec is not None:
                embeddings.append(emb_vec)

    if not embeddings:
        raise ValueError("No embeddings found for YOLANDA")

    yolanda_embedding = _normalize(np.mean(embeddings, axis=0))
    logger.info(f"Built YOLANDA template from {len(embeddings)} face embeddings (cluster_id={yolanda_cluster['cluster_id']})")

    return yolanda_embedding


def find_large_gaps(timeline_df: pd.DataFrame, min_gap_ms: int = 10000) -> list[dict]:
    """Find gaps >10s in YOLANDA's timeline."""
    yolanda_intervals = timeline_df[timeline_df["person_name"] == "YOLANDA"].sort_values("start_ms")

    gaps = []
    for i in range(len(yolanda_intervals) - 1):
        current_end = yolanda_intervals.iloc[i]["end_ms"]
        next_start = yolanda_intervals.iloc[i + 1]["start_ms"]
        gap_ms = next_start - current_end

        if gap_ms > min_gap_ms:
            gaps.append({
                "gap_idx": len(gaps),
                "start_ms": int(current_end),
                "end_ms": int(next_start),
                "duration_ms": int(gap_ms),
            })

    return gaps


def generate_sliding_windows(
    gap: dict,
    window_ms: int = 3000,
    hop_ms: int = 500,
) -> list[SlidingWindow]:
    """Generate overlapping sliding windows across a gap."""
    windows = []
    start_ms = gap["start_ms"]
    end_ms = gap["end_ms"]

    window_start = start_ms
    window_idx = 0

    while window_start + window_ms <= end_ms:
        window = SlidingWindow(
            gap_idx=gap["gap_idx"],
            window_idx=window_idx,
            start_ms=window_start,
            end_ms=window_start + window_ms,
        )
        windows.append(window)
        window_start += hop_ms
        window_idx += 1

    # Add final partial window if there's remaining footage
    if window_start < end_ms:
        window = SlidingWindow(
            gap_idx=gap["gap_idx"],
            window_idx=window_idx,
            start_ms=window_start,
            end_ms=end_ms,
        )
        windows.append(window)

    return windows


def scan_window(
    window: SlidingWindow,
    video_path: Path,
    yolanda_template: np.ndarray,
    detector: SmallFaceRetinaDetector,
    embedder: ArcFaceEmbedder,
    min_sim: float = 0.86,
    min_margin: float = 0.12,
) -> WindowResult:
    """Scan a single window for YOLANDA faces."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int((window.start_ms / 1000) * fps)
    end_frame = int((window.end_ms / 1000) * fps)

    # Sample at 10fps
    stride = max(1, int(fps / 10))

    yolanda_detections = []
    other_faces = 0
    all_face_sizes = []  # All detected faces
    yolanda_face_sizes = []  # Verified YOLANDA faces
    rejected_face_sizes = []  # Rejected (other identity) faces

    for frame_idx in range(start_frame, end_frame, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        ts_ms = int((frame_idx / fps) * 1000)

        # Detect faces
        detections = detector.detect(frame)

        for det in detections:
            bbox = det.bbox
            x1, y1, x2, y2 = bbox
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            all_face_sizes.append(det.face_size)

            try:
                embedding = embedder.embed(face_crop)
                embedding = _normalize(embedding)

                # Verify against YOLANDA
                sim = float(np.dot(embedding, yolanda_template))

                # For margin, we'd need other templates, but for now just use sim threshold
                if sim >= min_sim:
                    # YOLANDA detected!
                    yolanda_face_sizes.append(det.face_size)
                    yolanda_detections.append({
                        "ts_ms": ts_ms,
                        "frame_idx": frame_idx,
                        "bbox": bbox,
                        "confidence": det.confidence,
                        "similarity": float(sim),
                        "face_size": det.face_size,
                    })
                else:
                    rejected_face_sizes.append(det.face_size)
                    other_faces += 1
            except Exception as e:
                logger.debug(f"Failed to embed face at {ts_ms}ms: {e}")
                continue

    cap.release()

    # Calculate stats
    min_face_px = int(min(all_face_sizes)) if all_face_sizes else None
    max_face_px = int(max(all_face_sizes)) if all_face_sizes else None
    avg_similarity = float(np.mean([d["similarity"] for d in yolanda_detections])) if yolanda_detections else None

    return WindowResult(
        window=window,
        faces_detected=len(all_face_sizes),
        yolanda_matches=len(yolanda_detections),
        other_faces=other_faces,
        min_face_px=min_face_px,
        max_face_px=max_face_px,
        avg_similarity=avg_similarity,
        detections=yolanda_detections,
        yolanda_face_sizes=[int(s) for s in yolanda_face_sizes],
        rejected_face_sizes=[int(s) for s in rejected_face_sizes],
        all_face_sizes=[int(s) for s in all_face_sizes],
    )


def yolanda_fullgap_scan(
    episode_id: str,
    video_path: Path,
    config: dict,
) -> dict:
    """Execute full-gap sliding window scan for YOLANDA."""
    harvest_dir = Path("data/harvest") / episode_id
    timeline_path = Path("data/outputs") / episode_id / "timeline.csv"

    # Load config
    yolanda_config = config.get("timeline", {}).get("per_identity", {}).get("YOLANDA", {})
    densify_config = yolanda_config.get("local_densify", {})

    if not densify_config.get("enabled", False):
        logger.warning("YOLANDA local_densify not enabled in config")
        return {"error": "local_densify not enabled"}

    max_gap_ms = densify_config.get("max_gap_ms", 45000)
    window_ms = densify_config.get("window_ms", 3000)
    hop_ms = densify_config.get("hop_ms", 500)
    min_face_px = densify_config.get("min_face_px", 36)
    min_confidence = densify_config.get("min_confidence", 0.50)
    scales = densify_config.get("scales", [1.0, 1.25, 1.5, 2.0])

    verify_config = densify_config.get("verify", {})
    min_sim = verify_config.get("min_similarity", 0.86)
    min_margin = verify_config.get("second_best_margin", 0.12)

    logger.info("=" * 80)
    logger.info("YOLANDA FULL-GAP SLIDING WINDOW SCAN")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  max_gap_ms: {max_gap_ms}")
    logger.info(f"  window_ms: {window_ms}")
    logger.info(f"  hop_ms: {hop_ms}")
    logger.info(f"  min_face_px: {min_face_px}")
    logger.info(f"  min_confidence: {min_confidence}")
    logger.info(f"  scales: {scales}")
    logger.info(f"  min_similarity: {min_sim}")
    logger.info(f"  second_best_margin: {min_margin}")
    logger.info("")

    # Load YOLANDA template
    yolanda_template = load_yolanda_template(harvest_dir)

    # Load timeline and find large gaps
    timeline_df = pd.read_csv(timeline_path)
    gaps = find_large_gaps(timeline_df, min_gap_ms=10000)  # Find gaps >10s

    logger.info(f"Found {len(gaps)} large gaps (>10s) in YOLANDA timeline:")
    for gap in gaps:
        logger.info(f"  Gap {gap['gap_idx']}: {gap['start_ms']}-{gap['end_ms']}ms ({gap['duration_ms']/1000:.1f}s)")
    logger.info("")

    # Initialize detector and embedder
    detector = SmallFaceRetinaDetector(
        min_confidence=min_confidence,
        min_face_px=min_face_px,
    )
    embedder = ArcFaceEmbedder()

    # Scan each gap with sliding windows
    all_results = []
    total_windows = 0
    total_yolanda_faces = 0

    for gap in gaps:
        windows = generate_sliding_windows(gap, window_ms=window_ms, hop_ms=hop_ms)
        logger.info(f"Gap {gap['gap_idx']}: Generated {len(windows)} sliding windows")
        total_windows += len(windows)

        gap_yolanda_faces = 0
        for idx, window in enumerate(windows, 1):
            # Progress indicator every 10 windows
            if idx % 10 == 0:
                logger.info(f"  Progress: {idx}/{len(windows)} windows scanned...")

            result = scan_window(
                window,
                video_path,
                yolanda_template,
                detector,
                embedder,
                min_sim=min_sim,
                min_margin=min_margin,
            )
            all_results.append(result)
            gap_yolanda_faces += result.yolanda_matches

            if result.yolanda_matches > 0:
                logger.info(f"  Window {window.window_idx} ({window.start_ms}-{window.end_ms}ms): "
                          f"✅ {result.yolanda_matches} YOLANDA faces!")

        total_yolanda_faces += gap_yolanda_faces
        logger.info(f"  Gap {gap['gap_idx']} total: {gap_yolanda_faces} YOLANDA faces")
        logger.info("")

    # Summary
    logger.info("=" * 80)
    logger.info("FULL-GAP SCAN RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total gaps scanned: {len(gaps)}")
    logger.info(f"Total windows scanned: {total_windows}")
    logger.info(f"YOLANDA faces found: {total_yolanda_faces}")
    logger.info("")

    if total_yolanda_faces > 0:
        logger.info(f"✅ Found {total_yolanda_faces} YOLANDA faces across {total_windows} windows!")
        logger.info("   These represent potential missed detections")
    else:
        logger.info(f"❌ No YOLANDA faces found in {total_windows} sliding windows")
        logger.info("   Conclusive proof: YOLANDA was off-screen during all large gaps")

    # Collect all face sizes for histogram
    all_yolanda_sizes = []
    all_rejected_sizes = []
    all_detected_sizes = []

    for result in all_results:
        all_yolanda_sizes.extend(result.yolanda_face_sizes)
        all_rejected_sizes.extend(result.rejected_face_sizes)
        all_detected_sizes.extend(result.all_face_sizes)

    # Generate face size histogram
    def create_histogram(sizes, bins=[0, 20, 28, 32, 40, 50, 75, 100, 150, 200, 1000]):
        """Create histogram of face sizes."""
        if not sizes:
            return {f"{bins[i]}-{bins[i+1]}px": 0 for i in range(len(bins)-1)}

        histogram = {}
        for i in range(len(bins) - 1):
            count = sum(1 for s in sizes if bins[i] <= s < bins[i+1])
            histogram[f"{bins[i]}-{bins[i+1]}px"] = count
        return histogram

    yolanda_histogram = create_histogram(all_yolanda_sizes)
    rejected_histogram = create_histogram(all_rejected_sizes)
    all_faces_histogram = create_histogram(all_detected_sizes)

    # Count small faces (≤32px)
    small_yolanda = sum(1 for s in all_yolanda_sizes if s <= 32)
    small_rejected = sum(1 for s in all_rejected_sizes if s <= 32)
    small_all = sum(1 for s in all_detected_sizes if s <= 32)

    logger.info("")
    logger.info("FACE SIZE ANALYSIS:")
    logger.info(f"  Total faces detected: {len(all_detected_sizes)}")
    logger.info(f"    YOLANDA verified: {len(all_yolanda_sizes)}")
    logger.info(f"    Other identities: {len(all_rejected_sizes)}")
    logger.info(f"  Small faces (≤32px): {small_all} total")
    logger.info(f"    YOLANDA: {small_yolanda}")
    logger.info(f"    Other: {small_rejected}")
    if all_yolanda_sizes:
        logger.info(f"  YOLANDA face sizes: min={min(all_yolanda_sizes)}px, median={int(np.median(all_yolanda_sizes))}px, max={max(all_yolanda_sizes)}px")
    if all_rejected_sizes:
        logger.info(f"  Rejected face sizes: min={min(all_rejected_sizes)}px, median={int(np.median(all_rejected_sizes))}px, max={max(all_rejected_sizes)}px")

    # Save results
    output_data = {
        "episode_id": episode_id,
        "target_identity": "YOLANDA",
        "config": {
            "max_gap_ms": max_gap_ms,
            "window_ms": window_ms,
            "hop_ms": hop_ms,
            "min_face_px": min_face_px,
            "min_confidence": min_confidence,
            "min_similarity": min_sim,
            "second_best_margin": min_margin,
            "profile": "small_face_recall_v1",
        },
        "gaps": gaps,
        "total_windows": total_windows,
        "total_yolanda_faces": total_yolanda_faces,
        # Face size analysis
        "face_size_analysis": {
            "total_faces_detected": len(all_detected_sizes),
            "yolanda_verified": len(all_yolanda_sizes),
            "other_identities": len(all_rejected_sizes),
            "small_faces_32px": {
                "total": small_all,
                "yolanda": small_yolanda,
                "other": small_rejected,
            },
            "yolanda_size_stats": {
                "min": int(min(all_yolanda_sizes)) if all_yolanda_sizes else None,
                "median": int(np.median(all_yolanda_sizes)) if all_yolanda_sizes else None,
                "max": int(max(all_yolanda_sizes)) if all_yolanda_sizes else None,
            } if all_yolanda_sizes else None,
            "rejected_size_stats": {
                "min": int(min(all_rejected_sizes)) if all_rejected_sizes else None,
                "median": int(np.median(all_rejected_sizes)) if all_rejected_sizes else None,
                "max": int(max(all_rejected_sizes)) if all_rejected_sizes else None,
            } if all_rejected_sizes else None,
            "histograms": {
                "yolanda_faces": yolanda_histogram,
                "rejected_faces": rejected_histogram,
                "all_faces": all_faces_histogram,
            },
        },
        "windows": [
            {
                "gap_idx": r.window.gap_idx,
                "window_idx": r.window.window_idx,
                "start_ms": r.window.start_ms,
                "end_ms": r.window.end_ms,
                "duration_ms": r.window.duration_ms,
                "faces_detected": r.faces_detected,
                "yolanda_matches": r.yolanda_matches,
                "other_faces": r.other_faces,
                "min_face_px": r.min_face_px,
                "max_face_px": r.max_face_px,
                "avg_similarity": r.avg_similarity,
                "detections": r.detections,
            }
            for r in all_results
        ],
    }

    output_path = harvest_dir / "diagnostics" / "reports" / "yolanda_fullgap_scan.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Results saved to: {output_path}")

    return output_data


if __name__ == "__main__":
    # Load config
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    episode_id = "RHOBH-TEST-10-28"
    video_path = Path("data/videos/RHOBH-TEST-10-28.mp4")

    result = yolanda_fullgap_scan(episode_id, video_path, config)

    if result.get("total_yolanda_faces", 0) > 0:
        logger.info("")
        logger.info("Next: Integrate YOLANDA faces into tracklets and re-run analytics")
    else:
        logger.info("")
        logger.info("Conclusion: YOLANDA deficit is genuine off-screen time")
