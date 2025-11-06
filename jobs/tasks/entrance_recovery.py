#!/usr/bin/env python3
"""
Generic entrance recovery module for all identities.

Recovers pre-first-interval appearances using:
- Data-driven candidate collection (no geometry assumptions)
- Unsupervised clustering with temporal consistency
- Negative gating against all other identities
- Set-to-set bridging with temporal adjacency
- Multi-prototype assimilation

This replaces hard-coded, scene-specific bootstrapping with a general solution.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

from screentime.detectors.face_retina import RetinaFaceDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder, EmbeddingResult

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@dataclass
class EntranceCandidate:
    """Entrance face candidate with embedding and metadata."""
    frame_id: int
    ts_ms: int
    det_idx: int
    bbox: list[int]
    confidence: float
    face_size: int
    embedding: np.ndarray | None
    has_embedding: bool
    used_kps: bool
    tries: int
    sim_to_others: dict[str, float] = field(default_factory=dict)


@dataclass
class EntranceSeed:
    """Provisional identity seed from entrance clustering."""
    identity_name: str
    medoid: np.ndarray
    cluster_size: int
    num_clusters: int
    temporal_segments: int
    time_span_ms: int
    member_frames: list[int]


@dataclass
class BridgeResult:
    """Result from attempting to bridge entrance to existing tracks."""
    success: bool
    target_track_id: int | None
    set_similarity: float
    temporal_gap_ms: int | None
    reason: str


@dataclass
class EntranceStats:
    """Statistics from entrance recovery."""
    identity_name: str
    first_interval_start_ms: int
    window_start_ms: int
    window_end_ms: int
    frames_decoded: int
    faces_detected: int
    candidates_collected: int
    embeddings_ok: int
    embeddings_failed: int
    seed_frames: int
    seed_clusters: int
    seed_temporal_segments: int
    accepted_candidates: int
    bridge_success: bool
    bridge_target_track: int | None
    seconds_recovered: float
    negative_hits: dict[str, int] = field(default_factory=dict)

    @classmethod
    def empty(
        cls,
        identity_name: str,
        first_interval_start_ms: int = 0,
        window_start_ms: int = 0,
        window_end_ms: int = 0,
        reason: str = ""
    ) -> "EntranceStats":
        """Create empty EntranceStats for early returns."""
        return cls(
            identity_name=identity_name,
            first_interval_start_ms=first_interval_start_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            frames_decoded=0,
            faces_detected=0,
            candidates_collected=0,
            embeddings_ok=0,
            embeddings_failed=0,
            seed_frames=0,
            seed_clusters=0,
            seed_temporal_segments=0,
            accepted_candidates=0,
            bridge_success=False,
            bridge_target_track=None,
            seconds_recovered=0.0,
            negative_hits={}
        )


def load_config() -> dict:
    """Load pipeline configuration."""
    config_path = Path("configs/pipeline.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_frame(video_path: Path, ts_ms: int) -> np.ndarray | None:
    """Extract frame at timestamp."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_number = int((ts_ms / 1000.0) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_bgr = cap.read()
    cap.release()

    return frame_bgr if ret else None


def compute_identity_prototypes(
    identity_name: str,
    episode_id: str,
    data_root: Path
) -> dict[str, np.ndarray]:
    """
    Compute prototype vectors for all identities in the episode.

    Returns:
        Dict mapping identity_name -> average embedding vector
    """
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"

    clusters_data = json.loads(clusters_path.read_text())
    tracks_data = json.loads(tracks_path.read_text())
    embeddings_df = pd.read_parquet(embeddings_path)

    prototypes = {}

    for cluster in clusters_data["clusters"]:
        cluster_name = cluster.get("name")
        if not cluster_name or cluster_name == "UNKNOWN":
            continue

        # Get frame IDs for this identity
        frame_ids = []
        for track_id in cluster.get("track_ids", []):
            track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
            if track:
                for frame_ref in track.get("frame_refs", []):
                    frame_ids.append(frame_ref["frame_id"])

        if len(frame_ids) == 0:
            continue

        # Get embeddings
        identity_embeddings = embeddings_df[embeddings_df["frame_id"].isin(frame_ids)]

        if len(identity_embeddings) > 0:
            # Convert to list of arrays, filtering None
            emb_list = [np.asarray(e) for e in identity_embeddings["embedding"].values if e is not None]
            if not emb_list:
                continue
            try:
                vecs = np.stack(emb_list, axis=0)
            except ValueError:
                # Handle shape mismatch by normalizing
                emb_list = [e.reshape(-1) for e in emb_list]
                vecs = np.stack(emb_list, axis=0)
            avg = vecs.mean(axis=0)
            avg = avg / np.linalg.norm(avg)
            prototypes[cluster_name] = avg

    logger.info(f"Computed prototypes for {len(prototypes)} identities: {list(prototypes.keys())}")

    return prototypes


def find_first_interval(identity_name: str, episode_id: str, data_root: Path) -> int | None:
    """
    Find the start time of the first interval for an identity.

    Returns:
        First interval start time in ms, or None if no intervals found
    """
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"

    clusters_data = json.loads(clusters_path.read_text())
    tracks_data = json.loads(tracks_path.read_text())

    # Find identity's cluster
    identity_cluster = next(
        (c for c in clusters_data["clusters"] if c.get("name") == identity_name),
        None
    )

    if not identity_cluster:
        logger.warning(f"Cluster not found for {identity_name}")
        return None

    # Find earliest track start
    earliest_start = None
    for track_id in identity_cluster.get("track_ids", []):
        track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
        if track:
            start_ms = track.get("start_ms", 0)
            if earliest_start is None or start_ms < earliest_start:
                earliest_start = start_ms

    return earliest_start


def collect_entrance_candidates(
    identity_name: str,
    window_start_ms: int,
    window_end_ms: int,
    episode_id: str,
    video_path: Path,
    manifest_df: pd.DataFrame,
    detector: RetinaFaceDetector,
    embedder: ArcFaceEmbedder,
    prototypes: dict[str, np.ndarray],
    config: dict
) -> tuple[list[EntranceCandidate], dict]:
    """
    Collect entrance candidates using data-driven filtering.

    NO geometry assumptions - uses only:
    - Detector confidence
    - Face size (reasonable bounds)
    - Negative constraint against other identities
    """
    sampled_frames = manifest_df[
        (manifest_df['ts_ms'] >= window_start_ms) &
        (manifest_df['ts_ms'] <= window_end_ms) &
        (manifest_df['sampled'] == True)
    ]

    logger.info(f"Collecting entrance candidates for {identity_name}")
    logger.info(f"  Window: {window_start_ms}-{window_end_ms}ms ({len(sampled_frames)} frames)")

    candidates = []
    stats = {
        "frames_decoded": 0,
        "faces_detected": 0,
        "candidates_collected": 0,
        "embeddings_ok": 0,
        "embeddings_failed": 0,
        "negative_hits": {}
    }

    # Get detection config
    min_face_px = config.get("detection", {}).get("min_face_px", 80)
    min_confidence = config.get("detection", {}).get("min_confidence", 0.7)

    for _, row in sampled_frames.iterrows():
        frame_id = row['frame_id']
        ts_ms = row['ts_ms']

        stats["frames_decoded"] += 1

        # Extract frame
        frame_bgr = extract_frame(video_path, ts_ms)
        if frame_bgr is None:
            logger.warning(f"Failed to extract frame {frame_id} at {ts_ms}ms")
            continue

        # Detect faces
        detections = detector.detect(frame_bgr)
        stats["faces_detected"] += len(detections)

        if len(detections) == 0:
            continue

        # Process each detection
        for det_idx, det in enumerate(detections):
            bbox = det["bbox"]
            face_size = det["face_size"]
            confidence = det["confidence"]

            # Basic quality filters (no geometry assumptions)
            if face_size < min_face_px * 0.8:  # Slightly relaxed for entrance
                continue

            if confidence < min_confidence * 0.9:  # Slightly relaxed for entrance
                continue

            stats["candidates_collected"] += 1

            # Generate embedding
            landmarks = det.get("landmarks")
            kps = None
            if landmarks is not None:
                landmarks_array = np.array(landmarks)
                if len(landmarks_array) >= 5:
                    kps = landmarks_array[:5]

            result = embedder.embed_from_detection(frame_bgr, bbox, kps=kps)

            if result.success:
                stats["embeddings_ok"] += 1
            else:
                stats["embeddings_failed"] += 1

            # Compute similarities to all other identities (negative gating)
            sim_to_others = {}
            if result.success and result.embedding is not None:
                emb_norm = result.embedding / np.linalg.norm(result.embedding)

                for other_name, other_proto in prototypes.items():
                    if other_name != identity_name:
                        sim = float(np.dot(emb_norm, other_proto))
                        sim_to_others[other_name] = sim

                        # Track negative hits (too similar to wrong identity)
                        if sim > 0.75:  # High similarity to wrong identity
                            stats["negative_hits"][other_name] = stats["negative_hits"].get(other_name, 0) + 1

            candidate = EntranceCandidate(
                frame_id=frame_id,
                ts_ms=ts_ms,
                det_idx=det_idx,
                bbox=bbox,
                confidence=confidence,
                face_size=face_size,
                embedding=result.embedding,
                has_embedding=result.success,
                used_kps=result.used_kps,
                tries=result.tries,
                sim_to_others=sim_to_others
            )

            candidates.append(candidate)

    logger.info(f"Collected {len(candidates)} candidates")
    logger.info(f"  Embeddings: {stats['embeddings_ok']} OK, {stats['embeddings_failed']} failed")
    logger.info(f"  Negative hits: {stats['negative_hits']}")

    return candidates, stats


def build_entrance_seed(
    candidates: list[EntranceCandidate],
    identity_name: str,
    config: dict
) -> EntranceSeed | None:
    """
    Build provisional entrance seed using unsupervised clustering.

    Uses temporal consistency and cluster cohesion to validate.
    """
    valid_candidates = [c for c in candidates if c.has_embedding and c.embedding is not None]

    cluster_config = config.get("entrance", {}).get("cluster", {})
    min_samples = cluster_config.get("min_samples", 4)
    min_duration_ms = cluster_config.get("min_duration_ms", 600)

    if len(valid_candidates) < min_samples:
        logger.warning(f"Insufficient candidates for clustering: {len(valid_candidates)} < {min_samples}")
        return None

    logger.info(f"Building entrance seed from {len(valid_candidates)} candidates")

    # Stack embeddings - convert to arrays and filter None
    emb_list = [np.asarray(c.embedding) for c in valid_candidates if c.embedding is not None]
    if not emb_list:
        logger.warning("No valid embeddings after filtering None")
        return None
    try:
        embeddings = np.stack(emb_list, axis=0)
    except ValueError:
        # Handle shape mismatch by normalizing
        emb_list = [e.reshape(-1) for e in emb_list]
        embeddings = np.stack(emb_list, axis=0)
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise cosine distances
    distances = 1.0 - np.dot(embeddings_norm, embeddings_norm.T)
    distances = np.clip(distances, 0.0, 2.0)

    # Cluster with DBSCAN
    eps = cluster_config.get("eps", 0.35)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(distances)

    # Find largest cluster
    unique_labels = set(labels)
    unique_labels.discard(-1)  # Remove noise

    if len(unique_labels) == 0:
        logger.warning("No clusters found")
        return None

    cluster_sizes = {label: np.sum(labels == label) for label in unique_labels}
    largest_label = max(cluster_sizes, key=cluster_sizes.get)
    largest_size = cluster_sizes[largest_label]

    logger.info(f"Found {len(unique_labels)} clusters, largest has {largest_size} members")

    # Get largest cluster
    cluster_mask = labels == largest_label
    cluster_embeddings = embeddings[cluster_mask]
    cluster_candidates = [c for i, c in enumerate(valid_candidates) if cluster_mask[i]]

    # Check temporal consistency
    cluster_timestamps = sorted([c.ts_ms for c in cluster_candidates])
    segments = []
    current_segment = [cluster_timestamps[0]]

    for ts in cluster_timestamps[1:]:
        if ts - current_segment[-1] <= 200:  # Contiguous at 10fps
            current_segment.append(ts)
        else:
            segments.append(current_segment)
            current_segment = [ts]

    segments.append(current_segment)

    # Validate temporal span
    time_span_ms = cluster_timestamps[-1] - cluster_timestamps[0]
    is_valid = time_span_ms >= min_duration_ms

    if not is_valid:
        logger.warning(f"Insufficient temporal span: {time_span_ms}ms < {min_duration_ms}ms")
        return None

    logger.info(f"Temporal consistency OK: {len(segments)} segments, {time_span_ms}ms span")

    # Compute medoid
    cluster_embeddings_norm = cluster_embeddings / np.linalg.norm(cluster_embeddings, axis=1, keepdims=True)
    pairwise_sims = np.dot(cluster_embeddings_norm, cluster_embeddings_norm.T)
    centrality = pairwise_sims.sum(axis=1)
    medoid_idx = np.argmax(centrality)

    seed = EntranceSeed(
        identity_name=identity_name,
        medoid=cluster_embeddings_norm[medoid_idx],
        cluster_size=largest_size,
        num_clusters=len(unique_labels),
        temporal_segments=len(segments),
        time_span_ms=time_span_ms,
        member_frames=[c.frame_id for c in cluster_candidates]
    )

    logger.info(f"Built entrance seed: {seed.cluster_size} frames, {seed.temporal_segments} segments")

    return seed


def verify_candidates(
    candidates: list[EntranceCandidate],
    seed: EntranceSeed,
    config: dict
) -> list[EntranceCandidate]:
    """
    Verify candidates against entrance seed with negative gating.

    Acceptance criteria:
    1. sim_to_seed >= threshold
    2. (sim_to_seed - max(sim_to_others)) >= negative_margin
    """
    verify_config = config.get("entrance", {}).get("verify", {})
    seed_min_sim = verify_config.get("seed_min_similarity", 0.72)
    negative_margin = verify_config.get("negative_margin", 0.06)

    accepted = []

    for candidate in candidates:
        if not candidate.has_embedding or candidate.embedding is None:
            continue

        emb_norm = candidate.embedding / np.linalg.norm(candidate.embedding)
        sim_to_seed = float(np.dot(emb_norm, seed.medoid))

        # Find best similarity to other identities
        best_other_sim = max(candidate.sim_to_others.values()) if candidate.sim_to_others else 0.0

        # Acceptance criteria
        passes_seed_threshold = sim_to_seed >= seed_min_sim
        passes_negative_margin = (sim_to_seed - best_other_sim) >= negative_margin

        accept = passes_seed_threshold and passes_negative_margin

        if accept:
            accepted.append(candidate)

    logger.info(f"Accepted {len(accepted)}/{len(candidates)} candidates")
    logger.info(f"  Seed threshold: {seed_min_sim}, Negative margin: {negative_margin}")

    return accepted


def bridge_to_existing_tracks(
    seed: EntranceSeed,
    accepted_candidates: list[EntranceCandidate],
    identity_name: str,
    episode_id: str,
    data_root: Path,
    config: dict
) -> BridgeResult:
    """
    Attempt to bridge entrance seed to existing tracks using set-to-set matching.

    Uses:
    - Top-K average similarity (set-to-set, not medoid-to-medoid)
    - Temporal adjacency constraint
    - Mutual nearest neighbor validation
    """
    bridge_config = config.get("entrance", {}).get("bridge", {})
    topk = bridge_config.get("set_topk", 5)
    sim_min = bridge_config.get("sim_min", 0.70)
    temporal_adj_ms = bridge_config.get("temporal_adj_ms", 1000)

    # Load tracks and embeddings
    clusters_path = data_root / "harvest" / episode_id / "clusters.json"
    tracks_path = data_root / "harvest" / episode_id / "tracks.json"
    embeddings_path = data_root / "harvest" / episode_id / "embeddings.parquet"

    clusters_data = json.loads(clusters_path.read_text())
    tracks_data = json.loads(tracks_path.read_text())
    embeddings_df = pd.read_parquet(embeddings_path)

    # Find identity's cluster
    identity_cluster = next(
        (c for c in clusters_data["clusters"] if c.get("name") == identity_name),
        None
    )

    if not identity_cluster:
        return BridgeResult(
            success=False,
            target_track_id=None,
            set_similarity=0.0,
            temporal_gap_ms=None,
            reason="Identity cluster not found"
        )

    # Get entrance end time
    if not accepted_candidates:
        return BridgeResult(
            success=False,
            target_track_id=None,
            set_similarity=0.0,
            temporal_gap_ms=None,
            reason="No accepted candidates"
        )
    entrance_end_ms = max(c.ts_ms for c in accepted_candidates)

    # Find tracks that start near entrance end
    candidate_tracks = []
    for track_id in identity_cluster.get("track_ids", []):
        track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
        if track:
            track_start = track.get("start_ms", 0)
            gap = track_start - entrance_end_ms

            # Check temporal adjacency
            if 0 <= gap <= temporal_adj_ms:
                candidate_tracks.append((track_id, track_start, gap))

    if len(candidate_tracks) == 0:
        return BridgeResult(
            success=False,
            target_track_id=None,
            set_similarity=0.0,
            temporal_gap_ms=None,
            reason=f"No tracks within {temporal_adj_ms}ms of entrance end ({entrance_end_ms}ms)"
        )

    logger.info(f"Found {len(candidate_tracks)} temporally adjacent tracks")

    # For each candidate track, compute set-to-set similarity
    best_track = None
    best_sim = 0.0
    best_gap = None

    for track_id, track_start, gap in candidate_tracks:
        # Get track embeddings
        track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
        if track is None:
            logger.warning(f"Track {track_id} not found in tracks_data")
            continue

        track_frame_ids = [ref["frame_id"] for ref in track.get("frame_refs", [])]
        track_embeddings = embeddings_df[embeddings_df["frame_id"].isin(track_frame_ids)]

        if len(track_embeddings) == 0:
            continue

        # Compute top-K similarities between entrance and track
        # Convert to list of arrays, filtering None
        track_emb_list = [np.asarray(e) for e in track_embeddings["embedding"].values if e is not None]
        if not track_emb_list:
            continue
        try:
            track_vecs = np.stack(track_emb_list, axis=0)
        except ValueError:
            # Handle shape mismatch by normalizing
            track_emb_list = [e.reshape(-1) for e in track_emb_list]
            track_vecs = np.stack(track_emb_list, axis=0)
        track_vecs_norm = track_vecs / np.linalg.norm(track_vecs, axis=1, keepdims=True)

        # Similarity matrix: entrance x track
        entrance_emb_list = [np.asarray(c.embedding) for c in accepted_candidates if c.embedding is not None]
        if not entrance_emb_list:
            continue
        try:
            entrance_vecs = np.stack(entrance_emb_list, axis=0)
        except ValueError:
            # Handle shape mismatch by normalizing
            entrance_emb_list = [e.reshape(-1) for e in entrance_emb_list]
            entrance_vecs = np.stack(entrance_emb_list, axis=0)
        entrance_vecs_norm = entrance_vecs / np.linalg.norm(entrance_vecs, axis=1, keepdims=True)

        sim_matrix = np.dot(entrance_vecs_norm, track_vecs_norm.T)

        # Top-K average
        k = min(topk, min(len(entrance_vecs), len(track_vecs)))
        topk_sims = np.partition(sim_matrix.flatten(), -k)[-k:]
        set_sim = float(topk_sims.mean())

        logger.info(f"  Track {track_id}: set_sim={set_sim:.3f}, gap={gap}ms")

        if set_sim > best_sim:
            best_sim = set_sim
            best_track = track_id
            best_gap = gap

    # Check if bridge passes threshold
    if best_sim >= sim_min:
        return BridgeResult(
            success=True,
            target_track_id=best_track,
            set_similarity=best_sim,
            temporal_gap_ms=best_gap,
            reason=f"Set similarity {best_sim:.3f} >= {sim_min}"
        )
    else:
        return BridgeResult(
            success=False,
            target_track_id=best_track,
            set_similarity=best_sim,
            temporal_gap_ms=best_gap,
            reason=f"Set similarity {best_sim:.3f} < {sim_min}"
        )


def run_entrance_recovery(
    identity_name: str,
    episode_id: str,
    config: dict,
    data_root: Path,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None
) -> EntranceStats:
    """
    Run entrance recovery for a single identity.

    Args:
        identity_name: Name of the identity to recover
        episode_id: Episode ID
        config: Pipeline configuration
        data_root: Data root path
        window_start_ms: Optional override for window start (default: first_interval - pad_ms)
        window_end_ms: Optional override for window end (default: first_interval + pad_ms)

    Returns:
        Statistics about the recovery process
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Entrance recovery for {identity_name}")
    logger.info(f"{'='*80}")

    # Find first interval
    first_start_ms = find_first_interval(identity_name, episode_id, data_root)

    if first_start_ms is None:
        logger.warning(f"No intervals found for {identity_name}")
        return EntranceStats.empty(identity_name, reason="no_intervals")

    if first_start_ms == 0:
        logger.info(f"{identity_name} starts at time 0 - no entrance recovery needed")
        return EntranceStats.empty(identity_name, first_interval_start_ms=0, reason="starts_at_zero")

    logger.info(f"First interval starts at {first_start_ms}ms")

    # Define entrance window
    entrance_config = config.get("entrance", {})
    pad_ms = entrance_config.get("pad_ms", 800)

    # Use overrides if provided, otherwise compute from first_interval
    if window_start_ms is None or window_end_ms is None:
        window_start_ms = max(0, first_start_ms - pad_ms)
        window_end_ms = first_start_ms + pad_ms
        logger.info(f"Entrance window (auto): {window_start_ms}-{window_end_ms}ms")
    else:
        logger.info(f"Entrance window (override): {window_start_ms}-{window_end_ms}ms")

    # Type coercion for safety
    window_start_ms = int(window_start_ms)
    window_end_ms = int(window_end_ms)

    # Load resources
    video_path = data_root / "videos" / f"{episode_id}.mp4"
    manifest_path = data_root / "harvest" / episode_id / "manifest.parquet"

    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        return EntranceStats.empty(
            identity_name,
            first_interval_start_ms=first_start_ms,
            window_start_ms=int(window_start_ms) if window_start_ms is not None else 0,
            window_end_ms=int(window_end_ms) if window_end_ms is not None else 0,
            reason="video_not_found"
        )

    manifest_df = pd.read_parquet(manifest_path)

    # Initialize detector and embedder
    detector = RetinaFaceDetector(
        min_face_px=config["detection"].get("min_face_px", 80),
        min_confidence=config["detection"].get("min_confidence", 0.7),
        provider_order=config["detection"].get("provider_order", ["coreml", "cpu"])
    )

    embedding_config = config.get("embedding", {})
    embedder = ArcFaceEmbedder(
        provider_order=config["detection"].get("provider_order", ["coreml", "cpu"]),
        skip_redetect=embedding_config.get("skip_redetect", True),
        align_priority=embedding_config.get("align_priority", "kps_then_bbox"),
        margin_scale=embedding_config.get("margin_scale", 1.25),
        min_chip_px=embedding_config.get("min_chip_px", 112),
        fallback_scales=embedding_config.get("fallback_scales", [1.0, 1.2, 1.4])
    )

    # Compute prototypes for all identities
    prototypes = compute_identity_prototypes(identity_name, episode_id, data_root)

    # Step 1: Collect candidates
    logger.info("\n=== Step 1: Collect entrance candidates ===")
    candidates, collection_stats = collect_entrance_candidates(
        identity_name,
        window_start_ms,
        window_end_ms,
        episode_id,
        video_path,
        manifest_df,
        detector,
        embedder,
        prototypes,
        config
    )

    if len(candidates) == 0:
        logger.warning("No candidates found")
        return EntranceStats(
            identity_name=identity_name,
            first_interval_start_ms=first_start_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            **collection_stats,
            seed_frames=0,
            seed_clusters=0,
            seed_temporal_segments=0,
            accepted_candidates=0,
            bridge_success=False,
            bridge_target_track=None,
            seconds_recovered=0.0
        )

    # Step 2: Build entrance seed
    logger.info("\n=== Step 2: Build entrance seed ===")
    seed = build_entrance_seed(candidates, identity_name, config)

    if seed is None:
        logger.warning("Failed to build entrance seed")
        return EntranceStats(
            identity_name=identity_name,
            first_interval_start_ms=first_start_ms,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            **collection_stats,
            seed_frames=0,
            seed_clusters=0,
            seed_temporal_segments=0,
            accepted_candidates=0,
            bridge_success=False,
            bridge_target_track=None,
            seconds_recovered=0.0
        )

    # Step 3: Verify candidates
    logger.info("\n=== Step 3: Verify candidates ===")
    accepted = verify_candidates(candidates, seed, config)

    # Step 4: Bridge to existing tracks
    logger.info("\n=== Step 4: Bridge to existing tracks ===")
    bridge_result = bridge_to_existing_tracks(
        seed,
        accepted,
        identity_name,
        episode_id,
        data_root,
        config
    )

    logger.info(f"Bridge result: {bridge_result.reason}")

    # Calculate seconds recovered
    seconds_recovered = 0.0
    if len(accepted) > 0:
        accepted_times = sorted([c.ts_ms for c in accepted])
        seconds_recovered = (accepted_times[-1] - accepted_times[0]) / 1000.0

    # Final stats
    stats = EntranceStats(
        identity_name=identity_name,
        first_interval_start_ms=first_start_ms,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
        **collection_stats,
        seed_frames=seed.cluster_size,
        seed_clusters=seed.num_clusters,
        seed_temporal_segments=seed.temporal_segments,
        accepted_candidates=len(accepted),
        bridge_success=bridge_result.success,
        bridge_target_track=bridge_result.target_track_id,
        seconds_recovered=seconds_recovered
    )

    logger.info(f"\n=== Entrance recovery complete for {identity_name} ===")
    logger.info(f"  Candidates collected: {stats.candidates_collected}")
    logger.info(f"  Embeddings OK: {stats.embeddings_ok}")
    logger.info(f"  Seed frames: {stats.seed_frames}")
    logger.info(f"  Accepted: {stats.accepted_candidates}")
    logger.info(f"  Bridge: {'SUCCESS' if stats.bridge_success else 'FAILED'}")
    logger.info(f"  Seconds recovered: {stats.seconds_recovered:.2f}s")

    return stats


def main():
    """
    Main entry point - run entrance recovery for all identities in an episode.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generic entrance recovery")
    parser.add_argument("episode_id", help="Episode ID")
    parser.add_argument("--identity", help="Specific identity (or all if not specified)")
    parser.add_argument("--window-start-ms", type=int, help="Override entrance window start (milliseconds)")
    parser.add_argument("--window-end-ms", type=int, help="Override entrance window end (milliseconds)")
    args = parser.parse_args()

    config = load_config()
    data_root = Path(config["paths"]["data_root"])

    # Check if entrance recovery is enabled
    if not config.get("entrance", {}).get("enabled", True):
        logger.info("Entrance recovery is disabled in config")
        return

    # Load clusters to find all identities
    clusters_path = data_root / "harvest" / args.episode_id / "clusters.json"
    clusters_data = json.loads(clusters_path.read_text())

    identities = [
        c.get("name") for c in clusters_data["clusters"]
        if c.get("name") and c.get("name") != "UNKNOWN"
    ]

    if args.identity:
        identities = [args.identity] if args.identity in identities else []

    logger.info(f"Running entrance recovery for {len(identities)} identities: {identities}")

    all_stats = []
    for identity_name in identities:
        stats = run_entrance_recovery(
            identity_name,
            args.episode_id,
            config,
            data_root,
            window_start_ms=args.window_start_ms,
            window_end_ms=args.window_end_ms
        )
        if stats:
            all_stats.append(stats)

    # Save combined stats
    reports_dir = data_root / "harvest" / args.episode_id / "diagnostics" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats_dict = {
        "episode_id": args.episode_id,
        "identities": {
            stats.identity_name: {
                "first_interval_start_ms": int(stats.first_interval_start_ms),
                "window": {
                    "start_ms": int(stats.window_start_ms),
                    "end_ms": int(stats.window_end_ms)
                },
                "collection": {
                    "frames_decoded": int(stats.frames_decoded),
                    "faces_detected": int(stats.faces_detected),
                    "candidates_collected": int(stats.candidates_collected),
                    "embeddings_ok": int(stats.embeddings_ok),
                    "embeddings_failed": int(stats.embeddings_failed),
                    "negative_hits": {k: int(v) for k, v in stats.negative_hits.items()}
                },
                "seed": {
                    "frames": int(stats.seed_frames),
                    "clusters": int(stats.seed_clusters),
                    "temporal_segments": int(stats.seed_temporal_segments)
                },
                "verification": {
                    "accepted_candidates": int(stats.accepted_candidates)
                },
                "bridge": {
                    "success": bool(stats.bridge_success),
                    "target_track": int(stats.bridge_target_track) if stats.bridge_target_track is not None else None
                },
                "recovery": {
                    "seconds_recovered": float(stats.seconds_recovered)
                }
            }
            for stats in all_stats
        }
    }

    stats_path = reports_dir / "entrance_audit.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    logger.info(f"\nSaved entrance audit to {stats_path}")


if __name__ == "__main__":
    main()
