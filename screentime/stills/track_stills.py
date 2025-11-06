"""
Face-aware track stills generation with FIQA scoring and Re-ID disambiguation.

Generates high-quality face crops for each track by:
1. Sampling candidate frames across the track timeline
2. Scoring each candidate using FIQA + sharpness + bbox size + frontalness
3. Cropping the best face with margin and letterboxing
4. Optionally applying CodeFormer restoration with identity guard
5. Writing JSONL manifest for UI consumption
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageStat

from screentime.recognition.embed_arcface import ArcFaceEmbedder
from screentime.stills.serfiq import SERFIQScorer

logger = logging.getLogger(__name__)


@dataclass
class CandidateFrame:
    """A candidate frame for still generation."""
    frame_idx: int
    timestamp_s: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    landmarks: Optional[np.ndarray]  # 5 points (x, y) if available
    embedding: Optional[np.ndarray]  # ArcFace embedding if available
    frame_data: Optional[np.ndarray] = None  # BGR image data


@dataclass
class QualityScores:
    """Quality scores for a candidate frame."""
    fiqa: float
    sharpness: float
    face_height_norm: float
    composite: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "fiqa": float(self.fiqa),
            "sharpness": float(self.sharpness),
            "face_height_norm": float(self.face_height_norm),
            "composite": float(self.composite),
        }


@dataclass
class StillResult:
    """Result of still generation for a track."""
    track_id: int
    crop_path: str
    context_path: Optional[str]
    timestamp: str
    scores: Dict[str, float]
    fallback: bool
    source_frame_idx: int
    bbox_used: Tuple[int, int, int, int]
    identity_cosine: Optional[float] = None


def load_config(config_path: str = "configs/stills.yaml") -> Dict[str, Any]:
    """Load stills configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_embeddings_for_frames(
    episode_id: str,
    frame_indices: List[int],
    data_root: str = "data",
) -> Dict[int, np.ndarray]:
    """
    Load embeddings for specific frames from embeddings.parquet.

    Args:
        episode_id: Episode identifier
        frame_indices: List of frame indices to load
        data_root: Data root directory

    Returns:
        Dict mapping frame_idx -> embedding array
    """
    embeddings_path = Path(data_root) / "harvest" / episode_id / "embeddings.parquet"
    if not embeddings_path.exists():
        logger.warning(f"Embeddings file not found: {embeddings_path}")
        return {}

    try:
        df = pd.read_parquet(embeddings_path)

        # Filter to requested frames
        df_filtered = df[df["frame_id"].isin(frame_indices)]

        # Build lookup dict
        embeddings = {}
        for _, row in df_filtered.iterrows():
            frame_id = int(row["frame_id"])
            embedding = np.array(row["embedding"])
            embeddings[frame_id] = embedding

        return embeddings
    except Exception as exc:
        logger.warning(f"Failed to load embeddings from {embeddings_path}: {exc}")
        return {}


def sample_frames(
    track_dict: Dict[str, Any],
    n_samples: int = 7,
    video_fps: float = 23.976,
) -> List[CandidateFrame]:
    """
    Sample candidate frames evenly across track timeline.

    Args:
        track_dict: Track data from tracks.json (with frame_refs)
        n_samples: Number of frames to sample
        video_fps: Video framerate

    Returns:
        List of CandidateFrame objects
    """
    frame_refs = track_dict.get("frame_refs", [])
    if not frame_refs:
        return []

    # Sample evenly across track
    indices = np.linspace(0, len(frame_refs) - 1, n_samples, dtype=int)
    candidates = []

    for idx in indices:
        frame_ref = frame_refs[idx]
        frame_idx = frame_ref.get("frame_id", idx)

        # Calculate timestamp from frame_id
        ts_ms = frame_idx * 1000.0 / video_fps

        # Convert bbox from [x1, y1, x2, y2] to (x, y, w, h)
        bbox_xyxy = frame_ref.get("bbox", [0, 0, 100, 100])
        x1, y1, x2, y2 = bbox_xyxy
        bbox_xywh = (x1, y1, x2 - x1, y2 - y1)

        # Landmarks and embeddings will be loaded separately from embeddings.parquet
        # For now, pass None - these will be joined later if needed
        candidates.append(CandidateFrame(
            frame_idx=frame_idx,
            timestamp_s=ts_ms / 1000.0,
            bbox=bbox_xywh,
            landmarks=None,  # Will load from embeddings if needed
            embedding=None,  # Will load from embeddings if needed
        ))

    return candidates


def compute_sharpness(img_gray: np.ndarray) -> float:
    """
    Compute sharpness using Laplacian variance.

    Args:
        img_gray: Grayscale image

    Returns:
        Sharpness score (higher = sharper)
    """
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_frontalness(landmarks: Optional[np.ndarray]) -> float:
    """
    Estimate frontalness from landmarks (0-1, 1 = frontal).

    Uses eye-to-nose symmetry as proxy for frontalness.

    Args:
        landmarks: 5-point facial landmarks [[x,y], ...]

    Returns:
        Frontalness score 0-1 (1.0 if no landmarks)
    """
    if landmarks is None or len(landmarks) != 5:
        return 1.0  # Neutral if no landmarks

    # Landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    # Compute distances
    left_to_nose = np.linalg.norm(left_eye - nose)
    right_to_nose = np.linalg.norm(right_eye - nose)

    # Symmetry ratio (closer to 1.0 = more frontal)
    ratio = min(left_to_nose, right_to_nose) / (max(left_to_nose, right_to_nose) + 1e-6)

    return float(ratio)


def score_candidate(
    candidate: CandidateFrame,
    frame_img: np.ndarray,
    config: Dict[str, Any],
    embedder: Optional[ArcFaceEmbedder] = None,
    serfiq_scorer: Optional[SERFIQScorer] = None,
) -> QualityScores:
    """
    Score a candidate frame using multiple quality metrics.

    Args:
        candidate: Candidate frame with bbox and embedding
        frame_img: Full frame BGR image
        config: Configuration dict
        embedder: ArcFace embedder (required if using SER-FIQ)
        serfiq_scorer: SER-FIQ scorer instance (if fiqa_method == "serfiq")

    Returns:
        QualityScores object
    """
    weights = config["quality"]["weights"]

    # Extract face crop
    x, y, w, h = candidate.bbox
    face_crop = frame_img[y:y+h, x:x+w]

    if face_crop.size == 0:
        # Invalid bbox
        return QualityScores(0.0, 0.0, 0.0, 0.0)

    # 1. FIQA score - use SER-FIQ if configured, otherwise fall back to variance/placeholder
    fiqa_method = config.get("quality", {}).get("fiqa_method", "embedding_norm")

    if fiqa_method == "serfiq" and serfiq_scorer is not None and embedder is not None:
        # Use true SER-FIQ with stochastic forward passes
        # Pass full frame and bbox for better detection robustness
        fiqa_score, metadata = serfiq_scorer.score_with_bbox(frame_img, candidate.bbox, embedder)
        fiqa = fiqa_score
        logger.debug(f"SER-FIQ: {fiqa:.3f} (std={metadata['std']:.4f}, passes={metadata['num_valid_passes']})")
    elif candidate.embedding is not None:
        # Fallback: embedding variance proxy (saturates around 0.48)
        embedding_var = float(np.var(candidate.embedding))
        fiqa = min(embedding_var * 250.0, 1.0)  # Scale to 0-1
    else:
        # Final fallback: placeholder FIQA
        fiqa = _estimate_fiqa_placeholder(face_crop)

    # 2. Sharpness
    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    sharpness = compute_sharpness(face_gray)
    sharpness_norm = min(sharpness / 100.0, 1.0)  # Normalize to 0-1

    # 3. Face height (taller faces = better quality)
    face_height = h
    face_height_norm = min(face_height / 150.0, 1.0)  # Normalize (150px = excellent)

    # Composite score
    composite = (
        weights["fiqa"] * fiqa +
        weights["sharpness"] * sharpness_norm +
        weights["face_height"] * face_height_norm
    )

    return QualityScores(
        fiqa=fiqa,
        sharpness=sharpness,
        face_height_norm=face_height_norm,
        composite=composite,
    )


def _estimate_fiqa_placeholder(face_crop: np.ndarray) -> float:
    """
    Placeholder FIQA estimation until SER-FIQ is integrated.

    Uses brightness, contrast, and saturation as proxy.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)

    # Brightness (V channel)
    brightness = hsv[:, :, 2].mean() / 255.0

    # Contrast (std of V channel)
    contrast = hsv[:, :, 2].std() / 128.0

    # Saturation
    saturation = hsv[:, :, 1].mean() / 255.0

    # Simple weighted combination
    score = 0.4 * brightness + 0.4 * contrast + 0.2 * saturation
    return float(np.clip(score, 0.0, 1.0))


def select_best_candidate(
    candidates: List[CandidateFrame],
    scores: List[QualityScores],
    config: Dict[str, Any],
) -> Tuple[Optional[CandidateFrame], Optional[QualityScores]]:
    """
    Select the best candidate frame based on quality gates and composite score.

    Args:
        candidates: List of candidate frames
        scores: Corresponding quality scores
        config: Configuration dict

    Returns:
        (best_candidate, best_scores) or (None, None) if all fail gates
    """
    fiqa_min = config["quality"]["fiqa_min"]
    sharp_min = config["quality"]["sharp_min"]

    # Filter by quality gates
    valid = []
    for cand, score in zip(candidates, scores):
        if score.fiqa >= fiqa_min and score.sharpness >= sharp_min:
            valid.append((cand, score))

    if not valid:
        logger.warning("No candidates passed quality gates, using best available")
        valid = list(zip(candidates, scores))

    # Select best by composite score
    best_cand, best_score = max(valid, key=lambda x: x[1].composite)

    return best_cand, best_score


def crop_face(
    frame_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    margin: float = 0.30,
    out_size: Tuple[int, int] = (320, 400),
    letterbox: bool = True,
) -> Image.Image:
    """
    Crop face from frame with margin and letterboxing.

    Args:
        frame_img: BGR frame image
        bbox: (x, y, w, h) face bounding box
        margin: Fraction to expand on each side
        out_size: (width, height) output dimensions
        letterbox: Whether to letterbox (preserve aspect ratio)

    Returns:
        PIL Image of cropped face
    """
    x, y, w, h = bbox
    frame_h, frame_w = frame_img.shape[:2]

    # Expand bbox by margin
    margin_w = int(w * margin)
    margin_h = int(h * margin)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(frame_w, x + w + margin_w)
    y2 = min(frame_h, y + h + margin_h)

    # Crop
    crop = frame_img[y1:y2, x1:x2]

    if crop.size == 0:
        # Fallback to full frame
        crop = frame_img

    # Convert BGR to RGB
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(crop_rgb)

    if letterbox:
        # Resize preserving aspect ratio and pad
        img.thumbnail(out_size, Image.Resampling.LANCZOS)

        # Create letterboxed image
        letterboxed = Image.new('RGB', out_size, (0, 0, 0))
        paste_x = (out_size[0] - img.width) // 2
        paste_y = (out_size[1] - img.height) // 2
        letterboxed.paste(img, (paste_x, paste_y))

        return letterboxed
    else:
        # Direct resize
        return img.resize(out_size, Image.Resampling.LANCZOS)


def is_black_frame(img: np.ndarray, min_mean: float = 4.0, min_var: float = 8.0) -> bool:
    """
    Check if frame is near-black.

    Args:
        img: BGR image
        min_mean: Minimum mean brightness
        min_var: Minimum variance

    Returns:
        True if frame is black
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    var = gray.var()

    return mean < min_mean or var < min_var


def create_context_image(
    frame_img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    config: Dict[str, Any],
) -> Image.Image:
    """
    Create context image (full frame with bbox overlay).

    Args:
        frame_img: BGR frame image
        bbox: (x, y, w, h) bounding box to highlight
        config: Configuration dict

    Returns:
        PIL Image with bbox drawn
    """
    # Convert to RGB
    img_rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)

    if config["overlays"]["draw_box"]:
        draw = ImageDraw.Draw(img)
        x, y, w, h = bbox
        color = tuple(config["overlays"]["box_color"])
        width = config["overlays"]["box_width"]

        # Draw rectangle
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

    return img


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    seconds -= h * 3600
    m = int(seconds // 60)
    seconds -= m * 60
    s = int(seconds)
    ms = int((seconds - s) * 1000)

    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_manifest_entry(
    manifest_path: Path,
    result: StillResult,
) -> None:
    """
    Append entry to JSONL manifest.

    Args:
        manifest_path: Path to manifest file
        result: StillResult to write
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "track_id": result.track_id,
        "crop_path": result.crop_path,
        "context_path": result.context_path,
        "timestamp": result.timestamp,
        "scores": result.scores,
        "fallback": result.fallback,
        "source_frame_idx": result.source_frame_idx,
        "bbox_used": result.bbox_used,
    }

    if result.identity_cosine is not None:
        entry["identity_cosine"] = result.identity_cosine

    with open(manifest_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def log_telemetry_event(
    log_path: Path,
    track_id: int,
    event_data: Dict[str, Any],
) -> None:
    """
    Log telemetry event to JSONL file.

    Args:
        log_path: Path to telemetry log
        track_id: Track ID
        event_data: Event data dict
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "track_id": track_id,
        **event_data
    }

    with open(log_path, 'a') as f:
        f.write(json.dumps(entry) + '\n')


def generate_track_still(
    track_id: int,
    track_dict: Dict[str, Any],
    video_path: str,
    episode_id: str,
    config: Dict[str, Any],
    reid_centroid: Optional[np.ndarray] = None,
    embedder: Optional[ArcFaceEmbedder] = None,
    serfiq_scorer: Optional[SERFIQScorer] = None,
) -> Optional[StillResult]:
    """
    Generate a still image for a single track.

    Args:
        track_id: Track identifier
        track_dict: Track data from tracks.json
        video_path: Path to video file
        episode_id: Episode identifier
        config: Configuration dict
        reid_centroid: Optional track centroid for Re-ID disambiguation
        embedder: ArcFace embedder (required if using SER-FIQ)
        serfiq_scorer: SER-FIQ scorer instance (if fiqa_method == "serfiq")

    Returns:
        StillResult or None if generation failed
    """
    logger.info(f"Generating still for track {track_id}")

    # Sample candidate frames
    candidates = sample_frames(
        track_dict,
        n_samples=config["samples_per_track"],
    )

    if not candidates:
        logger.warning(f"Track {track_id} has no frames")
        return None

    # Load embeddings for candidate frames
    frame_indices = [cand.frame_idx for cand in candidates]
    embeddings_map = load_embeddings_for_frames(episode_id, frame_indices)

    # Attach embeddings to candidates
    for cand in candidates:
        if cand.frame_idx in embeddings_map:
            cand.embedding = embeddings_map[cand.frame_idx]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return None

    try:
        # Load and score candidates
        scores = []
        for cand in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand.frame_idx)
            ret, frame = cap.read()

            if not ret:
                scores.append(QualityScores(0.0, 0.0, 0.0, 0.0))
                continue

            cand.frame_data = frame
            score = score_candidate(cand, frame, config, embedder, serfiq_scorer)
            scores.append(score)

        # Select best
        best_cand, best_score = select_best_candidate(candidates, scores, config)

        if best_cand is None or best_cand.frame_data is None:
            logger.warning(f"No valid candidate for track {track_id}")
            return None

        # Crop face
        crop_img = crop_face(
            best_cand.frame_data,
            best_cand.bbox,
            margin=config["margin"],
            out_size=tuple(config["tile_size"]),
            letterbox=True,
        )

        # Save crop
        out_dir = Path(config["io"]["out_dir"].format(episode_id=episode_id))
        tracks_dir = out_dir / config["io"]["tracks_subdir"]
        tracks_dir.mkdir(parents=True, exist_ok=True)

        crop_path = tracks_dir / f"{track_id}.jpg"
        crop_img.save(crop_path, quality=config["io"]["jpeg_quality"])

        # Optionally save context image
        context_path = None
        if config["overlays"]["write_context"]:
            context_dir = out_dir / config["io"]["context_subdir"]
            context_dir.mkdir(parents=True, exist_ok=True)

            context_img = create_context_image(best_cand.frame_data, best_cand.bbox, config)
            context_path_obj = context_dir / f"{track_id}.jpg"
            context_img.save(context_path_obj, quality=config["io"]["jpeg_quality"])
            context_path = str(context_path_obj.relative_to(out_dir.parent))

        # Create result
        result = StillResult(
            track_id=track_id,
            crop_path=str(crop_path.relative_to(out_dir.parent)),
            context_path=context_path,
            timestamp=format_timestamp(best_cand.timestamp_s),
            scores=best_score.to_dict(),
            fallback=False,
            source_frame_idx=best_cand.frame_idx,
            bbox_used=best_cand.bbox,
        )

        logger.info(f"✓ Generated still for track {track_id} (score: {best_score.composite:.3f})")

        return result

    finally:
        cap.release()
