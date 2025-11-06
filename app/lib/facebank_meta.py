"""
Facebank metadata helpers for resolving featured thumbnails and managing person metadata.
"""

from pathlib import Path
import json
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple
from PIL import Image
import streamlit as st


def get_person_meta(show_id: str, season_id: str, person: str) -> dict:
    """Load person_meta.json for a given person."""
    person_dir = Path("data/facebank") / show_id / season_id / person
    meta_path = person_dir / "person_meta.json"

    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except:
            pass

    return {}


def save_person_meta(show_id: str, season_id: str, person: str, meta: dict):
    """Save person_meta.json atomically."""
    person_dir = Path("data/facebank") / show_id / season_id / person
    person_dir.mkdir(parents=True, exist_ok=True)

    meta_path = person_dir / "person_meta.json"
    tmp_path = person_dir / "person_meta.json.tmp"

    with open(tmp_path, 'w') as f:
        json.dump(meta, f, indent=2)

    tmp_path.replace(meta_path)


def choose_best_seed(show_id: str, season_id: str, person: str) -> Optional[Path]:
    """
    Choose best seed by quality heuristic:
    - Largest face with highest confidence
    - Falls back to first seed if no metadata
    """
    person_dir = Path("data/facebank") / show_id / season_id / person

    if not person_dir.exists():
        return None

    seed_files = sorted(person_dir.glob("seed_*.png"))
    if not seed_files:
        return None

    # Try to use seeds_metadata.json if available
    meta_path = person_dir / "seeds_metadata.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                seeds = meta.get('seeds', [])

                # Sort by face_height (largest) and confidence (highest)
                seeds_sorted = sorted(
                    seeds,
                    key=lambda s: (s.get('face_height', 0), s.get('confidence', 0)),
                    reverse=True
                )

                if seeds_sorted:
                    best_path = Path(seeds_sorted[0]['path'])
                    if best_path.exists():
                        return best_path
        except:
            pass

    # Fallback to first seed
    return seed_files[0] if seed_files else None


def resolve_person_thumbnail(show_id: str, season_id: str, person: str) -> Optional[str]:
    """
    Resolve person thumbnail in priority order:
    1. Featured seed from person_meta.json
    2. Best seed by quality heuristic
    3. First seed_*.png
    4. None (caller should use placeholder)
    """
    person_dir = Path("data/facebank") / show_id / season_id / person

    # Check for featured seed
    meta = get_person_meta(show_id, season_id, person)
    if meta and meta.get("featured_seed"):
        featured_path = person_dir / meta["featured_seed"]
        if featured_path.exists():
            return str(featured_path)

    # Fallback to best seed
    best = choose_best_seed(show_id, season_id, person)
    return str(best) if best else None


def calculate_seed_quality_metrics(seed_path: Path, seeds_metadata: Optional[dict] = None) -> Dict[str, Any]:
    """
    Calculate quality metrics for a seed image.

    Returns dict with:
    - filename: str
    - dimensions: tuple (width, height)
    - file_size_bytes: int
    - file_size_kb: float
    - face_bbox: tuple (x, y, w, h) if available
    - coverage_pct: float (face area / image area * 100)
    - sharpness_score: float (Laplacian variance)
    - brightness_mean: float (mean pixel value)
    - contrast_std: float (std dev of pixel values)
    - quality_score: float (composite 0-100)
    """
    metrics = {
        'filename': seed_path.name,
        'file_size_bytes': 0,
        'file_size_kb': 0.0,
        'dimensions': (0, 0),
        'face_bbox': None,
        'coverage_pct': 0.0,
        'sharpness_score': 0.0,
        'brightness_mean': 0.0,
        'contrast_std': 0.0,
        'quality_score': 0.0
    }

    if not seed_path.exists():
        return metrics

    # File size
    metrics['file_size_bytes'] = seed_path.stat().st_size
    metrics['file_size_kb'] = metrics['file_size_bytes'] / 1024.0

    # Load image
    img = cv2.imread(str(seed_path))
    if img is None:
        return metrics

    h, w = img.shape[:2]
    metrics['dimensions'] = (w, h)

    # Convert to grayscale for quality metrics
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sharpness (Laplacian variance) - higher is sharper
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    metrics['sharpness_score'] = float(laplacian.var())

    # Brightness (mean pixel value)
    metrics['brightness_mean'] = float(gray.mean())

    # Contrast (standard deviation)
    metrics['contrast_std'] = float(gray.std())

    # Try to get face bbox from seeds_metadata
    if seeds_metadata:
        seeds = seeds_metadata.get('seeds', [])
        for seed in seeds:
            if seed_path.name in seed.get('path', ''):
                # Face bbox from metadata (if available)
                face_height = seed.get('face_height', 0)
                if face_height > 0:
                    # Estimate bbox (we don't have full bbox in current metadata, just height)
                    # Assume square face and centered
                    face_w = face_height
                    face_h = face_height
                    face_area = face_w * face_h
                    img_area = w * h
                    metrics['coverage_pct'] = (face_area / img_area) * 100 if img_area > 0 else 0.0
                    metrics['face_bbox'] = (0, 0, int(face_w), int(face_h))  # Placeholder
                break

    # Composite quality score (0-100)
    # Normalize sharpness (typical range 0-1000+, cap at 500)
    sharpness_norm = min(metrics['sharpness_score'] / 500.0, 1.0) * 40

    # Normalize brightness (ideal around 127, penalize too dark/bright)
    brightness_score = (1.0 - abs(metrics['brightness_mean'] - 127) / 127) * 30

    # Normalize contrast (typical range 0-100, higher is better)
    contrast_norm = min(metrics['contrast_std'] / 100.0, 1.0) * 30

    metrics['quality_score'] = sharpness_norm + brightness_score + contrast_norm

    return metrics


def crop_to_ratio(img: Image.Image, ratio: float = 4/5, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Image.Image:
    """
    Crop image to target aspect ratio (width/height).

    Args:
        img: PIL Image to crop
        ratio: Target width/height ratio (default 4/5 for 4:5 aspect)
        face_bbox: Optional (x, y, w, h) face bounding box to center crop around

    Returns:
        Cropped PIL Image
    """
    w, h = img.size
    target = ratio
    cur = w / float(h)

    # Already correct ratio
    if abs(cur - target) < 1e-3:
        return img

    if cur > target:
        # Too wide → crop left/right
        new_w = int(h * target)

        # If face bbox provided, try to center around face
        if face_bbox:
            face_x, face_y, face_w, face_h = face_bbox
            face_center_x = face_x + face_w // 2
            x0 = max(0, min(w - new_w, face_center_x - new_w // 2))
        else:
            # Center crop
            x0 = (w - new_w) // 2

        return img.crop((x0, 0, x0 + new_w, h))
    else:
        # Too tall → crop top/bottom
        new_h = int(w / target)

        # If face bbox provided, try to center around face
        if face_bbox:
            face_x, face_y, face_w, face_h = face_bbox
            face_center_y = face_y + face_h // 2
            y0 = max(0, min(h - new_h, face_center_y - new_h // 2))
        else:
            # Center crop
            y0 = (h - new_h) // 2

        return img.crop((0, y0, w, y0 + new_h))


@st.cache_data(show_spinner=False, max_entries=2048)
def load_thumbnail(img_path: str, size: Tuple[int, int] = (160, 200), mtime: float = None) -> Image.Image:
    """
    Load and cache a 4:5 thumbnail from a seed image.

    Args:
        img_path: Path to the image file
        size: Target thumbnail size (width, height)
        mtime: File modification time (used for cache busting)

    Returns:
        PIL Image resized to target size with 4:5 aspect ratio
    """
    # mtime is only used to bust cache when file changes
    img = Image.open(img_path).convert("RGB")
    # Crop to 4:5 ratio first
    img = crop_to_ratio(img, ratio=4/5)
    # Resize to target size
    img = img.resize(size, Image.LANCZOS)
    return img


def get_thumbnail(img_path: Path, size: Tuple[int, int] = (160, 200)) -> Image.Image:
    """
    Get a cache-busted thumbnail for display.

    Args:
        img_path: Path object to the image
        size: Target thumbnail size

    Returns:
        PIL Image thumbnail
    """
    return load_thumbnail(str(img_path), size, mtime=img_path.stat().st_mtime)


def compute_quality_from_image(img: Image.Image, face_coverage: float = 0.0) -> Dict[str, Any]:
    """
    Compute quality metrics from a PIL Image.

    Args:
        img: PIL Image to analyze
        face_coverage: Optional face coverage percentage (0-100)

    Returns:
        Dict with quality_score, sharpness, brightness, contrast
    """
    # Convert to grayscale numpy array
    gray = np.array(img.convert("L"), dtype=np.uint8)

    # Sharpness (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(laplacian.var())

    # Brightness (mean pixel value)
    brightness = float(gray.mean())

    # Contrast (standard deviation)
    contrast = float(gray.std())

    # Normalize components to 0-100 scale
    # Sharpness: typical range 0-1000+, cap at 500
    sharp_norm = min(sharpness / 500.0, 1.0) * 40

    # Brightness: ideal around 127, penalize too dark/bright
    bright_norm = (1.0 - abs(brightness - 127) / 127) * 30

    # Contrast: typical range 0-100, higher is better
    contrast_norm = min(contrast / 100.0, 1.0) * 20

    # Face coverage bonus (if available)
    coverage_bonus = min(face_coverage / 100.0, 1.0) * 10

    # Composite score (0-100)
    quality_score = sharp_norm + bright_norm + contrast_norm + coverage_bonus

    return {
        'quality_score': round(quality_score, 1),
        'sharpness': round(sharpness, 1),
        'brightness': round(brightness, 1),
        'contrast': round(contrast, 1)
    }


def recompute_bank_metrics(show_id: str, season_id: str) -> None:
    """
    Recompute bank metrics for all persons in a facebank.

    Computes:
    - bank_conf_median_p25: Median of P25 confidence scores across seed embeddings
    - bank_contam_rate: Percentage of seeds with max similarity to another identity

    Writes results to person_meta.json for each person.

    Args:
        show_id: Show identifier (e.g., "rhobh")
        season_id: Season identifier (e.g., "s05")
    """
    from datetime import datetime
    import torch
    import torch.nn.functional as F

    facebank_root = Path("data/facebank") / show_id / season_id

    if not facebank_root.exists():
        return

    # Load all person embeddings
    person_embeddings = {}
    for person_dir in facebank_root.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        emb_files = list(person_dir.glob("seed_*.pt"))

        if not emb_files:
            continue

        # Load all seed embeddings for this person
        embeddings = []
        for emb_file in emb_files:
            try:
                emb = torch.load(emb_file, map_location="cpu")
                embeddings.append(emb)
            except:
                pass

        if embeddings:
            person_embeddings[person_name] = torch.stack(embeddings)

    if not person_embeddings:
        return

    # Compute metrics for each person
    for person_name, embeddings in person_embeddings.items():
        # Compute P25 confidence (similarity to own seeds)
        # For each seed, compute similarity to all other seeds of same person
        confidences = []
        for i, emb in enumerate(embeddings):
            # Get all other embeddings for this person
            other_embs = torch.cat([embeddings[:i], embeddings[i+1:]], dim=0) if len(embeddings) > 1 else embeddings

            if len(other_embs) > 0:
                # Compute cosine similarity
                sim = F.cosine_similarity(emb.unsqueeze(0), other_embs, dim=1)
                confidences.extend(sim.tolist())

        # Compute P25 (25th percentile)
        if confidences:
            confidences.sort()
            p25_idx = int(len(confidences) * 0.25)
            p25 = confidences[p25_idx] if p25_idx < len(confidences) else confidences[0]
        else:
            p25 = 0.0

        # Compute contamination rate
        # For each seed, check if max similarity to other identities exceeds threshold
        contaminated_count = 0
        threshold = 0.5  # Similarity threshold for contamination

        for emb in embeddings:
            max_other_sim = 0.0

            for other_person, other_embs in person_embeddings.items():
                if other_person == person_name:
                    continue

                # Compute similarity to all seeds of other person
                sim = F.cosine_similarity(emb.unsqueeze(0), other_embs, dim=1)
                max_sim = float(sim.max())

                if max_sim > max_other_sim:
                    max_other_sim = max_sim

            if max_other_sim > threshold:
                contaminated_count += 1

        contam_rate = contaminated_count / len(embeddings) if len(embeddings) > 0 else 0.0

        # Save to person_meta.json
        meta = get_person_meta(show_id, season_id, person_name)
        meta["bank"] = {
            "p25": round(p25, 3),
            "contam": round(contam_rate, 3),
            "updated_at": datetime.now().isoformat()
        }
        save_person_meta(show_id, season_id, person_name, meta)
