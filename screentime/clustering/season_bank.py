"""
Season bank builder: multi-prototype system for season-wide face recognition.

Bins seeds by pose × scale, keeps top-K per bin for robust open-set assignment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import cv2
import numpy as np


@dataclass
class Prototype:
    """Single prototype embedding with metadata."""
    embedding: List[float]
    source: str  # seed filename
    confidence: float
    face_height: float
    yaw: float
    bin_key: str  # e.g., "frontal_large"


def estimate_pose(landmarks: Optional[np.ndarray]) -> str:
    """
    Estimate face pose from landmarks.

    Args:
        landmarks: 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)

    Returns:
        "frontal", "three_quarter", or "profile"
    """
    if landmarks is None or len(landmarks) != 5:
        return "frontal"  # Default if no landmarks

    # Simple heuristic: eye distance vs face width
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    # Eye distance
    eye_dist = np.linalg.norm(right_eye - left_eye)

    # Distance from nose to eye center
    eye_center = (left_eye + right_eye) / 2
    nose_to_eye = np.linalg.norm(nose - eye_center)

    # Ratio (smaller = more profile)
    ratio = nose_to_eye / (eye_dist + 1e-6)

    if ratio > 0.5:
        return "profile"
    elif ratio > 0.25:
        return "three_quarter"
    else:
        return "frontal"


def build_season_bank(
    show_id: str,
    season_id: str,
    facebank_dir: Path,
    max_prototypes_per_bin: int = 5,
    min_face_height: float = 64.0
) -> Dict:
    """
    Build season bank from seed images.

    Args:
        show_id: Show ID (e.g., "rhobh")
        season_id: Season ID (e.g., "s05")
        facebank_dir: Path to facebank directory (e.g., data/facebank/rhobh/s05/)
        max_prototypes_per_bin: Max prototypes per pose×scale bin
        min_face_height: Minimum face height to include

    Returns:
        Season bank dict ready for JSON serialization
    """
    from screentime.detectors.face_retina import RetinaFaceDetector
    from screentime.recognition.embed_arcface import ArcFaceEmbedder

    detector = RetinaFaceDetector()
    embedder = ArcFaceEmbedder()

    identities = {}

    # Process each cast member
    for cast_dir in sorted(facebank_dir.iterdir()):
        if not cast_dir.is_dir():
            continue

        cast_name = cast_dir.name
        seed_files = sorted(cast_dir.glob("seed_*.png"))

        if not seed_files:
            continue

        print(f"Processing {cast_name}: {len(seed_files)} seeds")

        # Bins: pose × scale
        bins = {
            'frontal_small': [],
            'frontal_large': [],
            'three_quarter_small': [],
            'three_quarter_large': [],
            'profile_small': [],
            'profile_large': []
        }

        for seed_file in seed_files:
            try:
                # Load image
                img = cv2.imread(str(seed_file))
                if img is None:
                    print(f"  ⚠️ Cannot read {seed_file.name}")
                    continue

                # Detect face
                faces = detector.detect(img)

                if len(faces) != 1:
                    print(f"  ⚠️ {seed_file.name}: {len(faces)} faces (expected 1)")
                    continue

                face = faces[0]

                # Quality checks
                face_height = face['bbox'][3] - face['bbox'][1]

                if face['confidence'] < 0.65:
                    print(f"  ⚠️ {seed_file.name}: Low confidence ({face['confidence']:.2f})")
                    continue

                if face_height < min_face_height:
                    print(f"  ⚠️ {seed_file.name}: Too small ({face_height:.0f}px)")
                    continue

                # Embed directly from detection (embedder handles alignment)
                bbox = face['bbox']
                kps = face.get('kps')
                result = embedder.embed_from_detection(img, bbox, kps)

                if not result.success or result.embedding is None:
                    print(f"  ⚠️ {seed_file.name}: Embedding failed")
                    continue

                embedding = result.embedding

                # Determine bin
                pose = estimate_pose(face.get('kps'))
                scale = 'small' if face_height < 100 else 'large'
                bin_key = f"{pose}_{scale}"

                # Compute yaw from landmarks (simplified)
                yaw = 0.0
                if 'kps' in face and face['kps'] is not None:
                    kps = face['kps']
                    if len(kps) >= 2:
                        # Rough yaw estimate from eye positions
                        left_eye = kps[0]
                        right_eye = kps[1]
                        eye_center_x = (left_eye[0] + right_eye[0]) / 2
                        face_center_x = (face['bbox'][0] + face['bbox'][2]) / 2
                        offset = eye_center_x - face_center_x
                        face_width = face['bbox'][2] - face['bbox'][0]
                        yaw = (offset / face_width) * 90  # Rough degrees

                bins[bin_key].append(Prototype(
                    embedding=embedding.tolist(),
                    source=seed_file.name,
                    confidence=float(face['confidence']),
                    face_height=float(face_height),
                    yaw=float(yaw),
                    bin_key=bin_key
                ))

                print(f"  ✓ {seed_file.name}: {bin_key} (conf={face['confidence']:.2f}, h={face_height:.0f}px)")

            except Exception as e:
                print(f"  ✗ {seed_file.name}: {e}")

        # Keep top-K per bin (by confidence)
        for bin_key in bins:
            bins[bin_key] = sorted(
                bins[bin_key],
                key=lambda p: p.confidence,
                reverse=True
            )[:max_prototypes_per_bin]

        # Convert to serializable format
        identities[cast_name] = {
            bin_key: [asdict(p) for p in protos]
            for bin_key, protos in bins.items()
            if protos  # Only include non-empty bins
        }

        total_protos = sum(len(protos) for protos in bins.values())
        print(f"  → {cast_name}: {total_protos} prototypes across {len([b for b in bins.values() if b])} bins")

    # Build season bank
    season_bank = {
        'version': '1.0',
        'show_id': show_id,
        'season_id': season_id,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'config': {
            'max_prototypes_per_bin': max_prototypes_per_bin,
            'min_face_height': min_face_height,
            'pose_bins': ['frontal', 'three_quarter', 'profile'],
            'scale_bins': ['small', 'large']
        },
        'identities': identities
    }

    return season_bank


def save_season_bank(season_bank: Dict, output_path: Path) -> None:
    """
    Save season bank to JSON with atomic write.

    Args:
        season_bank: Season bank dict
        output_path: Where to save (e.g., data/facebank/rhobh/s05/multi_prototypes.json)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write
    tmp_path = output_path.with_suffix('.json.tmp')
    with open(tmp_path, 'w') as f:
        json.dump(season_bank, f, indent=2)

    tmp_path.replace(output_path)


def print_season_bank_summary(season_bank: Dict) -> None:
    """Print summary statistics for season bank."""
    identities = season_bank.get('identities', {})

    print("\n" + "="*60)
    print("SEASON BANK SUMMARY")
    print("="*60)
    print(f"Show: {season_bank.get('show_id')}")
    print(f"Season: {season_bank.get('season_id')}")
    print(f"Created: {season_bank.get('created_at')}")
    print(f"\nCast Members: {len(identities)}")
    print("-"*60)

    # Summary table
    print(f"{'Cast':<15} {'Frontal':<10} {'3/4':<10} {'Profile':<10} {'Total':<10}")
    print("-"*60)

    for cast_name, bins in sorted(identities.items()):
        frontal = sum(len(protos) for key, protos in bins.items() if 'frontal' in key)
        three_q = sum(len(protos) for key, protos in bins.items() if 'three_quarter' in key)
        profile = sum(len(protos) for key, protos in bins.items() if 'profile' in key)
        total = frontal + three_q + profile

        print(f"{cast_name:<15} {frontal:<10} {three_q:<10} {profile:<10} {total:<10}")

    print("="*60 + "\n")
