"""
Generate boundary proof videos/images for gap analysis.

Creates visual evidence showing frames before/after large gaps to confirm
whether faces were missed or if gaps represent true off-screen time.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from screentime.detectors.face_retina import RetinaFaceDetector

logger = logging.getLogger(__name__)


def generate_boundary_proofs(
    job_id: str,
    episode_id: str,
    video_path: str | Path,
    target_identity: str,
    gap_windows: list[dict],
    boundary_ms: int = 1500,
) -> dict:
    """
    Generate boundary proofs for large gaps.

    Args:
        job_id: Job ID
        episode_id: Episode ID
        video_path: Path to video file
        target_identity: Person name (e.g., "YOLANDA")
        gap_windows: List of gap dicts with start_ms, end_ms
        boundary_ms: Milliseconds before/after gap to capture

    Returns:
        Dict with proof results
    """
    logger.info(f"[{job_id}] Generating boundary proofs for {target_identity}")

    # Setup paths
    proofs_dir = Path("data/harvest") / episode_id / "proofs"
    proofs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize detector
    detector = RetinaFaceDetector(
        min_confidence=0.50,  # Lower for edge detection
        min_face_px=32,
        provider_order=["coreml", "cpu"],
    )

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)

    proofs = []

    for gap_idx, gap in enumerate(gap_windows):
        gap_start_ms = gap["start_ms"]
        gap_end_ms = gap["end_ms"]
        gap_duration_ms = gap_end_ms - gap_start_ms

        logger.info(
            f"[{job_id}] Processing gap {gap_idx + 1}: "
            f"{gap_start_ms}-{gap_end_ms}ms ({gap_duration_ms}ms)"
        )

        # Define boundary regions
        before_start_ms = max(0, gap_start_ms - boundary_ms)
        before_end_ms = gap_start_ms
        after_start_ms = gap_end_ms
        after_end_ms = gap_end_ms + boundary_ms

        # Collect frames from before/after boundaries
        boundary_frames = []
        face_detections = []

        # Before gap
        for ts_ms in range(before_start_ms, before_end_ms, 100):  # 10fps sampling
            frame_num = int((ts_ms / 1000) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                faces = detector.detect(frame)
                boundary_frames.append({
                    "ts_ms": ts_ms,
                    "frame_num": frame_num,
                    "region": "before_gap",
                    "faces": len(faces),
                })
                if faces:
                    face_detections.extend([{
                        "ts_ms": ts_ms,
                        "region": "before_gap",
                        "bbox": face["bbox"],
                        "confidence": face["confidence"],
                    } for face in faces])

        # After gap
        for ts_ms in range(after_start_ms, after_end_ms, 100):  # 10fps sampling
            frame_num = int((ts_ms / 1000) * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                faces = detector.detect(frame)
                boundary_frames.append({
                    "ts_ms": ts_ms,
                    "frame_num": frame_num,
                    "region": "after_gap",
                    "faces": len(faces),
                })
                if faces:
                    face_detections.extend([{
                        "ts_ms": ts_ms,
                        "region": "after_gap",
                        "bbox": face["bbox"],
                        "confidence": face["confidence"],
                    } for face in faces])

        # Calculate stats
        any_faces_detected = len(face_detections) > 0
        min_face_px = min([min(f["bbox"][2] - f["bbox"][0], f["bbox"][3] - f["bbox"][1])
                           for f in face_detections], default=0)
        avg_confidence = np.mean([f["confidence"] for f in face_detections]) if face_detections else 0.0

        proof_entry = {
            "gap_idx": gap_idx + 1,
            "gap_start_ms": int(gap_start_ms),
            "gap_end_ms": int(gap_end_ms),
            "gap_duration_ms": int(gap_duration_ms),
            "boundary_ms": int(boundary_ms),
            "before_boundary": {
                "start_ms": int(before_start_ms),
                "end_ms": int(before_end_ms),
                "frames_sampled": len([f for f in boundary_frames if f["region"] == "before_gap"]),
                "faces_detected": len([f for f in face_detections if f["region"] == "before_gap"]),
            },
            "after_boundary": {
                "start_ms": int(after_start_ms),
                "end_ms": int(after_end_ms),
                "frames_sampled": len([f for f in boundary_frames if f["region"] == "after_gap"]),
                "faces_detected": len([f for f in face_detections if f["region"] == "after_gap"]),
            },
            "summary": {
                "any_faces_detected_in_boundaries": any_faces_detected,
                "total_faces": len(face_detections),
                "min_face_px_in_boundaries": int(min_face_px),
                "avg_confidence": float(avg_confidence),
            },
            "conclusion": "Faces detected in boundaries - potential missed footage" if any_faces_detected
                         else "No faces in boundaries - confirms off-screen time",
        }

        proofs.append(proof_entry)

        logger.info(
            f"[{job_id}] Gap {gap_idx + 1} proof: "
            f"{len(face_detections)} faces detected in boundaries"
        )

    cap.release()

    # Save proof report
    proof_report = {
        "job_id": job_id,
        "episode_id": episode_id,
        "target_identity": target_identity,
        "boundary_ms": boundary_ms,
        "total_gaps_analyzed": len(gap_windows),
        "proofs": proofs,
    }

    output_path = proofs_dir / f"{target_identity.lower()}_gap_proofs.json"
    with open(output_path, "w") as f:
        json.dump(proof_report, f, indent=2)

    logger.info(f"[{job_id}] Boundary proofs saved to {output_path}")

    # Summary
    gaps_with_faces = sum(1 for p in proofs if p["summary"]["any_faces_detected_in_boundaries"])
    gaps_confirmed_offscreen = len(proofs) - gaps_with_faces

    logger.info(
        f"[{job_id}] Summary: {gaps_with_faces} gaps have faces in boundaries, "
        f"{gaps_confirmed_offscreen} confirmed off-screen"
    )

    return proof_report
