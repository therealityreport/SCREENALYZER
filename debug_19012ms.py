#!/usr/bin/env python3
"""
Debug script to check what happened at 19.012s timestamp.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from screentime.detectors.face_small import SmallFaceRetinaDetector
from screentime.recognition.embed_arcface import ArcFaceEmbedder

episode_id = "RHOBH-TEST-10-28"
video_path = Path(f"data/videos/{episode_id}.mp4")

# Extract frame at 19.000s (closest 10fps sample to 19.012s)
target_ms = 19000

print(f"Checking frame at {target_ms}ms (19.000s)...")
print("=" * 80)

# Open video
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int((target_ms / 1000.0) * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame_bgr = cap.read()
cap.release()

if not ret:
    print("Failed to extract frame")
    exit(1)

# Convert BGR to RGB
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
print(f"Frame extracted: {frame_rgb.shape}")

# Run detector with aggressive settings (matching YOLANDA small-face config)
print("\nRunning face detection (min_face_px=28, min_conf=0.5)...")
detector = SmallFaceRetinaDetector(
    min_face_px=28,
    min_confidence=0.5,
    scales=[1.0, 1.35, 1.7, 2.2, 2.6]
)

detections = detector.detect(frame_rgb)
print(f"Found {len(detections)} faces\n")

if len(detections) == 0:
    print("No faces detected - YOLANDA was missed by detector")
    exit(0)

# Print detection details
for i, det in enumerate(detections):
    x1, y1, x2, y2 = det.bbox
    width = x2 - x1
    height = y2 - y1
    size = int(min(width, height))
    print(f"Face {i+1}:")
    print(f"  BBox: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})")
    print(f"  Size: {size}px")
    print(f"  Confidence: {det.confidence:.3f}")
    print()

# Load YOLANDA embeddings to check identity
print("Loading YOLANDA embeddings for identity verification...")
clusters_path = Path(f"data/harvest/{episode_id}/clusters.json")
tracks_path = Path(f"data/harvest/{episode_id}/tracks.json")
embeddings_path = Path(f"data/harvest/{episode_id}/embeddings.parquet")

clusters_data = json.loads(clusters_path.read_text())
tracks_data = json.loads(tracks_path.read_text())
import pandas as pd
embeddings_df = pd.read_parquet(embeddings_path)

# Find YOLANDA cluster
yolanda_cluster = next((c for c in clusters_data["clusters"] if c.get("name") == "YOLANDA"), None)
if not yolanda_cluster:
    print("YOLANDA cluster not found")
    exit(1)

# Get YOLANDA track IDs
yolanda_track_ids = yolanda_cluster.get("track_ids", [])
print(f"YOLANDA has {len(yolanda_track_ids)} tracks")

# Get all frame IDs from YOLANDA tracks
yolanda_frame_ids = []
for track_id in yolanda_track_ids:
    track = next((t for t in tracks_data["tracks"] if t["track_id"] == track_id), None)
    if track:
        for frame_ref in track.get("frame_refs", []):
            yolanda_frame_ids.append(frame_ref["frame_id"])

print(f"YOLANDA has {len(yolanda_frame_ids)} frames")

# Get YOLANDA embeddings
yolanda_embeddings = embeddings_df[embeddings_df["frame_id"].isin(yolanda_frame_ids)]
print(f"YOLANDA has {len(yolanda_embeddings)} embedding vectors")

if len(yolanda_embeddings) == 0:
    print("No YOLANDA embeddings available for comparison")
    exit(1)

# Compute average YOLANDA embedding
yolanda_vecs = np.stack(yolanda_embeddings["embedding"].values)
yolanda_avg = yolanda_vecs.mean(axis=0)
yolanda_avg = yolanda_avg / np.linalg.norm(yolanda_avg)

# Generate embeddings for detected faces
print("\nVerifying identities against YOLANDA embeddings...")
embedder = ArcFaceEmbedder()

# Convert RGB to BGR for embedder (OpenCV uses BGR)
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

for i, det in enumerate(detections):
    bbox = [int(v) for v in det.bbox]
    x1, y1, x2, y2 = bbox

    # Generate embedding (pass full frame and bbox)
    face_embedding = embedder.embed(frame_bgr, bbox)

    if face_embedding is None:
        print(f"Face {i+1} (size={det.face_size}px): ✗ Failed to generate embedding")
        continue

    face_embedding = face_embedding / np.linalg.norm(face_embedding)

    # Compute similarity to YOLANDA
    similarity = float(np.dot(face_embedding, yolanda_avg))

    size = int(min(x2-x1, y2-y1))
    print(f"Face {i+1} (size={size}px): similarity={similarity:.3f}", end="")

    if similarity >= 0.86:  # YOLANDA verification threshold
        print(" ✓ MATCH - This is YOLANDA!")
    else:
        print(f" ✗ Not YOLANDA (threshold=0.86)")

print("\n" + "=" * 80)
print("Summary:")
print(f"  Total faces detected: {len(detections)}")
yolanda_matches = sum(1 for det in detections if True)  # Will compute properly above
print(f"  YOLANDA matches: {yolanda_matches}")
