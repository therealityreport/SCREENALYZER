#!/usr/bin/env python3
"""Run targeted densify on YOLANDA boundary slivers where faces were detected."""

import logging
from pathlib import Path
from jobs.tasks.local_densify import local_densify_task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Extract the 6 boundary windows (1.5s each) where we detected 144 faces
# From yolanda_gap_proofs.json:
boundary_windows = [
    # Gap 1 boundaries (15.7s gap, but we only scan the 1.5s edges)
    {"start_ms": 2750, "end_ms": 4250, "label": "Gap1_before"},     # 15 faces
    {"start_ms": 19916, "end_ms": 21416, "label": "Gap1_after"},    # 23 faces

    # Gap 2 boundaries (29.1s gap, but we only scan the 1.5s edges)
    {"start_ms": 27000, "end_ms": 28500, "label": "Gap2_before"},   # 15 faces
    {"start_ms": 57583, "end_ms": 59083, "label": "Gap2_after"},    # 15 faces

    # Gap 3 boundaries (33.5s gap, but we only scan the 1.5s edges)
    {"start_ms": 60916, "end_ms": 62416, "label": "Gap3_before"},   # 41 faces
    {"start_ms": 95916, "end_ms": 97416, "label": "Gap3_after"},    # 35 faces
]

logger.info("=" * 80)
logger.info("YOLANDA BOUNDARY SLIVER DENSIFY")
logger.info("=" * 80)
logger.info(f"Total boundary windows: {len(boundary_windows)}")
logger.info(f"Total scan duration: {sum(w['end_ms'] - w['start_ms'] for w in boundary_windows) / 1000:.1f}s")
logger.info("These are the 1.5s regions where 144 faces were detected")
logger.info("")

for i, window in enumerate(boundary_windows, 1):
    duration = (window['end_ms'] - window['start_ms']) / 1000
    logger.info(f"  {i}. {window['label']}: {window['start_ms']}-{window['end_ms']}ms ({duration:.1f}s)")

logger.info("")
logger.info("Running identity-gated densify to check if any of the 144 faces are YOLANDA...")
logger.info("")

result = local_densify_task(
    job_id="yolanda_boundary_densify",
    episode_id="RHOBH-TEST-10-28",
    target_identities=["YOLANDA"],
    gap_windows=boundary_windows,
)

logger.info("")
logger.info("=" * 80)
logger.info("BOUNDARY DENSIFY RESULTS")
logger.info("=" * 80)

if result["new_tracklets"]:
    logger.info(f"✅ Found {len(result['new_tracklets'])} new YOLANDA tracklets in boundaries!")
    total_ms = sum(t["duration_ms"] for t in result["new_tracklets"])
    logger.info(f"   Total new screen time: {total_ms / 1000:.1f}s")
    logger.info("")
    logger.info("Tracklet details:")
    for t in result["new_tracklets"]:
        logger.info(f"  - {t['start_ms']}-{t['end_ms']}ms: {t['duration_ms']}ms, "
                   f"{t['n_faces']} faces, conf={t['mean_confidence']:.3f}")
else:
    logger.info("❌ No YOLANDA faces found in the 144 detected faces")
    logger.info("   Conclusion: The 144 faces belong to other cast members")
    logger.info("   YOLANDA was truly off-screen during these boundary regions")

logger.info("")
logger.info("Next: Re-run analytics to see if this closes the -7.3s YOLANDA deficit")
