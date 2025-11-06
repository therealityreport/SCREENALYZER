#!/usr/bin/env python3
"""Generate YOLANDA boundary proofs for large off-screen gaps."""

import logging
from pathlib import Path
from jobs.tasks.generate_boundary_proofs import generate_boundary_proofs

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# YOLANDA's 3 large off-screen gaps (from interval analysis)
# These are the periods between her appearances where she's truly off-screen
gap_windows = [
    {"start_ms": 4250, "end_ms": 19916},    # 15.7s gap
    {"start_ms": 28500, "end_ms": 57583},   # 29.1s gap
    {"start_ms": 62416, "end_ms": 95916},   # 33.5s gap
]

video_path = Path("data/videos/RHOBH-TEST-10-28.mp4")

logger.info("Generating YOLANDA boundary proofs for 3 large gaps")
gap_desc = [f"{g['start_ms']}-{g['end_ms']}ms" for g in gap_windows]
logger.info(f"Gaps: {gap_desc}")

result = generate_boundary_proofs(
    job_id="boundary_proofs",
    episode_id="RHOBH-TEST-10-28",
    video_path=video_path,
    target_identity="YOLANDA",
    gap_windows=gap_windows,
    boundary_ms=1500,  # Check 1.5s before/after each gap
)

print("\n=== BOUNDARY PROOF RESULTS ===")
for proof in result["proofs"]:
    print(f"\nGap {proof['gap_idx']}: {proof['gap_start_ms']}-{proof['gap_end_ms']}ms "
          f"({proof['gap_duration_ms']/1000:.1f}s)")
    print(f"  Before boundary: {proof['before_boundary']['faces_detected']} faces detected")
    print(f"  After boundary: {proof['after_boundary']['faces_detected']} faces detected")
    print(f"  Total faces: {proof['summary']['total_faces']}")
    print(f"  Conclusion: {proof['conclusion']}")
