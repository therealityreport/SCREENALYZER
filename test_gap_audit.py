#!/usr/bin/env python3
"""Test gap audit for YOLANDA."""

import logging
from jobs.tasks.gap_audit import gap_audit_task

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Run gap audit for YOLANDA
result = gap_audit_task(
    job_id="gap_audit",
    episode_id="RHOBH-TEST-10-28",
    target_identity="YOLANDA",
    max_gap_ms=10000,
    pad_ms=800,
)

# Print summary
print("\n=== GAP AUDIT SUMMARY ===")
print(f"Total gaps: {result['summary']['total_gaps']}")
print(f"High-priority gaps (coverage < 20%): {result['summary']['high_priority_gaps']}")
print(f"High-priority duration: {result['summary']['high_priority_duration_ms']}ms ({result['summary']['high_priority_duration_ms']/1000:.1f}s)")

print("\n=== TOP 10 HIGH-PRIORITY GAPS ===")
high_priority = [w for w in result['gap_windows'] if w['priority'] == 'high'][:10]
for window in high_priority:
    print(f"Gap {window['window_idx']}: {window['gap_start_ms']}-{window['gap_end_ms']}ms "
          f"({window['gap_duration_ms']}ms), coverage={window['coverage_ratio']:.1%}")
