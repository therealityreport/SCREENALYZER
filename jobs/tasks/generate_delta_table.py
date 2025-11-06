#!/usr/bin/env python3
"""
Generate delta table comparing current totals to ground truth.
"""

import pandas as pd
from pathlib import Path

# Ground truth timings (ms)
GROUND_TRUTH = {
    'KIM': 48004,
    'KYLE': 21017,
    'RINNA': 25015,
    'EILEEN': 10001,
    'BRANDI': 10014,
    'YOLANDA': 16002,
    'LVP': 2018,
}

episode_id = "RHOBH-TEST-10-28"
totals_path = Path(f"data/outputs/{episode_id}/totals.csv")
delta_path = Path(f"data/outputs/{episode_id}/delta_table.csv")

# Load current totals
totals_df = pd.read_csv(totals_path)

# Generate delta table
rows = []
for person in sorted(GROUND_TRUTH.keys()):
    gt_ms = GROUND_TRUTH[person]

    person_row = totals_df[totals_df['person_name'] == person]
    if len(person_row) > 0:
        auto_ms = int(person_row['total_ms'].values[0])
    else:
        auto_ms = 0

    delta_ms = auto_ms - gt_ms
    error_pct = (delta_ms / gt_ms * 100) if gt_ms > 0 else 0

    # Status
    if abs(error_pct) <= 5:
        status = "PASS"
    elif abs(error_pct) <= 10:
        status = "WARN"
    else:
        status = "FAIL"

    rows.append({
        'person': person,
        'auto_ms': auto_ms,
        'gt_ms': gt_ms,
        'delta_ms': delta_ms,
        'delta_s': round(delta_ms / 1000.0, 2),
        'error_pct': round(error_pct, 1),
        'status': status
    })

delta_df = pd.DataFrame(rows)

# Save
delta_df.to_csv(delta_path, index=False)

print(f"Delta table saved to {delta_path}\n")
print("=" * 80)
print(f"{'Person':<10s} {'Auto (ms)':>10s} {'GT (ms)':>10s} {'Delta (s)':>10s} {'Error %':>8s}  Status")
print("=" * 80)

passing = 0
for _, row in delta_df.iterrows():
    symbol = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[row['status']]
    print(f"{row['person']:<10s} {row['auto_ms']:10d} {row['gt_ms']:10d} {row['delta_s']:+10.2f} {row['error_pct']:+7.1f}%  {symbol} {row['status']}")
    if row['status'] == "PASS":
        passing += 1

print("=" * 80)
print(f"Within ±5%: {passing}/7 cast members")
print()

# Special YOLANDA report
yolanda_row = delta_df[delta_df['person'] == 'YOLANDA'].iloc[0]
print("YOLANDA ENTRANCE RECOVERY:")
print(f"  Before: 8.75s (delta: -7.25s)")
print(f"  After:  {yolanda_row['auto_ms']/1000:.2f}s (delta: {yolanda_row['delta_s']:.2f}s)")
print(f"  Improvement: {abs(-7.25 - yolanda_row['delta_s']):.2f}s recovered")
print(f"  Absolute error: {abs(yolanda_row['delta_s']):.2f}s")
print(f"  Target: ≤4.5s")
abs_error = abs(yolanda_row['delta_s'])
print(f"  Status: {'PASS' if abs_error <= 4.5 else f'Within {abs_error - 4.5:.2f}s of target'}")
