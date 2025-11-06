#!/usr/bin/env python3
"""
Update totals.csv with entrance recovery results.
"""

import pandas as pd
from pathlib import Path

episode_id = "RHOBH-TEST-10-28"
timeline_path = Path(f"data/outputs/{episode_id}/timeline.csv")
totals_path = Path(f"data/outputs/{episode_id}/totals.csv")

# Load timeline
timeline_df = pd.read_csv(timeline_path)

# Compute totals by person
totals = []
for person_name in sorted(timeline_df['person_name'].unique()):
    person_intervals = timeline_df[timeline_df['person_name'] == person_name]

    total_ms = person_intervals['duration_ms'].sum()
    total_sec = total_ms / 1000.0
    appearances = len(person_intervals)
    first_ms = person_intervals['start_ms'].min()
    last_ms = person_intervals['end_ms'].max()
    mean_confidence = person_intervals['confidence'].mean()

    totals.append({
        'person_name': person_name,
        'total_ms': int(total_ms),
        'total_sec': round(total_sec, 1),
        'appearances': appearances,
        'first_ms': int(first_ms),
        'last_ms': int(last_ms),
        'mean_confidence': round(mean_confidence, 3),
    })

# Create dataframe
totals_df = pd.DataFrame(totals)

# Add percent column (assuming episode duration from max last_ms)
episode_duration_ms = totals_df['last_ms'].max()
totals_df['percent'] = (totals_df['total_ms'] / episode_duration_ms * 100).round(2)

# Save
totals_df.to_csv(totals_path, index=False)

print(f"Updated totals saved to {totals_path}")
print("\nUpdated totals:")
print(totals_df.to_string(index=False))
