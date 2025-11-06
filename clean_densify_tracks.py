#!/usr/bin/env python3
"""Clean up duplicate densify tracks from multiple test runs."""

import json
from pathlib import Path

episode_id = "RHOBH-TEST-10-28"
tracks_path = Path(f"data/harvest/{episode_id}/tracks.json")

with open(tracks_path) as f:
    data = json.load(f)

original_count = len(data['tracks'])
data['tracks'] = [t for t in data['tracks'] if t.get('source') != 'local_densify']
data['total_tracks'] = len(data['tracks'])

with open(tracks_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f'Removed {original_count - len(data["tracks"])} densify tracks')
print(f'Remaining tracks: {len(data["tracks"])}')
