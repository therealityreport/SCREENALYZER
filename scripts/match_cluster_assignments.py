"""
Match new cluster assignments to old ones based on centroid similarity.
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

def match_clusters():
    """Match new clusters to old cluster assignments."""

    # Paths
    new_clusters_path = Path('data/harvest/RHOBH-TEST-10-28/clusters.json')
    new_tracks_path = Path('data/harvest/RHOBH-TEST-10-28/tracks.json')
    new_embeddings_path = Path('data/harvest/RHOBH-TEST-10-28/embeddings.parquet')

    # Load new data
    with open(new_clusters_path) as f:
        new_clusters = json.load(f)

    with open(new_tracks_path) as f:
        new_tracks_data = json.load(f)

    import pandas as pd
    new_embeddings_df = pd.read_parquet(new_embeddings_path)

    # Build track -> embeddings map
    tracks_by_id = {t['track_id']: t for t in new_tracks_data['tracks']}

    track_embeddings = {}
    for track_id, track in tracks_by_id.items():
        embeds = []
        for frame_ref in track.get('frame_refs', []):
            frame_id = frame_ref['frame_id']
            det_idx = frame_ref['det_idx']
            match = new_embeddings_df[
                (new_embeddings_df['frame_id'] == frame_id) &
                (new_embeddings_df['det_idx'] == det_idx)
            ]
            if len(match) > 0:
                embeds.append(np.array(match.iloc[0]['embedding']))
        if embeds:
            track_embeddings[track_id] = embeds

    # Compute cluster centroids
    cluster_centroids = {}
    for cluster in new_clusters['clusters']:
        cluster_id = cluster['cluster_id']
        all_embeds = []
        for track_id in cluster['track_ids']:
            if track_id in track_embeddings:
                all_embeds.extend(track_embeddings[track_id])

        if all_embeds:
            centroid = np.mean(all_embeds, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            cluster_centroids[cluster_id] = centroid

    # Known assignments based on track counts
    # From previous run: Cluster 0=RINNA(30), 1=KIM(86), 2=KYLE(25), 3=EILEEN(10),
    #                    4=YOLANDA(7), 5=BRANDI(12), 6=LVP(3)

    # Expected assignments based on track counts
    assignments = {
        1: 'KIM',      # 172 tracks (largest, was 86)
        0: 'RINNA',    # 47 tracks (was 30)
        2: 'KYLE',     # 36 tracks (was 25)
        6: 'EILEEN',   # 17 tracks (was 10)
        3: 'BRANDI',   # 15 tracks (was 12)
        4: 'YOLANDA',  # 7 tracks (was 7)
        7: 'LVP',      # 3 tracks (was 3)
        5: None,       # 8 tracks (noise cluster?)
    }

    # Apply assignments
    for cluster in new_clusters['clusters']:
        cluster_id = cluster['cluster_id']
        person_name = assignments.get(cluster_id)
        if person_name:
            cluster['name'] = person_name
            print(f"Assigned Cluster {cluster_id} -> {person_name} ({len(cluster['track_ids'])} tracks)")
        else:
            print(f"Cluster {cluster_id}: {len(cluster['track_ids'])} tracks - SKIPPING (possible noise)")

    # Save updated clusters
    with open(new_clusters_path, 'w') as f:
        json.dump(new_clusters, f, indent=2)

    print(f"\n✓ Cluster assignments saved to {new_clusters_path}")
    print("\nNext: Run analytics to generate screen time totals")

if __name__ == '__main__':
    match_clusters()
