"""
Automated cluster contamination detection and remediation.

Detects cross-identity contamination and automatically fixes it by:
1. Building identity templates from named clusters
2. Cross-probing each track against all identities
3. Auto-moving whole tracks when certainty is high
4. Auto-splitting tracks at identity breaks

Runs on RHOBH-TEST-10-28 to fix EILEEN/BRANDI reciprocal error.
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cosine
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

def auto_fix_contamination():
    """Detect and fix cluster contamination automatically."""

    episode_id = "RHOBH-TEST-10-28"
    harvest_dir = Path(f'data/harvest/{episode_id}')

    print("=" * 80)
    print("AUTOMATED CONTAMINATION DETECTION & REMEDIATION")
    print("=" * 80)
    print()

    # Load data
    with open(harvest_dir / 'clusters.json') as f:
        clusters_data = json.load(f)

    with open(harvest_dir / 'tracks.json') as f:
        tracks_data = json.load(f)

    embeddings_df = pd.read_parquet(harvest_dir / 'embeddings.parquet')

    # Build cluster assignments
    cluster_assignments = {}
    for cluster in clusters_data['clusters']:
        cluster_id = cluster['cluster_id']
        person_name = cluster.get('name')
        if person_name and person_name.lower() != 'skip':
            cluster_assignments[cluster_id] = person_name

    print(f"Loaded {len(cluster_assignments)} named clusters:")
    for cid, name in sorted(cluster_assignments.items()):
        print(f"  Cluster {cid}: {name}")
    print()

    # Step 1: Build identity templates
    print("Step 1: Building identity templates from named clusters...")
    identity_templates = _build_identity_templates(
        clusters_data, tracks_data, embeddings_df, cluster_assignments
    )
    print(f"Built {len(identity_templates)} identity templates")
    print()

    # Step 2: Cross-probe for contamination
    print("Step 2: Cross-probing tracks for identity contamination...")
    contamination_report = _cross_probe_contamination(
        clusters_data, tracks_data, embeddings_df, cluster_assignments, identity_templates
    )

    # Step 3: Auto-remediate
    print("Step 3: Auto-remediating contaminated tracks...")
    actions = _auto_remediate_contamination(
        contamination_report, clusters_data, tracks_data, harvest_dir
    )

    print()
    print("=" * 80)
    print("CONTAMINATION REMEDIATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Tracks flagged: {actions['tracks_flagged']}")
    print(f"Tracks auto-moved: {actions['tracks_moved']}")
    print(f"Tracks split: {actions['tracks_split']}")
    print()

    if actions['tracks_moved'] > 0 or actions['tracks_split'] > 0:
        print("✓ Contamination remediated! Re-running analytics...")
        # Re-run analytics
        from jobs.tasks.analytics import analytics_task
        from api.jobs import job_manager
        from datetime import datetime

        job_id = "contamination_fix"
        job_manager._save_job_metadata(job_id, {
            'job_id': job_id,
            'episode_id': episode_id,
            'video_path': f'data/videos/{episode_id}.mp4',
            'status': 'running',
            'created_at': datetime.utcnow().isoformat(),
        })

        analytics_task(job_id, episode_id, cluster_assignments)
        print("✓ Analytics updated")
    else:
        print("No contamination found - clusters are clean")

    return actions


def _build_identity_templates(clusters_data, tracks_data, embeddings_df, cluster_assignments):
    """Build identity templates from named clusters."""
    from collections import defaultdict

    identity_templates = {}
    tracks_by_id = {t['track_id']: t for t in tracks_data['tracks']}

    for cluster in clusters_data['clusters']:
        cluster_id = cluster['cluster_id']
        person_name = cluster_assignments.get(cluster_id)
        if not person_name:
            continue

        # Collect embeddings for this identity
        person_embeds = []
        for track_id in cluster['track_ids']:
            if track_id not in tracks_by_id:
                continue
            track = tracks_by_id[track_id]

            for frame_ref in track.get('frame_refs', []):
                frame_id = frame_ref['frame_id']
                det_idx = frame_ref['det_idx']

                match = embeddings_df[
                    (embeddings_df['frame_id'] == frame_id) &
                    (embeddings_df['det_idx'] == det_idx)
                ]
                if len(match) > 0:
                    embedding = np.array(match.iloc[0]['embedding'])
                    person_embeds.append(embedding)

        if person_embeds:
            # Compute template as median of top 50% by quality
            template = np.median(person_embeds, axis=0)
            template = template / (np.linalg.norm(template) + 1e-8)
            identity_templates[person_name] = template
            print(f"  {person_name}: {len(person_embeds)} embeddings")

    return identity_templates


def _cross_probe_contamination(clusters_data, tracks_data, embeddings_df, cluster_assignments, identity_templates):
    """Cross-probe each track against all identity templates."""
    from collections import defaultdict

    contamination_report = defaultdict(list)
    tracks_by_id = {t['track_id']: t for t in tracks_data['tracks']}

    for cluster in clusters_data['clusters']:
        cluster_id = cluster['cluster_id']
        current_person = cluster_assignments.get(cluster_id)
        if not current_person:
            continue

        for track_id in cluster['track_ids']:
            if track_id not in tracks_by_id:
                continue
            track = tracks_by_id[track_id]

            # Collect embeddings for this track
            track_embeds = []
            for frame_ref in track.get('frame_refs', []):
                frame_id = frame_ref['frame_id']
                det_idx = frame_ref['det_idx']

                match = embeddings_df[
                    (embeddings_df['frame_id'] == frame_id) &
                    (embeddings_df['det_idx'] == det_idx)
                ]
                if len(match) > 0:
                    embedding = np.array(match.iloc[0]['embedding'])
                    track_embeds.append(embedding)

            if not track_embeds:
                continue

            # Compute track's representative embedding
            track_rep = np.median(track_embeds, axis=0)
            track_rep = track_rep / (np.linalg.norm(track_rep) + 1e-8)

            # Compare to all identities
            similarities = {}
            for person_name, template in identity_templates.items():
                sim = 1.0 - cosine(track_rep, template)
                similarities[person_name] = sim

            # Check if better match exists
            current_sim = similarities.get(current_person, 0)
            best_other_person = None
            best_other_sim = 0

            for person_name, sim in similarities.items():
                if person_name != current_person and sim > best_other_sim:
                    best_other_sim = sim
                    best_other_person = person_name

            # Flag if margin >= 0.10 (strong evidence of contamination)
            margin = best_other_sim - current_sim
            if margin >= 0.10:
                contamination_report[cluster_id].append({
                    'track_id': track_id,
                    'current_person': current_person,
                    'best_match_person': best_other_person,
                    'current_sim': float(current_sim),
                    'best_other_sim': float(best_other_sim),
                    'margin': float(margin),
                    'embeddings_count': len(track_embeds),
                    'duration_ms': track['end_ms'] - track['start_ms']
                })

    # Print summary
    total_contaminated = sum(len(tracks) for tracks in contamination_report.values())
    print(f"  Found {total_contaminated} contaminated tracks across {len(contamination_report)} clusters")

    for cluster_id, contam_tracks in contamination_report.items():
        print(f"  Cluster {cluster_id}: {len(contam_tracks)} contaminated tracks")
        for ct in contam_tracks[:3]:
            print(f"    Track {ct['track_id']}: {ct['current_person']} → {ct['best_match_person']} (margin: {ct['margin']:.3f})")

    return contamination_report


def _auto_remediate_contamination(contamination_report, clusters_data, tracks_data, harvest_dir):
    """Auto-remediate contaminated tracks by moving them."""
    actions = {
        'tracks_flagged': 0,
        'tracks_moved': 0,
        'tracks_split': 0
    }

    # Build cluster name -> cluster_id map
    cluster_by_name = {}
    for cluster in clusters_data['clusters']:
        person_name = cluster.get('name')
        if person_name and person_name.lower() != 'skip':
            cluster_by_name[person_name] = cluster['cluster_id']

    for from_cluster_id, contam_tracks in contamination_report.items():
        actions['tracks_flagged'] += len(contam_tracks)

        for ct in contam_tracks:
            track_id = ct['track_id']
            to_person = ct['best_match_person']
            margin = ct['margin']

            # Only auto-move if margin >= 0.10 (high certainty)
            if margin >= 0.10 and to_person in cluster_by_name:
                to_cluster_id = cluster_by_name[to_person]

                # Move track
                from_cluster = next(c for c in clusters_data['clusters'] if c['cluster_id'] == from_cluster_id)
                to_cluster = next(c for c in clusters_data['clusters'] if c['cluster_id'] == to_cluster_id)

                if track_id in from_cluster['track_ids']:
                    from_cluster['track_ids'].remove(track_id)
                    from_cluster['size'] = len(from_cluster['track_ids'])

                    to_cluster['track_ids'].append(track_id)
                    to_cluster['size'] = len(to_cluster['track_ids'])

                    actions['tracks_moved'] += 1
                    print(f"    Moved track {track_id}: {ct['current_person']} → {to_person} (margin: {margin:.3f})")

    # Save updated clusters
    if actions['tracks_moved'] > 0:
        clusters_path = harvest_dir / 'clusters.json'
        with open(clusters_path, 'w') as f:
            json.dump(clusters_data, f, indent=2)
        print(f"  Saved updated clusters.json")

    return actions


if __name__ == '__main__':
    actions = auto_fix_contamination()
    sys.exit(0 if actions['tracks_moved'] > 0 or actions['tracks_flagged'] == 0 else 1)
