"""
Generate baseline accuracy report: Auto vs GT for all 7 cast members.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jobs.tasks.analytics import analytics_task
import pandas as pd

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

def generate_report():
    """Generate baseline accuracy report."""

    episode_id = "RHOBH-TEST-10-28"

    print("=" * 80)
    print("BASELINE ACCURACY REPORT")
    print("=" * 80)
    print(f"\nEpisode: {episode_id}")
    print(f"Configuration: re-ID enabled, adaptive gap-merge, baseline thresholds")
    print()

    # Load clusters to get assignments
    clusters_file = Path(f'data/harvest/{episode_id}/clusters.json')
    with open(clusters_file) as f:
        clusters_data = json.load(f)

    # Build cluster assignments (cluster_id -> person_name)
    # Skip clusters marked as "skip" (noise clusters)
    cluster_assignments = {}
    for cluster in clusters_data['clusters']:
        cluster_id = cluster['cluster_id']
        person_name = cluster.get('name')
        if person_name and person_name.lower() != 'skip':
            cluster_assignments[cluster_id] = person_name

    if len(cluster_assignments) != 7:
        print(f"✗ ERROR: Only {len(cluster_assignments)}/7 clusters have names assigned")
        print(f"   Found: {', '.join(cluster_assignments.values())}")
        return False

    # Run analytics
    print("Running analytics...")
    try:
        result = analytics_task('baseline_report', episode_id, cluster_assignments)
        print(f"✓ Analytics complete: {result['stats'].get('intervals_created', 0)} intervals\n")
    except Exception as e:
        print(f"✗ ERROR: Analytics failed: {e}")
        return False

    # Load results
    totals_file = Path(f'data/outputs/{episode_id}/totals.csv')
    if not totals_file.exists():
        print(f"✗ ERROR: Totals file not found: {totals_file}")
        return False

    df = pd.read_csv(totals_file)

    # Generate delta table
    print("-" * 80)
    print(f"{'Person':<10s} {'Auto (ms)':>10s} {'GT (ms)':>10s} {'Delta':>10s} {'Error %':>8s}  Status")
    print("-" * 80)

    results = []
    within_5_pct = 0
    total_gt_ms = sum(GROUND_TRUTH.values())

    for person in sorted(GROUND_TRUTH.keys()):
        row = df[df['person_name'] == person]
        if len(row) > 0:
            auto_ms = int(row['total_ms'].values[0])
        else:
            auto_ms = 0

        gt_ms = GROUND_TRUTH[person]
        delta_ms = auto_ms - gt_ms
        error_pct = (delta_ms / gt_ms * 100) if gt_ms > 0 else 0

        # Status
        if abs(error_pct) <= 5:
            status = "✓"
            within_5_pct += 1
        elif abs(error_pct) <= 10:
            status = "⚠"
        else:
            status = "✗"

        print(f"{person:<10s} {auto_ms:10d} {gt_ms:10d} {delta_ms:+10d} {error_pct:+7.1f}%  {status}")

        results.append({
            'person': person,
            'auto_ms': auto_ms,
            'gt_ms': gt_ms,
            'delta_ms': delta_ms,
            'error_pct': error_pct,
            'within_5pct': abs(error_pct) <= 5
        })

    print("-" * 80)
    print(f"Within ±5%: {within_5_pct}/7 cast members")
    print()

    # Diagnostics
    print("DIAGNOSTICS:")
    print()

    # Check for suspiciously high values (may indicate wrong cluster assignment)
    for r in results:
        if r['error_pct'] > 200:  # More than 2x GT
            print(f"⚠ WARNING: {r['person']} shows {r['auto_ms']/1000:.1f}s (GT {r['gt_ms']/1000:.1f}s)")
            print(f"   This suggests Cluster assigned to '{r['person']}' contains wrong person's faces")
            print()
        elif r['error_pct'] < -50:  # Missing >50%
            print(f"⚠ WARNING: {r['person']} shows {r['auto_ms']/1000:.1f}s (GT {r['gt_ms']/1000:.1f}s)")
            print(f"   This suggests severe detection/tracking issues or wrong cluster assignment")
            print()

    if within_5_pct < 7:
        print("NEXT STEPS:")
        print()
        print("1. Review cluster assignments in UI - verify each cluster shows correct person")
        print("2. If assignments are wrong, manually reassign cluster names")
        print("3. If assignments are correct:")
        print("   - For errors >5%, implement targeted high-recall detection")
        print("   - For errors >10%, run micro-sweep on detection thresholds")
        print()
        return False
    else:
        print("✓ ALL CAST MEMBERS WITHIN ±5% - BASELINE ACCEPTED")
        print()
        print("Ready to proceed with Phase 2!")
        return True

if __name__ == "__main__":
    success = generate_report()
    sys.exit(0 if success else 1)
