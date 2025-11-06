"""
Verify cluster name assignments and generate instructions for manual review.
"""

import json
import sys
from pathlib import Path

def verify_clusters():
    """Check if all clusters have valid names assigned."""

    episode_id = "RHOBH-TEST-10-28"
    clusters_file = Path(f"data/harvest/{episode_id}/clusters.json")

    if not clusters_file.exists():
        print(f"✗ ERROR: Clusters file not found: {clusters_file}")
        return False

    with open(clusters_file) as f:
        data = json.load(f)

    clusters = data.get("clusters", [])

    print("=" * 70)
    print("CLUSTER NAME VERIFICATION")
    print("=" * 70)
    print(f"\nTotal clusters: {len(clusters)}")
    print()

    expected_names = {"KIM", "KYLE", "RINNA", "EILEEN", "BRANDI", "YOLANDA", "LVP"}
    assigned_names = set()
    unnamed = []

    for cluster in sorted(clusters, key=lambda x: x.get('size', 0), reverse=True):
        cid = cluster['cluster_id']
        size = cluster['size']
        name = cluster.get('name', None)

        if name and name in expected_names:
            assigned_names.add(name)
            print(f"  ✓ Cluster {cid}: {name:10s} ({size:2d} tracks)")
        elif name:
            print(f"  ⚠ Cluster {cid}: {name:10s} ({size:2d} tracks) - UNEXPECTED NAME")
        else:
            unnamed.append((cid, size))
            print(f"  ✗ Cluster {cid}: UNNAMED    ({size:2d} tracks) - NEEDS ASSIGNMENT")

    missing_names = expected_names - assigned_names

    print()
    print("-" * 70)

    if len(unnamed) == 0 and len(missing_names) == 0:
        print("✓ ALL CLUSTERS HAVE VALID NAMES ASSIGNED")
        print("\nYou can now run analytics to generate screen time totals.")
        return True
    else:
        print("✗ CLUSTERS NEED MANUAL NAME ASSIGNMENT")
        print()

        if unnamed:
            print(f"Unnamed clusters: {len(unnamed)}")
            for cid, size in unnamed:
                print(f"  - Cluster {cid} ({size} tracks)")

        if missing_names:
            print(f"\nMissing cast members: {', '.join(sorted(missing_names))}")

        print()
        print("NEXT STEPS:")
        print("1. Open the Screenalyzer UI in your browser")
        print("2. Navigate to the Review page for RHOBH-TEST-10-28")
        print("3. Look at the face samples in each cluster")
        print("4. Manually assign the correct name to each cluster")
        print("5. Once all 7 names are assigned, re-run analytics")
        print()
        print("Then run: python scripts/generate_baseline_report.py")

        return False

if __name__ == "__main__":
    success = verify_clusters()
    sys.exit(0 if success else 1)
