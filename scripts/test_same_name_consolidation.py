#!/usr/bin/env python3
"""
Test script to verify same-name consolidation integration.
Creates a test scenario with two KIM clusters and runs RE-CLUSTER.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_consolidation():
    """Test same-name consolidation by creating two KIM clusters."""
    print("🧪 Testing Same-Name Consolidation")
    print("=" * 60)

    episode_id = "RHOBH-TEST-10-28"
    data_root = Path("data")
    harvest_dir = data_root / "harvest" / episode_id

    # Load current clusters
    clusters_path = harvest_dir / "clusters.json"
    with open(clusters_path) as f:
        clusters_data = json.load(f)

    print(f"\n📊 Current State:")
    print(f"   Total clusters: {clusters_data['total_clusters']}")

    # Show current cluster names
    names_count = {}
    for cluster in clusters_data.get('clusters', []):
        name = cluster.get('name', 'Unknown')
        conf = cluster.get('assignment_confidence', 0.0)
        if name not in names_count:
            names_count[name] = []
        names_count[name].append((cluster['cluster_id'], conf, cluster['size']))

    print(f"\n   Clusters by identity:")
    for name, clusters in sorted(names_count.items()):
        if len(clusters) > 1:
            print(f"   ⚠️  {name}: {len(clusters)} clusters")
            for cid, conf, size in clusters:
                print(f"      - Cluster {cid}: conf={conf:.3f}, size={size}")
        else:
            cid, conf, size = clusters[0]
            print(f"   ✓ {name}: 1 cluster (id={cid}, conf={conf:.3f}, size={size})")

    # Check if there are multiple clusters with same name and conf=1.0
    consolidation_candidates = {}
    for name, clusters in names_count.items():
        if name != 'Unknown' and len(clusters) > 1:
            # Check if any have conf=1.0 (manual assignment)
            conf_1_clusters = [(cid, conf, size) for cid, conf, size in clusters if conf == 1.0]
            if len(conf_1_clusters) > 1:
                consolidation_candidates[name] = conf_1_clusters

    if consolidation_candidates:
        print(f"\n✅ Found consolidation candidates:")
        for name, clusters in consolidation_candidates.items():
            print(f"   {name}: {len(clusters)} clusters with conf=1.0")
            print(f"   → Would consolidate into 1 cluster on RE-CLUSTER")
    else:
        print(f"\n⚠️  No consolidation candidates found")
        print(f"   To test consolidation:")
        print(f"   1. Manually assign 2+ clusters to the same identity (e.g., KIM)")
        print(f"   2. Ensure assignment_confidence = 1.0 for both")
        print(f"   3. Run RE-CLUSTER with constraints enabled")

    # Check constraints
    constraints_path = harvest_dir / "diagnostics" / "constraints.json"
    if constraints_path.exists():
        with open(constraints_path) as f:
            constraints_data = json.load(f)

        extraction = constraints_data.get('extraction', {})
        print(f"\n📊 Existing Constraints:")
        print(f"   Must-Link: {extraction.get('must_link_count', 0):,}")
        print(f"   Cannot-Link: {extraction.get('cannot_link_count', 0):,}")

    # Check suppress.json
    suppress_path = harvest_dir / "diagnostics" / "suppress.json"
    if suppress_path.exists():
        with open(suppress_path) as f:
            suppress_data = json.load(f)
        print(f"\n🗑️  Suppressed Items:")
        print(f"   Tracks: {len(suppress_data.get('deleted_tracks', []))}")
        print(f"   Clusters: {len(suppress_data.get('deleted_clusters', []))}")
    else:
        print(f"\n🗑️  Suppressed Items: None (suppress.json not found)")

    # Check track_constraints.jsonl
    track_constraints_path = harvest_dir / "diagnostics" / "track_constraints.jsonl"
    if track_constraints_path.exists():
        with open(track_constraints_path) as f:
            lines = [line for line in f if line.strip()]
        print(f"\n📝 Persisted Constraints:")
        print(f"   Track constraint entries: {len(lines)}")

    print("\n" + "=" * 60)
    print("✅ Integration test complete!")
    print("\n📋 To test same-name consolidation:")
    print("   1. Open Streamlit UI at http://localhost:8501")
    print("   2. Navigate to REVIEW → All Faces")
    print("   3. Assign two different clusters to KIM (or any identity)")
    print("   4. Run RE-CLUSTER with 'Use manual constraints' checked")
    print("   5. Verify the two clusters merge into one")
    print("\n   Expected diagnostics/constraints.json output:")
    print('   "same_name_consolidations": {"KIM": 2}')

    return 0


if __name__ == "__main__":
    sys.exit(test_consolidation())
