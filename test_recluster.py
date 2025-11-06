"""
Test re-clustering with season bank.

Usage:
    python test_recluster.py --episode RHOBH-TEST-10-28 --show rhobh --season s05
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from jobs.tasks.recluster import recluster_task


def main():
    parser = argparse.ArgumentParser(description='Test re-clustering with season bank')
    parser.add_argument('--episode', required=True, help='Episode ID')
    parser.add_argument('--show', default='rhobh', help='Show ID')
    parser.add_argument('--season', default='s05', help='Season ID')
    parser.add_argument('--constraints', action='store_true', help='Use manual constraints')

    args = parser.parse_args()

    constraints_msg = " WITH CONSTRAINTS" if args.constraints else ""
    print(f"Re-clustering {args.episode} with season bank {args.show}/{args.season}{constraints_msg}")
    print("-" * 60)

    try:
        result = recluster_task(
            job_id="test-recluster",
            episode_id=args.episode,
            show_id=args.show,
            season_id=args.season,
            sources=["baseline", "entrance", "densify"],
            use_constraints=args.constraints
        )

        print("\n" + "=" * 60)
        print("RE-CLUSTER COMPLETE")
        print("=" * 60)
        print(f"Episode: {result['episode_id']}")
        print(f"EPS chosen: {result['eps_chosen']:.3f}")
        print(f"Clusters: {result['n_clusters']}")
        print(f"Noise: {result['n_noise']}")
        print(f"Assigned: {result['n_assigned']}")
        print(f"Season bank used: {result['season_bank_used']}")
        print(f"Constraints used: {result['constraints_used']}")
        if result['constraints_used']:
            constraints_info = result.get('constraints', {})
            print(f"  - Must-link pairs: {constraints_info.get('must_link_count', 0)}")
            print(f"  - Cannot-link pairs: {constraints_info.get('cannot_link_count', 0)}")
            print(f"  - ML components: {constraints_info.get('ml_components_count', 0)}")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
