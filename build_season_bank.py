"""
Build season bank from uploaded seed images.

Usage:
    python build_season_bank.py --show rhobh --season s05
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
from screentime.clustering.season_bank import (
    build_season_bank,
    save_season_bank,
    print_season_bank_summary
)


def main():
    parser = argparse.ArgumentParser(description='Build season bank from seed images')
    parser.add_argument('--show', default='rhobh', help='Show ID (default: rhobh)')
    parser.add_argument('--season', default='s05', help='Season ID (default: s05)')
    parser.add_argument('--max-per-bin', type=int, default=5, help='Max prototypes per bin (default: 5)')
    parser.add_argument('--min-face-height', type=float, default=64.0, help='Min face height (default: 64)')

    args = parser.parse_args()

    # Paths
    facebank_dir = Path(f"data/facebank/{args.show}/{args.season}")
    output_path = facebank_dir / "multi_prototypes.json"

    if not facebank_dir.exists():
        print(f"❌ Facebank directory not found: {facebank_dir}")
        print(f"   Please upload seed images first via Cast Images page")
        return 1

    cast_dirs = [d for d in facebank_dir.iterdir() if d.is_dir()]
    if not cast_dirs:
        print(f"❌ No cast members found in: {facebank_dir}")
        return 1

    print(f"Building season bank for {args.show}/{args.season}")
    print(f"Facebank: {facebank_dir}")
    print(f"Cast members: {len(cast_dirs)}")
    print("-" * 60)

    # Build season bank
    season_bank = build_season_bank(
        show_id=args.show,
        season_id=args.season,
        facebank_dir=facebank_dir,
        max_prototypes_per_bin=args.max_per_bin,
        min_face_height=args.min_face_height
    )

    # Save
    save_season_bank(season_bank, output_path)
    print(f"\n✅ Season bank saved to: {output_path}")

    # Print summary
    print_season_bank_summary(season_bank)

    return 0


if __name__ == "__main__":
    sys.exit(main())
