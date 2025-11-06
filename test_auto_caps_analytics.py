"""
Test analytics with auto-caps enabled.
"""

from jobs.tasks.analytics import analytics_task

# Known cluster assignments from previous successful run
# Based on delta_table showing: BRANDI, EILEEN, KIM, KYLE, LVP, RINNA, YOLANDA
cluster_assignments = {
    0: "KIM",
    1: "KYLE",
    2: "EILEEN",
    3: "RINNA",
    4: "YOLANDA",
    5: "BRANDI",
    6: "LVP",
}

print("=" * 60)
print("Testing Analytics with Auto-Caps")
print("=" * 60)

try:
    result = analytics_task(
        job_id="test_auto_caps",
        episode_id="RHOBH-TEST-10-28",
        cluster_assignments=cluster_assignments
    )

    print("\n✅ Analytics completed successfully!")
    print(f"\nStats: {result.get('stats', {})}")

    # Check if auto-caps were computed
    import json
    from pathlib import Path

    caps_path = Path("data/harvest/RHOBH-TEST-10-28/diagnostics/per_identity_caps.json")
    if caps_path.exists():
        with open(caps_path) as f:
            caps = json.load(f)

        print("\n" + "=" * 60)
        print("Auto-Caps Results")
        print("=" * 60)
        for identity, data in caps.items():
            print(f"{identity}:")
            print(f"  auto_cap_ms: {data.get('auto_cap_ms')}ms")
            print(f"  safe_gap_count: {data.get('safe_gap_count')}")
            print(f"  safe_gap_p80: {data.get('safe_gap_p80')}ms")

    # Check delta table
    delta_path = Path("data/outputs/RHOBH-TEST-10-28/delta_table.csv")
    if delta_path.exists():
        print("\n" + "=" * 60)
        print("Delta Table (with Auto-Caps)")
        print("=" * 60)
        with open(delta_path) as f:
            print(f.read())

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
