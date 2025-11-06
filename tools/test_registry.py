#!/usr/bin/env python3
"""
Unit tests for episode registry functionality.

Tests the Phase 2 registry implementation without requiring full pipeline runs.

Usage:
    python tools/test_registry.py
"""

import json
import sys
import tempfile
from pathlib import Path


def test_job_manager_registry():
    """Test JobManager registry functions."""
    print("🧪 Testing JobManager registry functions...")

    # Import job_manager
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.jobs import job_manager

    # Test 1: Normalize episode key
    print("\n  Test 1: normalize_episode_key()")
    test_cases = [
        ("RHOBH_S05_E03_11062025", "rhobh_s05_e03"),
        ("RHOSLC_S06_E01", "rhoslc_s06_e01"),
        ("rhobh_s05_e03", "rhobh_s05_e03"),
    ]

    for input_id, expected in test_cases:
        result = job_manager.normalize_episode_key(input_id)
        if result == expected:
            print(f"     ✅ {input_id} -> {result}")
        else:
            print(f"     ❌ {input_id} -> {result} (expected {expected})")
            return False

    # Test 2: Create and load registry
    print("\n  Test 2: write_episode_registry() and load_episode_registry()")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override DATA_ROOT
        import api.jobs
        original_data_root = api.jobs.DATA_ROOT
        api.jobs.DATA_ROOT = Path(tmpdir)

        episode_key = "test_show_s01_e01"
        test_registry = {
            "episode_key": episode_key,
            "episode_id": "TEST_SHOW_S01_E01_12345678",
            "show": "TEST_SHOW",
            "season": "S01",
            "episode": "E01",
            "video_path": "videos/test.mp4",
            "states": {
                "detected": False,
                "tracked": False,
            },
        }

        # Write registry
        job_manager.write_episode_registry(episode_key, test_registry)
        print(f"     ✅ Wrote registry to {tmpdir}/episodes/{episode_key}/state.json")

        # Load registry
        loaded = job_manager.load_episode_registry(episode_key)
        if loaded:
            if loaded["episode_key"] == episode_key:
                print(f"     ✅ Loaded registry successfully")
            else:
                print(f"     ❌ Loaded registry has wrong episode_key")
                return False
        else:
            print(f"     ❌ Failed to load registry")
            return False

        # Test 3: Update registry state
        print("\n  Test 3: update_registry_state()")
        job_manager.update_registry_state(episode_key, "detected", True)

        loaded = job_manager.load_episode_registry(episode_key)
        if loaded and loaded["states"]["detected"]:
            print(f"     ✅ Updated detected state to True")
        else:
            print(f"     ❌ Failed to update state")
            return False

        # Test 4: Ensure episode registry (create if missing)
        print("\n  Test 4: ensure_episode_registry()")
        new_episode_id = "ANOTHER_SHOW_S02_E05_99999999"
        new_video_path = "videos/another.mp4"

        returned_key = job_manager.ensure_episode_registry(new_episode_id, new_video_path)
        expected_key = "another_show_s02_e05"

        if returned_key == expected_key:
            print(f"     ✅ Created registry with key: {returned_key}")
        else:
            print(f"     ❌ Wrong key returned: {returned_key} (expected {expected_key})")
            return False

        # Verify it was actually created
        loaded = job_manager.load_episode_registry(returned_key)
        if loaded and loaded["episode_id"] == new_episode_id:
            print(f"     ✅ Registry persisted correctly")
        else:
            print(f"     ❌ Registry not persisted correctly")
            return False

        # Restore original DATA_ROOT
        api.jobs.DATA_ROOT = original_data_root

    return True


def test_job_envelope_structure():
    """Test job envelope has correct structure with registry reference."""
    print("\n🧪 Testing job envelope structure...")

    # Import job_manager
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from api.jobs import job_manager

    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override DATA_ROOT
        import api.jobs
        original_data_root = api.jobs.DATA_ROOT
        api.jobs.DATA_ROOT = Path(tmpdir)

        # Create a test envelope
        job_id = "prepare_TEST_S01_E01_12345678"
        episode_id = "TEST_S01_E01_12345678"
        episode_key = job_manager.normalize_episode_key(episode_id)

        envelope = {
            "job_id": job_id,
            "episode_id": episode_id,
            "episode_key": episode_key,
            "video_path": "videos/test.mp4",
            "mode": "prepare",
            "created_at": 1234567890.0,
            "registry_path": f"episodes/{episode_key}/state.json",
            "stages": {
                "detect": {"status": "pending"},
                "track": {"status": "pending"},
            },
        }

        # Write envelope
        job_manager.write_job_envelope(job_id, envelope)
        print(f"  ✅ Wrote job envelope to {tmpdir}/jobs/{job_id}/meta.json")

        # Load envelope
        loaded = job_manager.load_job_envelope(job_id)
        if not loaded:
            print(f"  ❌ Failed to load envelope")
            return False

        # Verify structure
        required_fields = ["job_id", "episode_key", "registry_path", "stages"]
        for field in required_fields:
            if field in loaded:
                print(f"  ✅ Envelope has required field: {field}")
            else:
                print(f"  ❌ Envelope missing required field: {field}")
                return False

        # Verify registry_path format
        if loaded["registry_path"] == f"episodes/{episode_key}/state.json":
            print(f"  ✅ Registry path format correct")
        else:
            print(f"  ❌ Registry path format wrong: {loaded['registry_path']}")
            return False

        # Restore original DATA_ROOT
        api.jobs.DATA_ROOT = original_data_root

    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("Episode Registry Unit Tests")
    print("="*70 + "\n")

    tests = [
        ("Registry Functions", test_job_manager_registry),
        ("Envelope Structure", test_job_envelope_structure),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"\n✅ {test_name}: PASSED\n")
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED\n")
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name}: ERROR - {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("="*70)
    print(f"\nTest Results: {passed} passed, {failed} failed\n")
    print("="*70 + "\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
