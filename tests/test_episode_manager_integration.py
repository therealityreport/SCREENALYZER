"""
Integration tests for episode_manager workflows.

Tests complete Move/Remove/Restore flows with pipeline gating and audit logging.
"""

import json
import pytest
from pathlib import Path

from app.lib.episode_manager import (
    move_episode,
    remove_episode,
    restore_episode,
    rehash_episode,
    list_episodes,
)
from app.lib.pipeline import (
    check_pipeline_can_run,
    is_maintenance_mode,
    get_maintenance_reason,
)
from screentime.diagnostics.migrations import get_audit_history


@pytest.fixture
def integration_data_root(tmp_path):
    """Create full integration test environment."""
    data_root = tmp_path / "data"

    # Create harvest directories
    harvest_root = data_root / "harvest"
    harvest_root.mkdir(parents=True)

    # Create test episode
    ep_id = "INT_TEST_S01_E01"
    ep_dir = harvest_root / ep_id
    ep_dir.mkdir()

    # Create diagnostics
    diag_dir = ep_dir / "diagnostics"
    diag_dir.mkdir()

    # Create key files
    (ep_dir / "manifest.parquet").write_bytes(b"integration_test_manifest")
    (ep_dir / "embeddings.parquet").write_bytes(b"integration_test_embeddings")
    (ep_dir / "tracks.json").write_text('{"tracks": []}')
    (ep_dir / "clusters.json").write_text('{"clusters": []}')

    # Create stills
    stills_dir = ep_dir / "stills"
    stills_dir.mkdir()
    (stills_dir / "track_001.jpg").write_bytes(b"mock_still_image")

    # Create mock registry
    registry_path = data_root / "registry.json"
    registry = {
        "shows": [
            {
                "show_id": "srcshow",
                "seasons": [
                    {
                        "season_id": "s01",
                        "season_number": 1,
                        "episodes": [
                            {
                                "episode_id": ep_id,
                                "archived": False,
                                "episode_hash": "int_test_hash",
                            },
                        ],
                    },
                ],
            },
            {
                "show_id": "dstshow",
                "seasons": [
                    {
                        "season_id": "s02",
                        "season_number": 2,
                        "episodes": [],
                    },
                ],
            },
        ],
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Override DATA_ROOT
    import app.lib.episode_manager as em
    import app.lib.pipeline as pipeline
    original_data_root_em = em.DATA_ROOT
    em.DATA_ROOT = data_root
    pipeline.DATA_ROOT = data_root

    yield data_root, ep_id

    # Restore
    em.DATA_ROOT = original_data_root_em


class TestMoveEpisodeWorkflow:
    """Integration tests for complete move workflow."""

    def test_move_episode_complete_workflow(self, integration_data_root):
        """Test complete move workflow with audit logging."""
        data_root, ep_id = integration_data_root

        # Verify starting state
        episodes = list_episodes(show="srcshow")
        assert len(episodes) == 1
        assert episodes[0].episode_id == ep_id

        # Move episode
        move_episode(ep_id, "dstshow", 2, actor="integration_test", reason="Testing move workflow")

        # Verify episode moved in registry
        episodes_src = list_episodes(show="srcshow")
        assert len(episodes_src) == 0

        episodes_dst = list_episodes(show="dstshow")
        assert len(episodes_dst) == 1
        assert episodes_dst[0].episode_id == ep_id
        assert episodes_dst[0].season_number == 2

        # Verify data preserved on disk
        ep_dir = data_root / "harvest" / ep_id
        assert (ep_dir / "manifest.parquet").exists()
        assert (ep_dir / "tracks.json").exists()
        assert (ep_dir / "stills").exists()

        # Verify audit log created
        audit_events = get_audit_history(ep_id)
        assert len(audit_events) > 0

        move_event = [e for e in audit_events if e.get("op_type") == "move"][0]
        assert move_event["before_state"]["show"] == "srcshow"
        assert move_event["after_state"]["show"] == "dstshow"
        assert move_event["actor"] == "integration_test"

    def test_move_blocks_during_prepare(self, integration_data_root):
        """Test that move blocks pipeline operations."""
        data_root, ep_id = integration_data_root

        # Simulate prepare pipeline running
        state_file = data_root / "harvest" / ep_id / "diagnostics" / "pipeline_state.json"
        with open(state_file, "w") as f:
            json.dump({
                "status": "running",
                "current_step": "Detect/Embed",
                "maintenance_mode": False,
            }, f)

        # Try to move - should work (no maintenance mode yet)
        # But let's simulate that move sets maintenance mode
        from app.lib.episode_manager import _set_maintenance_mode
        _set_maintenance_mode(ep_id, "move", True)

        # Now pipeline should be blocked
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is False
        assert "maintenance_mode" in check["reason"].lower() or "move" in check["reason"].lower()

        # Clear maintenance mode
        _set_maintenance_mode(ep_id, "move", False)

        # Pipeline should be unblocked
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is True or check["reason"] == ""


class TestRemoveRestoreWorkflow:
    """Integration tests for remove and restore workflows."""

    def test_remove_and_restore_complete_workflow(self, integration_data_root):
        """Test complete remove → restore workflow with audit logging."""
        data_root, ep_id = integration_data_root

        # Initial state
        episodes = list_episodes()
        assert len(episodes) == 1

        # Remove episode
        remove_episode(ep_id, soft=True, actor="integration_test", reason="Testing removal")

        # Verify episode hidden from default listings
        episodes = list_episodes()
        assert len(episodes) == 0

        # Verify episode appears with include_archived
        episodes = list_episodes(include_archived=True)
        assert len(episodes) == 1
        assert episodes[0].archived is True

        # Verify data still on disk
        ep_dir = data_root / "harvest" / ep_id
        assert ep_dir.exists()
        assert (ep_dir / "manifest.parquet").exists()

        # Verify remove audit event
        audit_events = get_audit_history(ep_id)
        remove_event = [e for e in audit_events if e.get("op_type") == "remove"][0]
        assert remove_event["after_state"]["archived"] is True
        assert remove_event["actor"] == "integration_test"

        # Restore episode
        restore_episode(ep_id, actor="integration_test", reason="Testing restore")

        # Verify episode visible again
        episodes = list_episodes()
        assert len(episodes) == 1
        assert episodes[0].archived is False

        # Verify restore audit event
        audit_events = get_audit_history(ep_id)
        restore_event = [e for e in audit_events if e.get("op_type") == "restore"][0]
        assert restore_event["after_state"]["archived"] is False

    def test_remove_blocks_pipeline_during_operation(self, integration_data_root):
        """Test that pipeline blocked during remove operation."""
        data_root, ep_id = integration_data_root

        # Simulate remove setting maintenance mode
        from app.lib.episode_manager import _set_maintenance_mode
        _set_maintenance_mode(ep_id, "remove", True, {
            "current": 0,
            "total": 1,
            "stage": "archiving"
        })

        # Check if pipeline can run
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is False
        assert check["maintenance_mode"] is True

        # Get reason
        reason = get_maintenance_reason(ep_id, data_root)
        assert "archive" in reason.lower() or "remove" in reason.lower()

        # Clear maintenance mode
        _set_maintenance_mode(ep_id, "remove", False)

        # Pipeline should be unblocked
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is True


class TestRehashWorkflow:
    """Integration tests for rehash workflow."""

    def test_rehash_complete_workflow(self, integration_data_root):
        """Test complete rehash workflow with validation and audit."""
        data_root, ep_id = integration_data_root

        # Get original hash from registry
        from app.lib.registry import load_registry
        reg = load_registry()
        original_hash = reg["shows"][0]["seasons"][0]["episodes"][0]["episode_hash"]

        # Rehash episode
        result = rehash_episode(ep_id, actor="integration_test", reason="Testing rehash")

        # Verify result structure
        assert result.episode_id == ep_id
        assert result.old_hash == original_hash
        assert result.new_hash != original_hash
        assert len(result.new_hash) == 12

        # Verify all files validated
        assert result.validated_files["manifest"] is True
        assert result.validated_files["embeddings"] is True
        assert result.validated_files["tracks"] is True
        assert result.validated_files["clusters"] is True
        assert result.validated_files["stills"] is True

        # Verify no errors
        assert len(result.errors) == 0

        # Verify registry updated
        reg = load_registry()
        ep_data = reg["shows"][0]["seasons"][0]["episodes"][0]
        assert ep_data["episode_hash"] == result.new_hash
        assert "last_rehash" in ep_data

        # Verify audit event
        audit_events = get_audit_history(ep_id)
        rehash_event = [e for e in audit_events if e.get("op_type") == "rehash"][0]
        assert rehash_event["before_state"]["hash"] == original_hash
        assert rehash_event["after_state"]["hash"] == result.new_hash
        assert rehash_event["after_state"]["validated"]

    def test_rehash_detects_missing_files(self, integration_data_root):
        """Test that rehash detects and reports missing files."""
        data_root, ep_id = integration_data_root

        # Remove a key file
        (data_root / "harvest" / ep_id / "clusters.json").unlink()

        # Rehash
        result = rehash_episode(ep_id, actor="integration_test", reason="Test validation")

        # Verify validation caught the issue
        assert result.validated_files["clusters"] is False

        # Verify audit log includes errors
        audit_events = get_audit_history(ep_id)
        rehash_event = [e for e in audit_events if e.get("op_type") == "rehash"][-1]
        assert "errors" in rehash_event.get("metadata", {})


class TestPipelineGating:
    """Integration tests for pipeline gating during episode operations."""

    def test_prepare_blocked_during_move(self, integration_data_root):
        """Test that Prepare is blocked when Move operation is active."""
        data_root, ep_id = integration_data_root

        # Set maintenance mode as if move is in progress
        from app.lib.episode_manager import _set_maintenance_mode
        _set_maintenance_mode(ep_id, "move", True, {
            "current": 1,
            "total": 3,
            "stage": "updating_registry"
        })

        # Check pipeline
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is False
        assert check["maintenance_mode"] is True
        assert "move" in check["reason"].lower() or "updating_registry" in check["reason"].lower()

        # Verify maintenance mode detected
        assert is_maintenance_mode(ep_id, data_root) is True

        # Get human-readable reason
        reason = get_maintenance_reason(ep_id, data_root)
        assert "move" in reason.lower()
        assert "1/3" in reason or "updating_registry" in reason.lower()

    def test_cluster_blocked_during_rehash(self, integration_data_root):
        """Test that Cluster is blocked when Rehash operation is active."""
        data_root, ep_id = integration_data_root

        # Set maintenance mode for rehash
        from app.lib.episode_manager import _set_maintenance_mode
        _set_maintenance_mode(ep_id, "rehash", True, {
            "current": 2,
            "total": 3,
            "stage": "validating"
        })

        # Check pipeline
        check = check_pipeline_can_run(ep_id, data_root)
        assert check["can_run"] is False

        # Get reason
        reason = get_maintenance_reason(ep_id, data_root)
        assert "rehash" in reason.lower()

    def test_multiple_operations_sequence(self, integration_data_root):
        """Test sequence of operations with proper gating."""
        data_root, ep_id = integration_data_root

        # Operation 1: Move
        from app.lib.episode_manager import _set_maintenance_mode
        _set_maintenance_mode(ep_id, "move", True)
        assert check_pipeline_can_run(ep_id, data_root)["can_run"] is False
        _set_maintenance_mode(ep_id, "move", False)

        # Pipeline should be unblocked
        assert check_pipeline_can_run(ep_id, data_root)["can_run"] is True

        # Operation 2: Rehash
        _set_maintenance_mode(ep_id, "rehash", True)
        assert check_pipeline_can_run(ep_id, data_root)["can_run"] is False
        _set_maintenance_mode(ep_id, "rehash", False)

        # Pipeline should be unblocked again
        assert check_pipeline_can_run(ep_id, data_root)["can_run"] is True


class TestAuditLogging:
    """Integration tests for audit logging across operations."""

    def test_all_operations_create_audit_events(self, integration_data_root):
        """Test that all operations create proper audit events."""
        data_root, ep_id = integration_data_root

        # Perform all operations
        remove_episode(ep_id, soft=True, actor="test", reason="test remove")
        restore_episode(ep_id, actor="test", reason="test restore")
        rehash_episode(ep_id, actor="test", reason="test rehash")
        move_episode(ep_id, "dstshow", 2, actor="test", reason="test move")

        # Get audit history
        audit_events = get_audit_history(ep_id)

        # Verify all operation types present
        op_types = {e.get("op_type") for e in audit_events}
        assert "remove" in op_types
        assert "restore" in op_types
        assert "rehash" in op_types
        assert "move" in op_types

        # Verify all have required fields
        for event in audit_events:
            assert "timestamp" in event
            assert "actor" in event
            assert "reason" in event
            assert "before_state" in event
            assert "after_state" in event

    def test_audit_history_ordering(self, integration_data_root):
        """Test that audit history is ordered by timestamp."""
        data_root, ep_id = integration_data_root

        # Perform operations
        remove_episode(ep_id, soft=True, actor="test", reason="first")
        restore_episode(ep_id, actor="test", reason="second")
        rehash_episode(ep_id, actor="test", reason="third")

        # Get audit history
        audit_events = get_audit_history(ep_id)

        # Verify ordering (should be most recent first)
        timestamps = [e.get("timestamp") for e in audit_events]
        assert timestamps == sorted(timestamps, reverse=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
