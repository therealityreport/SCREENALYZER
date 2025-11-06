"""
Unit tests for episode_manager operations.

Tests Move, Remove, Restore, and Rehash operations with maintenance mode gating.
"""

import json
import pytest
import shutil
from pathlib import Path
from datetime import datetime

from app.lib.episode_manager import (
    list_episodes,
    move_episode,
    remove_episode,
    restore_episode,
    rehash_episode,
    _compute_episode_hash,
    _set_maintenance_mode,
    EpisodeMeta,
    EpisodeHash,
)
from app.lib.registry import load_registry, save_registry


@pytest.fixture
def temp_data_root(tmp_path):
    """Create temporary data directory structure."""
    data_root = tmp_path / "data"

    # Create harvest directories
    harvest_root = data_root / "harvest"
    harvest_root.mkdir(parents=True)

    # Create test episode directories
    for ep_id in ["TEST_S01_E01", "TEST_S01_E02"]:
        ep_dir = harvest_root / ep_id
        ep_dir.mkdir()

        # Create diagnostics
        diag_dir = ep_dir / "diagnostics"
        diag_dir.mkdir()

        # Create key files
        (ep_dir / "manifest.parquet").write_bytes(b"mock_manifest_data")
        (ep_dir / "tracks.json").write_text("{}")
        (ep_dir / "clusters.json").write_text("{}")

        # Create stills directory with mock files
        stills_dir = ep_dir / "stills"
        stills_dir.mkdir()
        (stills_dir / "track_001.jpg").write_bytes(b"mock_image")

    # Create mock registry
    registry_path = data_root / "registry.json"
    registry = {
        "shows": [
            {
                "show_id": "testshow",
                "seasons": [
                    {
                        "season_id": "s01",
                        "season_number": 1,
                        "episodes": [
                            {
                                "episode_id": "TEST_S01_E01",
                                "archived": False,
                                "episode_hash": "abc123",
                            },
                            {
                                "episode_id": "TEST_S01_E02",
                                "archived": False,
                                "episode_hash": "def456",
                            },
                        ],
                    },
                    {
                        "season_id": "s02",
                        "season_number": 2,
                        "episodes": [],
                    },
                ],
            },
            {
                "show_id": "othershow",
                "seasons": [
                    {
                        "season_id": "s01",
                        "season_number": 1,
                        "episodes": [],
                    },
                ],
            },
        ],
    }

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)

    # Temporarily override DATA_ROOT
    import app.lib.episode_manager as em
    original_data_root = em.DATA_ROOT
    em.DATA_ROOT = data_root

    yield data_root

    # Restore original
    em.DATA_ROOT = original_data_root


class TestListEpisodes:
    """Tests for list_episodes function."""

    def test_list_all_episodes(self, temp_data_root):
        """Should list all non-archived episodes."""
        episodes = list_episodes()

        assert len(episodes) == 2
        assert all(isinstance(ep, EpisodeMeta) for ep in episodes)
        assert {ep.episode_id for ep in episodes} == {"TEST_S01_E01", "TEST_S01_E02"}

    def test_list_by_show(self, temp_data_root):
        """Should filter episodes by show."""
        episodes = list_episodes(show="testshow")

        assert len(episodes) == 2
        assert all(ep.show_id == "testshow" for ep in episodes)

    def test_list_by_season(self, temp_data_root):
        """Should filter episodes by season number."""
        episodes = list_episodes(season=1)

        assert len(episodes) == 2
        assert all(ep.season_number == 1 for ep in episodes)

    def test_list_archived_episodes(self, temp_data_root):
        """Should include archived episodes when requested."""
        # First archive an episode
        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test")

        # Should not appear by default
        episodes = list_episodes()
        assert len(episodes) == 1
        assert episodes[0].episode_id == "TEST_S01_E02"

        # Should appear when include_archived=True
        episodes = list_episodes(include_archived=True)
        assert len(episodes) == 2
        assert any(ep.archived for ep in episodes)


class TestMoveEpisode:
    """Tests for move_episode function."""

    def test_move_to_different_season(self, temp_data_root):
        """Should move episode to different season in same show."""
        move_episode("TEST_S01_E01", "testshow", 2, actor="test", reason="test move")

        # Check registry was updated
        reg = load_registry()

        # Should be gone from S01
        s01_episodes = reg["shows"][0]["seasons"][0]["episodes"]
        assert not any(ep["episode_id"] == "TEST_S01_E01" for ep in s01_episodes)

        # Should appear in S02
        s02_episodes = reg["shows"][0]["seasons"][1]["episodes"]
        assert any(ep["episode_id"] == "TEST_S01_E01" for ep in s02_episodes)

    def test_move_to_different_show(self, temp_data_root):
        """Should move episode to different show."""
        move_episode("TEST_S01_E01", "othershow", 1, actor="test", reason="test move")

        reg = load_registry()

        # Should be gone from testshow
        testshow_all_eps = []
        for season in reg["shows"][0]["seasons"]:
            testshow_all_eps.extend(ep["episode_id"] for ep in season["episodes"])
        assert "TEST_S01_E01" not in testshow_all_eps

        # Should appear in othershow
        othershow_eps = reg["shows"][1]["seasons"][0]["episodes"]
        assert any(ep["episode_id"] == "TEST_S01_E01" for ep in othershow_eps)

    def test_move_sets_maintenance_mode(self, temp_data_root):
        """Should set maintenance mode during move."""
        import threading

        maintenance_seen = []

        def check_maintenance():
            # Check during move
            from app.lib.pipeline import is_maintenance_mode
            maintenance_seen.append(is_maintenance_mode("TEST_S01_E01", temp_data_root))

        # Move in thread to check during operation
        import time
        check_maintenance()  # Before
        move_episode("TEST_S01_E01", "testshow", 2, actor="test", reason="test")

        # After move, maintenance should be cleared
        from app.lib.pipeline import is_maintenance_mode
        assert not is_maintenance_mode("TEST_S01_E01", temp_data_root)

    def test_move_same_location_raises_error(self, temp_data_root):
        """Should raise error if source equals destination."""
        with pytest.raises(ValueError, match="Source and destination are the same"):
            move_episode("TEST_S01_E01", "testshow", 1, actor="test", reason="test")

    def test_move_nonexistent_episode_raises_error(self, temp_data_root):
        """Should raise error if episode not found."""
        with pytest.raises(ValueError, match="not found in registry"):
            move_episode("NONEXISTENT", "testshow", 2, actor="test", reason="test")

    def test_move_to_nonexistent_show_raises_error(self, temp_data_root):
        """Should raise error if destination show doesn't exist."""
        with pytest.raises(ValueError, match="Destination show.*not found"):
            move_episode("TEST_S01_E01", "fakeshow", 1, actor="test", reason="test")


class TestRemoveEpisode:
    """Tests for remove_episode function."""

    def test_soft_remove_marks_archived(self, temp_data_root):
        """Should mark episode as archived."""
        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test removal")

        reg = load_registry()
        ep_data = reg["shows"][0]["seasons"][0]["episodes"][0]

        assert ep_data["archived"] is True
        assert "archived_at" in ep_data

    def test_soft_remove_preserves_data(self, temp_data_root):
        """Should keep all files on disk."""
        ep_dir = temp_data_root / "harvest" / "TEST_S01_E01"

        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test")

        # All files should still exist
        assert (ep_dir / "manifest.parquet").exists()
        assert (ep_dir / "tracks.json").exists()
        assert (ep_dir / "stills").exists()

    def test_hard_remove_raises_error(self, temp_data_root):
        """Should raise error for hard delete (not implemented)."""
        with pytest.raises(ValueError, match="Hard delete not supported"):
            remove_episode("TEST_S01_E01", soft=False, actor="test", reason="test")

    def test_remove_nonexistent_raises_error(self, temp_data_root):
        """Should raise error if episode not found."""
        with pytest.raises(ValueError, match="not found in registry"):
            remove_episode("NONEXISTENT", soft=True, actor="test", reason="test")

    def test_remove_sets_maintenance_mode(self, temp_data_root):
        """Should set maintenance mode during removal."""
        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test")

        # After removal, maintenance should be cleared
        from app.lib.pipeline import is_maintenance_mode
        assert not is_maintenance_mode("TEST_S01_E01", temp_data_root)


class TestRestoreEpisode:
    """Tests for restore_episode function."""

    def test_restore_clears_archived_flag(self, temp_data_root):
        """Should clear archived flag and timestamp."""
        # First archive
        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test")

        # Then restore
        restore_episode("TEST_S01_E01", actor="test", reason="test restore")

        reg = load_registry()
        ep_data = reg["shows"][0]["seasons"][0]["episodes"][0]

        assert ep_data["archived"] is False
        assert "archived_at" not in ep_data

    def test_restore_makes_visible_in_listings(self, temp_data_root):
        """Should make episode appear in default listings."""
        # Archive
        remove_episode("TEST_S01_E01", soft=True, actor="test", reason="test")
        assert len(list_episodes()) == 1

        # Restore
        restore_episode("TEST_S01_E01", actor="test", reason="test")
        assert len(list_episodes()) == 2

    def test_restore_nonexistent_raises_error(self, temp_data_root):
        """Should raise error if episode not found."""
        with pytest.raises(ValueError, match="not found in registry"):
            restore_episode("NONEXISTENT", actor="test", reason="test")


class TestRehashEpisode:
    """Tests for rehash_episode function."""

    def test_rehash_updates_hash(self, temp_data_root):
        """Should compute and save new hash."""
        result = rehash_episode("TEST_S01_E01", actor="test", reason="test rehash")

        assert isinstance(result, EpisodeHash)
        assert result.episode_id == "TEST_S01_E01"
        assert result.old_hash == "abc123"
        assert result.new_hash != result.old_hash
        assert len(result.new_hash) == 12  # MD5 truncated to 12 chars

    def test_rehash_validates_files(self, temp_data_root):
        """Should validate key files exist and are readable."""
        result = rehash_episode("TEST_S01_E01", actor="test", reason="test")

        assert "manifest" in result.validated_files
        assert "tracks" in result.validated_files
        assert "clusters" in result.validated_files
        assert "stills" in result.validated_files

        # All should be valid
        assert result.validated_files["manifest"] is True
        assert result.validated_files["tracks"] is True
        assert result.validated_files["clusters"] is True
        assert result.validated_files["stills"] is True

    def test_rehash_detects_missing_files(self, temp_data_root):
        """Should detect missing files."""
        # Remove a key file
        (temp_data_root / "harvest" / "TEST_S01_E01" / "clusters.json").unlink()

        result = rehash_episode("TEST_S01_E01", actor="test", reason="test")

        assert result.validated_files["clusters"] is False

    def test_rehash_detects_empty_stills(self, temp_data_root):
        """Should detect empty stills directory."""
        # Empty the stills directory
        stills_dir = temp_data_root / "harvest" / "TEST_S01_E01" / "stills"
        for file in stills_dir.iterdir():
            file.unlink()

        result = rehash_episode("TEST_S01_E01", actor="test", reason="test")

        assert result.validated_files["stills"] is False
        assert any("stills" in err and "empty" in err for err in result.errors)

    def test_rehash_updates_registry(self, temp_data_root):
        """Should update registry with new hash."""
        result = rehash_episode("TEST_S01_E01", actor="test", reason="test")

        reg = load_registry()
        ep_data = reg["shows"][0]["seasons"][0]["episodes"][0]

        assert ep_data["episode_hash"] == result.new_hash
        assert "last_rehash" in ep_data

    def test_rehash_nonexistent_raises_error(self, temp_data_root):
        """Should raise error if episode not found."""
        with pytest.raises(ValueError, match="not found in registry"):
            rehash_episode("NONEXISTENT", actor="test", reason="test")


class TestMaintenanceMode:
    """Tests for maintenance mode helpers."""

    def test_set_maintenance_mode(self, temp_data_root):
        """Should create or update pipeline_state.json."""
        ep_id = "TEST_S01_E01"
        _set_maintenance_mode(ep_id, "test_op", True, {"current": 1, "total": 3, "stage": "testing"})

        state_file = temp_data_root / "harvest" / ep_id / "diagnostics" / "pipeline_state.json"
        assert state_file.exists()

        with open(state_file) as f:
            state = json.load(f)

        assert state["maintenance_mode"] is True
        assert state["active_op"]["type"] == "test_op"
        assert state["active_op_progress"]["current"] == 1

    def test_clear_maintenance_mode(self, temp_data_root):
        """Should clear maintenance mode fields."""
        ep_id = "TEST_S01_E01"

        # Set maintenance mode
        _set_maintenance_mode(ep_id, "test_op", True)

        # Clear it
        _set_maintenance_mode(ep_id, "test_op", False)

        state_file = temp_data_root / "harvest" / ep_id / "diagnostics" / "pipeline_state.json"
        with open(state_file) as f:
            state = json.load(f)

        assert state["maintenance_mode"] is False
        assert "active_op" not in state
        assert "active_op_progress" not in state


class TestComputeEpisodeHash:
    """Tests for _compute_episode_hash helper."""

    def test_hash_changes_when_files_change(self, temp_data_root):
        """Should produce different hash when file content changes."""
        ep_id = "TEST_S01_E01"

        hash1 = _compute_episode_hash(ep_id)

        # Modify a file
        tracks_file = temp_data_root / "harvest" / ep_id / "tracks.json"
        tracks_file.write_text('{"modified": true}')

        hash2 = _compute_episode_hash(ep_id)

        assert hash1 != hash2

    def test_hash_consistent_for_same_content(self, temp_data_root):
        """Should produce same hash for same content."""
        ep_id = "TEST_S01_E01"

        hash1 = _compute_episode_hash(ep_id)
        hash2 = _compute_episode_hash(ep_id)

        assert hash1 == hash2

    def test_hash_handles_missing_files(self, temp_data_root):
        """Should handle missing optional files."""
        ep_id = "TEST_S01_E01"

        # Remove clusters.json
        (temp_data_root / "harvest" / ep_id / "clusters.json").unlink()

        # Should still compute hash
        hash_val = _compute_episode_hash(ep_id)
        assert isinstance(hash_val, str)
        assert len(hash_val) == 12


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
