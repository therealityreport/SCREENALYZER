#!/usr/bin/env python3
"""
State Inspection Tool

Inspect job envelope and episode registry sync status.

Usage:
    python tools/inspect_state.py <episode_key>
    python tools/inspect_state.py rhobh_s05_e03
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def load_json_file(file_path: Path) -> Optional[dict]:
    """Load JSON file or return None if not found."""
    if not file_path.exists():
        return None
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading {file_path}: {e}")
        return None


def format_timestamp(ts_str: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts_str


def inspect_episode_state(episode_key: str, data_root: Path = Path("data")) -> None:
    """Inspect and display episode state from registry and job envelope."""

    print(f"\n{'='*70}")
    print(f"State Inspection: {episode_key}")
    print(f"{'='*70}\n")

    # Load episode registry
    registry_path = data_root / "episodes" / episode_key / "state.json"
    registry = load_json_file(registry_path)

    print("📋 Episode Registry")
    print(f"   Path: {registry_path}")

    if registry:
        print(f"   ✅ EXISTS\n")
        print(f"   Episode ID: {registry.get('episode_id', 'N/A')}")
        print(f"   Show: {registry.get('show', 'N/A')}")
        print(f"   Season: {registry.get('season', 'N/A')}")
        print(f"   Episode: {registry.get('episode', 'N/A')}")
        print(f"   Video Path: {registry.get('video_path', 'N/A')}\n")

        # States
        states = registry.get("states", {})
        print("   States:")
        for state_name, state_value in states.items():
            icon = "✅" if state_value else "⏸️ "
            print(f"      {icon} {state_name}: {state_value}")

        # Timestamps
        timestamps = registry.get("timestamps", {})
        if timestamps:
            print("\n   Timestamps:")
            if "created" in timestamps:
                print(f"      Created: {format_timestamp(timestamps['created'])}")
            if "last_modified" in timestamps:
                print(f"      Last Modified: {format_timestamp(timestamps['last_modified'])}")
    else:
        print(f"   ❌ NOT FOUND\n")

    print(f"\n{'─'*70}\n")

    # Find related job envelopes
    jobs_dir = data_root / "jobs"
    if not jobs_dir.exists():
        print("📦 Job Envelopes")
        print(f"   Jobs directory not found: {jobs_dir}\n")
        return

    # Look for jobs matching this episode_key
    related_jobs = []
    for job_dir in jobs_dir.iterdir():
        if job_dir.is_dir():
            meta_file = job_dir / "meta.json"
            if meta_file.exists():
                envelope = load_json_file(meta_file)
                if envelope and envelope.get("episode_key") == episode_key:
                    related_jobs.append((job_dir.name, envelope))

    print("📦 Related Job Envelopes")
    if related_jobs:
        print(f"   Found {len(related_jobs)} job(s)\n")

        for job_id, envelope in related_jobs:
            print(f"   Job ID: {job_id}")
            print(f"   Mode: {envelope.get('mode', 'N/A')}")
            print(f"   Registry Path: {envelope.get('registry_path', 'N/A')}")

            if "self_healed" in envelope:
                print(f"   ⚠️  Self-Healed: {envelope.get('self_healed')}")

            # Created timestamp
            created_ts = envelope.get("created_at")
            if created_ts:
                try:
                    created_dt = datetime.fromtimestamp(created_ts)
                    print(f"   Created: {created_dt.strftime('%Y-%m-%d %H:%M:%S')}")
                except Exception:
                    print(f"   Created: {created_ts}")

            # Stages
            stages = envelope.get("stages", {})
            if stages:
                print(f"\n   Stages:")
                for stage_name, stage_data in stages.items():
                    status = stage_data.get("status", "unknown")
                    icon_map = {
                        "pending": "⏸️ ",
                        "running": "🔄",
                        "ok": "✅",
                        "skipped": "⏭️ ",
                        "error": "❌",
                    }
                    icon = icon_map.get(status, "❓")
                    print(f"      {icon} {stage_name}: {status}")

                    if status == "error" and "error" in stage_data:
                        error_msg = stage_data["error"]
                        # Truncate long errors
                        if len(error_msg) > 80:
                            error_msg = error_msg[:77] + "..."
                        print(f"         Error: {error_msg}")

            print()
    else:
        print(f"   No jobs found for episode_key: {episode_key}\n")

    print(f"{'─'*70}\n")

    # Sync status
    print("🔗 Registry-Job Sync Status")

    if not registry:
        print("   ❌ Registry missing - cannot verify sync\n")
        return

    if not related_jobs:
        print("   ⚠️  No jobs found - registry exists but no jobs yet\n")
        return

    # Check if states match
    registry_states = registry.get("states", {})
    all_synced = True

    for job_id, envelope in related_jobs:
        stages = envelope.get("stages", {})

        # Check detect <-> detected
        if "detect" in stages:
            detect_status = stages["detect"].get("status")
            detected_state = registry_states.get("detected", False)

            if detect_status == "ok" and not detected_state:
                print(f"   ⚠️  Job {job_id}: detect=ok but registry.detected=false")
                all_synced = False
            elif detect_status == "ok" and detected_state:
                print(f"   ✅ Job {job_id}: detect stage synced")

        # Check track <-> tracked
        if "track" in stages:
            track_status = stages["track"].get("status")
            tracked_state = registry_states.get("tracked", False)

            if track_status == "ok" and not tracked_state:
                print(f"   ⚠️  Job {job_id}: track=ok but registry.tracked=false")
                all_synced = False
            elif track_status == "ok" and tracked_state:
                print(f"   ✅ Job {job_id}: track stage synced")

    if all_synced and related_jobs:
        print(f"   ✅ All stages synced with registry\n")
    elif all_synced:
        print()

    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python tools/inspect_state.py <episode_key>")
        print("Example: python tools/inspect_state.py rhobh_s05_e03")
        sys.exit(1)

    episode_key = sys.argv[1]

    # Support both canonical (rhobh_s05_e03) and full (RHOBH_S05_E03_11062025) formats
    if "_" in episode_key and len(episode_key.split("_")) > 3:
        # Extract first 3 parts and lowercase
        parts = episode_key.split("_")
        episode_key = "_".join(parts[:3]).lower()

    inspect_episode_state(episode_key)


if __name__ == "__main__":
    main()
