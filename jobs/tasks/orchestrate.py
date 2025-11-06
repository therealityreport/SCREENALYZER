"""
Full pipeline orchestrator with Prepare mode.

Workflow:
1. Prepare: Detect/Embed → Track → Generate Face Stills (SER-FIQ + 160×200 thumbs)
   [STOP HERE - Manual facebank curation]
2. Cluster: Cluster prepared tracks against current facebank (on demand)
3. Enhance: Mine seeds, rebuild bank (optional)
4. Analyze: Generate timeline/totals from final labels (on demand)

No auto-clustering after Prepare.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from screentime.diagnostics.utils import emit_progress, json_safe, archive_pipeline_state
from app.lib.pipeline import update_step_stats

logger = logging.getLogger(__name__)

# Prepare stages (run before manual facebank curation)
PREPARE_STAGES = [
    ("Detect/Embed", "detect"),
    ("Track", "track"),
    ("Generate Face Stills", "stills"),
]

# Full pipeline stages (all stages including cluster/analytics)
FULL_STAGES = [
    ("Detect/Embed", "detect"),
    ("Track", "track"),
    ("Generate Face Stills", "stills"),
    ("Cluster", "cluster"),
    ("Analytics", "analytics"),
]


def check_artifacts(episode_id: str, data_root: Path = Path("data")) -> Dict[str, bool]:
    """Check which pipeline artifacts exist."""
    harvest_dir = data_root / "harvest" / episode_id
    outputs_dir = data_root / "outputs" / episode_id

    thumbs_dir = harvest_dir / "stills" / "thumbs"
    has_thumbs = thumbs_dir.exists() and len(list(thumbs_dir.glob("*.jpg"))) > 0

    # Check for key analytics outputs
    has_analytics = (
        (outputs_dir / "timeline.csv").exists() and
        (outputs_dir / "totals.csv").exists()
    )

    return {
        "manifest": (harvest_dir / "manifest.parquet").exists(),
        "embeddings": (harvest_dir / "embeddings.parquet").exists(),
        "tracks": (harvest_dir / "tracks.json").exists(),
        "clusters": (harvest_dir / "clusters.json").exists(),
        "stills_manifest": (harvest_dir / "stills" / "track_stills.jsonl").exists(),
        "stills_thumbs": has_thumbs,
        "analytics": has_analytics,
    }


def is_prepared(episode_id: str, data_root: Path = Path("data")) -> bool:
    """Check if episode has completed Prepare stage (tracks + stills ready)."""
    artifacts = check_artifacts(episode_id, data_root)
    return (
        artifacts["embeddings"] and
        artifacts["tracks"] and
        artifacts["stills_manifest"] and
        artifacts["stills_thumbs"]
    )


def mark_prepared(episode_id: str, data_root: Path = Path("data")) -> None:
    """Mark episode as prepared (tracks + stills complete)."""
    harvest_dir = data_root / "harvest" / episode_id
    diagnostics_dir = harvest_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    state_file = diagnostics_dir / "pipeline_state.json"

    # Update or create state with prepared=true
    state = {}
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

    state["prepared"] = True
    state["prepared_at"] = json_safe({"timestamp": "utcnow"})

    with open(state_file, "w") as f:
        json.dump(json_safe(state), f, indent=2)


def needs_step(step_key: str, artifacts: Dict[str, bool], force: bool = False) -> bool:
    """Check if a pipeline step needs to run."""
    if force:
        return True

    if step_key == "detect":
        return not artifacts["embeddings"]
    elif step_key == "track":
        return not artifacts["tracks"]
    elif step_key == "cluster":
        return not artifacts["clusters"]
    elif step_key == "stills":
        return not (artifacts["stills_manifest"] and artifacts["stills_thumbs"])
    elif step_key == "analytics":
        return not artifacts["analytics"]

    return False


def orchestrate_prepare(
    episode_id: str,
    job_id: Optional[str] = None,
    data_root: Path = Path("data"),
    force: bool = False,
    resume: bool = True,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Prepare pipeline: Detect/Embed → Track → Generate Face Stills (SER-FIQ + 160×200 thumbs).

    Stops after stills generation. Does NOT cluster.
    User should curate facebank, then click Cluster separately.

    Args:
        episode_id: Episode identifier
        job_id: Optional job identifier
        data_root: Root data directory
        force: Force re-run all stages
        resume: Resume stills generation from existing manifest
        progress_callback: Optional callback(progress_dict) for real-time updates

    Returns:
        Dict with per-stage results and final state
    """
    if job_id is None:
        job_id = f"prepare_{episode_id}"

    harvest_dir = data_root / "harvest" / episode_id
    diagnostics_dir = harvest_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    state_file = diagnostics_dir / "pipeline_state.json"

    # Check current state
    artifacts = check_artifacts(episode_id, data_root)
    logger.info(f"[{job_id}] Current artifacts: {artifacts}")

    # Auto-harvest if manifest doesn't exist
    if not artifacts["manifest"]:
        logger.info(f"[{job_id}] Manifest not found, running harvest first...")

        # Find video path from registry
        from screentime.models import get_registry
        registry = get_registry()

        video_path = None
        for show in getattr(registry, "shows", []):
            for season in getattr(show, "seasons", []):
                for ep in getattr(season, "episodes", []):
                    if ep.episode_id == episode_id:
                        video_path = str(data_root / "videos" / show.show_id.lower() / f"s{season.season_number:02d}" / f"{episode_id}.mp4")
                        break
                if video_path:
                    break
            if video_path:
                break

        if not video_path or not Path(video_path).exists():
            raise ValueError(f"Video file not found for episode {episode_id}")

        logger.info(f"[{job_id}] Running harvest from {video_path}")

        # Run harvest
        from jobs.tasks.harvest import harvest_task
        harvest_result = harvest_task(
            job_id=f"harvest_{episode_id}",
            episode_id=episode_id,
            video_path=video_path,
            resume_from=None,
        )

        logger.info(f"[{job_id}] Harvest complete: {harvest_result}")

        # Re-check artifacts after harvest
        artifacts = check_artifacts(episode_id, data_root)
        logger.info(f"[{job_id}] Artifacts after harvest: {artifacts}")

    results = {
        "episode_id": episode_id,
        "job_id": job_id,
        "mode": "prepare",
        "stages": {},
        "final_state": artifacts.copy(),
    }

    total_steps = len(PREPARE_STAGES)

    # Execute each Prepare stage
    for step_index, (step_name, step_key) in enumerate(PREPARE_STAGES, start=1):

        if not needs_step(step_key, artifacts, force):
            logger.info(f"[{job_id}] {step_name} artifacts exist, skipping")
            results["stages"][step_key] = {"status": "skipped"}

            # Emit skipped progress
            emit_progress(
                episode_id=episode_id,
                step=step_name,
                step_index=step_index,
                total_steps=total_steps,
                status="skipped",
                message=f"{step_name} already complete",
            )

            if progress_callback:
                progress_callback({
                    "step": step_name,
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "status": "skipped",
                })

            continue

        # Emit running progress
        logger.info(f"[{job_id}] Running {step_name}...")
        emit_progress(
            episode_id=episode_id,
            step=step_name,
            step_index=step_index,
            total_steps=total_steps,
            status="running",
            message=f"Running {step_name}...",
        )

        if progress_callback:
            progress_callback({
                "step": step_name,
                "step_index": step_index,
                "total_steps": total_steps,
                "status": "running",
            })

        # Run the stage
        step_start_time = time.time()
        try:
            if step_key == "detect":
                result = _run_detect(job_id, episode_id)
            elif step_key == "track":
                result = _run_track(job_id, episode_id)
            elif step_key == "stills":
                result = _run_stills(episode_id, data_root, force=force, resume=resume)
            else:
                raise ValueError(f"Unknown step: {step_key}")

            # Update ETA stats
            step_elapsed = time.time() - step_start_time
            try:
                update_step_stats(
                    episode_id=episode_id,
                    operation="prepare",
                    step_name=step_key,
                    duration_seconds=step_elapsed,
                    data_root=data_root,
                )
            except Exception as eta_exc:
                logger.warning(f"Failed to update ETA stats for {step_key}: {eta_exc}")

            results["stages"][step_key] = {"status": "ok", "result": json_safe(result)}
            artifacts = check_artifacts(episode_id, data_root)

            # Emit success progress
            emit_progress(
                episode_id=episode_id,
                step=step_name,
                step_index=step_index,
                total_steps=total_steps,
                status="ok",
                message=f"{step_name} complete",
                extra={"result": json_safe(result)},
            )

            if progress_callback:
                progress_callback({
                    "step": step_name,
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "status": "ok",
                    "result": result,
                })

        except Exception as exc:
            logger.error(f"[{job_id}] {step_name} failed: {exc}", exc_info=True)
            results["stages"][step_key] = {"status": "error", "error": str(exc)}

            # Emit error progress
            emit_progress(
                episode_id=episode_id,
                step=step_name,
                step_index=step_index,
                total_steps=total_steps,
                status="error",
                message=f"{step_name} failed: {exc}",
            )

            if progress_callback:
                progress_callback({
                    "step": step_name,
                    "step_index": step_index,
                    "total_steps": total_steps,
                    "status": "error",
                    "error": str(exc),
                })

            # Save state and abort
            results["final_state"] = check_artifacts(episode_id, data_root)
            _save_state(state_file, results)
            return results

    # Update final state
    results["final_state"] = check_artifacts(episode_id, data_root)

    # Mark episode as prepared
    mark_prepared(episode_id, data_root)
    results["prepared"] = True

    # Emit final "done" progress
    last_result = results.get("stages", {}).get("stills", {}).get("result", {})
    emit_progress(
        episode_id=episode_id,
        step="Prepare Complete",
        step_index=total_steps,
        total_steps=total_steps,
        status="done",
        message="Tracks & Stills ready → Curate facebank, then Cluster",
        extra={"result": last_result, "prepared": True},
    )

    if progress_callback:
        progress_callback({
            "step": "Prepare Complete",
            "step_index": total_steps,
            "total_steps": total_steps,
            "status": "done",
            "prepared": True,
        })

    _save_state(state_file, results)
    logger.info(f"[{job_id}] Prepare complete: {results['final_state']}")
    return results


def orchestrate_cluster_only(
    episode_id: str,
    job_id: Optional[str] = None,
    data_root: Path = Path("data"),
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Cluster stage only (requires prepared episode).

    Clusters prepared tracks against current facebank.
    Does NOT run detect/track/stills.

    Args:
        episode_id: Episode identifier
        job_id: Optional job identifier
        data_root: Root data directory
        progress_callback: Optional callback(progress_dict) for real-time updates

    Returns:
        Dict with cluster result
    """
    if job_id is None:
        job_id = f"cluster_{episode_id}"

    # Check if prepared
    if not is_prepared(episode_id, data_root):
        error_msg = "Episode not prepared. Run Prepare Tracks & Stills first."
        logger.error(f"[{job_id}] {error_msg}")
        return {
            "episode_id": episode_id,
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
        }

    harvest_dir = data_root / "harvest" / episode_id
    diagnostics_dir = harvest_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    state_file = diagnostics_dir / "pipeline_state.json"

    logger.info(f"[{job_id}] Running cluster for {episode_id}")

    # Emit starting progress
    emit_progress(
        episode_id=episode_id,
        step="Cluster",
        step_index=1,
        total_steps=1,
        status="running",
        message="Clustering tracks against current facebank...",
    )

    if progress_callback:
        progress_callback({
            "step": "Cluster",
            "step_index": 1,
            "total_steps": 1,
            "status": "running",
        })

    step_start_time = time.time()
    try:
        result = _run_cluster(job_id, episode_id)

        # Update ETA stats
        step_elapsed = time.time() - step_start_time
        try:
            update_step_stats(
                episode_id=episode_id,
                operation="cluster",
                step_name="cluster",
                duration_seconds=step_elapsed,
                data_root=data_root,
            )
        except Exception as eta_exc:
            logger.warning(f"Failed to update ETA stats for cluster: {eta_exc}")

        # Emit success
        emit_progress(
            episode_id=episode_id,
            step="Cluster",
            step_index=1,
            total_steps=1,
            status="done",
            message="Clustering complete",
            extra={"result": json_safe(result)},
        )

        if progress_callback:
            progress_callback({
                "step": "Cluster",
                "step_index": 1,
                "total_steps": 1,
                "status": "done",
                "result": result,
            })

        logger.info(f"[{job_id}] Cluster complete: {result}")
        return {"status": "ok", "result": result}

    except Exception as exc:
        logger.error(f"[{job_id}] Cluster failed: {exc}", exc_info=True)

        # Emit error
        emit_progress(
            episode_id=episode_id,
            step="Cluster",
            step_index=1,
            total_steps=1,
            status="error",
            message=f"Clustering failed: {exc}",
        )

        if progress_callback:
            progress_callback({
                "step": "Cluster",
                "step_index": 1,
                "total_steps": 1,
                "status": "error",
                "error": str(exc),
            })

        return {"status": "error", "error": str(exc)}


def orchestrate_analytics_only(
    episode_id: str,
    job_id: Optional[str] = None,
    data_root: Path = Path("data"),
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run Analytics stage only (requires clusters).

    Generates timeline.csv and totals.csv from final labels.

    Args:
        episode_id: Episode identifier
        job_id: Optional job identifier
        data_root: Root data directory
        progress_callback: Optional callback(progress_dict) for real-time updates

    Returns:
        Dict with analytics result
    """
    if job_id is None:
        job_id = f"analytics_{episode_id}"

    # Check if clusters exist
    artifacts = check_artifacts(episode_id, data_root)
    if not artifacts["clusters"]:
        error_msg = "Clusters not found. Run Cluster first."
        logger.error(f"[{job_id}] {error_msg}")
        return {
            "episode_id": episode_id,
            "job_id": job_id,
            "status": "error",
            "error": error_msg,
        }

    logger.info(f"[{job_id}] Running analytics for {episode_id}")

    # Emit starting progress
    emit_progress(
        episode_id=episode_id,
        step="Analytics",
        step_index=1,
        total_steps=1,
        status="running",
        message="Generating timeline and totals...",
    )

    if progress_callback:
        progress_callback({
            "step": "Analytics",
            "step_index": 1,
            "total_steps": 1,
            "status": "running",
        })

    step_start_time = time.time()
    try:
        result = _run_analytics(job_id, episode_id, data_root)

        # Update ETA stats
        step_elapsed = time.time() - step_start_time
        try:
            update_step_stats(
                episode_id=episode_id,
                operation="analytics",
                step_name="analytics",
                duration_seconds=step_elapsed,
                data_root=data_root,
            )
        except Exception as eta_exc:
            logger.warning(f"Failed to update ETA stats for analytics: {eta_exc}")

        # Emit success
        emit_progress(
            episode_id=episode_id,
            step="Analytics",
            step_index=1,
            total_steps=1,
            status="done",
            message="Analytics complete",
            extra={"result": json_safe(result)},
        )

        if progress_callback:
            progress_callback({
                "step": "Analytics",
                "step_index": 1,
                "total_steps": 1,
                "status": "done",
                "result": result,
            })

        logger.info(f"[{job_id}] Analytics complete")
        return {"status": "ok", "result": result}

    except Exception as exc:
        logger.error(f"[{job_id}] Analytics failed: {exc}", exc_info=True)

        # Emit error
        emit_progress(
            episode_id=episode_id,
            step="Analytics",
            step_index=1,
            total_steps=1,
            status="error",
            message=f"Analytics failed: {exc}",
        )

        if progress_callback:
            progress_callback({
                "step": "Analytics",
                "step_index": 1,
                "total_steps": 1,
                "status": "error",
                "error": str(exc),
            })

        return {"status": "error", "error": str(exc)}


def _run_detect(job_id: str, episode_id: str) -> Dict[str, Any]:
    """Run detect/embed stage."""
    from jobs.tasks.detect_embed import detect_embed_task
    return detect_embed_task(job_id, episode_id)


def _run_track(job_id: str, episode_id: str) -> Dict[str, Any]:
    """Run tracking stage."""
    from jobs.tasks.track import track_task
    return track_task(job_id, episode_id)


def _run_cluster(job_id: str, episode_id: str) -> Dict[str, Any]:
    """Run clustering stage."""
    from jobs.tasks.cluster import cluster_task
    return cluster_task(job_id, episode_id)


def _run_stills(episode_id: str, data_root: Path, force: bool, resume: bool = True) -> Dict[str, Any]:
    """
    Run stills generation stage.

    Includes SER-FIQ face-aware crops AND thumbnail materialization (160×200).
    Always materializes thumbs to guarantee 100% coverage.
    """
    logger.info(f"[{episode_id}] Stage 1/2: SER-FIQ face-aware crops (320×400)")

    # Step 1: Generate SER-FIQ stills
    from jobs.tasks.generate_face_stills import generate_face_stills_task
    stills_result = generate_face_stills_task(
        episode_id,
        data_root=data_root,
        force=force,
        resume=resume,
    )

    logger.info(f"[{episode_id}] Stage 2/2: Materialize thumbnails (160×200)")

    # Step 2: Materialize thumbs from crops (always run to guarantee coverage)
    from jobs.tasks.materialize_thumbs import materialize_thumbs
    try:
        thumb_result = materialize_thumbs(episode_id, data_root)

        # Combine results
        return {
            "serfiq_stills": stills_result,
            "thumbnails": thumb_result,
            "generated": stills_result.get("generated", 0) + stills_result.get("skipped", 0),
            "total_tracks": stills_result.get("total_tracks", 0),
        }

    except Exception as exc:
        logger.error(f"Thumb materialization failed: {exc}")
        return {
            "serfiq_stills": stills_result,
            "thumbnails": {"error": str(exc)},
            "generated": stills_result.get("generated", 0) + stills_result.get("skipped", 0),
            "total_tracks": stills_result.get("total_tracks", 0),
        }


def _run_analytics(job_id: str, episode_id: str, data_root: Path) -> Dict[str, Any]:
    """
    Run analytics stage.

    Generates timeline.csv, totals.csv, and totals_by_identity.json.
    """
    logger.info(f"[{job_id}] Running analytics for {episode_id}")

    from jobs.tasks.analytics import analytics_task
    from app.lib.data import load_clusters

    # Load clusters to get cluster assignments
    clusters_data = load_clusters(episode_id, data_root)
    if not clusters_data:
        logger.warning(f"No clusters found for {episode_id}, skipping analytics")
        return {"status": "skipped", "reason": "no_clusters"}

    # Build cluster assignments map
    cluster_assignments = {}
    for cluster in clusters_data.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        name = cluster.get("name", "Unknown")
        if cluster_id is not None:
            cluster_assignments[cluster_id] = name

    # Run analytics task
    result = analytics_task(job_id, episode_id, cluster_assignments)

    # Extract totals for summary
    outputs_dir = data_root / "outputs" / episode_id
    totals_csv = outputs_dir / "totals.csv"

    totals_by_identity = {}
    if totals_csv.exists():
        try:
            import pandas as pd
            df = pd.read_csv(totals_csv)
            for _, row in df.iterrows():
                identity = row.get("identity", "Unknown")
                total_ms = row.get("total_ms", 0)
                totals_by_identity[identity] = int(total_ms)
        except Exception as exc:
            logger.warning(f"Failed to read totals.csv: {exc}")

    # Write totals_by_identity.json for easy access
    if totals_by_identity:
        totals_json_path = outputs_dir / "totals_by_identity.json"
        with open(totals_json_path, "w") as f:
            json.dump(totals_by_identity, f, indent=2)

    return {
        **result,
        "totals_by_identity": totals_by_identity,
    }


def _save_state(state_file: Path, state: Dict[str, Any]):
    """Save pipeline state to JSON with numpy type conversion."""
    with open(state_file, "w") as f:
        json.dump(json_safe(state), f, indent=2)
