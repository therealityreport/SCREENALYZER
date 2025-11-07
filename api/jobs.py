"""
Job management API for async processing with RQ.

Endpoints:
- POST /jobs/enqueue - Create new processing job
- GET /jobs/status/{job_id} - Get job status
- POST /jobs/cancel/{job_id} - Cancel job
- POST /jobs/resume/{job_id} - Resume from checkpoint
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import redis
from dotenv import load_dotenv
from rq import Queue
from rq.job import Job

from screentime.diagnostics.telemetry import telemetry, TelemetryEvent
from screentime.types import JobStatus, ProcessingJob

load_dotenv()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))

# Redis connection
redis_conn = redis.Redis.from_url(REDIS_URL)

# RQ queues by priority
harvest_queue = Queue("harvest.q", connection=redis_conn)
inference_queue = Queue("inference.q", connection=redis_conn)
tracking_queue = Queue("tracking.q", connection=redis_conn)
cluster_queue = Queue("cluster.q", connection=redis_conn)
assist_queue = Queue("assist.q", connection=redis_conn)
analytics_queue = Queue("analytics.q", connection=redis_conn)


class JobManager:
    """Manages processing jobs with RQ."""

    def __init__(self, redis_conn: redis.Redis):
        self.redis = redis_conn
        self.redis_conn = redis_conn  # Alias for compatibility
        self.harvest_dir = DATA_ROOT / "harvest"
        self.harvest_dir.mkdir(parents=True, exist_ok=True)

        # Expose queues
        self.harvest_queue = harvest_queue
        self.inference_queue = inference_queue
        self.tracking_queue = tracking_queue
        self.cluster_queue = cluster_queue
        self.assist_queue = assist_queue
        self.analytics_queue = analytics_queue

    def enqueue_processing_job(self, episode_id: str, video_path: Path) -> dict:
        """
        Enqueue a new processing job.

        Args:
            episode_id: Unique episode identifier
            video_path: Path to video file

        Returns:
            Dict with job_id and status

        Raises:
            ValueError: If video doesn't exist
        """
        if not video_path.exists():
            raise ValueError(f"Video not found: {video_path}")

        # Create job ID
        job_id = f"job_{episode_id}_{int(time.time())}"

        # Enqueue harvest task first
        from jobs.tasks.harvest import harvest_task

        rq_job = harvest_queue.enqueue(
            harvest_task,
            job_id=job_id,
            episode_id=episode_id,
            video_path=str(video_path),
            job_timeout="30m",
        )

        # Initialize job metadata in Redis
        job_data = {
            "job_id": job_id,
            "episode_id": episode_id,
            "video_path": str(video_path),
            "status": JobStatus.PENDING.value,
            "stage": "harvest",
            "progress_pct": 0.0,
            "eta_ms": None,
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "message": "Job queued",
            "rq_job_id": rq_job.id,
        }

        self._save_job_metadata(job_id, job_data)

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_STARTED,
            metadata={
                "job_id": job_id,
                "episode_id": episode_id,
                "queue": "harvest.q",
            },
        )

        return {
            "job_id": job_id,
            "rq_job_id": rq_job.id,
            "status": JobStatus.PENDING.value,
            "message": "Job enqueued for processing",
        }

    def get_job_status(self, job_id: str) -> dict:
        """
        Get job status and progress.

        Args:
            job_id: Job ID

        Returns:
            Dict with status, stage, progress, ETA

        Raises:
            ValueError: If job not found
        """
        job_data = self._get_job_metadata(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        # Get progress from Redis
        progress_key = f"job:{job_id}:progress"
        progress_data = self.redis.get(progress_key)
        if progress_data:
            progress_info = json.loads(progress_data)
            job_data.update(progress_info)

        # Get ETA
        eta_key = f"job:{job_id}:eta_ms"
        eta_ms = self.redis.get(eta_key)
        if eta_ms:
            job_data["eta_ms"] = int(eta_ms)

        return {
            "job_id": job_id,
            "status": job_data.get("status", JobStatus.PENDING.value),
            "stage": job_data.get("stage", "unknown"),
            "progress_pct": job_data.get("progress_pct", 0.0),
            "eta_ms": job_data.get("eta_ms"),
            "message": job_data.get("message", ""),
            "created_at": job_data.get("created_at"),
            "started_at": job_data.get("started_at"),
        }

    def cancel_job(self, job_id: str) -> dict:
        """
        Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            Dict with cancellation status

        Raises:
            ValueError: If job not found
        """
        job_data = self._get_job_metadata(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        # Get RQ job and cancel
        rq_job_id = job_data.get("rq_job_id")
        if rq_job_id:
            try:
                rq_job = Job.fetch(rq_job_id, connection=self.redis)
                rq_job.cancel()
            except Exception:
                pass  # Job may already be finished

        # Update status
        job_data["status"] = JobStatus.CANCELLED.value
        job_data["message"] = "Job cancelled by user"
        self._save_job_metadata(job_id, job_data)

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_CANCELLED,
            metadata={"job_id": job_id},
        )

        return {
            "job_id": job_id,
            "status": JobStatus.CANCELLED.value,
            "message": "Job cancelled successfully",
        }

    def resume_job(self, job_id: str) -> dict:
        """
        Resume job from last checkpoint.

        Args:
            job_id: Job ID

        Returns:
            Dict with resume status

        Raises:
            ValueError: If job not found or can't resume
        """
        job_data = self._get_job_metadata(job_id)
        if not job_data:
            raise ValueError(f"Job {job_id} not found")

        # Check if job can be resumed
        status = job_data.get("status")
        if status not in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            raise ValueError(f"Job {job_id} cannot be resumed (status: {status})")

        # Load checkpoint
        episode_id = job_data["episode_id"]
        checkpoint_path = self.harvest_dir / episode_id / "checkpoints" / "job.json"

        if not checkpoint_path.exists():
            raise ValueError(f"No checkpoint found for job {job_id}")

        with open(checkpoint_path) as f:
            checkpoint = json.load(f)

        # Resume from last completed stage
        last_stage = checkpoint.get("last_completed_stage", "harvest")
        video_path = job_data["video_path"]

        # Re-enqueue from checkpoint
        from jobs.tasks.harvest import harvest_task

        rq_job = harvest_queue.enqueue(
            harvest_task,
            job_id=job_id,
            episode_id=episode_id,
            video_path=video_path,
            resume_from=last_stage,
            job_timeout="30m",
        )

        # Update job metadata
        job_data["status"] = JobStatus.RUNNING.value
        job_data["message"] = f"Resumed from {last_stage}"
        job_data["rq_job_id"] = rq_job.id
        self._save_job_metadata(job_id, job_data)

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_STARTED,
            metadata={
                "job_id": job_id,
                "resumed": True,
                "from_stage": last_stage,
            },
        )

        return {
            "job_id": job_id,
            "status": JobStatus.RUNNING.value,
            "message": f"Job resumed from {last_stage}",
            "checkpoint": checkpoint,
        }

    def update_job_progress(
        self, job_id: str, stage: str, progress_pct: float, message: str = ""
    ) -> None:
        """
        Update job progress (called by workers).

        Args:
            job_id: Job ID
            stage: Current stage name
            progress_pct: Progress percentage (0-100)
            message: Optional status message
        """
        progress_data = {
            "stage": stage,
            "progress_pct": progress_pct,
            "message": message,
            "updated_at": datetime.utcnow().isoformat(),
        }

        # Store in Redis with short TTL (will be refreshed by heartbeat)
        progress_key = f"job:{job_id}:progress"
        self.redis.setex(progress_key, 300, json.dumps(progress_data))  # 5 min TTL

        # Update job metadata
        job_data = self._get_job_metadata(job_id)
        if job_data:
            job_data.update(progress_data)
            self._save_job_metadata(job_id, job_data)

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_STARTED,
            metadata={
                "job_id": job_id,
                "stage": stage,
                "progress_pct": progress_pct,
            },
        )

    def update_job_eta(self, job_id: str, eta_ms: int) -> None:
        """
        Update job ETA.

        Args:
            job_id: Job ID
            eta_ms: Estimated time remaining in milliseconds
        """
        eta_key = f"job:{job_id}:eta_ms"
        self.redis.setex(eta_key, 300, eta_ms)  # 5 min TTL

    def complete_job(self, job_id: str, message: str = "Job completed successfully") -> None:
        """
        Mark job as completed.

        Args:
            job_id: Job ID
            message: Completion message
        """
        job_data = self._get_job_metadata(job_id)
        if job_data:
            job_data["status"] = JobStatus.COMPLETED.value
            job_data["message"] = message
            job_data["completed_at"] = datetime.utcnow().isoformat()
            self._save_job_metadata(job_id, job_data)

        # Telemetry
        telemetry.log(
            TelemetryEvent.JOB_COMPLETED,
            metadata={"job_id": job_id},
        )

    def create_job(self, task: str, **kwargs) -> str:
        """
        Create and enqueue a new job.

        Args:
            task: Task type (e.g., 'recluster', 'contamination_audit')
            **kwargs: Task-specific arguments

        Returns:
            Job ID string
        """
        # Generate job ID
        timestamp = int(time.time())
        episode_id = kwargs.get('episode_id', 'unknown')
        job_id = f"{task}_{episode_id}_{timestamp}"

        # Select queue based on task type
        queue_map = {
            'recluster': self.cluster_queue,
            'contamination_audit': self.assist_queue,
            'merge': self.assist_queue,
            'analytics': self.analytics_queue,
        }

        queue = queue_map.get(task, self.cluster_queue)

        # Import task function
        task_map = {
            'recluster': 'jobs.tasks.recluster.recluster_task',
        }

        task_path = task_map.get(task)
        if not task_path:
            raise ValueError(f"Unknown task type: {task}")

        # Dynamically import task function
        module_path, func_name = task_path.rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        task_func = getattr(module, func_name)

        # Enqueue job
        rq_job = queue.enqueue(
            task_func,
            job_id=job_id,
            **kwargs,
            job_timeout="30m"
        )

        # Save metadata
        job_data = {
            "job_id": job_id,
            "rq_job_id": rq_job.id,
            "task": task,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat() + 'Z',
            **kwargs
        }
        self._save_job_metadata(job_id, job_data)

        return job_id

    def get_active_jobs(self) -> list[dict]:
        """
        Get all active jobs (queued or started).

        Returns:
            List of job metadata dicts
        """
        active_jobs = []

        # Scan all job keys in Redis
        for key in self.redis.scan_iter("job:*:metadata"):
            try:
                data = self.redis.get(key)
                if data:
                    job_data = json.loads(data)
                    status = job_data.get('status', 'unknown')
                    if status in ['queued', 'started', 'processing']:
                        active_jobs.append(job_data)
            except Exception:
                continue

        return active_jobs

    def cancel_all(self, prefix: str) -> dict:
        """
        Cancel all jobs matching a prefix.

        Args:
            prefix: Job ID prefix to match (e.g., 'recluster_rhobh-test-10-28')

        Returns:
            Dict with count of cancelled jobs and list of job_ids
        """
        cancelled = []

        # Scan all job keys in Redis
        for key in self.redis.scan_iter("job:*:metadata"):
            try:
                data = self.redis.get(key)
                if data:
                    job_data = json.loads(data)
                    job_id = job_data.get('job_id', '')

                    # Check if job_id matches prefix and is active
                    if job_id.startswith(prefix):
                        status = job_data.get('status', 'unknown')
                        if status in ['queued', 'started', 'processing']:
                            # Cancel the job
                            try:
                                self.cancel_job(job_id)
                                cancelled.append(job_id)
                            except Exception as e:
                                # Log but continue
                                print(f"Failed to cancel {job_id}: {e}")
            except Exception:
                continue

        return {
            "cancelled_count": len(cancelled),
            "cancelled_jobs": cancelled,
            "message": f"Cancelled {len(cancelled)} jobs matching prefix '{prefix}'"
        }

    def clear_active_recluster(self, episode_id: str) -> None:
        """
        Clear active recluster job state for an episode.

        Args:
            episode_id: Episode ID
        """
        # Cancel all recluster jobs for this episode
        prefix = f"recluster_{episode_id}"
        self.cancel_all(prefix)

        # Write cancellation marker
        harvest_dir = DATA_ROOT / "harvest" / episode_id / "diagnostics"
        harvest_dir.mkdir(parents=True, exist_ok=True)

        marker_path = harvest_dir / "recluster_last.json"
        marker_data = {
            "status": "canceled",
            "job_id": prefix,
            "ended_at": datetime.utcnow().isoformat() + 'Z'
        }

        with open(marker_path, "w") as f:
            json.dump(marker_data, f, indent=2)

    def _save_job_metadata(self, job_id: str, job_data: dict) -> None:
        """Save job metadata to Redis."""
        key = f"job:{job_id}:metadata"
        self.redis.setex(key, 86400, json.dumps(job_data))  # 24h TTL

    def _get_job_metadata(self, job_id: str) -> Optional[dict]:
        """Get job metadata from Redis."""
        key = f"job:{job_id}:metadata"
        data = self.redis.get(key)
        if data:
            return json.loads(data)
        return None

    def write_job_envelope(self, job_id: str, envelope: dict) -> None:
        """
        Write job envelope to disk (idempotent, atomic).

        Creates jobs/{job_id}/meta.json with job metadata.
        Used for self-healing workers when Redis metadata expires.

        Uses atomic write with flush+fsync to prevent corruption.

        Args:
            job_id: Job ID (e.g., 'prepare_RHOBH_S05_E03_11062025')
            envelope: Job envelope dict with job_id, episode_key, mode, stages
        """
        jobs_dir = DATA_ROOT / "jobs" / job_id
        jobs_dir.mkdir(parents=True, exist_ok=True)

        meta_file = jobs_dir / "meta.json"
        tmp_file = jobs_dir / "meta.json.tmp"

        # Atomic write with flush+fsync
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(envelope, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_file.replace(meta_file)

    def load_job_envelope(self, job_id: str) -> Optional[dict]:
        """
        Load job envelope from disk with safe JSON loading.

        Args:
            job_id: Job ID

        Returns:
            Job envelope dict, or None if not found
        """
        meta_file = DATA_ROOT / "jobs" / job_id / "meta.json"
        if not meta_file.exists():
            return None

        # Use safe_load_json to handle corruption
        from screentime.diagnostics.utils import safe_load_json
        result = safe_load_json(meta_file)
        return result if result else None

    def update_stage_status(self, job_id: str, stage_key: str, status: str, result: Optional[dict] = None, error: Optional[str] = None) -> None:
        """
        Update stage status in job envelope.

        Args:
            job_id: Job ID
            stage_key: Stage key (e.g., 'detect', 'track')
            status: Stage status ('pending', 'running', 'ok', 'skipped', 'error')
            result: Optional result dict for successful stages
            error: Optional error message for failed stages
        """
        envelope = self.load_job_envelope(job_id)
        if not envelope:
            return

        if "stages" not in envelope:
            envelope["stages"] = {}

        envelope["stages"][stage_key] = {"status": status}
        if result:
            envelope["stages"][stage_key]["result"] = result
        if error:
            envelope["stages"][stage_key]["error"] = error

        self.write_job_envelope(job_id, envelope)



    # ========================================================================
    # Episode Registry Management
    # ========================================================================

    def normalize_episode_key(self, episode_id: str) -> str:
        """
        Normalize episode ID to canonical key format.

        Converts RHOBH_S05_E03_11062025 -> rhobh_s05_e03
        Converts ANOTHER_SHOW_S02_E05_99999999 -> another_show_s02_e05
        Strips timestamp and lowercases.

        Args:
            episode_id: Episode ID in any format

        Returns:
            Canonical episode key (lowercase, underscores, no timestamp)
        """
        import re

        # Find season (S##) and episode (E##) patterns
        season_match = re.search(r'(S\d+)', episode_id, re.IGNORECASE)
        episode_match = re.search(r'(E\d+)', episode_id, re.IGNORECASE)

        if season_match and episode_match:
            # Extract show name (everything before season)
            season_pos = season_match.start()
            show_part = episode_id[:season_pos].rstrip("_")

            # Build canonical key: show_season_episode
            canonical = f"{show_part}_{season_match.group(1)}_{episode_match.group(1)}".lower()
            return canonical

        # Fallback: just lowercase
        return episode_id.lower()

    def write_episode_registry(self, episode_key: str, registry_data: dict) -> None:
        """
        Write episode registry to disk (idempotent, atomic).

        Creates episodes/{episode_key}/state.json with episode state.

        Args:
            episode_key: Canonical episode key (e.g., 'rhobh_s05_e03')
            registry_data: Registry dict with states, paths, timestamps
        """
        episodes_dir = DATA_ROOT / "episodes" / episode_key
        episodes_dir.mkdir(parents=True, exist_ok=True)

        state_file = episodes_dir / "state.json"
        tmp_file = episodes_dir / "state.json.tmp"

        # Ensure timestamps
        if "timestamps" not in registry_data:
            registry_data["timestamps"] = {}
        registry_data["timestamps"]["last_modified"] = datetime.utcnow().isoformat() + 'Z'

        # Atomic write
        with open(tmp_file, "w") as f:
            json.dump(registry_data, f, indent=2)
        tmp_file.replace(state_file)

    def load_episode_registry(self, episode_key: str) -> Optional[dict]:
        """
        Load episode registry from disk.

        Args:
            episode_key: Canonical episode key

        Returns:
            Registry dict, or None if not found
        """
        state_file = DATA_ROOT / "episodes" / episode_key / "state.json"
        if not state_file.exists():
            return None

        with open(state_file) as f:
            return json.load(f)

    def update_registry_state(self, episode_key: str, state_name: str, value: bool) -> None:
        """
        Update a specific state in the episode registry.

        Args:
            episode_key: Canonical episode key
            state_name: State name (e.g., 'detected', 'tracked', 'clustered')
            value: State value (True/False)
        """
        registry = self.load_episode_registry(episode_key)
        if not registry:
            return

        if "states" not in registry:
            registry["states"] = {}

        registry["states"][state_name] = value
        self.write_episode_registry(episode_key, registry)

    def ensure_episode_registry(self, episode_id: str, video_path: str) -> str:
        """
        Ensure episode registry exists, create if missing.

        Args:
            episode_id: Episode ID (e.g., 'RHOBH_S05_E03_11062025')
            video_path: Path to video file

        Returns:
            Canonical episode_key
        """
        episode_key = self.normalize_episode_key(episode_id)
        registry = self.load_episode_registry(episode_key)

        if not registry:
            # Parse episode components
            parts = episode_id.split("_")
            show = parts[0].upper() if len(parts) > 0 else "UNKNOWN"
            season = parts[1].upper() if len(parts) > 1 else "S00"
            episode = parts[2].upper() if len(parts) > 2 else "E00"

            # Create new registry
            registry = {
                "episode_key": episode_key,
                "episode_id": episode_id,
                "show": show,
                "season": season,
                "episode": episode,
                "video_path": video_path,
                "paths": {
                    "root": f"data/harvest/{episode_id}",
                    "frames_dir": f"data/harvest/{episode_id}/frames",
                    "manifests_dir": f"data/harvest/{episode_id}/diagnostics",
                },
                "contracts": {
                    "frame_ext": "jpg",
                    "frame_pad": 6,
                },
                "states": {
                    "validated": False,
                    "extracted_frames": False,
                    "detected": False,
                    "tracked": False,
                    "clustered": False,
                    "assigned": False,
                },
                "timestamps": {
                    "created": datetime.utcnow().isoformat() + 'Z',
                },
            }
            self.write_episode_registry(episode_key, registry)

        return episode_key


# Global job manager instance
job_manager = JobManager(redis_conn)
