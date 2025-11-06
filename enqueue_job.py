#!/usr/bin/env python3
"""Simple script to enqueue jobs to RQ."""
import sys
import uuid
from redis import Redis
from rq import Queue

def main():
    if len(sys.argv) < 3:
        print("Usage: python enqueue_job.py <task_name> <episode_id>")
        sys.exit(1)

    task_name = sys.argv[1]
    episode_id = sys.argv[2]

    # Import the task function
    if task_name == "detect_embed":
        from jobs.tasks.detect_embed import detect_embed_task
        task_func = detect_embed_task
    elif task_name == "recluster":
        from jobs.tasks.recluster import recluster_task
        task_func = recluster_task
    else:
        print(f"Unknown task: {task_name}")
        sys.exit(1)

    # Connect to Redis and enqueue
    redis_conn = Redis(host='localhost', port=6379, db=0)
    queue = Queue('default', connection=redis_conn)

    # Use "manual" workflow (uses episode registry instead of job manager)
    job_id_arg = "manual"

    # Enqueue task with job_id="manual" and episode_id as args
    job = queue.enqueue(
        task_func,
        job_id_arg,
        episode_id,
        job_timeout='2h'
    )

    print(f"✅ Enqueued {task_name} for {episode_id}")
    print(f"Job ID: {job.id}")
    print(f"Queue: {queue.name}")

if __name__ == "__main__":
    main()
