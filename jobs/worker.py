"""
RQ worker bootstrap with shared logging and telemetry.

Usage:
    python jobs/worker.py [queue_name]

Queues:
    - harvest.q (default)
    - inference.q
    - tracking.q
    - cluster.q
    - assist.q
    - analytics.q
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Fix macOS fork() crash with Objective-C runtime
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import redis
from dotenv import load_dotenv
from rq import Worker, SimpleWorker
from rq.logutils import setup_loghandlers

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/diagnostics/worker.log"),
    ],
)

logger = logging.getLogger("screenalyzer.worker")


def main():
    """Start RQ worker."""
    # Get queue name from args
    queue_name = sys.argv[1] if len(sys.argv) > 1 else "harvest.q"

    logger.info(f"Starting RQ worker for queue: {queue_name}")

    # Connect to Redis
    redis_conn = redis.Redis.from_url(REDIS_URL)

    # Create worker - SimpleWorker on macOS (no fork), Worker on Linux
    # Set DEV_NO_FORK=1 to use SimpleWorker (single-process, no fork)
    use_simple_worker = os.getenv("DEV_NO_FORK", "0") == "1" or sys.platform == "darwin"

    if use_simple_worker:
        logger.info(f"Using SimpleWorker (no fork) for {queue_name}")
        worker = SimpleWorker(
            [queue_name],
            connection=redis_conn,
            name=f"worker-{queue_name}-{os.getpid()}",
        )
    else:
        logger.info(f"Using Worker (forking) for {queue_name}")
        worker = Worker(
            [queue_name],
            connection=redis_conn,
            name=f"worker-{queue_name}-{os.getpid()}",
        )

    # Setup RQ log handlers
    setup_loghandlers("INFO")

    # Start worker
    logger.info(f"Worker listening on {queue_name}")
    worker.work(with_scheduler=False, logging_level="INFO")


if __name__ == "__main__":
    main()
