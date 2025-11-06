#!/bin/bash
# Start RQ workers with macOS fork safety disabled

export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
cd "$(dirname "$0")/.."
source .venv/bin/activate

# Start workers in background
python jobs/worker.py harvest.q > logs/worker_harvest.log 2>&1 &
python jobs/worker.py inference.q > logs/worker_inference.log 2>&1 &
python jobs/worker.py tracking.q > logs/worker_tracking.log 2>&1 &
python jobs/worker.py cluster.q > logs/worker_cluster.log 2>&1 &

echo "Workers started with PIDs:"
pgrep -f "python jobs/worker.py"
