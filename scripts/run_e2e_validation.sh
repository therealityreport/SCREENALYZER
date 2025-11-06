#!/bin/bash
# End-to-end validation runner for Screenalyzer
# Starts workers and runs full pipeline validation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "============================================================"
echo "Screenalyzer E2E Validation Runner"
echo "============================================================"
echo ""

# Activate venv
source .venv/bin/activate

# Check Redis
echo "Checking Redis..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi
echo "✅ Redis is running"
echo ""

# Start workers in background
echo "Starting RQ workers..."

python jobs/worker.py harvest.q > logs/harvest.log 2>&1 &
HARVEST_PID=$!
echo "  ✅ Started harvest.q worker (PID: $HARVEST_PID)"

python jobs/worker.py inference.q > logs/inference.log 2>&1 &
INFERENCE_PID=$!
echo "  ✅ Started inference.q worker (PID: $INFERENCE_PID)"

python jobs/worker.py tracking.q > logs/tracking.log 2>&1 &
TRACKING_PID=$!
echo "  ✅ Started tracking.q worker (PID: $TRACKING_PID)"

python jobs/worker.py cluster.q > logs/cluster.log 2>&1 &
CLUSTER_PID=$!
echo "  ✅ Started cluster.q worker (PID: $CLUSTER_PID)"

echo ""
echo "Workers started. Waiting 2 seconds for initialization..."
sleep 2

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping workers..."
    kill $HARVEST_PID $INFERENCE_PID $TRACKING_PID $CLUSTER_PID 2>/dev/null || true
    echo "✅ Workers stopped"
}

trap cleanup EXIT

# Run E2E test
echo ""
echo "Running E2E test..."
echo "============================================================"
python scripts/test_e2e.py

# Test result
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ E2E Validation: PASSED"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "  1. Start UI: streamlit run app/labeler.py"
    echo "  2. Test Review tab with real thumbnails"
    echo "  3. Test merge/split operations"
    echo "  4. Generate analytics and verify exports"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "❌ E2E Validation: FAILED"
    echo "============================================================"
    echo ""
    echo "Check logs in logs/ directory for details"
    echo ""
    exit 1
fi
