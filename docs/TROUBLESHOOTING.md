# Troubleshooting Guide

## Common Issues & Resolutions

### 1. RQ Worker Crashes on macOS (Fork Safety)

**Symptom:**
```
objc[PID]: +[NSNumber initialize] may have been in progress in another thread when fork() was called.
Work-horse terminated unexpectedly; waitpid returned 6 (signal 6)
```

**Root Cause:**
macOS prohibits Objective-C library calls between fork() and exec(). RQ workers fork without exec, and PyArrow/OpenCV trigger CF/ObjC initializers.

**Resolution:**
✅ **Automatic** - `jobs/worker.py` detects macOS (`sys.platform == "darwin"`) and uses `SimpleWorker` (no fork).

**Manual Override (if needed):**
```bash
# Force SimpleWorker on any platform
export DEV_NO_FORK=1
python jobs/worker.py harvest.q
```

**Verify:**
```bash
tail -f logs/worker_harvest.log
# Should see: "Using SimpleWorker (no fork) for harvest.q"
```

---

### 2. OpenCV (cv2) Import Fails

**Symptom:**
```
ModuleNotFoundError: No module named 'cv2'
```

**Cause:** Wrong Python version (3.14+) or mixed wheel installations

**Resolution:**
```bash
# 1. Check Python version
python --version  # Must be 3.11.x or 3.12.x

# 2. If wrong version, recreate venv
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Reinstall opencv
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.10.0.84

# 4. Verify
python -c "import cv2; print(f'OpenCV {cv2.__version__} OK')"
```

---

### 3. LAP Solver Failures ("too many values to unpack")

**Symptom:**
```
LAP solver failed: too many values to unpack (expected 2), falling back to greedy matching
```

**Root Cause:**
`lap==0.5.12` returns 3 values `(row_ind, col_ind, cost)`, but ByteTrack expects 2 values (old API).

**Current Behavior:**
✅ **Graceful fallback** - Automatically uses greedy matching (acceptable for Phase 1).

**Future Fix:**
Update `screentime/tracking/bytetrack_wrap.py` to handle new API:
```python
# BEFORE (expects 2 values)
row_ind, col_ind = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)

# AFTER (handle 3 values)
result = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=threshold)
if len(result) == 3:
    row_ind, col_ind, _ = result  # Ignore cost
else:
    row_ind, col_ind = result      # Old API
```

**Workaround (if lap 0.5.9 available):**
```bash
pip install lap==0.5.9  # Only if wheel exists for your Python version
```

---

### 4. ONNX Runtime Provider Errors

**Symptom:**
```
Failed to find CoreMLExecutionProvider
```

**Cause:** ONNX Runtime provider not available on platform

**Resolution:**
✅ **Automatic** - Code falls back to CPUExecutionProvider

**Verify:**
```bash
python -c "from screentime.detectors.face_retina import RetinaFaceDetector; print('Provider fallback OK')"
```

**Performance Note:**
- macOS with CoreML: ~60-70s for 207 frames
- CPU only: ~90-120s for 207 frames

---

### 5. Merge Suggestions Not Generated

**Symptom:**
```
❌ Merge suggestions: data/harvest/EPISODE/assist/merge_suggestions.parquet
```

**Root Cause:**
No cluster pairs met similarity threshold (e.g., all faces very distinct).

**Verification:**
```bash
cat data/harvest/EPISODE/diagnostics/reports/cluster_stats.json
# Check: "suggestions_enqueued": 0
```

**Expected Behavior:**
✅ **Correct** - `merge_suggestions.parquet` only created when suggestions exist (≥1).

**If unexpected:**
- Check clustering config (`configs/pipeline.yaml`):
  ```yaml
  clustering:
    similarity_threshold: 0.75  # Lower = more suggestions
  ```
- Verify clusters have sufficient similarity:
  ```python
  import json
  with open('data/harvest/EPISODE/clusters.json') as f:
      data = json.load(f)
      print(f"Total clusters: {data['total_clusters']}")
  ```

---

### 6. Redis Connection Errors

**Symptom:**
```
redis.exceptions.ConnectionError: Error connecting to Redis
```

**Resolution:**
```bash
# Check Redis is running
redis-cli ping  # Should return "PONG"

# macOS
brew services start redis

# Linux
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:latest
```

**Verify Connection:**
```bash
python -c "import redis; r=redis.Redis(host='localhost', port=6379); r.ping(); print('✓ Redis OK')"
```

---

### 7. Numpy int64 JSON Serialization Error

**Symptom:**
```
TypeError: Object of type int64 is not JSON serializable
```

**Resolution:**
✅ **Fixed** - Added `default=int` converter to json.dump in `jobs/tasks/track.py`

**Verification:**
```bash
python -c "import json, numpy as np; json.dumps({'x': np.int64(5)}, default=int)"
```

---

### 8. TelemetryEvent Import Errors

**Symptom:**
```
AttributeError: 'TelemetryLogger' object has no attribute 'TelemetryEvent'
```

**Resolution:**
✅ **Fixed** - All task files now import `TelemetryEvent` directly:
```python
from screentime.diagnostics.telemetry import telemetry, TelemetryEvent
```

**Verify:**
```bash
python -c "from jobs.tasks.harvest import harvest_task; print('✓ Imports OK')"
```

---

## Validation Failures

### E2E Validation Script Fails

**Diagnosis:**
```bash
# Run with verbose output
source .venv/bin/activate
python scripts/validate_pipeline_direct.py 2>&1 | tee validation.log

# Check for specific errors
grep "❌" validation.log
grep "Traceback" validation.log
```

**Common Issues:**
1. **Redis not running** → See section 6
2. **OpenCV not installed** → See section 2
3. **Video file missing** → Check path: `/Volumes/HardDrive/SCREENALYZER/data/videos/RHOBH-TEST-10-28.mp4`

---

## Performance Issues

### Detection/Embeddings Slow (>120s for 3-min video)

**Check:**
1. ONNX Runtime provider:
   ```python
   import onnxruntime
   print(onnxruntime.get_available_providers())
   # Look for 'CoreMLExecutionProvider' (macOS) or 'CUDAExecutionProvider' (GPU)
   ```

2. CPU usage:
   ```bash
   top -pid $(pgrep -f "python.*detect_embed")
   ```

3. Video resolution (higher = slower):
   ```bash
   ffprobe data/videos/VIDEO.mp4 2>&1 | grep Stream.*Video
   ```

**Optimization:**
- Reduce frame sampling rate (`configs/pipeline.yaml`):
  ```yaml
  video:
    fps_rate: 1  # Sample 1 frame/sec (default: 2)
  ```

---

## Getting Help

1. **Check logs:**
   ```bash
   # Worker logs
   tail -f logs/worker_*.log

   # Telemetry
   tail -f data/diagnostics/telemetry.jsonl

   # Task stats
   cat data/harvest/EPISODE/diagnostics/reports/*.json
   ```

2. **Environment info:**
   ```bash
   python --version
   pip list | grep -E "opencv|onnx|redis|rq|insightface|lap"
   uname -a
   ```

3. **Minimal repro:**
   ```bash
   # Test each stage independently
   source .venv/bin/activate
   python -c "from jobs.tasks.harvest import harvest_task; print('✓ Import OK')"
   python scripts/validate_pipeline_direct.py
   ```

4. **Open issue:** Include logs + environment info from steps 1-2 above.
