# Environment Setup Guide

## Supported Python Versions

- **Recommended**: Python 3.11.x (tested with 3.11.14)
- **Supported**: Python 3.11-3.12
- **Not supported**: Python 3.14+ (onnxruntime incompatibility)
- **Not supported**: Python 3.9 or earlier

## Platform-Specific Notes

### macOS (Apple Silicon/Intel)

**Fork-Safety Issues with RQ Workers:**

macOS has strict fork-safety checks for Objective-C libraries. When RQ workers fork, libraries like PyArrow and OpenCV may crash with:
```
objc[PID]: +[NSNumber initialize] may have been in progress in another thread when fork() was called.
SIGABRT (signal 6)
```

**Solution:**
The worker.py automatically detects macOS and uses `SimpleWorker` (single-process, no fork) instead of the standard `Worker`. On Linux/production, it uses standard forking workers.

**Environment Variables:**
```bash
# For macOS development
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES  # Safety valve (auto-set by worker.py)
export DEV_NO_FORK=1                             # Force SimpleWorker on any platform
```

### Linux

RQ workers use standard forking model. No special configuration needed.

## Dependencies

### Core Requirements

```
# requirements.txt (key dependencies)
python==3.11.*
opencv-python-headless==4.10.0.84  # Headless for server/macOS compatibility
onnxruntime==1.18.0                 # Python 3.11-3.12 only
insightface==0.7.3
redis==5.0.4
rq==1.16.2
streamlit==1.38.0
pandas==2.2.2
pyarrow==16.1.0
scikit-learn==1.5.1
scipy==1.13.1
lap==0.5.12                         # 0.5.10 incompatible with Python 3.11
plotly==5.24.1
openpyxl==3.1.5
```

### Known Dependency Issues

#### 1. LAP Solver Version (lap 0.5.12 vs 0.5.10)

**Issue:** lap 0.5.10 is not available for Python 3.11+
**Solution:** Using lap 0.5.12
**Impact:** API changed - returns 3 values instead of 2
**Workaround:** ByteTrack falls back to greedy matching (acceptable for Phase 1)
**Future Fix:** Update ByteTrack to handle lap 0.5.12 API properly

#### 2. OpenCV Import on Python 3.14

**Issue:** opencv-python fails to import on Python 3.14
**Solution:** Use Python 3.11.x

#### 3. ONNX Runtime Platform Support

**Issue:** onnxruntime==1.18.0 has limited Python 3.14 wheels
**Solution:** Use Python 3.11.x

## Virtual Environment Setup

### macOS

```bash
# Ensure Python 3.11 is installed
brew install python@3.11

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify critical imports
python -c "import cv2, onnxruntime, insightface; print('✓ All imports successful')"
```

### Linux/Docker

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Redis Setup

```bash
# macOS
brew install redis
brew services start redis

# Linux (Ubuntu/Debian)
sudo apt-get install redis-server
sudo systemctl start redis

# Verify
redis-cli ping  # Should return "PONG"
```

## Common Setup Issues

### Issue: cv2 module not found despite opencv-python installed

**Cause:** Wrong Python version in venv, or mixed wheel sources
**Solution:**
```bash
# Check Python version
python --version  # Should be 3.11.x

# Reinstall opencv with correct Python
pip uninstall opencv-python opencv-python-headless
pip install opencv-python-headless==4.10.0.84
python -c "import cv2; print(cv2.__version__)"
```

### Issue: RQ workers crash on macOS with fork errors

**Cause:** macOS fork-safety with Objective-C libraries
**Solution:** Automatic - worker.py detects macOS and uses SimpleWorker
**Verify:**
```bash
# Check worker logs
tail -f logs/worker_harvest.log
# Should see: "Using SimpleWorker (no fork) for harvest.q"
```

### Issue: lap package fails to install

**Cause:** No wheel for your Python version
**Solution:** Use Python 3.11 and lap==0.5.12

## Development Environment Verification

Run the environment check script:

```bash
source .venv/bin/activate
python scripts/validate_pipeline_direct.py
```

## Appendix: Local Densify Toolkit (Optional)

Local densify runs 30 fps decode + high-recall detection inside short windows. If you plan to run it locally:

- **ffmpeg / PyAV**: Install a build with `--enable-libvpx` and `--enable-nonfree` to handle 30 fps H.264 re-encode without quality loss. On macOS:
  ```bash
  brew install ffmpeg --with-fdk-aac --with-libvpx
  pip install av==12.3.0
  ```
- **onnxruntime**: Keep both `onnxruntime` (CPU) and `onnxruntime-silicon` (if you prefer the Apple Neural Engine path). Switch via `ORT_EXECUTION_PROVIDER=COREML_EP` when the CoreML delegate is stable; otherwise default to CPU for reproducibility.
- **Apple Silicon flags**: Set `export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` (already part of the worker bootstrap) and `export PYTORCH_ENABLE_MPS_FALLBACK=1` if you experiment with PyTorch-based detectors inside densify.
- **Performance checks**: Once the profiling helper lands, run `python scripts/profile_local_densify.py --episode RHOBH-TEST-10-28 --max-windows 5` to keep decode throughput above 120 fps and RAM under ~6 GB; until then, spot-check with `ffprobe` and Activity Monitor.

Expected output:
- ✅ All stages complete (Harvest → Detection → Tracking → Clustering)
- ✅ 8/9 artifacts created (merge_suggestions.parquet only created when suggestions exist)
- Total time: ~70-120s for 3-minute test video

## Appendix: Local Densify Requirements

- **Decoder tooling**: ffmpeg (build with `--enable-libvpx --enable-nonfree`) or PyAV ≥ 12.3.0 to support 30 fps segment extraction without drift.
- **ONNX Runtime**: install `onnxruntime==1.18.0`; optional `onnxruntime-silicon` for Apple Neural Engine. Switch via `ORT_EXECUTION_PROVIDER` when experimenting.
- **Apple Silicon**: set `OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES` and prefer CPU execution for repeatability; monitor memory when tiling (2×2) is enabled.
- **High-recall invocation flags**: export `LOCAL_DENSIFY_MIN_CONF=0.56`, `LOCAL_DENSIFY_MIN_FACE=45`, `LOCAL_DENSIFY_SCALES="1.0,1.3,1.6,2.0"` as overrides when testing scripts; ensure `LOCAL_DENSIFY_PAD_MS=300` and `LOCAL_DENSIFY_MAX_GAP_MS=3200` match the preset.
- **Telemetry checks**: confirm `recall_stats.json` is emitted and that `densify_windows_scanned`, `densify_tracks_built`, `recall_seconds_recovered` counters increment during runs.

## Production Deployment Notes

### Linux/Docker (Recommended)

- Use standard RQ Worker (forking) for better performance
- No fork-safety issues
- Set `DEV_NO_FORK=0` or omit (default)

### macOS (Dev Only)

- SimpleWorker automatically used
- Single-process per queue (no parallelism within worker)
- Acceptable for development, not recommended for production scale

## Environment Variables Reference

```bash
# Redis
export REDIS_URL="redis://localhost:6379/0"

# Worker behavior
export DEV_NO_FORK=1                             # Force SimpleWorker (optional, auto on macOS)
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES   # macOS fork safety (auto-set by worker.py)

# Pipeline config
export DATA_ROOT="data"                          # Default data directory
```

## Troubleshooting

See [docs/TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for detailed error resolution.
