"""Application package init."""

import os

# Limit BLAS threading to avoid runaway CPU usage during Streamlit renders.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

