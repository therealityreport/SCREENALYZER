# Proposed Documentation Updates — 2025-10-29

- Draft `docs/LOCAL_DENSIFY.md` detailing the gap-window heuristics (≤ 3.2 s + 300 ms pad), 30 fps recall decode, detection thresholds, verification gates, proposed function signatures, outputs, and acceptance criteria for YOLANDA/RINNA/BRANDI/EILEEN follow-up.
- Extend `docs/SOLUTION_ARCHITECTURE.md` with a “Local Densify (Gap-Focused High-Recall)” subsection clarifying that the global pipeline runs at 10 fps with 30 fps decode only inside selected windows.
- Refresh `docs/ACCEPTANCE_MATRIX.csv` to capture the current baseline status: KIM/KYLE/LVP passing, RINNA/BRANDI pending densify, YOLANDA/EILEEN blocked; add new FR rows for YOLANDA densify and EILEEN auto cleanup.
- Optional: append a densify appendix to `docs/ENV_SETUP.md` (ffmpeg / pyav build flags, onnxruntime notes for Apple Silicon).
- Update QA collateral so `docs/REVIEWS/2025-10-29/*` and `validation_summary.txt` reflect the present artifacts (merge suggestions parquet now emitted, bootstrap stage still pending integration).
