"""Workspace UI constants and stage labels.

Centralized constants for Workspace UI and pipeline stage labels.
All UI components should import from this module for consistency.
"""

# Pipeline stage labels for UI display
STAGE_LABELS = {
    # Individual stages (numbered)
    "detect_embed": "1. RetinaFace + ArcFace (Detect & Embed)",
    "detect": "1. RetinaFace + ArcFace (Detect & Embed)",  # Alias for detect_embed
    "track": "2. ByteTrack (Track Faces)",
    "cluster": "3. Agglomerative Clustering (Group Tracks)",
    "stills": "4. Generate Stills",
    "analytics": "5. Screenalyzer Analytics (Re-Analyze)",

    # Composite workflows
    "full_pipeline": "(1–4) Run Full Pipeline",
    "full": "(1–4) Run Full Pipeline",  # Alias
    "cluster_button": "4. Clustering (Group Tracks)",
    "analytics_button": "5. Screenalyzer Analytics (Re-Analyze)",
}

# Stage help text (tooltips)
STAGE_HELP = {
    "detect_embed": "Runs RetinaFace for detection and ArcFace for embeddings. Must be completed before tracking or clustering.",
    "detect": "Runs RetinaFace for detection and ArcFace for embeddings. Must be completed before tracking or clustering.",
    "track": "Uses ByteTrack to track faces across frames. Creates continuous tracks from detections.",
    "cluster": "Groups tracks into clusters by identity using Agglomerative Clustering on ArcFace embeddings.",
    "stills": "Generates representative stills per cluster for review and facebank updates.",
    "analytics": "Computes per-person screen time from labeled clusters.",
    "full": "Runs the entire pipeline automatically: Detect → Track → Cluster → Stills.",
}

# Short stage names (for internal use)
STAGE_KEYS = ["detect_embed", "track", "cluster", "stills"]

# Full pipeline stages (runs automatically in sequence)
FULL_PIPELINE_STAGES = STAGE_KEYS

# Registry state keys (explicit stage flags - replaces legacy "prepared")
REGISTRY_STATE_KEYS = {
    "validated": "validated",
    "extracted_frames": "extracted_frames",
    "detected": "detected",                      # NEW: replaces "prepared"
    "tracked": "tracked",
    "stills_generated": "stills_generated",
    "clustered": "clustered",
    "assigned": "assigned",
    "analytics_computed": "analytics_computed",
}
