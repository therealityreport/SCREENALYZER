"""Shared helpers for workspace views."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional, Iterable, List, Sequence, Tuple

import streamlit as st

from screentime.viz.thumbnails import ThumbnailGenerator

PLACEHOLDER_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAJYAAADICAIAAACF548yAAAB5UlEQVR4nO3RMQ0AIQDAQHhDbKz4d/UiGEiTOwVNOtc+g7LvdQC3LMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzMM/CPAvzLMyzcNT9eYoCNCvOwuYAAAAASUVORK5CYII="
)


def ensure_workspace_styles() -> None:
    if st.session_state.get("_workspace_css_loaded"):
        return

    st.markdown(
        """
        <style>
        .workspace-card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 4px 0;
        }
        .badge-row {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge-high { background: #163d23; color: #61ff9c; }
        .badge-mid { background: #3b3612; color: #ffd861; }
        .badge-low { background: #3c1513; color: #ff6a69; }
        .badge-neutral { background: #2f363f; color: #d8dee9; }
        .why-pill {
            display: inline-block;
            margin-bottom: 6px;
            background: #2f363f;
            color: #ffd861;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_workspace_css_loaded"] = True


def confidence_badge(label: str, value: float | None, mid: float, high: float) -> str:
    if value is None:
        color = "neutral"
        text = "n/a"
    elif value >= high:
        color = "high"
        text = f"{value:.2f}"
    elif value >= mid:
        color = "mid"
        text = f"{value:.2f}"
    else:
        color = "low"
        text = f"{value:.2f}"
    return f'<span class="badge badge-{color}">{label}: {text}</span>'


def contamination_badge(label: str, value: float | None, high: float) -> str:
    if value is None:
        color = "neutral"
        text = "n/a"
    elif value >= high:
        color = "low"
        text = f"{value:.2f}"
    elif value >= high / 2:
        color = "mid"
        text = f"{value:.2f}"
    else:
        color = "high"
        text = f"{value:.2f}"
    return f'<span class="badge badge-{color}">{label}: {text}</span>'


def get_thumbnail_generator() -> ThumbnailGenerator:
    if "thumbnail_generator" not in st.session_state:
        cache_dir = Path("data/cache/thumbnails")
        st.session_state.thumbnail_generator = ThumbnailGenerator(cache_dir)
    return st.session_state.thumbnail_generator


def track_preview_image(
    track: dict | None,
    video_path: Path,
    episode_id: str,
    thumb_gen: ThumbnailGenerator,
) -> str:
    if not track:
        return PLACEHOLDER_DATA_URI

    for ref in track.get("frame_refs", []) or []:
        bbox = ref.get("bbox")
        if not bbox:
            continue
        thumb_path = thumb_gen.generate_thumbnail(
            video_path,
            ref["frame_id"],
            bbox,
            episode_id,
            track["track_id"],
        )
        if thumb_path:
            return str(thumb_path)
    return PLACEHOLDER_DATA_URI


def track_frame_images(
    track: dict | None,
    video_path: Path,
    episode_id: str,
    thumb_gen: ThumbnailGenerator,
) -> Tuple[List[int], List[str]]:
    if not track:
        return [], []

    frame_ids: List[int] = []
    images: List[str] = []
    for ref in track.get("frame_refs", []) or []:
        frame_id = int(ref.get("frame_id"))
        bbox = ref.get("bbox")
        frame_ids.append(frame_id)
        if bbox:
            thumb_path = thumb_gen.generate_frame_thumbnail(
                video_path,
                frame_id,
                bbox,
                episode_id,
                track["track_id"],
            )
            images.append(str(thumb_path) if thumb_path else PLACEHOLDER_DATA_URI)
        else:
            images.append(PLACEHOLDER_DATA_URI)
    return frame_ids, images


def count_total_tracks(episode_id: str, data_root: str = "data") -> int:
    """Count total tracks from tracks.json or embeddings.parquet."""
    from pathlib import Path
    import json

    # Try tracks.json first
    tracks_path = Path(data_root) / "harvest" / episode_id / "tracks.json"
    if tracks_path.exists():
        try:
            with open(tracks_path) as f:
                data = json.load(f)
                return len(data.get("tracks", []))
        except Exception:
            pass

    # Fallback to embeddings.parquet
    try:
        import pandas as pd
        emb_path = Path(data_root) / "harvest" / episode_id / "embeddings.parquet"
        if emb_path.exists():
            df = pd.read_parquet(emb_path)
            return int(df["track_id"].nunique())
    except Exception:
        pass

    return 0


def count_stills_manifest(episode_id: str, data_root: str = "data") -> int:
    """Count completed stills from track_stills.jsonl manifest."""
    from pathlib import Path

    manifest_path = Path(data_root) / "harvest" / episode_id / "stills" / "track_stills.jsonl"
    if not manifest_path.exists():
        return 0

    try:
        with open(manifest_path) as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def render_stills_progress(episode_id: str, data_root: str = "data", key_suffix: str = "") -> tuple[int, int, float]:
    """
    Render stills generation progress bar if incomplete.

    Returns:
        (done, total, coverage_ratio)
    """
    total = count_total_tracks(episode_id, data_root)
    done = count_stills_manifest(episode_id, data_root)

    if total == 0:
        return 0, 0, 0.0

    coverage = done / total

    if coverage < 1.0:
        st.progress(coverage, text=f"Face stills: {done}/{total} ({int(100*coverage)}%)")
        st.caption("⏳ Generating face-aware crops with FIQA scoring... Refresh page to update.")

    return done, total, coverage


def read_pipeline_state(episode_id: str, data_root: str = "data") -> Optional[Dict[str, Any]]:
    """
    Read current pipeline state from diagnostics/pipeline_state.json.

    Returns None if file doesn't exist or is invalid.
    """
    state_path = Path(data_root) / "harvest" / episode_id / "diagnostics" / "pipeline_state.json"
    if not state_path.exists():
        return None

    try:
        with open(state_path) as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_track_stills_manifest(episode_id: str, data_root: str, mtime: float) -> Dict[str, Dict[str, Any]]:
    """
    Load track stills manifest, cached by file modification time.

    Returns dict mapping track_id -> manifest entry.
    """
    manifest_path = Path(data_root) / "harvest" / episode_id / "stills" / "track_stills.jsonl"
    if not manifest_path.exists():
        return {}

    manifest = {}
    try:
        with open(manifest_path) as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    manifest[entry["track_id"]] = entry
    except Exception:
        pass

    return manifest


def get_track_still_path(
    episode_id: str,
    track_id: str,
    data_root: str = "data",
    prefer_thumb: bool = True
) -> Optional[Path]:
    """
    Get track still image path with 4-tier fallback:
    1. SER-FIQ thumb (160x200) from manifest
    2. SER-FIQ crop (320x400) from manifest
    3. Legacy ffmpeg thumbnail
    4. None (caller should use placeholder)

    Args:
        episode_id: Episode identifier
        track_id: Track identifier
        data_root: Root data directory
        prefer_thumb: Prefer 160x200 thumb over 320x400 crop

    Returns:
        Path to image file, or None if not found
    """
    harvest_dir = Path(data_root) / "harvest" / episode_id

    # Try manifest first
    manifest_path = harvest_dir / "stills" / "track_stills.jsonl"
    if manifest_path.exists():
        try:
            mtime = manifest_path.stat().st_mtime
            manifest = load_track_stills_manifest(episode_id, data_root, mtime)

            if track_id in manifest:
                entry = manifest[track_id]

                # Tier 1: 160x200 thumbnail
                if prefer_thumb and "thumb_path" in entry:
                    thumb_path = Path(entry["thumb_path"])
                    if thumb_path.exists():
                        return thumb_path

                # Tier 2: 320x400 crop
                if "crop_path" in entry:
                    crop_path = Path(entry["crop_path"])
                    if crop_path.exists():
                        return crop_path

                # Tier 1b: Try thumb even if not preferred
                if not prefer_thumb and "thumb_path" in entry:
                    thumb_path = Path(entry["thumb_path"])
                    if thumb_path.exists():
                        return thumb_path
        except Exception:
            pass

    # Tier 3: Legacy ffmpeg thumbnails
    legacy_thumb = harvest_dir / "thumbnails" / f"{track_id}.jpg"
    if legacy_thumb.exists():
        return legacy_thumb

    # Tier 4: Not found
    return None


def check_artifacts_status(episode_id: str, data_root: str = "data") -> Dict[str, Any]:
    """
    Check which pipeline artifacts exist and what needs to run.

    Returns dict with:
        - artifacts: dict of artifact existence flags
        - needs_detect: bool
        - needs_track: bool
        - needs_cluster: bool
        - needs_stills: bool
        - message: human-readable status
        - next_action: suggested action button text
    """
    from jobs.tasks.orchestrate import check_artifacts

    artifacts = check_artifacts(episode_id, Path(data_root))

    result = {
        "artifacts": artifacts,
        "needs_detect": not artifacts["embeddings"],
        "needs_track": not artifacts["tracks"],
        "needs_cluster": not artifacts["clusters"],
        "needs_stills": not (artifacts["stills_manifest"] and artifacts["stills_thumbs"]),
        "needs_analytics": not artifacts.get("analytics", False),
    }

    # Determine message and next action
    if result["needs_detect"]:
        result["message"] = "Face detection and embeddings not found."
        result["next_action"] = "Run Detect/Embed"
    elif result["needs_track"]:
        result["message"] = "Face tracking data not found."
        result["next_action"] = "Run Track"
    elif result["needs_cluster"]:
        result["message"] = "Cluster data not found."
        result["next_action"] = "Run Cluster"
    elif result["needs_stills"]:
        result["message"] = "Face stills (thumbnails) not found or incomplete."
        result["next_action"] = "Generate Stills"
    elif result["needs_analytics"]:
        result["message"] = "Analytics not found or outdated."
        result["next_action"] = "Run Analytics"
    else:
        result["message"] = "All pipeline artifacts are ready."
        result["next_action"] = None

    return result
