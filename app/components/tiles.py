"""
Reusable tile components for Workspace pages.

Tiles render compact cards with 4:5 thumbnails (160x200) with confidence badges and metadata.
Each renderer returns True if the tile was clicked (button press) so callers
can update selection state.
"""

from __future__ import annotations

from typing import Dict, Optional
from pathlib import Path

import streamlit as st


def _confidence_badge(value: Optional[float], low: float, high: float) -> str:
    """Return HTML badge for confidence value."""
    if value is None:
        return '<span class="badge badge-neutral">n/a</span>'

    if value >= high:
        tone = "badge-high"
    elif value >= low:
        tone = "badge-mid"
    else:
        tone = "badge-low"

    return f'<span class="badge {tone}">{value:.2f}</span>'


def _ensure_styles():
    """Inject lightweight CSS once per session."""
    if st.session_state.get("_workspace_tile_css_loaded"):
        return

    st.markdown(
        """
        <style>
            .workspace-tile {
                background-color: #20252b;
                border: 1px solid #2f363f;
                border-radius: 12px;
                padding: 12px;
                height: 220px;
                width: 150px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
            }
            .workspace-tile h4 {
                margin: 0 0 2px 0;
                font-size: 1rem;
            }
            .workspace-tile .counts {
                font-size: 0.8rem;
                color: #a8b2c1;
            }
            .workspace-tile .avatar {
                width: 128px;
                height: 128px;
                border-radius: 64px;
                object-fit: cover;
                margin: 8px auto;
                display: block;
            }
            .workspace-tile .placeholder {
                width: 128px;
                height: 128px;
                border-radius: 64px;
                background: #2f363f;
                margin: 8px auto;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 48px;
            }
            .badge {
                display: inline-block;
                padding: 2px 6px;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
            }
            .badge-high { background: #163d23; color: #61ff9c; }
            .badge-mid { background: #3b3612; color: #ffd861; }
            .badge-low { background: #3c1513; color: #ff6a69; }
            .badge-neutral { background: #2e3440; color: #d8dee9; }
            .metric-row {
                display: flex;
                justify-content: space-between;
                font-size: 0.75rem;
                margin-top: 4px;
            }
            .metric-row span.label {
                color: #8f97a4;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["_workspace_tile_css_loaded"] = True


def render_person_tile(person: Dict, thresholds: Dict[str, float], key: str, show_id: str = "rhobh", season_id: str = "s05") -> bool:
    """Render person overview tile with 4:5 thumbnail (160x200)."""
    _ensure_styles()

    low = float(thresholds.get("person_low_p25", 0.7))
    high = float(thresholds.get("track_high_p25", 0.7))
    badge = _confidence_badge(person.get("bank_conf_median_p25"), low, high)
    contam = person.get("bank_contam_rate")
    contam_badge = _confidence_badge(contam if contam is not None else 0.0, 0.0, 0.1)

    person_name = person.get('person', 'Unknown')

    # Try to resolve thumbnail
    from app.lib.facebank_meta import resolve_person_thumbnail, get_thumbnail
    thumbnail_path = resolve_person_thumbnail(show_id, season_id, person_name)

    # Show cache-busted 4:5 thumbnail (160x200 enforced by Workspace CSS)
    if thumbnail_path and Path(thumbnail_path).exists():
        thumb = get_thumbnail(Path(thumbnail_path), size=(160, 200))
        st.image(thumb, use_column_width=False, width=160)
    else:
        # Placeholder for missing thumbnail (4:5 ratio)
        st.markdown(
            '<div style="width:160px; height:200px; background:#2f363f; display:flex; align-items:center; justify-content:center; border-radius:8px;'
            'font-size:48px;">👤</div>',
            unsafe_allow_html=True
        )

    # Person info and metrics
    st.markdown(f"**{person_name}**")
    st.caption(f"{person.get('n_clusters', 0)} clusters · {person.get('n_tracks', 0)} tracks")

    # Metrics row
    st.markdown(f'<div class="metric-row"><span class="label">Bank p25</span>{badge}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="metric-row"><span class="label">Contam</span>{contam_badge}</div>', unsafe_allow_html=True)

    return st.button("Open", key=key, help="Open person details", use_container_width=True)


def render_cluster_tile(cluster: Dict, thresholds: Dict[str, float], key: str) -> bool:
    """Render cluster tile with confidence metrics."""
    _ensure_styles()
    low = float(thresholds.get("cluster_low_p25", 0.6))
    high = float(thresholds.get("track_high_p25", 0.7))
    badge = _confidence_badge(cluster.get("tracks_conf_p25_median"), low, high)
    contam_badge = _confidence_badge(
        cluster.get("contam_rate"), 0.0, float(thresholds.get("cluster_contam_high", 0.2))
    )

    content = f"""
    <div class="workspace-tile">
        <div>
            <h4>Cluster {cluster.get('cluster_id')}</h4>
            <div class="counts">{cluster.get('name', 'Unknown')}</div>
            <div class="counts">{cluster.get('n_tracks', 0)} tracks</div>
        </div>
        <div>
            <div class="metric-row"><span class="label">Median p25</span>{badge}</div>
            <div class="metric-row"><span class="label">Contam</span>{contam_badge}</div>
        </div>
    </div>
    """
    st.markdown(content, unsafe_allow_html=True)
    return st.button("Inspect", key=key, help="Open cluster details")


def render_track_tile(track: Dict, thresholds: Dict[str, float], key: str) -> bool:
    """Render track tile."""
    _ensure_styles()
    low = float(thresholds.get("track_low_p25", 0.55))
    high = float(thresholds.get("track_high_p25", 0.7))
    badge_p25 = _confidence_badge(track.get("conf_p25"), low, high)
    badge_mean = _confidence_badge(track.get("conf_mean"), low, high)

    content = f"""
    <div class="workspace-tile">
        <div>
            <h4>Track {track.get('track_id')}</h4>
            <div class="counts">{track.get('n_frames', 0)} frames</div>
        </div>
        <div>
            <div class="metric-row"><span class="label">p25</span>{badge_p25}</div>
            <div class="metric-row"><span class="label">mean</span>{badge_mean}</div>
        </div>
    </div>
    """
    st.markdown(content, unsafe_allow_html=True)
    return st.button("View", key=key, help="View track frames")
