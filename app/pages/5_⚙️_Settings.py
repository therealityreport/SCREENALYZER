"""
Settings - Configure application settings and thresholds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import yaml
import streamlit as st
from dotenv import load_dotenv

from app.utils.ui_keys import wkey, safe_rerun

load_dotenv()

st.set_page_config(
    page_title="Settings",
    page_icon="⚙️",
    layout="wide",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))
CONFIG_PATH = Path("configs/pipeline.yaml")


# Default configuration (used for "Restore Defaults")
DEFAULT_CONFIG = {
    'confidence': {
        'frame_low': 0.55,
        'track_low_p25': 0.55,
        'track_high_p25': 0.70,
        'track_conflict_frac_high': 0.20,
        'cluster_low_p25': 0.60,
        'cluster_contam_high': 0.20,
        'person_low_p25': 0.70,
        'person_contam_high': 0.15,
        'top2_margin_low': 0.08,
    },
    'clustering': {
        'use_constraints': True,
        'use_season_bank': True,
    },
    'advanced': {
        'enable_body_linking': False,
    }
}


def load_config():
    """Load pipeline configuration from YAML."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    return {}


def save_config(config):
    """Save pipeline configuration to YAML atomically."""
    # Ensure directory exists
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first
    tmp_path = CONFIG_PATH.with_suffix('.yaml.tmp')
    with open(tmp_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Atomic replace
    tmp_path.replace(CONFIG_PATH)


def render_settings_page():
    """Render settings page."""
    st.title("⚙️ Settings")

    st.markdown("""
    Configure confidence thresholds, paths, and advanced options for the pipeline.
    Changes are saved to `configs/pipeline.yaml` and take effect on the next Re-Cluster + Analyze.
    """)

    # Load current config
    config = load_config()

    # Initialize session state for form values
    if 'settings_modified' not in st.session_state:
        st.session_state.settings_modified = False

    # ========================================
    # 1. System Information (Read-only)
    # ========================================
    st.subheader("1. System Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Data Root", str(DATA_ROOT))

    with col2:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        st.metric("Redis URL", redis_url.replace("redis://", ""))

    with col3:
        videos_dir = DATA_ROOT / "videos"
        if videos_dir.exists():
            video_count = len(list(videos_dir.glob("**/*.mp4")))
            st.metric("Uploaded Videos", video_count)
        else:
            st.metric("Uploaded Videos", 0)

    # ========================================
    # 2. Confidence Thresholds
    # ========================================
    st.markdown("---")
    st.subheader("2. Confidence Thresholds")

    st.markdown("""
    These thresholds determine what counts as "low confidence" in the UI and affect filtering.
    Lower values are more permissive (show more items), higher values are stricter.
    """)

    confidence = config.get('confidence', {})

    # Frame-level
    with st.expander("🔹 Frame Confidence", expanded=True):
        st.caption("Individual face detection confidence")

        frame_low = st.slider(
            "Frame Low Confidence",
            min_value=0.0,
            max_value=1.0,
            value=float(confidence.get('frame_low', 0.55)),
            step=0.01,
            help="Minimum confidence for a single face detection",
            key=wkey("settings_frame_low")
        )

        if frame_low != confidence.get('frame_low'):
            st.session_state.settings_modified = True

    # Track-level
    with st.expander("🔹 Track Confidence", expanded=True):
        st.caption("Confidence thresholds for tracks (groups of faces)")

        col1, col2, col3 = st.columns(3)

        with col1:
            track_low_p25 = st.slider(
                "Track Low P25",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('track_low_p25', 0.55)),
                step=0.01,
                help="25th percentile confidence threshold for tracks",
                key=wkey("settings_track_low_p25")
            )

        with col2:
            track_high_p25 = st.slider(
                "Track High P25",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('track_high_p25', 0.70)),
                step=0.01,
                help="High confidence threshold for track P25",
                key=wkey("settings_track_high_p25")
            )

        with col3:
            track_conflict_frac_high = st.slider(
                "Track Conflict Fraction High",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('track_conflict_frac_high', 0.20)),
                step=0.01,
                help="Maximum fraction of conflicting assignments",
                key=wkey("settings_track_conflict_frac_high")
            )

        if (track_low_p25 != confidence.get('track_low_p25') or
            track_high_p25 != confidence.get('track_high_p25') or
            track_conflict_frac_high != confidence.get('track_conflict_frac_high')):
            st.session_state.settings_modified = True

    # Cluster-level
    with st.expander("🔹 Cluster Confidence", expanded=True):
        st.caption("Confidence thresholds for clusters (groups of tracks)")

        col1, col2 = st.columns(2)

        with col1:
            cluster_low_p25 = st.slider(
                "Cluster Low P25",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('cluster_low_p25', 0.60)),
                step=0.01,
                help="25th percentile confidence threshold for clusters",
                key=wkey("settings_cluster_low_p25")
            )

        with col2:
            cluster_contam_high = st.slider(
                "Cluster Contamination High",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('cluster_contam_high', 0.20)),
                step=0.01,
                help="Maximum contamination rate for clusters",
                key=wkey("settings_cluster_contam_high")
            )

        if (cluster_low_p25 != confidence.get('cluster_low_p25') or
            cluster_contam_high != confidence.get('cluster_contam_high')):
            st.session_state.settings_modified = True

    # Person-level
    with st.expander("🔹 Person Confidence", expanded=True):
        st.caption("Confidence thresholds for person identities")

        col1, col2, col3 = st.columns(3)

        with col1:
            person_low_p25 = st.slider(
                "Person Low P25",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('person_low_p25', 0.70)),
                step=0.01,
                help="25th percentile confidence threshold for person assignments",
                key=wkey("settings_person_low_p25")
            )

        with col2:
            person_contam_high = st.slider(
                "Person Contamination High",
                min_value=0.0,
                max_value=1.0,
                value=float(confidence.get('person_contam_high', 0.15)),
                step=0.01,
                help="Maximum contamination rate for person identities",
                key=wkey("settings_person_contam_high")
            )

        with col3:
            top2_margin_low = st.slider(
                "Top2 Margin Low",
                min_value=0.0,
                max_value=0.5,
                value=float(confidence.get('top2_margin_low', 0.08)),
                step=0.01,
                help="Minimum margin between top 2 identity matches",
                key=wkey("settings_top2_margin_low")
            )

        if (person_low_p25 != confidence.get('person_low_p25') or
            person_contam_high != confidence.get('person_contam_high') or
            top2_margin_low != confidence.get('top2_margin_low')):
            st.session_state.settings_modified = True

    # ========================================
    # 3. Advanced Options
    # ========================================
    st.markdown("---")
    st.subheader("3. Advanced Options")

    # Note: These settings are informational for now
    # The actual implementation may store these in different parts of the config

    col1, col2 = st.columns(2)

    with col1:
        use_constraints = st.checkbox(
            "Use Manual Constraints",
            value=True,  # Always default to True
            help="Respect manual track assignments during re-clustering (must-link / cannot-link)",
            key=wkey("settings_use_constraints")
        )
        st.caption("Recommended: ON - preserves your manual edits during re-clustering")

    with col2:
        use_season_bank = st.checkbox(
            "Use Season Bank",
            value=True,  # Always default to True
            disabled=True,  # Always on
            help="Use season facebank for open-set assignment (min_sim=0.60, min_margin=0.08)",
            key=wkey("settings_use_season_bank")
        )
        st.caption("Always ON - required for identity assignment")

    # Future option (not yet implemented)
    enable_body_linking = st.checkbox(
        "Enable Body Linking (Experimental)",
        value=False,
        disabled=True,  # Not implemented yet
        help="Use body detection + tracking to link faces (future feature)",
        key=wkey("settings_enable_body_linking")
    )
    st.caption("Coming soon - track identities using body re-identification")

    # ========================================
    # 4. Paths Configuration
    # ========================================
    st.markdown("---")
    st.subheader("4. Paths")

    paths = config.get('paths', {})

    col1, col2 = st.columns(2)

    with col1:
        st.text_input(
            "Data Root",
            value=str(paths.get('data_root', 'data')),
            disabled=True,
            help="Root directory for all data (read-only)",
            key=wkey("settings_path_data_root")
        )

        st.text_input(
            "Facebank",
            value=str(paths.get('facebank', 'data/facebank')),
            disabled=True,
            help="Directory for cast reference images (read-only)",
            key=wkey("settings_path_facebank")
        )

    with col2:
        st.text_input(
            "Videos",
            value=str(paths.get('videos', 'data/videos')),
            disabled=True,
            help="Directory for uploaded videos (read-only)",
            key=wkey("settings_path_videos")
        )

        st.text_input(
            "Outputs",
            value=str(paths.get('outputs', 'data/outputs')),
            disabled=True,
            help="Directory for analytics outputs (read-only)",
            key=wkey("settings_path_outputs")
        )

    # ========================================
    # 5. Save / Restore Actions
    # ========================================
    st.markdown("---")

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 Save Settings", type="primary", key=wkey("settings_save")):
            # Update config with new values
            if 'confidence' not in config:
                config['confidence'] = {}

            config['confidence']['frame_low'] = frame_low
            config['confidence']['track_low_p25'] = track_low_p25
            config['confidence']['track_high_p25'] = track_high_p25
            config['confidence']['track_conflict_frac_high'] = track_conflict_frac_high
            config['confidence']['cluster_low_p25'] = cluster_low_p25
            config['confidence']['cluster_contam_high'] = cluster_contam_high
            config['confidence']['person_low_p25'] = person_low_p25
            config['confidence']['person_contam_high'] = person_contam_high
            config['confidence']['top2_margin_low'] = top2_margin_low

            # Save config
            save_config(config)

            st.success("✅ Settings saved to configs/pipeline.yaml")
            st.info("💡 Changes will take effect on the next Re-Cluster + Analyze")
            st.session_state.settings_modified = False

            # Wait a moment then rerun to show the success message
            import time
            time.sleep(1)
            safe_rerun()

    with col2:
        if st.button("🔄 Restore Defaults", key=wkey("settings_restore")):
            # Restore default confidence values
            if 'confidence' not in config:
                config['confidence'] = {}

            config['confidence'].update(DEFAULT_CONFIG['confidence'])

            # Save config
            save_config(config)

            st.success("✅ Settings restored to defaults")
            st.session_state.settings_modified = False

            # Wait a moment then rerun
            import time
            time.sleep(1)
            safe_rerun()

    with col3:
        if st.session_state.settings_modified:
            st.warning("⚠️ You have unsaved changes")

    # ========================================
    # 6. Current Configuration Preview
    # ========================================
    with st.expander("📋 View Full Configuration", expanded=False):
        st.caption("Current pipeline.yaml contents:")
        st.code(yaml.dump(config, default_flow_style=False, sort_keys=False), language='yaml')


if __name__ == "__main__":
    render_settings_page()
