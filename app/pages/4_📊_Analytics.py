"""
Analytics - View analytics and metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv

from app.utils.ui_keys import wkey, safe_rerun
from app.lib.data import load_clusters
from app.lib.analytics_dirty import is_analytics_dirty, get_analytics_freshness

load_dotenv()

st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide",
)

DATA_ROOT = Path(os.getenv("DATA_ROOT", "./data"))


def render_analytics_page():
    """Render analytics page with exports."""
    st.title("📊 Analytics & Exports")

    # Episode selector
    harvest_dir = DATA_ROOT / "harvest"
    if not harvest_dir.exists():
        st.warning("No episodes found. Please upload and process a video first.")
        return

    episodes = [d.name for d in harvest_dir.iterdir() if d.is_dir()]
    if not episodes:
        st.warning("No episodes found.")
        return

    selected_episode = st.selectbox("Select Episode", episodes, index=0, key=wkey("analytics_episode_select"))

    # Check Analytics freshness
    is_dirty = is_analytics_dirty(selected_episode, DATA_ROOT)
    freshness_status = get_analytics_freshness(selected_episode, DATA_ROOT)

    # Show freshness indicator
    if is_dirty or freshness_status == 'stale':
        st.warning("⚠️ Analytics are **stale** - clusters have been modified since last Analyze run")
        st.info("**Recommended workflow for best results:**\n1. Go to **Workspace** and run **Re-Cluster (constrained)**\n2. Return here and click **Analyze** to rebuild analytics from current state")
    elif freshness_status == 'fresh':
        st.success("✅ Analytics are **fresh** - up to date with current clusters")
    elif freshness_status == 'unknown':
        st.info("ℹ️ Analytics status: **unknown** - no previous Analyze run detected")

    # Check for analytics outputs
    outputs_dir = DATA_ROOT / "outputs" / selected_episode
    totals_csv = outputs_dir / "totals.csv"
    timeline_csv = outputs_dir / "timeline.csv"
    excel_file = outputs_dir / "totals.xlsx"

    # Check if files exist and are not empty
    files_valid = False
    if totals_csv.exists() and timeline_csv.exists():
        # Check if files have content (more than just header/empty)
        try:
            import os
            totals_size = os.path.getsize(totals_csv)
            timeline_size = os.path.getsize(timeline_csv)
            # Files should be more than 1 byte (not just newline)
            if totals_size > 10 and timeline_size > 10:
                # Try to read them
                test_df = pd.read_csv(totals_csv)
                if len(test_df) > 0:
                    files_valid = True
        except:
            files_valid = False

    if not files_valid:
        st.info(f"Analytics not yet generated for {selected_episode} or files are empty.")
        st.info("**Click below to rebuild analytics from current clusters** (always rebuilds from scratch, no cache)")

        # Button to generate analytics
        if st.button("📊 Analyze", type="primary", help="Rebuild analytics from current clusters.json (suppression-aware, always fresh)", key=wkey("analytics_analyze")):
            with st.spinner("Generating analytics..."):
                try:
                    # Load cluster assignments from clusters.json
                    clusters_data = load_clusters(selected_episode, DATA_ROOT)
                    cluster_assignments = {}
                    if clusters_data:
                        for cluster in clusters_data.get("clusters", []):
                            if "name" in cluster:
                                cluster_assignments[cluster["cluster_id"]] = cluster["name"]

                    from jobs.tasks.analytics import analytics_task

                    result = analytics_task("manual", selected_episode, cluster_assignments)
                    st.success(
                        f"✅ Analytics generated! {result['stats']['intervals_created']} intervals created."
                    )
                    safe_rerun()
                except Exception as e:
                    st.error(f"Failed to generate analytics: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    # Load analytics data
    try:
        totals_df = pd.read_csv(totals_csv)
        timeline_df = pd.read_csv(timeline_csv)
    except Exception as e:
        st.error(f"Failed to load analytics: {str(e)}")
        st.info("Try clicking 'Generate Analytics' above to regenerate the files.")
        return

    # Display summary
    st.subheader("Screen Time Summary")

    # Regenerate button
    if st.button("🔄 Rebuild Analytics", help="Rebuild analytics from current clusters.json (always rebuilds from scratch, suppression-aware)", key=wkey("analytics_rebuild")):
        with st.spinner("Regenerating analytics..."):
            try:
                # Load cluster assignments from clusters.json
                clusters_data = load_clusters(selected_episode, DATA_ROOT)
                cluster_assignments = {}
                if clusters_data:
                    for cluster in clusters_data.get("clusters", []):
                        if "name" in cluster:
                            cluster_assignments[cluster["cluster_id"]] = cluster["name"]

                from jobs.tasks.analytics import analytics_task

                result = analytics_task("manual", selected_episode, cluster_assignments)
                st.success(
                    f"Analytics regenerated! {result['stats']['intervals_created']} intervals created."
                )
                safe_rerun()
            except Exception as e:
                st.error(f"Failed to regenerate analytics: {str(e)}")

    if len(totals_df) == 0:
        st.info("No people detected or assigned in this episode.")
        return

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("People Detected", len(totals_df))
    with col2:
        total_time_sec = totals_df["total_sec"].sum()
        minutes = int(total_time_sec // 60)
        seconds = int(total_time_sec % 60)
        st.metric("Total Screen Time", f"{minutes:02d}:{seconds:02d}")
    with col3:
        st.metric("Total Intervals", len(timeline_df))
    with col4:
        mean_time_sec = totals_df["total_sec"].mean()
        st.metric("Avg Time/Person", f"{mean_time_sec:.1f} sec")

    # Totals table
    st.subheader("Top Cast Members")

    # Format display columns
    display_df = totals_df[["person_name", "total_sec", "appearances", "percent"]].copy()
    # Format as MM:SS:MS
    display_df["total_time"] = display_df["total_sec"].apply(
        lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}:{int((x % 1) * 1000):03d}"
    )
    display_df = display_df.rename(
        columns={
            "person_name": "Person",
            "total_time": "Screen Time",
            "appearances": "Appearances",
            "percent": "% of Episode",
        }
    )
    display_df = display_df[["Person", "Screen Time", "Appearances", "% of Episode"]]

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Bar chart
    st.subheader("Screen Time Distribution")

    fig = px.bar(
        totals_df.head(10),
        x="person_name",
        y="total_sec",
        labels={"person_name": "Person", "total_sec": "Screen Time (seconds)"},
        title="Top 10 Cast Members by Screen Time",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Timeline preview
    with st.expander("Timeline Preview (First 20 intervals)", expanded=False):
        timeline_preview = timeline_df.head(20)[
            ["person_name", "start_ms", "end_ms", "duration_ms", "confidence"]
        ]
        st.dataframe(timeline_preview, use_container_width=True, hide_index=True)

    # Export section
    st.markdown("---")
    st.subheader("Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.download_button(
            label="📥 Download Totals (CSV)",
            data=open(totals_csv, "rb").read(),
            file_name=f"{selected_episode}_totals.csv",
            mime="text/csv",
            key=wkey("analytics_download_totals")
        )

    with col2:
        st.download_button(
            label="📥 Download Timeline (CSV)",
            data=open(timeline_csv, "rb").read(),
            file_name=f"{selected_episode}_timeline.csv",
            mime="text/csv",
            key=wkey("analytics_download_timeline")
        )

    with col3:
        if excel_file.exists():
            st.download_button(
                label="📥 Download Excel",
                data=open(excel_file, "rb").read(),
                file_name=f"{selected_episode}_totals.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=wkey("analytics_download_excel")
            )
        else:
            st.button("📥 Download Excel", disabled=True, key=wkey("analytics_download_excel_disabled"))

    # Analytics debug info
    with st.expander("📋 Debug Info", expanded=False):
        debug_file = outputs_dir / "analytics_debug.json"
        if debug_file.exists():
            import json
            with open(debug_file, 'r') as f:
                debug_data = json.load(f)
            st.json(debug_data)
        else:
            st.info("No debug info available")


if __name__ == "__main__":
    render_analytics_page()
