"""Deterministic UI key helpers for Streamlit widgets."""

from __future__ import annotations

from typing import Any

import streamlit as st


def wkey(*parts: Any) -> str:
    """
    Build a deterministic widget key scoped by episode.

    Args:
        *parts: Hashable components that identify the widget.

    Returns:
        Unique string key.
    """
    episode = (
        st.session_state.get("episode_id")
        or st.session_state.get("workspace_episode")
        or st.session_state.get("selected_episode")
        or "noep"
    )
    normalized = "_".join(str(part).replace(" ", "_") for part in parts)
    return f"k_{episode}_{normalized}"


def safe_rerun():
    """
    Safe rerun that handles both old and new Streamlit API.

    Uses st.rerun() if available (Streamlit >= 1.27), otherwise falls back
    to st.experimental_rerun() for older versions.
    """
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        raise RuntimeError("No rerun method available in Streamlit")
