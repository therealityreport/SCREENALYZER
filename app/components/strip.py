"""Reusable thumbnail strip renderer."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image

import streamlit as st

from app.utils.ui_keys import wkey

logger = logging.getLogger(__name__)

STRIP_TILE_W = 160
STRIP_TILE_H = 200
STRIP_PAGE_SIZE = 24


@st.cache_data(show_spinner=False)
def _load_pil_thumb(path: str, mtime: float) -> Image.Image:
    """Load thumbnail image from disk with cache keyed by mtime."""
    with Image.open(path) as img:
        return img.copy()



def render_strip(
    images: Sequence[str | Path],
    ids: Optional[Sequence[int | str]] = None,
    *,
    key_prefix: str,
    strip_id: Optional[str] = None,
    selectable: bool = False,
    selected_ids: Optional[Iterable[int | str]] = None,
    image_loader: Optional[
        Callable[[Sequence[int | str]], Sequence[str | Path | Image.Image]]
    ] = None,
) -> Tuple[List[int | str], List[int | str]]:
    """
    Render a paginated strip of thumbnails with optional selection checkboxes.

    Args:
        images: Sequence representing all thumbnails (can be placeholders when using loader).
        ids: Optional stable identifiers parallel to images; falls back to index.
        key_prefix: Base prefix for widget keys (e.g., "ws_clusters", "ws_tracks").
        strip_id: Unique identifier for this strip instance to prevent key collisions.
                 Example: f"c{cluster_id}_all" or f"t{track_id}"
        selectable: When True, show checkboxes and return selected IDs.
        selected_ids: Optional initial selection iterable.
        image_loader: Optional callable to lazily load visible page thumbnails.

    Returns:
        Tuple (page_ids, selected_ids) where:
            - page_ids: IDs currently visible on this page.
            - selected_ids: IDs selected across the entire strip.
    """
    total = len(images)
    if total == 0:
        st.info("No thumbnails available.")
        return [], []

    ids = list(ids) if ids is not None else list(range(total))
    if len(ids) != total:
        raise ValueError("ids length must match images length")

    # Build fully qualified prefix with strip_id to ensure uniqueness
    full_prefix = f"{key_prefix}_{strip_id}" if strip_id else key_prefix

    selected_key = wkey(full_prefix, "selected")
    page_key = wkey(full_prefix, "page")

    if selectable:
        if selected_key not in st.session_state:
            st.session_state[selected_key] = set(selected_ids or [])
    elif selected_key in st.session_state:
        del st.session_state[selected_key]

    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    total_pages = max(1, math.ceil(total / STRIP_PAGE_SIZE))
    current_page = max(0, min(st.session_state[page_key], total_pages - 1))

    start = current_page * STRIP_PAGE_SIZE
    end = min(start + STRIP_PAGE_SIZE, total)
    raw_page_images = list(images[start:end])
    page_ids = ids[start:end]

    if image_loader is not None:
        page_images = list(image_loader(page_ids))
    else:
        page_images = raw_page_images

    nav_left, nav_mid, nav_right = st.columns([1, 6, 1])
    with nav_left:
        if st.button(
            "◀",
            key=wkey(full_prefix, "prev"),
            disabled=current_page == 0,
        ):
            st.session_state[page_key] = current_page - 1
            st.rerun()
    with nav_mid:
        st.caption(f"Showing {start + 1}–{end} of {total}")
    with nav_right:
        if st.button(
            "▶",
            key=wkey(full_prefix, "next"),
            disabled=current_page >= total_pages - 1,
        ):
            st.session_state[page_key] = current_page + 1
            st.rerun()

    # Render thumbnails in a flexible row
    _render_grid(
        page_images,
        page_ids,
        selectable=selectable,
        full_prefix=full_prefix,
        current_page=current_page,
        selected_key=selected_key,
    )

    selected_items: List[int | str] = []
    if selectable:
        selected_items = list(st.session_state[selected_key])

    return page_ids, selected_items


def _render_grid(
    images: Sequence[str | Path],
    ids: Sequence[int | str],
    *,
    selectable: bool,
    full_prefix: str,
    current_page: int,
    selected_key: str,
) -> None:
    """Render thumbnails using Streamlit columns."""
    total = len(images)
    if total == 0:
        return

    cols_per_row = min(6, total) or 1
    if selectable:
        current_selected = st.session_state[selected_key]
    else:
        current_selected = set()

    for row_start in range(0, total, cols_per_row):
        row_images = images[row_start : row_start + cols_per_row]
        row_ids = ids[row_start : row_start + cols_per_row]
        columns = st.columns(len(row_images) or 1)
        for idx, (col, img, pid) in enumerate(zip(columns, row_images, row_ids)):
            with col:
                resolved = _resolve_image(img)
                st.image(resolved, width=STRIP_TILE_W)
                if selectable:
                    checkbox_key = wkey(full_prefix, "chk", pid, current_page, row_start + idx)
                    checked = pid in current_selected
                    new_val = st.checkbox(
                        "",
                        value=checked,
                        key=checkbox_key,
                    )
                    if new_val and not checked:
                        current_selected.add(pid)
                    elif not new_val and checked:
                        current_selected.remove(pid)


def _resolve_image(img: Union[str, Path, Image.Image]) -> Image.Image | str:
    """Return cached PIL image for given path, or original object."""
    if isinstance(img, Image.Image):
        return img

    if isinstance(img, Path):
        path = img
    else:
        try:
            path = Path(str(img))
        except TypeError:
            return str(img)

    if path.exists():
        try:
            mtime = path.stat().st_mtime
            return _load_pil_thumb(str(path), mtime)
        except Exception as exc:
            logger.warning("Failed to load image %s: %s", path, exc)
            return str(path)
    else:
        # Log missing paths (limit to first 10 per session to avoid spam)
        if not hasattr(_resolve_image, "_logged_missing"):
            _resolve_image._logged_missing = set()
        path_str = str(path)
        if path_str not in _resolve_image._logged_missing and len(_resolve_image._logged_missing) < 10:
            logger.warning("Image not found: %s", path)
            _resolve_image._logged_missing.add(path_str)

    # Allow pre-baked data URIs or URLs to pass through
    if isinstance(img, str):
        return img

    return str(path)
