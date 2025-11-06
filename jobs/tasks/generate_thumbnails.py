"""Generate persistent track thumbnails for an episode."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime
from io import BytesIO
from pathlib import Path
from shutil import which
from typing import Dict, List, Optional

import cv2
from PIL import Image

from app.lib.data import load_tracks
from app.media.thumbnails import (
    THUMB_QUALITY,
    center_crop_fill,
    ensure_placeholder_thumbnail,
    record_thumb_path,
)
from screentime.episode_registry import episode_registry
from screentime.utils import get_video_path

logger = logging.getLogger(__name__)

THUMB_SUBDIR = "thumbnails/tracks"
THUMB_STATS_FILENAME = "thumbnails_stats.json"


def generate_thumbnails_task(
    episode_id: str,
    *,
    data_root: Path = Path("data"),
    force: bool = False,
) -> dict:
    """Generate thumbnails for all tracks in an episode."""

    harvest_dir = data_root / "harvest" / episode_id
    thumbs_dir = harvest_dir / THUMB_SUBDIR
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    tracks_data = load_tracks(episode_id, data_root) or {"tracks": []}
    track_list = tracks_data.get("tracks") or []

    if not track_list:
        logger.warning("No tracks found for episode %s; skipping thumbnails", episode_id)
        return _write_stats(thumbs_dir, episode_id, 0, 0, 0, [], {}, {})

    track_map: Dict[int, dict] = {}
    for track in track_list:
        try:
            track_map[int(track["track_id"])] = track
        except (TypeError, ValueError, KeyError):
            continue

    total_tracks = len(track_map)
    generated = 0
    placeholders = 0
    skipped_existing = 0
    errors: List[dict] = []

    video_path = _resolve_video_path(episode_id, data_root)
    fps = _probe_video_fps(video_path) if video_path else None
    ffmpeg_available = has_ffmpeg()
    opencv_available = _probe_opencv_decoder(video_path) if video_path else False

    diagnostics = {
        "ffmpeg": ffmpeg_available,
        "opencv": opencv_available,
        "fps": fps,
    }

    if not video_path or not video_path.exists():
        reason = f"video_not_found:{video_path}"
        logger.warning("Video for episode %s not found (%s)", episode_id, video_path)
        for track_id in track_map:
            thumb_path = thumbs_dir / f"{track_id}.jpg"
            if force or not thumb_path.exists():
                ensure_placeholder_thumbnail(thumb_path)
                record_thumb_path(episode_id, track_id, thumb_path)
                placeholders += 1
                errors.append({"track_id": track_id, "reason": reason})
        return _write_stats(
            thumbs_dir,
            episode_id,
            total_tracks,
            generated,
            placeholders,
            errors,
            diagnostics,
            {"video_path": str(video_path) if video_path else None},
            skipped_existing=skipped_existing,
        )

    if not ffmpeg_available and not opencv_available:
        reason = "no_decoder"
        logger.error(
            "Neither ffmpeg nor OpenCV decoding available for %s – thumbnails cannot be generated",
            episode_id,
        )
        return _write_stats(
            thumbs_dir,
            episode_id,
            total_tracks,
            generated,
            placeholders,
            [{"track_id": None, "reason": reason}],
            diagnostics,
            {"video_path": str(video_path)},
            skipped_existing=skipped_existing,
        )

    for track_id, track in track_map.items():
        thumb_path = thumbs_dir / f"{track_id}.jpg"

        if thumb_path.exists() and not force:
            skipped_existing += 1
            record_thumb_path(episode_id, track_id, thumb_path)
            continue

        ts_ms = _select_timestamp_ms(track, fps)
        if ts_ms is None:
            reason = "no_timestamp"
            placeholders += 1
            ensure_placeholder_thumbnail(thumb_path)
            record_thumb_path(episode_id, track_id, thumb_path)
            errors.append({"track_id": track_id, "reason": reason})
            logger.warning("Track %s placeholder: %s", track_id, reason)
            continue

        frame_image = None
        try:
            frame_image = _grab_frame(video_path, ts_ms, ffmpeg_available, opencv_available, fps)
        except FileNotFoundError:
            reason = "ffmpeg_missing"
            errors.append({"track_id": track_id, "reason": reason})
            logger.warning("FFmpeg not available when processing track %s", track_id)
        except RuntimeError as exc:
            reason = f"decode_failed:{exc}"
            errors.append({"track_id": track_id, "reason": reason})
            logger.warning("Frame decode failed for track %s at %sms: %s", track_id, ts_ms, exc)

        if frame_image is None:
            placeholders += 1
            ensure_placeholder_thumbnail(thumb_path)
            record_thumb_path(episode_id, track_id, thumb_path)
            continue

        thumb_image = center_crop_fill(frame_image)
        thumb_image.save(thumb_path, quality=THUMB_QUALITY)
        record_thumb_path(episode_id, track_id, thumb_path)
        generated += 1

    return _write_stats(
        thumbs_dir,
        episode_id,
        total_tracks,
        generated,
        placeholders,
        errors,
        diagnostics,
        {"video_path": str(video_path)},
        skipped_existing=skipped_existing,
    )


def _write_stats(
    thumbs_dir: Path,
    episode_id: str,
    total_tracks: int,
    generated: int,
    placeholders: int,
    errors: List[dict],
    diagnostics: Dict,
    meta: Dict,
    *,
    skipped_existing: int = 0,
) -> dict:
    stats = {
        "episode_id": episode_id,
        "total_tracks": total_tracks,
        "generated": generated,
        "placeholders": placeholders,
        "skipped_existing": skipped_existing,
        "errors": errors,
        "diagnostics": diagnostics,
        "meta": {
            "video_path": meta.get("video_path"),
            "completed_at": datetime.utcnow().isoformat(),
        },
    }

    stats_path = thumbs_dir.parent / THUMB_STATS_FILENAME
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info(
        "Thumbnail generation complete for %s: %d generated, %d placeholders, %d skipped",
        episode_id,
        generated,
        placeholders,
        skipped_existing,
    )
    if errors:
        logger.info("Thumbnail generation recorded %d errors", len(errors))
    return stats


def _resolve_video_path(episode_id: str, data_root: Path) -> Optional[Path]:
    episode_info = episode_registry.get_episode(episode_id)
    if episode_info and episode_info.get("video_path"):
        candidate = data_root / episode_info["video_path"]
        if candidate.exists():
            return candidate

    # Fallback to canonical lookup
    return get_video_path(episode_id, data_root)


def has_ffmpeg() -> bool:
    return bool(which("ffmpeg"))


def _probe_video_fps(video_path: Path) -> Optional[float]:
    if not video_path or not video_path.exists():
        return None
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    if fps and fps > 0:
        return float(fps)
    return None


def _probe_opencv_decoder(video_path: Optional[Path]) -> bool:
    if not video_path or not video_path.exists():
        return False
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return False
    capture.release()
    return True


def _select_timestamp_ms(track: dict, fps: Optional[float]) -> Optional[int]:
    frame_refs = track.get("frame_refs") or []
    ts_values = sorted(
        int(ref["ts_ms"])
        for ref in frame_refs
        if isinstance(ref, dict) and ref.get("ts_ms") is not None
    )
    if ts_values:
        return ts_values[len(ts_values) // 2]

    start_ms = track.get("start_ms")
    end_ms = track.get("end_ms")
    if isinstance(start_ms, (int, float)) and isinstance(end_ms, (int, float)) and end_ms >= start_ms:
        return int(start_ms + (end_ms - start_ms) / 2)

    frame_ids = sorted(
        int(ref["frame_id"])
        for ref in frame_refs
        if isinstance(ref, dict) and ref.get("frame_id") is not None
    )
    if frame_ids and fps:
        mid_frame = frame_ids[len(frame_ids) // 2]
        return int((mid_frame / fps) * 1000)

    return None


def _grab_frame(
    video_path: Path,
    ts_ms: int,
    ffmpeg_available: bool,
    opencv_available: bool,
    fps: Optional[float],
) -> Optional[Image.Image]:
    if not video_path.exists():
        raise RuntimeError("video_missing")

    if ffmpeg_available:
        ts_sec = ts_ms / 1000.0
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{ts_sec:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-",
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                image = Image.open(BytesIO(result.stdout))
                return image.convert("RGB")
            else:
                error_msg = result.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg_error"
                logger.debug("FFmpeg decode failed (%s), attempting OpenCV fallback", error_msg)
        except FileNotFoundError as exc:
            raise FileNotFoundError("ffmpeg_not_found") from exc
        except Exception as exc:
            logger.debug("FFmpeg execution raised %s, falling back to OpenCV", exc)

    if not opencv_available:
        raise RuntimeError("no_decoder")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cv2_open_failed")

    try:
        if fps and fps > 0:
            frame_index = int((ts_ms / 1000.0) * fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        else:
            capture.set(cv2.CAP_PROP_POS_MSEC, ts_ms)

        success, frame = capture.read()
        if not success or frame is None:
            raise RuntimeError("cv2_read_failed")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        capture.release()
