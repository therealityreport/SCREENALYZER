#!/usr/bin/env python3
"""
Generate representative stills for tracks using ffmpeg.

Creates one stable still per track with black-frame guards and exact seeking.
Outputs to: data/harvest/{episode_id}/frames/tracks/{track_id}.jpg
Manifest: data/harvest/{episode_id}/frames/frames_manifest.json
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image, ImageStat

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def find_ffmpeg() -> str:
    """Find ffmpeg binary in PATH or common locations."""
    # Try system PATH first
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    # Try common locations
    for path in ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"]:
        if os.path.exists(path):
            return path

    raise RuntimeError(
        "ffmpeg not found. Install with: brew install ffmpeg"
    )


FFMPEG_BIN = find_ffmpeg()
logger.info(f"Using ffmpeg: {FFMPEG_BIN}")


def ts_fmt(s: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg."""
    h = int(s // 3600)
    s -= h * 3600
    m = int(s // 60)
    s -= m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def ffmpeg_grab(
    src: str,
    ts_s: float,
    out_jpg: str,
    w: int = 160,
    h: int = 200,
) -> tuple[bool, str]:
    """
    Extract a single frame at timestamp using ffmpeg.

    Args:
        src: Video file path
        ts_s: Timestamp in seconds
        out_jpg: Output JPEG path
        w: Target width (with letterbox)
        h: Target height (with letterbox)

    Returns:
        Tuple of (success, stderr_output)
    """
    vf = f"yadif=0:-1:0,scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-ss", f"{ts_s:.3f}",
        "-copyts",
        "-i", src,
        "-frames:v", "1",
        "-vf", vf,
        "-q:v", "3",
        out_jpg,
    ]
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        return p.returncode == 0, p.stderr.decode()
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timeout"
    except Exception as e:
        return False, str(e)


def nonblack(jpg: str, min_mean: float = 4, min_var: float = 8) -> bool:
    """
    Check if image is non-black (has sufficient brightness and variance).

    Args:
        jpg: Path to JPEG file
        min_mean: Minimum mean brightness (0-255)
        min_var: Minimum variance

    Returns:
        True if image passes black-frame check
    """
    try:
        im = Image.open(jpg).convert("L")
        st = ImageStat.Stat(im)
        return st.mean[0] >= min_mean and st.var[0] >= min_var
    except Exception:
        return False


def make_track_still(
    src: str,
    t_start: float,
    t_end: float,
    out_jpg: str,
    attempts: tuple = (0.00, 0.12, -0.12, 0.24, -0.24, 0.36, -0.36, 0.48, -0.48),
    w: int = 160,
    h: int = 200,
) -> tuple[bool, Optional[float]]:
    """
    Generate a representative still for a track with black-frame guards.

    Tries multiple timestamps around the track midpoint, skipping black frames.

    Args:
        src: Video file path
        t_start: Track start time in seconds
        t_end: Track end time in seconds
        out_jpg: Output JPEG path
        attempts: Offset attempts from midpoint
        w: Target width
        h: Target height

    Returns:
        Tuple of (success, timestamp_used)
    """
    t_mid = 0.5 * (t_start + t_end)
    os.makedirs(os.path.dirname(out_jpg), exist_ok=True)

    for dt in attempts:
        t = max(0.0, t_mid + dt)
        ok, stderr = ffmpeg_grab(src, t, out_jpg, w, h)

        if ok and nonblack(out_jpg):
            logger.debug(f"✓ Track still at t={t:.3f}s (offset {dt:+.3f}s)")
            return True, t

        if not ok:
            logger.debug(f"✗ ffmpeg failed at t={t:.3f}s: {stderr[:100]}")

    logger.warning(f"Failed to generate still for track ({t_start:.1f}-{t_end:.1f}s)")
    return False, None


def generate_track_stills(
    episode_id: str,
    video_path: str,
    tracks_json_path: str,
    output_dir: Path,
    w: int = 160,
    h: int = 200,
) -> None:
    """
    Generate stills for all tracks in an episode.

    Args:
        episode_id: Episode identifier
        video_path: Path to video file
        tracks_json_path: Path to tracks.json
        output_dir: Output directory (will create frames/tracks/ subdirs)
        w: Target width
        h: Target height
    """
    # Load tracks
    with open(tracks_json_path) as f:
        tracks_data = json.load(f)

    tracks = tracks_data.get("tracks", [])
    if not tracks:
        logger.error("No tracks found in tracks.json")
        return

    logger.info(f"Generating stills for {len(tracks)} tracks from {episode_id}")

    # Create output directories
    frames_dir = output_dir / "frames" / "tracks"
    frames_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    success_count = 0

    for track in tracks:
        track_id = track["track_id"]

        # Get track time range from frames
        frames = track.get("frames", [])
        if not frames:
            logger.warning(f"Track {track_id} has no frames, skipping")
            continue

        # Convert frame timestamps from ms to seconds
        t_start = frames[0]["ts_ms"] / 1000.0
        t_end = frames[-1]["ts_ms"] / 1000.0

        out_jpg = frames_dir / f"{track_id}.jpg"

        success, timestamp = make_track_still(
            video_path,
            t_start,
            t_end,
            str(out_jpg),
            w=w,
            h=h,
        )

        if success:
            success_count += 1
            manifest.append({
                "track_id": track_id,
                "path": str(out_jpg.relative_to(output_dir)),
                "timestamp": ts_fmt(timestamp),
                "timestamp_seconds": timestamp,
            })

            if success_count % 10 == 0:
                logger.info(f"Progress: {success_count}/{len(tracks)} stills generated")

    # Write manifest
    manifest_path = output_dir / "frames" / "frames_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"✅ Generated {success_count}/{len(tracks)} stills")
    logger.info(f"✅ Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate representative stills for tracks using ffmpeg"
    )
    parser.add_argument("episode_id", help="Episode ID (e.g., RHOBH_S05_E15_11052025)")
    parser.add_argument("--width", type=int, default=160, help="Target width (default: 160)")
    parser.add_argument("--height", type=int, default=200, help="Target height (default: 200)")
    parser.add_argument("--data-root", default="data", help="Data root directory")

    args = parser.parse_args()

    # Resolve paths
    data_root = Path(args.data_root)
    harvest_dir = data_root / "harvest" / args.episode_id
    tracks_json_path = harvest_dir / "tracks.json"

    if not tracks_json_path.exists():
        logger.error(f"tracks.json not found: {tracks_json_path}")
        return 1

    # Get video path from episode registry
    from screentime.episode_registry import episode_registry
    episode_data = episode_registry.get_episode(args.episode_id)
    if not episode_data:
        logger.error(f"Episode {args.episode_id} not found in registry")
        return 1

    video_path = str(data_root / episode_data["video_path"])
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return 1

    logger.info(f"Video: {video_path}")
    logger.info(f"Tracks: {tracks_json_path}")
    logger.info(f"Output: {harvest_dir / 'frames' / 'tracks'}")

    generate_track_stills(
        args.episode_id,
        video_path,
        str(tracks_json_path),
        harvest_dir,
        w=args.width,
        h=args.height,
    )

    return 0


if __name__ == "__main__":
    exit(main())
