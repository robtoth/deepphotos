#!/usr/bin/env python3
"""Extract still frames from every video in a folder at a fixed time interval.

Usage:
  python scripts/extract_video_frames.py \
      --input-dir /path/to/videos \
      --output-dir /path/to/output \
      --interval 5

Behavior:
- Recursively finds common video files in --input-dir.
- Creates one subfolder per video under --output-dir.
- Extracts one PNG frame every N seconds, starting at 0 seconds.
- Names frames with the source video stem and timestamp in milliseconds.
- Fails clearly if ffmpeg / ffprobe are not installed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpeg", ".mpg",
}


def require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Required binary not found in PATH: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract PNG frames from videos at a fixed interval."
    )
    parser.add_argument("--input-dir", required=True, help="Folder containing videos")
    parser.add_argument("--output-dir", required=True, help="Folder to write extracted frames")
    parser.add_argument(
        "--interval",
        type=float,
        required=True,
        help="Seconds between frames, for example 5 or 2.5",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively search for videos (default: on)",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write all frames directly into output-dir instead of per-video subfolders",
    )
    return parser.parse_args()


def find_videos(input_dir: Path) -> list[Path]:
    videos = [
        path for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return sorted(videos)


def ffprobe_duration_seconds(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")
    payload = json.loads(proc.stdout)
    duration = payload.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError(f"Could not determine duration for {video_path}")
    return float(duration)


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _extract_frames_primary(video_path: Path, output_dir: Path, interval: float) -> int:
    output_pattern = output_dir / f"{video_path.stem}_%010d.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1/{interval}",
        "-frame_pts",
        "1",
        str(output_pattern),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {video_path}: {proc.stderr.strip()}")
    return len(list(output_dir.glob(f"{video_path.stem}_*.png")))


def _extract_single_fallback_frame(video_path: Path, output_dir: Path) -> int:
    output_path = output_dir / f"{video_path.stem}_fallback_0000000000.png"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        "0",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg fallback failed for {video_path}: {proc.stderr.strip()}")
    return 1 if output_path.exists() and output_path.stat().st_size > 0 else 0


def extract_frames(video_path: Path, output_dir: Path, interval: float) -> int:
    duration = ffprobe_duration_seconds(video_path)
    if duration <= 0:
        return 0

    frame_count = _extract_frames_primary(video_path, output_dir, interval)
    if frame_count > 0:
        return frame_count

    # Some videos decode successfully but produce no frames through the fps filter path.
    # Fall back to a direct single-frame grab so valid videos never silently yield zero output.
    return _extract_single_fallback_frame(video_path, output_dir)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    interval = args.interval

    if interval <= 0:
        raise SystemExit("--interval must be greater than 0")
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    require_binary("ffmpeg")
    require_binary("ffprobe")
    ensure_clean_dir(output_dir)

    videos = find_videos(input_dir)
    if not videos:
        print(f"No supported video files found in {input_dir}")
        return 0

    total_frames = 0
    print(f"Found {len(videos)} video(s) in {input_dir}")

    for video in videos:
        target_dir = output_dir if args.flat else output_dir / video.stem
        ensure_clean_dir(target_dir)
        print(f"Extracting frames from: {video}")
        frame_count = extract_frames(video, target_dir, interval)
        total_frames += frame_count
        print(f"  Wrote {frame_count} frame(s) to {target_dir}")

    print(f"Done. Wrote {total_frames} total frame(s).")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
