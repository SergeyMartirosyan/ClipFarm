"""Thin wrappers around ffmpeg / ffprobe."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class MediaError(RuntimeError):
    pass


def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"{cmd[0]} failed ({proc.returncode}):\n{tail}")
    return proc


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    has_audio: bool


def probe(path: Path) -> VideoInfo:
    proc = run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,width,height:format=duration",
            "-of", "json", str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not video:
        raise MediaError(f"No video stream in {path}")
    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        duration=float(data.get("format", {}).get("duration") or 0),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def extract_audio(src: Path, dst: Path, start: float | None = None, end: float | None = None) -> Path:
    """16 kHz mono WAV: the format Whisper and the loudness scan both use."""
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if end is not None:
        cmd += ["-to", f"{end:.3f}"]
    cmd += ["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return dst


_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


def scene_cuts(src: Path, start: float, end: float, threshold: float = 0.3) -> list[float]:
    """Absolute timestamps of hard cuts between start and end (ffmpeg scene score)."""
    start = max(0.0, start)
    if end <= start:
        return []
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "info", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
            "-i", str(src), "-an",
            "-vf", f"scale=320:-2,select='gt(scene,{threshold})',showinfo",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    return [start + float(m) for m in _PTS_RE.findall(proc.stderr)]
