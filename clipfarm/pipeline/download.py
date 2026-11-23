"""Fetch a video (and whatever free signals the site exposes) with yt-dlp."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

log = logging.getLogger(__name__)

# Best mp4 up to 1080p; falls back to anything up to 1080p, then anything.
FORMAT = "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]/bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"


@dataclass
class Source:
    url: str
    video_path: Path
    title: str
    uploader: str
    duration: float
    extractor: str
    # YouTube "most replayed" graph: [{"start_time", "end_time", "value" 0..1}]
    heatmap: list[dict[str, float]] = field(default_factory=list)
    # json3 subtitle file, if the site had English subtitles
    subtitles_path: Path | None = None


def _base_opts(cookies: str) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 5,
        "fragment_retries": 10,
        "concurrent_fragment_downloads": 4,
    }
    if cookies:
        opts["cookiefile"] = cookies
    return opts


def probe_info(url: str, cookies: str = "") -> dict[str, Any]:
    """Metadata only (no download) — used to reject over-long videos early."""
    with YoutubeDL(_base_opts(cookies)) as ydl:
        return ydl.extract_info(url, download=False)


def download(url: str, workdir: Path, cookies: str = "") -> Source:
    workdir.mkdir(parents=True, exist_ok=True)
    opts = _base_opts(cookies) | {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not video_path.exists():
        candidates = sorted(workdir.glob("source.*"))
        candidates = [p for p in candidates if p.suffix in {".mp4", ".mkv", ".webm"}]
        if not candidates:
            raise RuntimeError("yt-dlp finished but no video file was written")
        video_path = candidates[0]

    source = Source(
        url=url,
        video_path=video_path,
        title=info.get("title") or "",
        uploader=info.get("uploader") or info.get("channel") or info.get("uploader_id") or "",
        duration=float(info.get("duration") or 0),
        extractor=(info.get("extractor_key") or info.get("extractor") or "").lower(),
        heatmap=list(info.get("heatmap") or []),
    )
    source.subtitles_path = _download_subtitles(url, workdir, cookies, info)
    return source


def _download_subtitles(url: str, workdir: Path, cookies: str, info: dict[str, Any]) -> Path | None:
    """English subtitles as json3, separately so a subtitle 429 never kills the job."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    lang = next((l for l in ("en", "en-US", "en-GB") if l in manual), None)
    if lang is None:
        lang = next((l for l in ("en-orig", "en") if l in auto), None)
    if lang is None:
        return None
    opts = _base_opts(cookies) | {
        "skip_download": True,
        "writesubtitles": lang in manual,
        "writeautomaticsub": lang not in manual,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",
        "outtmpl": str(workdir / "subs.%(ext)s"),
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        log.warning("Subtitle download failed, will transcribe instead: %s", exc)
        return None
    found = sorted(workdir.glob("subs*.json3"))
    return found[0] if found else None
