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
    video_id: str = ""
    uploader_id: str = ""      # channel login on Twitch
    timestamp: float = 0.0     # when the video/stream was published (unix time), if known
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


def download(url: str, workdir: Path, cookies: str = "", section: tuple[float, float] | None = None) -> Source:
    """Download the video (or only `section` = (start, end) seconds of it) to workdir/source.mp4."""
    workdir.mkdir(parents=True, exist_ok=True)
    if section is not None:
        info, video_path = _download_section(url, workdir, cookies, section)
    else:
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
        video_id=str(info.get("id") or ""),
        uploader_id=str(info.get("uploader_id") or info.get("channel_id") or ""),
        timestamp=float(info.get("timestamp") or info.get("release_timestamp") or 0),
        heatmap=list(info.get("heatmap") or []),
    )
    source.subtitles_path = _download_subtitles(url, workdir, cookies, info)
    return source


def _download_section(url: str, workdir: Path, cookies: str, section: tuple[float, float]) -> tuple[dict, Path]:
    """Only [start, end] of the video. HLS streams (Twitch) go straight to ffmpeg with an input
    seek, which jumps to the right 10-second segment; yt-dlp's own range download would read the
    whole stream from the beginning first (hours for a late part). Checked for a video track and
    retried once; a finished section is reused on a re-run."""
    with YoutubeDL(_base_opts(cookies) | {"format": FORMAT}) as ydl:
        info = ydl.extract_info(url, download=False)
    out = workdir / "source.mp4"
    done = workdir / "source.done"
    if done.exists() and out.exists():
        return info, out
    done.unlink(missing_ok=True)
    last_error = None
    for attempt in range(2):
        try:
            if info.get("url") and str(info.get("protocol", "")).startswith("m3u8"):
                hls_section(info["url"], info.get("http_headers") or {}, section, out)
            else:  # e.g. separate video + audio (YouTube): yt-dlp cuts it with HTTP range requests
                _ytdlp_section(url, workdir, cookies, section, out)
            from ..media import probe

            probe(out)  # raises if there is no video track
            done.touch()
            return info, out
        except Exception as exc:  # broken or empty section: try once more from scratch
            last_error = exc
            log.warning("Section download attempt %d failed: %s", attempt + 1, exc)
            out.unlink(missing_ok=True)
    raise RuntimeError(f"Couldn't download {section[0]:.0f}-{section[1]:.0f}s of the video: {last_error}")


def hls_section(playlist: str, headers: dict, section: tuple[float, float], out: Path) -> Path:
    import subprocess

    start, end = section
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if playlist.startswith("http"):
        cmd += ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "30"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += [
        "-ss", f"{start:.3f}", "-i", playlist, "-t", f"{end - start:.3f}",
        "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy", "-bsf:a", "aac_adtstoasc",
        "-movflags", "+faststart", str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=6 * 3600)
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError("ffmpeg: " + " ".join(proc.stderr.strip().splitlines()[-3:]))
    tmp.replace(out)
    return out


def _ytdlp_section(url: str, workdir: Path, cookies: str, section: tuple[float, float], out: Path) -> None:
    from yt_dlp.utils import download_range_func

    opts = _base_opts(cookies) | {
        "format": FORMAT,
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "section.%(ext)s"),
        "download_ranges": download_range_func(None, [(float(section[0]), float(section[1]))]),
        "force_keyframes_at_cuts": False,
    }
    with YoutubeDL(opts) as ydl:
        ydl.download([url])
    found = [p for p in sorted(workdir.glob("section.*")) if p.suffix in {".mp4", ".mkv", ".webm"}]
    if not found:
        raise RuntimeError("yt-dlp wrote no file for the section")
    found[0].replace(out)


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
