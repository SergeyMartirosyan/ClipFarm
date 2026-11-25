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
    # Where the downloaded file starts in the full video when only a section was fetched. HLS
    # sections start on a segment boundary, so this can be a few seconds before the asked start.
    section_start: float | None = None


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
    section_start = None
    if section is not None:
        info, video_path, section_start = _download_section(url, workdir, cookies, section)
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
        section_start=section_start,
    )
    source.subtitles_path = _download_subtitles(url, workdir, cookies, info)
    return source


def _download_section(url: str, workdir: Path, cookies: str, section: tuple[float, float]) -> tuple[dict, Path, float]:
    """Only [start, end] of the video. Returns (info, path, where the file really starts).

    HLS streams (Twitch): we read the media playlist ourselves, keep only the ~10 s segments that
    cover the range and hand ffmpeg that short playlist, with no seeking at all. (ffmpeg's own
    seek in a Twitch playlist read hours of stream and still lost the video track.) The file then
    starts at the first kept segment, up to one segment before `start`, and that real start is
    returned so chat/clip times line up exactly. Checked for a video track and retried once; a
    finished section is reused on a re-run."""
    out = workdir / "source.mp4"
    done = workdir / "source.done"
    with YoutubeDL(_base_opts(cookies) | {"format": FORMAT}) as ydl:
        info = ydl.extract_info(url, download=False)
        if done.exists() and out.exists():
            return info, out, _done_start(done, section[0])
        done.unlink(missing_ok=True)
        fmt = pick_hls_video(info)
        last_error = None
        for attempt in range(2):
            try:
                if fmt is not None:
                    text = ydl.urlopen(fmt["url"]).read().decode("utf-8", "replace")
                    playlist, real_start = trim_playlist(text, fmt["url"], section)
                    log.info("Section %s: %s, file starts at %.1fs", section, fmt.get("format_id"), real_start)
                    hls_section(playlist, workdir / "section.m3u8", out)
                else:  # no HLS video variant (e.g. YouTube): yt-dlp cuts it with HTTP range requests
                    real_start = float(section[0])
                    _ytdlp_section(url, workdir, cookies, section, out)
                from ..media import probe

                probe(out)  # raises if there is no video track
                done.write_text(f"{real_start:.3f}")
                return info, out, real_start
            except Exception as exc:  # broken or empty section: try once more from scratch
                last_error = exc
                log.warning("Section download attempt %d failed: %s", attempt + 1, exc)
                out.unlink(missing_ok=True)
    raise RuntimeError(f"Couldn't download {section[0]:.0f}-{section[1]:.0f}s of the video: {last_error}")


def _done_start(done: Path, default: float) -> float:
    try:
        return float(done.read_text().strip())
    except ValueError:
        return float(default)


def pick_hls_video(info: dict) -> dict | None:
    """Best HLS variant with video, up to 1080p; H.264 first (cheap to decode on the VM)."""
    formats = [f for f in info.get("formats") or [info]
               if f.get("url") and str(f.get("protocol", "")).startswith("m3u8")
               and f.get("vcodec") != "none" and (f.get("height") or 0) <= 1080]
    if not formats:
        return None

    def key(f):
        return (str(f.get("vcodec") or "").startswith(("avc1", "h264")), f.get("height") or 0,
                f.get("fps") or 0, f.get("tbr") or 0)

    return max(formats, key=key)


def trim_playlist(text: str, base_url: str, section: tuple[float, float]) -> tuple[str, float]:
    """A standalone VOD playlist with only the segments overlapping `section`, absolute URLs.
    Returns (playlist text, start time of its first segment)."""
    from urllib.parse import urljoin

    if "#EXT-X-STREAM-INF" in text:
        raise RuntimeError("Got a master playlist instead of a video playlist")
    start, end = section
    head = ["#EXTM3U", "#EXT-X-VERSION:6", "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-MEDIA-SEQUENCE:0"]
    body: list[str] = []
    t = 0.0
    seg_len = target = 0.0
    pending: list[str] = []
    first_start: float | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-TARGETDURATION:"):
            target = float(line.split(":", 1)[1] or 0)
        elif line.startswith("#EXT-X-MAP:"):
            uri = line.split('URI="', 1)[1].split('"', 1)[0]
            head.append(line.replace(f'URI="{uri}"', f'URI="{urljoin(base_url, uri)}"'))
        elif line.startswith("#EXTINF:"):
            seg_len = float(line.split(":", 1)[1].split(",", 1)[0])
            pending.append(line)
        elif line.startswith("#EXT-X-DISCONTINUITY") and not line.startswith("#EXT-X-DISCONTINUITY-SEQ"):
            pending.append(line)
        elif not line.startswith("#"):
            seg_start, seg_end = t, t + seg_len
            t = seg_end
            if seg_end > start and seg_start < end:
                if first_start is None:
                    first_start = seg_start
                # Twitch lists muted parts as "-unmuted", which 403s; the "-muted" file exists.
                uri = urljoin(base_url, line).replace("-unmuted.ts", "-muted.ts")
                body += [p for p in pending if first_start != seg_start or not p.startswith("#EXT-X-DISC")] + [uri]
            pending = []
            if seg_start >= end:
                break
    if first_start is None:
        raise RuntimeError(f"The playlist ({t:.0f}s) has no segments in {start:.0f}-{end:.0f}s")
    head.append(f"#EXT-X-TARGETDURATION:{int(max(target, 1))}")
    return "\n".join(head + body + ["#EXT-X-ENDLIST", ""]), first_start


def hls_section(playlist: str, playlist_path: Path, out: Path) -> Path:
    """Copy every segment of a (trimmed) HLS playlist into one mp4, without re-encoding. (A local
    playlist can't carry HTTP headers; Twitch's segment CDN doesn't need any.)"""
    import subprocess

    playlist_path.write_text(playlist)
    tmp = out.with_name(out.stem + ".part.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error",
           "-protocol_whitelist", "file,http,https,tcp,tls,crypto", "-allowed_extensions", "ALL"]
    cmd += [
        "-i", str(playlist_path),
        "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy", "-bsf:a", "aac_adtstoasc",
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
